"""Movable-antenna geometric constraints C3 (min spacing) and C4 (feasible region).

Provides
  * feasible random initialisation (rejection + repair),
  * a deterministic repair operator used *identically* by every outer optimiser,
  * the fixed-position-antenna (FPA) reference layout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .secrecy import pairwise_distances, spacing_violation

FDTYPE = np.float64


def clip_region(T: np.ndarray, half: float) -> np.ndarray:
    return np.clip(T, -half, half)


def repair(T: np.ndarray, min_spacing: float, half: float, iters: int = 200) -> np.ndarray:
    """Push antennas apart until C3 holds, keeping them inside C4.

    A simple projection-style loop: at each step every violating pair is separated along
    the line joining them, then all coordinates are clipped back into the region.
    Deterministic, batched over leading dims, and a no-op on already feasible inputs.
    """
    T = clip_region(np.array(T, dtype=FDTYPE, copy=True), half)
    M = T.shape[-2]
    if M < 2:
        return T
    for _ in range(iters):
        diff = T[..., :, None, :] - T[..., None, :, :]              # (..., M, M, 2)
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        eye = np.eye(M, dtype=bool)
        dist = np.where(eye, np.inf, dist)
        viol = min_spacing - dist
        if np.all(viol <= 1e-12):
            break
        need = viol > 1e-12
        # direction: if two antennas coincide, use a deterministic tie-breaking direction
        safe = np.maximum(dist, 1e-12)
        unit = diff / safe[..., None]
        degen = dist < 1e-12
        if np.any(degen):
            idx = np.arange(M)
            fallback = np.stack(
                [np.cos(idx * 2.399963), np.sin(idx * 2.399963)], axis=-1
            )                                                       # golden-angle spread
            fb = np.broadcast_to(fallback[:, None, :], unit.shape)
            unit = np.where(degen[..., None], fb, unit)
        push = 0.5 * np.where(need, viol, 0.0)[..., None] * unit    # (..., M, M, 2)
        T = T + np.sum(push, axis=-2) * 1.05
        T = clip_region(T, half)
    return T


def random_feasible(
    rng: np.random.Generator, shape_lead, M: int, min_spacing: float, half: float,
    max_tries: int = 100,
) -> np.ndarray:
    """Uniform-in-region sample repaired to satisfy C3.  Shape (*shape_lead, M, 2)."""
    lead = tuple(shape_lead)
    T = rng.uniform(-half, half, lead + (M, 2))
    T = repair(T, min_spacing, half)
    return T


def fpa_layout(M: int, wavelength: float, half: float, spacing_factor: float = 0.5) -> np.ndarray:
    """Fixed-position antenna reference layout: centred UPA with lambda/2 spacing.

    Used by the FPA-RIS baseline.  If the nominal array does not fit inside the region it is
    shrunk isotropically so that C4 still holds (this only happens for very small A).
    """
    from .channel import factor_upa

    rows, cols = factor_upa(M)
    d = spacing_factor * wavelength
    xs = (np.arange(cols, dtype=FDTYPE) - (cols - 1) / 2.0) * d
    ys = (np.arange(rows, dtype=FDTYPE) - (rows - 1) / 2.0) * d
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    T = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    ext = np.max(np.abs(T)) if M > 1 else 0.0
    if ext > half + 1e-12:
        T = T * (half / ext)
    return T


def feasibility_report(T: np.ndarray, min_spacing: float, half: float) -> dict:
    sv = spacing_violation(T, min_spacing)
    over = np.maximum(np.abs(T) - half, 0.0).sum(axis=(-1, -2))
    return {
        "max_spacing_violation": float(np.max(sv)),
        "frac_spacing_infeasible": float(np.mean(sv > 1e-9)),
        "max_region_violation": float(np.max(over)),
        "frac_region_infeasible": float(np.mean(over > 1e-9)),
    }
