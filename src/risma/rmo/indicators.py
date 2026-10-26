"""Multi-objective quality indicators, written for **maximisation** of both objectives.

Both RMO objectives (`F_mean`, `F_tail = LCVaR`) are maximised, so every indicator here takes
maximised objective vectors directly; nothing is silently negated.

  * `hypervolume_2d` -- exact 2-D hypervolume dominated by a front w.r.t. a reference point that
    must be *dominated by* every point considered (i.e. component-wise no larger).
  * `igd_plus`       -- the Pareto-compliant modification of IGD (Ishibuchi et al., 2015), adapted
    to maximisation: only the components where the approximation is *worse* than the reference
    point contribute.
  * `epsilon_additive` -- the additive epsilon indicator: how much every reference point would have
    to be weakened for the approximation to dominate it.
  * `reference_front` -- the non-dominated union of several fronts, used when no analytical front
    exists (always the case here).

Smaller is better for `igd_plus` and `epsilon_additive`; larger is better for the hypervolume.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .objectives import nondominated_mask

FDTYPE = np.float64


def hypervolume_2d(F: np.ndarray, ref: Sequence[float]) -> float:
    """Exact hypervolume of a 2-objective **maximisation** front.

    ``F`` is ``(n, 2)``; ``ref`` is the reference (nadir-side) point. Points that do not dominate
    the reference point contribute nothing.
    """
    F = np.atleast_2d(np.asarray(F, dtype=FDTYPE))
    if F.size == 0:
        return 0.0
    r = np.asarray(ref, dtype=FDTYPE)
    keep = np.all(F >= r, axis=1)
    F = F[keep]
    if F.size == 0:
        return 0.0
    F = F[nondominated_mask(F)]
    order = np.lexsort((-F[:, 1], -F[:, 0]))            # f1 descending, f2 descending on ties
    F = F[order]
    hv = 0.0
    prev_f2 = r[1]
    for f1, f2 in F:
        if f2 <= prev_f2:
            continue
        hv += (f1 - r[0]) * (f2 - prev_f2)
        prev_f2 = f2
    return float(hv)


def igd_plus(F: np.ndarray, ref_front: np.ndarray) -> float:
    """IGD+ for maximisation: mean over reference points of the modified distance to ``F``."""
    F = np.atleast_2d(np.asarray(F, dtype=FDTYPE))
    Z = np.atleast_2d(np.asarray(ref_front, dtype=FDTYPE))
    if F.size == 0:
        return float("inf")
    # only the shortfall of the approximation below a reference point counts
    diff = np.maximum(Z[:, None, :] - F[None, :, :], 0.0)      # (|Z|, |F|, m)
    d = np.sqrt(np.sum(diff ** 2, axis=-1))
    return float(np.mean(np.min(d, axis=1)))


def epsilon_additive(F: np.ndarray, ref_front: np.ndarray) -> float:
    """Additive epsilon for maximisation: ``max_z min_a max_m (z_m - a_m)``."""
    F = np.atleast_2d(np.asarray(F, dtype=FDTYPE))
    Z = np.atleast_2d(np.asarray(ref_front, dtype=FDTYPE))
    if F.size == 0:
        return float("inf")
    d = np.max(Z[:, None, :] - F[None, :, :], axis=-1)
    return float(np.max(np.min(d, axis=1)))


def reference_front(fronts: Iterable[np.ndarray]) -> np.ndarray:
    """Non-dominated union of several approximation fronts (maximisation)."""
    stack = [np.atleast_2d(np.asarray(f, dtype=FDTYPE)) for f in fronts if np.size(f)]
    if not stack:
        return np.zeros((0, 2), dtype=FDTYPE)
    U = np.unique(np.concatenate(stack, axis=0), axis=0)
    return U[nondominated_mask(U)]


def nadir_reference_point(fronts: Iterable[np.ndarray],
                          margin: float = 0.0) -> np.ndarray:
    """A common HV reference point: component-wise minimum over all fronts, minus ``margin``.

    Using one reference point for every method in a comparison is what makes their hypervolumes
    comparable; it must be fixed from the pooled data **before** any method is ranked.
    """
    stack = [np.atleast_2d(np.asarray(f, dtype=FDTYPE)) for f in fronts if np.size(f)]
    if not stack:
        return np.zeros(2, dtype=FDTYPE)
    U = np.concatenate(stack, axis=0)
    return np.min(U, axis=0) - margin


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance on one non-dominated set (orientation-independent)."""
    F = np.atleast_2d(np.asarray(F, dtype=FDTYPE))
    n, m = F.shape
    d = np.zeros(n, dtype=FDTYPE)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(F[:, j])
        fj = F[order, j]
        d[order[0]] = d[order[-1]] = np.inf
        span = fj[-1] - fj[0]
        if span <= 0:
            continue
        d[order[1:-1]] += (fj[2:] - fj[:-2]) / span
    return d


def fast_nondominated_sort(F: np.ndarray) -> np.ndarray:
    """Pareto-front index of every row (0 = best front), for **maximisation**."""
    F = np.atleast_2d(np.asarray(F, dtype=FDTYPE))
    n = F.shape[0]
    rank = np.full(n, -1, dtype=int)
    remaining = np.ones(n, dtype=bool)
    r = 0
    while np.any(remaining):
        idx = np.flatnonzero(remaining)
        sub = F[idx]
        front = nondominated_mask(sub)
        rank[idx[front]] = r
        remaining[idx[front]] = False
        r += 1
    return rank
