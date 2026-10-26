"""Secrecy-rate evaluation (paper Eq. 7) and MA feasibility penalty (paper Eq. 20)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

CDTYPE = np.complex128


def snrs(
    H: np.ndarray,
    h_b: np.ndarray,
    h_e: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    sigma_b2: float | np.ndarray,
    sigma_e2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gamma_b, gamma_e) with shapes (...,) and (..., J).

    ``v`` is the RIS vector of the paper, i.e. ``v_n = conj(phi_n)``; therefore
    ``h^H Phi H w = sum_n conj(v_n) * conj(h_n) * (H w)_n``.
    """
    Hw = np.matmul(H, w[..., None])[..., 0]                 # (..., N)
    yb = np.sum(np.conj(v) * np.conj(h_b) * Hw, axis=-1)    # (...)
    ye = np.sum(np.conj(v)[..., None, :] * np.conj(h_e) * Hw[..., None, :], axis=-1)  # (..., J)
    gb = np.abs(yb) ** 2 / sigma_b2
    ge = np.abs(ye) ** 2 / sigma_e2
    return gb, ge


def secrecy_rate(
    H: np.ndarray,
    h_b: np.ndarray,
    h_e: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    sigma_b2: float | np.ndarray,
    sigma_e2: np.ndarray,
    clip: bool = True,
) -> np.ndarray:
    """Colluding-Eve secrecy rate, paper Eq. (7).

    ``clip=True`` applies the [.]^+ operator.  ``clip=False`` returns the raw log-ratio,
    which is what the AO solver monotonically increases.
    """
    gb, ge = snrs(H, h_b, h_e, v, w, sigma_b2, sigma_e2)
    rs = np.log2(1.0 + gb) - np.log2(1.0 + np.sum(ge, axis=-1))
    return np.maximum(rs, 0.0) if clip else rs


# --------------------------------------------------------------------------------------
# MA feasibility
# --------------------------------------------------------------------------------------


def pairwise_distances(T: np.ndarray) -> np.ndarray:
    """(..., M, 2) -> (..., M, M) Euclidean distances."""
    d = T[..., :, None, :] - T[..., None, :, :]
    return np.sqrt(np.sum(d * d, axis=-1))


def spacing_violation(T: np.ndarray, min_spacing: float) -> np.ndarray:
    """sum_{i<j} [ min_spacing - ||t_i - t_j|| ]^+   (paper Eq. 20)."""
    M = T.shape[-2]
    if M < 2:
        return np.zeros(T.shape[:-2])
    D = pairwise_distances(T)
    iu = np.triu_indices(M, k=1)
    viol = np.maximum(min_spacing - D[..., iu[0], iu[1]], 0.0)
    return np.sum(viol, axis=-1)


def region_violation(T: np.ndarray, half: float) -> np.ndarray:
    """sum of coordinate excursions outside the square region [-half, half]^2."""
    over = np.maximum(np.abs(T) - half, 0.0)
    return np.sum(over, axis=(-1, -2))


def is_feasible(T: np.ndarray, min_spacing: float, half: float, tol: float = 1e-9) -> np.ndarray:
    return (spacing_violation(T, min_spacing) <= tol) & (region_violation(T, half) <= tol)


def penalised_objective(
    rs: np.ndarray, T: np.ndarray, min_spacing: float, chi: float
) -> np.ndarray:
    """Paper Eq. (20): r = R_s - chi * sum_{i<j} [lambda/2 - d_ij]^+."""
    return rs - chi * spacing_violation(T, min_spacing)
