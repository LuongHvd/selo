"""Robust objectives and evaluation metrics for the RMO branch.

Optimisation uses **only** ``F_mean`` and ``F_tail = LCVaR_alpha``; everything else in
:func:`robust_summary` is an evaluation metric.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

FDTYPE = np.float64

ALPHA = 0.10
SOP_THRESHOLDS: Sequence[float] = (0.5, 1.0, 2.0, 3.0)


def lcvar(rs: np.ndarray, alpha: float = ALPHA, axis: int = 0) -> np.ndarray:
    """Empirical lower-tail CVaR: the mean of the ``K = max(1, ceil(alpha S))`` smallest samples.

    Both objectives are maximised, so a *larger* LCVaR is better.
    """
    S = rs.shape[axis]
    K = max(1, int(np.ceil(alpha * S)))
    part = np.partition(rs, K - 1, axis=axis)
    sl = [slice(None)] * rs.ndim
    sl[axis] = slice(0, K)
    return np.mean(part[tuple(sl)], axis=axis)


def f_mean(rs: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.mean(rs, axis=axis)


def objectives(rs: np.ndarray, alpha: float = ALPHA, axis: int = 0) -> np.ndarray:
    """Stack the two maximised objectives: returns ``(..., 2)`` = [F_mean, F_tail]."""
    return np.stack([f_mean(rs, axis=axis), lcvar(rs, alpha, axis=axis)], axis=-1)


def sop(rs: np.ndarray, threshold: float, axis: int = 0) -> np.ndarray:
    """Secrecy outage probability  Pr[R_s < R_th]."""
    return np.mean(rs < threshold, axis=axis)


def robust_summary(rs: np.ndarray, alpha: float = ALPHA,
                   thresholds: Sequence[float] = SOP_THRESHOLDS,
                   axis: int = 0) -> Dict[str, np.ndarray]:
    """Full distributional summary of a design's secrecy rate over a scenario set."""
    out = {
        "mean_Rs": f_mean(rs, axis=axis),
        "LCVaR": lcvar(rs, alpha, axis=axis),
        "Q05": np.quantile(rs, 0.05, axis=axis),
        "median_Rs": np.median(rs, axis=axis),
        "min_Rs": np.min(rs, axis=axis),
        "std_Rs": np.std(rs, axis=axis),
    }
    for t in thresholds:
        out[f"SOP_{t:g}"] = sop(rs, t, axis=axis)
    return out


# --------------------------------------------------------------------------------------
# Pareto helpers (maximisation)
# --------------------------------------------------------------------------------------


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    """``F`` is ``(n, m)`` of **maximised** objectives; returns the non-dominated boolean mask."""
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    order = np.argsort(-F[:, 0], kind="mergesort")
    for i in order:
        if not keep[i]:
            continue
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        keep &= ~dominated
        keep[i] = True
    return keep


def pareto_front(F: np.ndarray, X: np.ndarray = None):
    m = nondominated_mask(F)
    return (F[m], None if X is None else X[m], m)
