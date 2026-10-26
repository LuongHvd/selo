"""Non-parametric significance tests + effect sizes (project spec section 16).

We never report a p-value without an effect size.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta in [-1, 1]: P(x>y) - P(x<y).  Non-parametric effect size."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    # rank-based O((n+m) log(n+m)) formulation
    n, m = len(x), len(y)
    allv = np.concatenate([x, y])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n + m + 1)
    # average ranks for ties
    ranks = stats.rankdata(allv)
    rx = ranks[:n].sum()
    u = rx - n * (n + 1) / 2.0          # Mann-Whitney U for x
    return float(2.0 * u / (n * m) - 1.0)


def magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def paired_compare(a: np.ndarray, b: np.ndarray, name_a="A", name_b="B") -> Dict:
    """Wilcoxon signed-rank on paired per-instance values + matched-pairs effect size."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    assert a.shape == b.shape, "paired comparison needs equal-length samples"
    d = a - b
    nz = np.count_nonzero(d)
    if nz == 0:
        p = 1.0
        stat = 0.0
    else:
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    win = float(np.mean(d > 0))
    loss = float(np.mean(d < 0))
    return {
        "name_a": name_a, "name_b": name_b, "n": int(a.size),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(d.mean()),
        "rel_diff_pct": float(100.0 * d.mean() / abs(b.mean())) if b.mean() != 0 else np.nan,
        "median_diff": float(np.median(d)),
        "wilcoxon_stat": float(stat), "p_value": float(p),
        "win_rate": win, "loss_rate": loss,
        "cliffs_delta": win - loss,
        "effect_magnitude": magnitude(win - loss),
        "ci95_diff": tuple(float(x) for x in bootstrap_ci(d)),
    }


def unpaired_compare(a: np.ndarray, b: np.ndarray, name_a="A", name_b="B") -> Dict:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    stat, p = stats.ranksums(a, b)
    d = cliffs_delta(a, b)
    return {
        "name_a": name_a, "name_b": name_b, "n_a": int(a.size), "n_b": int(b.size),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "rel_diff_pct": float(100.0 * (a.mean() / b.mean() - 1.0)) if b.mean() != 0 else np.nan,
        "ranksum_stat": float(stat), "p_value": float(p),
        "cliffs_delta": d, "effect_magnitude": magnitude(d),
    }


def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    idx = rng.integers(0, x.size, (n_boot, x.size))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarise(x: np.ndarray) -> Dict:
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = bootstrap_ci(x)
    return {
        "mean": float(x.mean()), "std": float(x.std(ddof=1)),
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)), "q95": float(np.quantile(x, 0.95)),
        "ci95_lo": lo, "ci95_hi": hi, "n": int(x.size),
    }
