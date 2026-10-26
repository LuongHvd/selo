"""Phase P2b -- direct test of H2/H3: is cross-task transfer actually *useful*?

P2 measures whether task landscapes are correlated.  Correlation alone does not imply that an
optimiser can exploit it, so this module runs the smallest experiment that answers the question
directly, before any multitask algorithm is built.

**Matched-budget protocol.**  A multitask setting solves ``S`` tasks with a total budget of
``S * B`` candidate evaluations.  Four ways of spending it are compared:

| scheme | how each task is started |
|---|---|
| `independent` | every task starts from its own random feasible configuration |
| `transfer_seq` | task 1 solved cold; tasks 2..S warm-started at task 1's solution |
| `transfer_random` | every task warm-started at a *randomly chosen* other task's cold solution |
| `transfer_similar` | every task warm-started at the solution of its *most similar* task |

Every scheme uses the identical (1+1) hill climber, the identical inner AO solver and exactly
``S * B`` candidate evaluations, so the only difference is where the search starts.

It also produces the per-pair data for Figure H (transfer gain vs task similarity) and the
negative-transfer statistics of project spec section 18.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ..channel import ChannelBatch, sample_channels
from ..config import ExperimentConfig
from ..ma_constraints import random_feasible
from ..problem import MAPositionProblem
from .p0 import TEST_SEED_BASE, config_key
from .p2 import build_candidates, make_task_channels, make_tasks

FDTYPE = np.float64


def hill_climb(prob: MAPositionProblem, budget: int, rng: np.random.Generator,
               T0: Optional[np.ndarray] = None, sigma0: float = 0.25,
               trace_at: Optional[List[int]] = None):
    """(1+1) hill climber; returns (best_T, best_obj, trace).

    ``T0`` is the (warm) start.  ``trace_at`` gives FE counts at which to record the incumbent.
    """
    half = prob.half
    T = prob.random_T(rng) if T0 is None else np.array(T0, dtype=FDTYPE, copy=True)
    best_obj = prob.evaluate(T).obj
    best_T = T.copy()
    sigma = np.full(prob.R, sigma0 * half)
    trace = {1: best_obj.copy()} if trace_at else {}
    for fe in range(2, budget + 1):
        cand = np.clip(best_T + sigma[:, None, None] * rng.standard_normal(best_T.shape),
                       -half, half)
        obj = prob.evaluate(cand).obj
        better = obj > best_obj
        best_obj = np.where(better, obj, best_obj)
        best_T = np.where(better[:, None, None], cand, best_T)
        sigma = np.clip(np.where(better, sigma * 1.15, sigma * 0.97), 1e-4 * half, half)
        if trace_at and fe in trace_at:
            trace[fe] = best_obj.copy()
    return best_T, best_obj, trace


@dataclass
class TransferResult:
    scheme_obj: Dict[str, np.ndarray]        # scheme -> (S, R) final objective per task
    pair_gain: np.ndarray                    # (S, S, R) gain of warm-starting j from i
    similarity: np.ndarray                   # (R, S, S) Spearman correlation of landscapes
    distance: np.ndarray                     # (R, S, S)
    traces: Dict[str, Dict[int, np.ndarray]]
    meta: Dict


def run_transfer_probe(
    cfg: ExperimentConfig, family: str = "csi_error", eta: float = 0.10,
    n_tasks: int = 8, n_inst: int = 20, budget: int = 100, seed: int = 0,
    K_sim: int = 150,
) -> TransferResult:
    t0 = time.time()
    key = config_key(cfg.system)
    ch = sample_channels(cfg.system, n_inst, TEST_SEED_BASE + key + 31)
    eve, _ = make_tasks(family, cfg, ch, n_tasks, eta, seed + 1)
    chans = make_task_channels(family, cfg, ch, eve, seed + 1)
    probs = [MAPositionProblem(c, cfg) for c in chans]

    # ---- task similarity on a shared probe set (this is what a similarity-aware method
    #      would have to estimate; here we give it the *exact* value, i.e. an upper bound
    #      on what any practical estimator could achieve) --------------------------------
    from .p2 import channel_distances, evaluate_landscapes, pairwise_correlations
    Xs = random_feasible(np.random.default_rng(seed + 77), (n_inst, K_sim), cfg.system.M,
                         cfg.system.min_spacing, cfg.system.region_half)
    _, F_raw = evaluate_landscapes(cfg, chans, Xs)
    sim = pairwise_correlations(F_raw)["spearman"]              # (R, S, S)
    dist = channel_distances(eve)                              # (R, S, S)

    trace_at = sorted(set(np.unique(np.round(np.geomspace(1, budget, 12)).astype(int)).tolist()))

    # ---- cold (independent) solutions -------------------------------------------------
    cold_T, cold_obj, cold_tr = [], [], []
    for s in range(n_tasks):
        rng = np.random.default_rng(seed * 1000 + s)
        T, o, tr = hill_climb(probs[s], budget, rng, trace_at=trace_at)
        cold_T.append(T); cold_obj.append(o); cold_tr.append(tr)
    cold_obj = np.stack(cold_obj)                              # (S, R)

    # ---- all ordered warm starts i -> j ------------------------------------------------
    pair_obj = np.empty((n_tasks, n_tasks, n_inst))
    pair_tr: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {}
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i == j:
                pair_obj[i, j] = cold_obj[j]
                continue
            rng = np.random.default_rng(seed * 7919 + i * 97 + j)
            _, o, tr = hill_climb(probs[j], budget, rng, T0=cold_T[i], trace_at=trace_at)
            pair_obj[i, j] = o
            pair_tr[(i, j)] = tr
    denom = np.maximum(np.abs(cold_obj), 1e-9)
    pair_gain = (pair_obj - cold_obj[None, :, :]) / denom[None, :, :]

    # ---- matched-budget schemes --------------------------------------------------------
    schemes: Dict[str, np.ndarray] = {"independent": cold_obj}
    seq = cold_obj.copy()
    seq[1:] = pair_obj[0, 1:]
    schemes["transfer_seq"] = seq

    rng = np.random.default_rng(seed + 4242)
    rnd = cold_obj.copy()
    for j in range(n_tasks):
        donors = [i for i in range(n_tasks) if i != j]
        d = rng.choice(donors)
        rnd[j] = pair_obj[d, j]
    schemes["transfer_random"] = rnd

    sim_mean = sim.mean(axis=0)                                # (S, S) averaged over instances
    best = cold_obj.copy()
    worst = cold_obj.copy()
    for j in range(n_tasks):
        order = np.argsort(-sim_mean[j])
        donors = [i for i in order if i != j]
        best[j] = pair_obj[donors[0], j]
        worst[j] = pair_obj[donors[-1], j]
    schemes["transfer_similar"] = best
    schemes["transfer_dissimilar"] = worst

    # ---- ALIEN control: warm start from a solution of a *completely unrelated* problem -----
    # (a task built from an independent channel realisation, landscape correlation ~ 0).
    # If this helps as much as the related donors, then whatever benefit "transfer" shows is a
    # warm-start effect, not exploitation of task relatedness.  This control is given one extra
    # cold solve, i.e. a small budget advantage, which only strengthens that conclusion.
    alien_eve, _ = make_tasks("full_channel", cfg, ch, 1, eta, seed + 991)
    alien_ch = make_task_channels("full_channel", cfg, ch, alien_eve, seed + 991)
    alien_prob = MAPositionProblem(alien_ch[0], cfg)
    aT, _, _ = hill_climb(alien_prob, budget, np.random.default_rng(seed + 5150))
    alien = cold_obj.copy()
    for j in range(n_tasks):
        rng = np.random.default_rng(seed * 6151 + j)
        _, o, _ = hill_climb(probs[j], budget, rng, T0=aT)
        alien[j] = o
    schemes["transfer_alien"] = alien

    # ---- RANDOM-RESTART control: same budget, two independent cold runs, keep the better ----
    # isolates "spending the budget in two shorter runs" from "transfer".
    restart = np.empty_like(cold_obj)
    for j in range(n_tasks):
        r1 = np.random.default_rng(seed * 313 + j)
        r2 = np.random.default_rng(seed * 313 + j + 55_555)
        _, o1, _ = hill_climb(probs[j], budget // 2, r1)
        _, o2, _ = hill_climb(probs[j], budget - budget // 2, r2)
        restart[j] = np.maximum(o1, o2)
    schemes["restart_x2"] = restart

    iu = np.triu_indices(n_tasks, k=1)
    g = pair_gain[iu[0], iu[1]].mean(axis=-1)                  # per-pair mean gain
    r_sim = sim.mean(axis=0)[iu[0], iu[1]]
    corr_gain_sim = stats.spearmanr(r_sim, g) if np.std(r_sim) > 1e-12 else (np.nan, np.nan)

    off = ~np.eye(n_tasks, dtype=bool)
    gains_all = pair_gain[off]                                 # (S*(S-1), R)
    med_sim = np.median(sim.mean(axis=0)[off])
    hi = sim.mean(axis=0)[off] >= med_sim
    meta = {
        "family": family, "eta": eta, "n_tasks": n_tasks, "n_inst": n_inst,
        "budget": budget, "seed": seed, "runtime_s": time.time() - t0,
        "mean_similarity": float(sim.mean(axis=0)[off].mean()),
        "mean_pair_gain_pct": float(100 * gains_all.mean()),
        "p_negative_transfer": float(np.mean(gains_all < 0)),
        "p_neg_given_high_sim": float(np.mean(gains_all[hi] < 0)),
        "p_neg_given_low_sim": float(np.mean(gains_all[~hi] < 0)),
        "gain_vs_similarity_spearman": float(corr_gain_sim[0]),
        "gain_vs_similarity_p": float(corr_gain_sim[1]),
        **{f"obj_{k}": float(v.mean()) for k, v in schemes.items()},
    }
    return TransferResult(scheme_obj=schemes, pair_gain=pair_gain, similarity=sim,
                          distance=dist, traces={"cold": cold_tr[0]}, meta=meta)
