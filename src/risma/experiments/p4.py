"""Phase P4 harness: method registry, oracle similarity, and one-cell execution.

A *cell* is (scale, task family, eta, budget, seed).  Within a cell every method sees the identical
task set and the identical instance set (common random numbers), so all comparisons are paired on
(instance, task, seed).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import ExperimentConfig
from ..multitask import TaskSet, build_task_set
from ..optimizers.mtpso import (
    MTResult, PSOConfig, TransferConfig, multitask_pso, pooled_pso,
)

FDTYPE = np.float64

# Instance seeds.  DEV and TEST are disjoint: hyper-parameters are locked on DEV only.
DEV_BASE_SEED = 610_001
TEST_BASE_SEED = 620_001

SCALES: Dict[str, Dict[str, int]] = {
    "CONTROL": dict(M=4, N=16, J=3),     # the reference paper's configuration (control only)
    "L1": dict(M=4, N=8, J=8),           # leakage-limited, primary scale
    "L2": dict(M=8, N=8, J=8),
    "L3": dict(M=6, N=12, J=8),
    "L4": dict(M=8, N=16, J=12),
}


# --------------------------------------------------------------------------------------
# method registry
# --------------------------------------------------------------------------------------


def method_transfer(name: str, hp: Dict) -> Optional[TransferConfig]:
    """Map a method/ablation name to its TransferConfig (None => not a mtpso method)."""
    base = dict(lambda_max=hp["lambda_max"], donor_form=hp["donor_form"], beta=hp["beta"],
                tau=hp["tau"], alpha=hp["alpha"], channel_mode=hp["channel_mode"],
                succ_a=hp["succ_a"], succ_b=hp["succ_b"], forget=hp["forget"])
    table = {
        # ---- headline methods ----------------------------------------------------------
        "Independent-PSO":     dict(base, lambda_max=0.0, similarity="none"),
        "Random-Transfer-PSO": dict(base, similarity="random", use_success_strength=False),
        "Channel-PSO":         dict(base, similarity="channel", use_success_strength=False),
        "Success-PSO":         dict(base, similarity="success", use_success_strength=True),
        "Hybrid-PSO":          dict(base, similarity="hybrid", use_success_strength=True),
        "Oracle-PSO":          dict(base, similarity="oracle", use_success_strength=True),
        # ---- ablations (A1..A10) -------------------------------------------------------
        "A1-independent":      dict(base, lambda_max=0.0, similarity="none"),
        "A2-random-donor":     dict(base, similarity="random", use_success_strength=True),
        "A3-fixed-strength":   dict(base, similarity="random", use_success_strength=False),
        "A4-channel-only":     dict(base, similarity="channel", use_success_strength=False),
        "A5-success-only":     dict(base, similarity="success", use_success_strength=True),
        "A6-hybrid":           dict(base, similarity="hybrid", use_success_strength=True),
        "A7-no-suppression":   dict(base, similarity="hybrid", use_success_strength=False),
        "A8-oracle":           dict(base, similarity="oracle", use_success_strength=True),
        "A9-no-fallback":      dict(base, similarity="hybrid", use_success_strength=True,
                                    a_floor=0.5),
        "A10-displacement":    dict(base, similarity="hybrid", use_success_strength=True,
                                    donor_form="displacement"),
        "A10-gbest":           dict(base, similarity="hybrid", use_success_strength=True,
                                    donor_form="gbest"),
        "A10-inject":          dict(base, similarity="hybrid", use_success_strength=True,
                                    donor_form="inject"),
    }
    if name in ("Pooled-PSO", "MFEA-II"):
        return None
    if name not in table:
        raise ValueError(f"unknown P4 method {name!r}")
    return TransferConfig(**table[name])


HEADLINE_METHODS = ["Independent-PSO", "Pooled-PSO", "Random-Transfer-PSO", "Channel-PSO",
                    "Success-PSO", "Hybrid-PSO", "Oracle-PSO", "MFEA-II"]
CORE_METHODS = ["Independent-PSO", "Random-Transfer-PSO", "Hybrid-PSO", "MFEA-II"]
ABLATIONS = ["A1-independent", "A2-random-donor", "A3-fixed-strength", "A4-channel-only",
             "A5-success-only", "A6-hybrid", "A7-no-suppression", "A8-oracle",
             "A9-no-fallback", "A10-gbest", "A10-displacement", "A10-inject"]


# Locked on the development set only -- see configs/P4_locked_hyperparameters.yaml.
DEFAULT_HP = dict(lambda_max=1.0, donor_form="inject", beta=6.0, tau=1.0, alpha=0.5,
                  channel_mode="eve", succ_a=1.0, succ_b=1.0, forget=1.0,
                  oracle_probe=100)


# --------------------------------------------------------------------------------------
# oracle similarity (CHARGED)
# --------------------------------------------------------------------------------------


def oracle_similarity(ts: TaskSet, cfg: ExperimentConfig, k_probe: int,
                      seed: int) -> Tuple[np.ndarray, int]:
    """Exact landscape rank correlation on a shared probe set.  Returns (sim (R,S,S), FE/task).

    Diagnostic only: it costs ``k_probe`` candidate evaluations per task per instance, which the
    "charged oracle" comparison subtracts from the optimisation budget.
    """
    from scipy import stats

    from ..ma_constraints import random_feasible
    from ..problem import MAPositionProblem

    S, R, M = ts.S, ts.R, ts.M
    rng = np.random.default_rng(900_000 + seed)
    X = random_feasible(rng, (R, k_probe), M, cfg.system.min_spacing, cfg.system.region_half)
    prob = MAPositionProblem(ts.stacked, cfg)
    big = np.broadcast_to(X[None], (S, R, k_probe, M, 2)).reshape(S * R, k_probe, M, 2)
    F = prob.evaluate(big, count=False, keep_wv=False).rs_raw.reshape(S, R, k_probe)
    sim = np.ones((R, S, S), dtype=FDTYPE)
    for r in range(R):
        ranks = np.apply_along_axis(stats.rankdata, 1, F[:, r, :])
        c = np.corrcoef(ranks)
        sim[r] = np.nan_to_num(c, nan=0.0)
    return sim, k_probe


# --------------------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------------------


def run_p4_cell(
    method: str,
    scale: str,
    family: str,
    eta: float,
    budget: int,
    seed: int,
    n_tasks: int = 9,
    n_inst: int = 20,
    hp: Optional[Dict] = None,
    dev: bool = False,
    charge_oracle: bool = False,
    cfg_base: Optional[ExperimentConfig] = None,
) -> Dict:
    """Execute one (method, cell, seed) and return everything the analysis needs."""
    t0 = time.time()
    hp = dict(DEFAULT_HP, **(hp or {}))
    cfg = (cfg_base or ExperimentConfig(name="P4", phase="P4", n_mc=1)).with_system(
        **SCALES[scale])
    base_seed = DEV_BASE_SEED if dev else TEST_BASE_SEED
    ts = build_task_set(cfg, family, eta, n_tasks, n_inst, seed=seed, base_seed=base_seed)

    pso = PSOConfig()
    probe_fe = 0
    oracle_sim = None
    eff_budget = budget

    if method in ("Oracle-PSO", "A8-oracle"):
        oracle_sim, probe_fe = oracle_similarity(ts, cfg, hp["oracle_probe"], seed)
        if charge_oracle:
            eff_budget = max(budget - probe_fe, pso.NP)

    if method == "Pooled-PSO":
        res = pooled_pso(ts, cfg, eff_budget, seed=seed, pso=pso)
    elif method == "MFEA-II":
        from ..optimizers.mfea2 import mfea2
        res = mfea2(ts, cfg, eff_budget, seed=seed, NP_per_task=pso.NP)
    else:
        tr = method_transfer(method, hp)
        res = multitask_pso(ts, cfg, eff_budget, seed=seed, pso=pso, tr=tr,
                            oracle_sim=oracle_sim)

    res.counters["similarity_probe_fe"] = probe_fe
    obj = res.best_obj                                   # (S, R)
    out = {
        "method": method, "scale": scale, "family": family, "eta": eta, "budget": budget,
        "seed": seed, "n_tasks": n_tasks, "n_inst": n_inst, "dev": dev,
        "charge_oracle": charge_oracle,
        "best_obj": obj,
        "anytime": res.anytime.astype(np.float32),
        "anytime_fe": res.anytime_fe.astype(np.float32),
        "lambda_trace": res.lambda_trace.astype(np.float32),
        "donor_entropy": res.donor_entropy.astype(np.float32),
        "labels": ts.labels,
        "channel_dist": ts.channel_distance(hp["channel_mode"]).astype(np.float32),
        "mean_obj": float(np.mean(obj)),
        "q05_obj": float(np.quantile(obj, 0.05)),
        "median_obj": float(np.median(obj)),
        "neg_transfer_rate": res.neg_transfer_rate,
        "n_transfer": res.n_transfer,
        "counters": res.counters,
        "runtime_s": time.time() - t0,
        "hp": hp,
    }
    return out
