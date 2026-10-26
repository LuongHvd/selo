"""Phase P1 -- robust MA optimisation under imperfect eavesdropper CSI.

Five *design strategies* are compared, chosen so that the contribution of robustifying the
inner variables ``(w, Phi)`` can be separated from the contribution of robustifying the outer
variable ``T``:

| strategy | inner AO sees | outer score on design scenarios |
|---|---|---|
| `nominal`    | the estimate `hhat_e` only            | `R_s` at `hhat_e` (P0 design, uncertainty-blind) |
| `inner_only` | all `S_design` scenarios (averaged)   | `R_s` at `hhat_e` |
| `mean`       | all `S_design` scenarios (averaged)   | mean over scenarios |
| `worst`      | all `S_design` scenarios (averaged)   | min over scenarios (sample worst case) |
| `cvar10`     | all `S_design` scenarios (averaged)   | CVaR at alpha = 0.10 |

Every strategy is optimised by the *same* outer optimiser with the *same* candidate-FE budget,
and every result is finally scored on `S_test = 1000` **locked, unseen** uncertainty realisations.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from ..channel import sample_channels
from ..config import ExperimentConfig
from ..ma_constraints import repair
from ..problem import BudgetCounters
from ..robust import RobustMAProblem, aggregate_robust
from ..uncertainty import EveUncertainty, design_scenarios, locked_test_scenarios
from .p0 import TEST_SEED_BASE, config_key

FDTYPE = np.float64

STRATEGIES = {
    #  name          -> (objective, inner uses all scenarios)
    "nominal":    ("nominal", False),
    "inner_only": ("nominal", True),
    "mean":       ("mean", True),
    "worst":      ("worst", True),
    "cvar10":     ("cvar", True),
}

SOP_THRESHOLDS = (0.5, 1.0, 2.0, 3.0, 4.0)


# --------------------------------------------------------------------------------------
# outer optimisers (identical code paths as P0, but driven by the robust problem)
# --------------------------------------------------------------------------------------


def _local_search(prob, budget: int, seed: int, sigma0: float = 0.25):
    rng = np.random.default_rng(seed)
    half = prob.half
    T = prob.random_T(rng)
    best_obj = prob.design(T).obj
    best_T = T.copy()
    sigma = np.full(prob.R, sigma0 * half)
    for _ in range(budget - 1):
        cand = np.clip(best_T + sigma[:, None, None] * rng.standard_normal(best_T.shape),
                       -half, half)
        obj = prob.design(cand).obj
        better = obj > best_obj
        best_obj = np.where(better, obj, best_obj)
        best_T = np.where(better[:, None, None], cand, best_T)
        sigma = np.clip(np.where(better, sigma * 1.15, sigma * 0.97), 1e-4 * half, half)
    return best_T, best_obj


def _random_search(prob, budget: int, seed: int, chunk: int = 25):
    rng = np.random.default_rng(seed)
    best_obj = np.full(prob.R, -np.inf)
    best_T = prob.random_T(rng)
    done = 0
    while done < budget:
        k = min(chunk, budget - done)
        T = prob.random_T(rng, k=k)
        obj = prob.design(T).obj
        for i in range(k):
            better = obj[:, i] > best_obj
            best_obj = np.where(better, obj[:, i], best_obj)
            best_T = np.where(better[:, None, None], T[:, i], best_T)
        done += k
    return best_T, best_obj


OPTIMIZERS = {"LS": _local_search, "RS": _random_search}


# --------------------------------------------------------------------------------------


def run_p1(
    strategy: str,
    unc_model: str,
    eta: float,
    cfg: ExperimentConfig,
    seed: int,
    S_design: int = 20,
    S_test: int = 1000,
    budget: int = 300,
    optimizer: str = "LS",
    keep_rs_test: bool = False,
) -> Dict:
    """One P1 run: design under uncertainty, then score on the locked test scenarios."""
    t0 = time.time()
    objective, use_all_inner = STRATEGIES[strategy]
    key = config_key(cfg.system)
    ch = sample_channels(cfg.system, cfg.n_mc, TEST_SEED_BASE + key)
    unc = EveUncertainty(unc_model, eta)

    d_eve = design_scenarios(ch, unc, S_design, key, seed)
    t_eve = locked_test_scenarios(ch, unc, S_test, key)

    prob = RobustMAProblem(
        ch, cfg, d_eve, objective=objective, alpha=0.10,
        inner_scenarios=None if use_all_inner else 1,
        score_on_nominal=(objective == "nominal"),
    )
    best_T, best_obj = OPTIMIZERS[optimizer](prob, budget, seed)
    rep = prob.report_on_test(best_T, t_eve, thresholds=SOP_THRESHOLDS, keep_rs=keep_rs_test)

    return {
        "strategy": strategy, "objective": objective, "unc_model": unc_model, "eta": eta,
        "S_design": S_design, "S_test": S_test, "budget": budget, "optimizer": optimizer,
        "seed": seed,
        "T": repair(best_T, cfg.system.min_spacing, cfg.system.region_half),
        "design_obj": best_obj,
        "test_mean": rep.mean, "test_median": rep.median, "test_q05": rep.q05,
        "test_cvar10": rep.cvar10,
        "test_sop": {str(k): v for k, v in rep.sop.items()},
        "rs_test": rep.rs_test,
        "counters": prob.counters.as_dict(),
        "runtime_s": time.time() - t0,
        "n_mc": cfg.n_mc,
    }
