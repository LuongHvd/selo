"""Non-learning outer optimisers / reference schemes.

* ``ma_random``   -- a single random feasible MA configuration (lower bound on MA schemes)
* ``random_search`` -- budget-matched uniform random search over the MA region
* ``fpa_ris``     -- FPA-RIS: fixed-position UPA at the BS, AO optimises (w, Phi)
* ``ma_rps``      -- MA-RPS: MA positions optimised, RIS phases drawn at random
* ``local_search`` -- (1+1) Gaussian hill climber, a cheap strong reference

Every one of them calls :class:`~risma.problem.MAPositionProblem`, i.e. the same inner AO
solver used by the DRL agents, so the comparison is fair.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..channel import ChannelBatch
from ..config import ExperimentConfig
from ..ma_constraints import fpa_layout, repair
from ..problem import BudgetCounters, MAPositionProblem
from ..secrecy import spacing_violation

FDTYPE = np.float64


@dataclass
class OuterResult:
    """Standard return type of every outer optimiser."""

    T: np.ndarray                     # (R, M, 2) best MA configuration per instance
    rs: np.ndarray                    # (R,) reported secrecy rate at the repaired T
    obj: np.ndarray                   # (R,) penalised objective at the returned T
    counters: BudgetCounters
    anytime: Optional[np.ndarray] = None      # (n_points, R) best-so-far objective
    anytime_fe: Optional[np.ndarray] = None   # (n_points,) candidate FEs consumed
    meta: Dict = None


def _finish(prob: MAPositionProblem, T: np.ndarray, obj: np.ndarray,
            anytime=None, anytime_fe=None, meta=None, t0: float = 0.0) -> OuterResult:
    rep = prob.report_solution(T)
    prob.counters.wall_time = time.time() - t0
    return OuterResult(
        T=rep["T"], rs=rep["rs"], obj=obj, counters=prob.counters,
        anytime=anytime, anytime_fe=anytime_fe,
        meta={**(meta or {}), "raw_feasible_frac": float(np.mean(rep["raw_feasible"]))},
    )


# --------------------------------------------------------------------------------------


def ma_random(prob: MAPositionProblem, seed: int = 0) -> OuterResult:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    T = prob.random_T(rng)
    ev = prob.evaluate(T)
    return _finish(prob, T, ev.obj, t0=t0)


def random_search(
    prob: MAPositionProblem, budget: int, seed: int = 0, chunk: int = 32,
    n_anytime: int = 25,
) -> OuterResult:
    """Uniform random search over feasible MA configurations, ``budget`` candidates/instance."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    R = prob.R
    best_obj = np.full(R, -np.inf)
    best_T = prob.random_T(rng)
    anytime, anytime_fe = [], []
    log_pts = np.unique(np.round(np.geomspace(1, budget, n_anytime)).astype(int))
    done = 0
    while done < budget:
        k = min(chunk, budget - done)
        T = prob.random_T(rng, k=k)
        ev = prob.evaluate(T)
        # running max including the anytime trace
        for i in range(k):
            better = ev.obj[:, i] > best_obj
            best_obj = np.where(better, ev.obj[:, i], best_obj)
            best_T = np.where(better[:, None, None], T[:, i], best_T)
            done += 1
            if done in log_pts:
                anytime.append(best_obj.copy())
                anytime_fe.append(done)
    return _finish(prob, best_T, best_obj,
                   anytime=np.array(anytime), anytime_fe=np.array(anytime_fe), t0=t0)


def local_search(
    prob: MAPositionProblem, budget: int, seed: int = 0, sigma0: float = 0.25,
    n_anytime: int = 25,
) -> OuterResult:
    """(1+1) Gaussian hill climber with 1/5-success-rule step adaptation."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    R = prob.R
    half = prob.half
    T = prob.random_T(rng)
    ev = prob.evaluate(T)
    best_obj = ev.obj.copy()
    best_T = T.copy()
    sigma = np.full(R, sigma0 * half)
    anytime, anytime_fe = [], []
    log_pts = set(np.unique(np.round(np.geomspace(1, max(budget, 2), n_anytime)).astype(int)))
    for step in range(2, budget + 1):
        cand = np.clip(best_T + sigma[:, None, None] * rng.standard_normal(best_T.shape),
                       -half, half)
        ev = prob.evaluate(cand)
        better = ev.obj > best_obj
        best_obj = np.where(better, ev.obj, best_obj)
        best_T = np.where(better[:, None, None], cand, best_T)
        sigma = np.where(better, sigma * 1.15, sigma * 0.97)
        sigma = np.clip(sigma, 1e-4 * half, half)
        if step in log_pts:
            anytime.append(best_obj.copy())
            anytime_fe.append(step)
    return _finish(prob, best_T, best_obj,
                   anytime=np.array(anytime), anytime_fe=np.array(anytime_fe), t0=t0)


def fpa_ris(ch: ChannelBatch, cfg: ExperimentConfig) -> OuterResult:
    """FPA-RIS: fixed UPA antenna positions, AO optimises beamforming + RIS phases."""
    t0 = time.time()
    prob = MAPositionProblem(ch, cfg)
    T0 = fpa_layout(cfg.system.M, cfg.system.wavelength, cfg.system.region_half)
    T = np.broadcast_to(T0, (ch.R, cfg.system.M, 2)).copy()
    ev = prob.evaluate(T)
    return _finish(prob, T, ev.obj, t0=t0, meta={"scheme": "FPA-RIS"})


def ma_rps_random(ch: ChannelBatch, cfg: ExperimentConfig, budget: int, seed: int = 0
                  ) -> OuterResult:
    """MA-RPS: MA positions searched, RIS phases random (not optimised)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    v_fixed = np.exp(1j * rng.uniform(0, 2 * np.pi, (ch.R, cfg.system.N)))
    prob = MAPositionProblem(ch, cfg, optimize_phase=False, v_fixed=v_fixed)
    res = random_search(prob, budget, seed=seed + 1)
    res.meta = {**(res.meta or {}), "scheme": "MA-RPS"}
    return res
