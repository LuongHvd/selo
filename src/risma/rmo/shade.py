"""SHADE, L-SHADE and L-SHADE+Normal for Phase R0.

These are drop-in replacements for ``risma.optimizers.evolutionary.{de,pso,cmaes}``: same
signature ``(prob, budget, seed, ...) -> OuterResult``, same objective, same constraint handling,
same budget accounting, vectorised over the instance axis of a ``MAPositionProblem``.

Canonical components (Tanabe & Fukunaga, CEC 2013 / CEC 2014):
  * current-to-pbest/1 mutation with an external archive,
  * binomial crossover,
  * success-history memories for F and CR with weighted Lehmer means,
  * p-best selection,
  * linear population-size reduction (L-SHADE only),
  * the midpoint bound-repair rule of the original papers.

Small-budget adaptation (documented deviation): the published setting ``NP_init = 18 D`` with
``max_FE = 10^4 D`` gives only a handful of generations at the budgets used here (1 000 / 5 000).
We therefore use ``NP_init = min(18 D, max(20, budget // 25))``, which guarantees >= 25 generations
at every budget screened while never exceeding the published initial size. Everything else is
canonical.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..optimizers.baselines import OuterResult, _finish
from ..optimizers.evolutionary import _Tracker, _as_T, _eval
from ..problem import MAPositionProblem

FDTYPE = np.float64


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------


@dataclass
class ShadeConfig:
    H: int = 6                 # success-history memory size
    p_best: float = 0.11       # p-best fraction (L-SHADE); SHADE samples p in [2/NP, 0.2]
    r_arc: float = 2.6         # archive size = r_arc * NP  (L-SHADE); SHADE uses 1.0
    NP_min: int = 4            # terminal population size for linear reduction
    bound_rule: str = "midpoint"   # "midpoint" (canonical) | "clip"
    # --- L-SHADE + Normal only -------------------------------------------------------
    local_frac: float = 0.15   # fraction of the generation replaced by a local Gaussian sample
    sigma0: float = 0.20       # initial local step, as a fraction of the box half-width
    sigma_end: float = 0.02


def _np_init(D: int, budget: int, lshade: bool = True) -> int:
    """``NP_init = min(18 D, max(20, budget // 25))`` -- see the module docstring."""
    return int(min(18 * D, max(20, budget // 25)))


def _weighted_lehmer(vals: np.ndarray, wts: np.ndarray) -> float:
    w = wts / max(wts.sum(), 1e-300)
    num = float(np.sum(w * vals * vals))
    den = float(np.sum(w * vals))
    return num / den if den > 1e-300 else 0.0


def _bound_repair(v: np.ndarray, x: np.ndarray, lo: float, hi: float, rule: str) -> np.ndarray:
    if rule == "clip":
        return np.clip(v, lo, hi)
    # canonical SHADE/L-SHADE midpoint rule
    v = np.where(v < lo, 0.5 * (lo + x), v)
    v = np.where(v > hi, 0.5 * (hi + x), v)
    return np.clip(v, lo, hi)      # guards the pathological case x itself out of bounds


# --------------------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------------------


def _shade_core(
    prob: MAPositionProblem, budget: int, seed: int, cfg: ShadeConfig,
    lshade: bool, local: bool, probe: Optional[Dict] = None,
) -> OuterResult:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    R, M, half = prob.R, prob.M, prob.half
    D = 2 * M
    lo, hi = -half, half

    NP0 = _np_init(D, budget, lshade)
    NP0 = max(NP0, 5)
    NP = NP0
    Amax = max(int(round(cfg.r_arc * NP0)), 1) if lshade else NP0

    tr = _Tracker(R, budget)
    pop = prob.random_T(rng, k=NP).reshape(R, NP, D)
    fit = _eval(prob, pop, M, half)
    tr.update(pop, fit)

    arch = np.zeros((R, Amax, D), dtype=FDTYPE)
    acnt = np.zeros(R, dtype=int)
    MF = np.full((R, cfg.H), 0.5)
    MCR = np.full((R, cfg.H), 0.5)
    kmem = np.zeros(R, dtype=int)

    gen = 0
    while tr.used < budget:
        gen += 1
        frac = tr.used / max(budget, 1)
        # ---- sample F and CR from the success-history memories --------------------------
        ridx = rng.integers(0, cfg.H, (R, NP))
        mcr = np.take_along_axis(MCR, ridx, axis=1)
        mf = np.take_along_axis(MF, ridx, axis=1)
        CR = np.where(mcr < 0, 0.0, np.clip(rng.normal(mcr, 0.1), 0.0, 1.0))
        # Cauchy(mf, 0.1) truncated to (0, 1]; resample non-positive values
        draw = mf + 0.1 * np.tan(np.pi * (rng.random((R, NP)) - 0.5))
        for _ in range(12):
            bad = draw <= 0.0
            if not np.any(bad):
                break
            draw = np.where(bad, mf + 0.1 * np.tan(np.pi * (rng.random((R, NP)) - 0.5)), draw)
        F = np.clip(np.where(draw <= 0.0, 0.05, draw), 1e-6, 1.0)

        # ---- current-to-pbest/1 with archive --------------------------------------------
        order = np.argsort(-fit, axis=1)                                  # best first
        if lshade:
            p_num = max(2, int(round(cfg.p_best * NP)))
            pb_rank = rng.integers(0, p_num, (R, NP))
        else:
            p_i = rng.uniform(2.0 / NP, 0.2, (R, NP))
            p_num_i = np.maximum(2, np.round(p_i * NP).astype(int))
            pb_rank = (rng.random((R, NP)) * p_num_i).astype(int)
        pb_idx = np.take_along_axis(order, pb_rank, axis=1)
        xpb = np.take_along_axis(pop, pb_idx[..., None], axis=1)

        r1 = rng.integers(0, NP, (R, NP))
        tot = NP + acnt                                                   # (R,)
        r2raw = (rng.random((R, NP)) * tot[:, None]).astype(int)
        xr1 = np.take_along_axis(pop, r1[..., None], axis=1)
        xr2 = np.empty_like(xr1)
        for r in range(R):
            fromP = r2raw[r] < NP
            xr2[r][fromP] = pop[r][r2raw[r][fromP]]
            if np.any(~fromP):
                xr2[r][~fromP] = arch[r][r2raw[r][~fromP] - NP]

        v = pop + F[..., None] * (xpb - pop) + F[..., None] * (xr1 - xr2)
        v = _bound_repair(v, pop, lo, hi, cfg.bound_rule)

        jr = rng.integers(0, D, (R, NP))
        cross = rng.random((R, NP, D)) < CR[..., None]
        cross[np.arange(R)[:, None], np.arange(NP)[None, :], jr] = True
        u = np.where(cross, v, pop)

        # ---- optional lightweight Gaussian refinement (L-SHADE + Normal) ---------------
        n_loc = 0
        if local:
            n_loc = int(round(cfg.local_frac * NP))
            if n_loc > 0:
                sigma = cfg.sigma0 + (cfg.sigma_end - cfg.sigma0) * frac
                spread = np.std(pop, axis=1, keepdims=True)               # (R,1,D)
                scale = np.maximum(spread, 1e-6 * half)
                pick = rng.integers(0, NP, (R, n_loc))
                best_idx = order[:, 0]                                     # (R,)
                elite = np.take_along_axis(pop, best_idx[:, None, None], axis=1)   # (R,1,D)
                loc = elite + sigma * scale * rng.standard_normal((R, n_loc, D))
                loc = np.clip(loc, lo, hi)
                for r in range(R):
                    u[r, pick[r]] = loc[r]

        take = min(NP, budget - tr.used)
        u = u[:, :take]
        fu = _eval(prob, u, M, half)
        tr.update(u, fu)

        # ---- selection + memory / archive update ----------------------------------------
        better = fu > fit[:, :take]
        delta = np.maximum(fu - fit[:, :take], 0.0)
        for r in range(R):
            win = np.flatnonzero(better[r])
            if win.size:
                for i in win:                                             # archive the parent
                    if acnt[r] < Amax:
                        arch[r, acnt[r]] = pop[r, i]
                        acnt[r] += 1
                    else:
                        arch[r, rng.integers(0, Amax)] = pop[r, i]
                sF, sCR, wts = F[r, win], CR[r, win], delta[r, win]
                if wts.sum() > 1e-300:
                    k = kmem[r]
                    MF[r, k] = _weighted_lehmer(sF, wts)
                    if MCR[r, k] < 0 or np.max(sCR) < 1e-12:
                        MCR[r, k] = -1.0
                    else:
                        MCR[r, k] = _weighted_lehmer(sCR, wts)
                    kmem[r] = (k + 1) % cfg.H
        pop[:, :take] = np.where(better[..., None], u, pop[:, :take])
        fit[:, :take] = np.where(better, fu, fit[:, :take])

        if probe is not None:
            probe.setdefault("NP", []).append(int(NP))
            probe.setdefault("Amax", []).append(int(Amax))
            probe.setdefault("acnt", []).append(acnt.copy())
            probe.setdefault("MF", []).append(MF.copy())
            probe.setdefault("MCR", []).append(MCR.copy())
            probe.setdefault("fe", []).append(int(tr.used))
            probe.setdefault("u_min", []).append(float(u.min()))
            probe.setdefault("u_max", []).append(float(u.max()))
            probe.setdefault("n_local", []).append(int(n_loc))

        # ---- linear population-size reduction (L-SHADE only) ----------------------------
        if lshade:
            NP_new = int(round((cfg.NP_min - NP0) * (tr.used / max(budget, 1)) + NP0))
            NP_new = max(cfg.NP_min, min(NP_new, NP))
            if probe is not None:
                probe.setdefault("NP_after", []).append(int(NP_new))
                probe.setdefault("fe_after", []).append(int(tr.used))
            if NP_new < NP:
                keep = np.argsort(-fit, axis=1)[:, :NP_new]
                pop = np.take_along_axis(pop, keep[..., None], axis=1)
                fit = np.take_along_axis(fit, keep, axis=1)
                NP = NP_new
                Amax_new = max(int(round(cfg.r_arc * NP)), 1)
                if Amax_new < Amax:
                    arch = arch[:, :Amax_new]
                    acnt = np.minimum(acnt, Amax_new)
                    Amax = Amax_new

    name = ("L-SHADE+Normal" if local else ("L-SHADE" if lshade else "SHADE"))
    return _finish(prob, _as_T(tr.best_x, M), tr.best_obj,
                   anytime=np.array(tr.anytime), anytime_fe=np.array(tr.anytime_fe),
                   meta={"optimizer": name, "NP_init": NP0, "NP_final": NP,
                         "H": cfg.H, "p_best": cfg.p_best, "r_arc": cfg.r_arc,
                         "bound_rule": cfg.bound_rule, "generations": gen,
                         "local_frac": cfg.local_frac if local else 0.0,
                         "sigma0": cfg.sigma0 if local else 0.0}, t0=t0)


# --------------------------------------------------------------------------------------
# public entry points (drop-in for evolutionary.{de,pso,cmaes})
# --------------------------------------------------------------------------------------


def shade(prob: MAPositionProblem, budget: int, seed: int = 0,
          cfg: Optional[ShadeConfig] = None, probe: Optional[Dict] = None) -> OuterResult:
    c = cfg or ShadeConfig(r_arc=1.0)
    return _shade_core(prob, budget, seed, c, lshade=False, local=False, probe=probe)


def lshade(prob: MAPositionProblem, budget: int, seed: int = 0,
           cfg: Optional[ShadeConfig] = None, probe: Optional[Dict] = None) -> OuterResult:
    return _shade_core(prob, budget, seed, cfg or ShadeConfig(), lshade=True, local=False,
                       probe=probe)


def lshade_normal(prob: MAPositionProblem, budget: int, seed: int = 0,
                  cfg: Optional[ShadeConfig] = None,
                  probe: Optional[Dict] = None) -> OuterResult:
    return _shade_core(prob, budget, seed, cfg or ShadeConfig(), lshade=True, local=True,
                       probe=probe)


def lshade_clip(prob: MAPositionProblem, budget: int, seed: int = 0,
                cfg: Optional[ShadeConfig] = None,
                probe: Optional[Dict] = None) -> OuterResult:
    """Bound-handling control: canonical L-SHADE but with the clipping rule used by PSO/DE/CMA-ES."""
    c = cfg or ShadeConfig()
    c = ShadeConfig(**{**c.__dict__, "bound_rule": "clip"})
    return _shade_core(prob, budget, seed, c, lshade=True, local=False, probe=probe)


OPTIMIZERS = {"SHADE": shade, "L-SHADE": lshade, "L-SHADE+Normal": lshade_normal,
              "L-SHADE-clip": lshade_clip}
