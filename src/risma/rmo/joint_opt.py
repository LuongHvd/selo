"""Single-objective L-SHADE on the **joint** encoding `y = [vec(T), theta, Re w, Im w]`.

This is a separate implementation from `risma.rmo.shade`, which searches the legacy T-only problem
with scalar bounds and an inner AO call per candidate.  `shade.py` is left untouched so that the R0
screening stays reproducible; the two share only the pure helpers (`_np_init`, `_weighted_lehmer`,
`_bound_repair`), which are unit-tested once.

Differences from the T-only version:
  * per-dimension bounds (`JointSpec.lower/upper`), since the three blocks have different ranges;
  * the budget is expressed in **scenario evaluations (SE)**, the primary currency of the RMO
    branch, and `SE = candidate_FE x S` for a fixed scenario count;
  * a **variable-group mask** freezes any subset of the blocks at a common reference design, which
    is what the R2 ablation needs; with `mask = all True` nothing is frozen and the search is the
    plain joint L-SHADE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .encoding import JointSpec, random_feasible_joint, repair_vector
from .evaluator import JointRobustEvaluator
from .objectives import lcvar, objectives
from .shade import ShadeConfig, _bound_repair, _np_init, _weighted_lehmer

FDTYPE = np.float64

GROUPS = ("T", "theta", "w")


def group_mask(spec: JointSpec, groups: Sequence[str]) -> np.ndarray:
    """Boolean mask over the `D` genes selecting the given blocks."""
    m = np.zeros(spec.D, dtype=bool)
    for g in groups:
        if g == "T":
            m[spec.sl_T] = True
        elif g == "theta":
            m[spec.sl_theta] = True
        elif g == "w":
            m[spec.sl_wr] = True
            m[spec.sl_wi] = True
        else:
            raise ValueError(f"unknown variable group {g!r}; expected one of {GROUPS}")
    return m


@dataclass
class JointResult:
    y: np.ndarray                      # (R, D) best design per instance
    obj: np.ndarray                    # (R,) scalarised design objective
    f_mean: np.ndarray                 # (R,) mean secrecy rate on the design bank
    f_tail: np.ndarray                 # (R,) LCVaR on the design bank
    candidate_fe: int = 0
    scenario_evals: int = 0
    runtime_s: float = 0.0
    anytime: Optional[np.ndarray] = None       # (n_points, R) best-so-far objective
    anytime_se: Optional[np.ndarray] = None    # (n_points,) SE consumed
    meta: Dict = field(default_factory=dict)


class JointProblem:
    """Scalarised robust objective over a scenario bank, with optional frozen blocks.

    ``objective = weights[0] * mean_hat + weights[1] * LCVaR_hat`` evaluated on the first
    ``n_scen`` scenarios of the design bank.  The scalarisation is a **diagnostic** device for the
    R2 ablation (plan section 9); the multi-objective study of R3 never uses it.
    """

    def __init__(self, ev: JointRobustEvaluator, spec: JointSpec,
                 reference: Optional[np.ndarray] = None,
                 groups: Sequence[str] = GROUPS, n_scen: int = 50,
                 alpha: float = 0.10, weights: Tuple[float, float] = (0.5, 0.5),
                 seed_reference: bool = False):
        self.ev = ev
        self.spec = spec
        self.mask = group_mask(spec, groups)
        self.groups = tuple(groups)
        self.n_scen = int(n_scen)
        self.alpha = float(alpha)
        self.weights = tuple(weights)
        self.R = ev.ch.R
        self.D = spec.D
        if reference is None and not self.mask.all():
            raise ValueError("a reference design is required when any block is frozen")
        self.ref = None if reference is None else np.asarray(reference, dtype=FDTYPE)
        if self.ref is not None and self.ref.shape != (self.R, spec.D):
            raise ValueError(f"reference must have shape {(self.R, spec.D)}, got {self.ref.shape}")
        # `seed_reference` places the shared nominal reference design in the initial population of
        # every group (D-2026-09-11-02).  In a group that freezes a block this changes nothing for
        # that block; it removes the asymmetry whereby a group optimising `w` would have to
        # rediscover the AO beamformer that the restricted groups are handed for free.
        self.seed_reference = bool(seed_reference)
        if self.seed_reference and self.ref is None:
            raise ValueError("seed_reference=True requires a reference design")

    # ------------------------------------------------------------------ genotype handling
    def freeze(self, y: np.ndarray) -> np.ndarray:
        """Overwrite the non-optimised genes with the reference design."""
        if self.ref is None or self.mask.all():
            return y
        ref = self.ref[:, None, :] if y.ndim == 3 else self.ref
        return np.where(self.mask, y, ref)

    def repair(self, y: np.ndarray) -> np.ndarray:
        """Freeze, canonicalise, freeze again (the repair of a frozen feasible block is a no-op)."""
        return self.freeze(repair_vector(self.freeze(y), self.spec))

    def random_pop(self, rng: np.random.Generator, NP: int) -> np.ndarray:
        pop = self.repair(random_feasible_joint(rng, (self.R, NP), self.spec))
        if self.seed_reference:
            pop[:, 0, :] = self.repair(self.ref)
        return pop

    # ------------------------------------------------------------------ objective
    def rates(self, y: np.ndarray, count: bool = True) -> np.ndarray:
        return self.ev.rates(y, n_scen=self.n_scen, count=count)

    def parts(self, y: np.ndarray, count: bool = True):
        rs = self.rates(y, count=count)
        return rs.mean(axis=0), lcvar(rs, self.alpha, axis=0)

    def objective(self, y: np.ndarray, count: bool = True) -> np.ndarray:
        fm, ft = self.parts(y, count=count)
        return self.weights[0] * fm + self.weights[1] * ft

    def F(self, y: np.ndarray, count: bool = True,
          n_scen: Optional[int] = None) -> np.ndarray:
        """The two **maximised** objectives, shape ``(..., 2)`` = [F_mean, F_tail].

        ``n_scen`` overrides the default scenario count for one call, which is what the
        adaptive-fidelity method needs; the nested bank guarantees that a larger ``n_scen``
        refines the same estimate instead of replacing it.
        """
        rs = self.ev.rates(y, n_scen=self.n_scen if n_scen is None else n_scen, count=count)
        return objectives(rs, self.alpha, axis=0)


def _anytime_points(budget: int, n: int = 24) -> np.ndarray:
    pts = np.unique(np.round(np.geomspace(max(budget // 100, 1), budget, n)).astype(int))
    return pts


def lshade_joint(prob: JointProblem, budget_se: int, seed: int = 0,
                 cfg: Optional[ShadeConfig] = None,
                 probe: Optional[Dict] = None, local: bool = True) -> JointResult:
    """L-SHADE (current-to-pbest/1 + archive + success history + LPSR) on the joint encoding.

    ``budget_se`` is the total number of scenario evaluations **per channel instance**; with a
    fixed scenario count this is exactly ``candidate_FE x n_scen``.

    ``local=True`` (the default) adds the elite Gaussian refinement of **L-SHADE+Normal**, which is
    the operator R0 selected (`reports/R0_operator_screening.md` section 4): a fraction
    ``cfg.local_frac`` of each generation's offspring is replaced by Gaussian samples around the
    incumbent elite, with a width annealed from ``cfg.sigma0`` to ``cfg.sigma_end`` of the
    population spread.  ``local=False`` recovers plain L-SHADE exactly.
    """
    t0 = time.time()
    cfg = cfg or ShadeConfig()
    rng = np.random.default_rng(seed)
    spec, D, R = prob.spec, prob.D, prob.R
    lo, hi = spec.lower(), spec.upper()
    S = prob.n_scen

    budget_fe = max(int(budget_se) // S, 5)
    NP0 = max(_np_init(D, budget_fe, lshade=True), 5)
    NP = NP0
    Amax = max(int(round(cfg.r_arc * NP0)), 1)

    pop = prob.random_pop(rng, NP)
    fit = prob.objective(pop)
    used = NP

    pts = _anytime_points(budget_fe)
    trace, trace_fe = [], []
    best_idx = np.argmax(fit, axis=1)
    best_y = np.take_along_axis(pop, best_idx[:, None, None], axis=1)[:, 0]
    best_f = np.max(fit, axis=1)

    arch = np.zeros((R, Amax, D), dtype=FDTYPE)
    acnt = np.zeros(R, dtype=int)
    MF = np.full((R, cfg.H), 0.5)
    MCR = np.full((R, cfg.H), 0.5)
    kmem = np.zeros(R, dtype=int)
    gen = 0

    while used < budget_fe:
        gen += 1
        ridx = rng.integers(0, cfg.H, (R, NP))
        mcr = np.take_along_axis(MCR, ridx, axis=1)
        mf = np.take_along_axis(MF, ridx, axis=1)
        CR = np.where(mcr < 0, 0.0, np.clip(rng.normal(mcr, 0.1), 0.0, 1.0))
        draw = mf + 0.1 * np.tan(np.pi * (rng.random((R, NP)) - 0.5))
        for _ in range(12):
            bad = draw <= 0.0
            if not np.any(bad):
                break
            draw = np.where(bad, mf + 0.1 * np.tan(np.pi * (rng.random((R, NP)) - 0.5)), draw)
        F = np.clip(np.where(draw <= 0.0, 0.05, draw), 1e-6, 1.0)

        order = np.argsort(-fit, axis=1)
        p_num = max(2, int(round(cfg.p_best * NP)))
        pb_idx = np.take_along_axis(order, rng.integers(0, p_num, (R, NP)), axis=1)
        xpb = np.take_along_axis(pop, pb_idx[..., None], axis=1)

        r1 = rng.integers(0, NP, (R, NP))
        tot = NP + acnt
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

        # ---- elite Gaussian refinement (L-SHADE+Normal, the R0 winner) ------------------
        n_loc = 0
        if local:
            n_loc = int(round(cfg.local_frac * NP))
            if n_loc > 0:
                frac = used / max(budget_fe, 1)
                sigma = cfg.sigma0 + (cfg.sigma_end - cfg.sigma0) * frac
                spread = np.std(pop, axis=1, keepdims=True)              # (R,1,D)
                scale = np.maximum(spread, 1e-6 * (hi - lo))
                pick = rng.integers(0, NP, (R, n_loc))
                elite = np.take_along_axis(pop, order[:, :1, None], axis=1)   # (R,1,D)
                loc = np.clip(elite + sigma * scale * rng.standard_normal((R, n_loc, D)),
                              lo, hi)
                for r in range(R):
                    u[r, pick[r]] = loc[r]
        u = prob.repair(u)

        take = min(NP, budget_fe - used)
        u = u[:, :take]
        fu = prob.objective(u)
        used += take

        better = fu > fit[:, :take]
        delta = np.maximum(fu - fit[:, :take], 0.0)
        for r in range(R):
            win = np.flatnonzero(better[r])
            if win.size:
                for i in win:
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

        imp = np.max(fit, axis=1) > best_f
        if np.any(imp):
            bi = np.argmax(fit, axis=1)
            cand = np.take_along_axis(pop, bi[:, None, None], axis=1)[:, 0]
            best_y = np.where(imp[:, None], cand, best_y)
            best_f = np.where(imp, np.max(fit, axis=1), best_f)
        while pts.size and used >= pts[0]:
            trace.append(best_f.copy())
            trace_fe.append(int(pts[0]))
            pts = pts[1:]

        if probe is not None:
            probe.setdefault("NP", []).append(NP)
            probe.setdefault("fe", []).append(used)
            probe.setdefault("MF", []).append(MF.copy())
            probe.setdefault("MCR", []).append(MCR.copy())

        # ---- linear population size reduction ------------------------------------------
        NP_new = int(round(NP0 + (cfg.NP_min - NP0) * (used / budget_fe)))
        NP_new = max(cfg.NP_min, min(NP, NP_new))
        if NP_new < NP:
            keep = np.argsort(-fit, axis=1)[:, :NP_new]
            pop = np.take_along_axis(pop, keep[..., None], axis=1)
            fit = np.take_along_axis(fit, keep, axis=1)
            NP = NP_new
        if probe is not None:
            probe.setdefault("NP_after", []).append(NP)
            probe.setdefault("fe_after", []).append(used)

    if not trace:
        trace.append(best_f.copy())
        trace_fe.append(used)

    fm, ft = prob.parts(best_y, count=False)
    return JointResult(
        y=best_y, obj=best_f, f_mean=fm, f_tail=ft,
        candidate_fe=used, scenario_evals=used * S, runtime_s=time.time() - t0,
        anytime=np.asarray(trace), anytime_se=np.asarray(trace_fe) * S,
        meta=dict(optimizer="L-SHADE+Normal-joint" if local else "L-SHADE-joint",
                  local_refinement=bool(local), local_frac=cfg.local_frac if local else 0.0,
                  NP_init=NP0, NP_final=NP, generations=gen,
                  groups=list(prob.groups), n_scen=S, alpha=prob.alpha,
                  weights=list(prob.weights), budget_se=int(budget_se), budget_fe=budget_fe),
    )
