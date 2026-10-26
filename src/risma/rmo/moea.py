"""Multi-objective evolutionary algorithms for the robust joint design.

Both objectives are **maximised**: `F = [F_mean, F_tail = LCVaR_alpha]`.

Implemented
  * `nsga2`      -- NSGA-II (SBX + polynomial mutation, non-dominated sorting + crowding);
  * `moead_de`   -- MOEA/D-DE (Tchebycheff, DE/rand/1, neighbourhood replacement limit `nr`);
  * `shade_moo`  -- GDE3-style differential evolution with SHADE's success-history adaptation of
    `F`/`CR`, i.e. the multi-objective counterpart of the R0 operator family;
  * `sa_rmoea`   -- **preliminary** scenario-adaptive variant: the same backbone as `shade_moo` but
    the scenario count is raised along the run (20 -> 50 -> 100) instead of being fixed at 50.

Every method
  * is vectorised over the `R` channel instances (one independent run per instance, shared code);
  * spends a budget measured in **scenario evaluations (SE)**, so a method that evaluates cheaply
    at low fidelity is allowed more candidate evaluations -- that is exactly the trade the adaptive
    method is meant to exploit, and it is why SE, not FE, is the pre-registered currency;
  * returns the final non-dominated archive per instance, evaluated at the design fidelity.

The archive discipline of the plan (section 7) is enforced here: a solution may only enter the
archive after being evaluated at `S >= S_mid`, so no reported front rests on a 20-scenario tail
estimate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .indicators import crowding_distance, fast_nondominated_sort
from .joint_opt import JointProblem
from .objectives import nondominated_mask
from .shade import _weighted_lehmer

FDTYPE = np.float64
S_LOW, S_MID, S_HIGH = 20, 50, 100


@dataclass
class MOConfig:
    NP: int = 60
    # NSGA-II operators
    eta_c: float = 20.0            # SBX distribution index
    eta_m: float = 20.0            # polynomial mutation distribution index
    p_c: float = 0.9
    p_m: Optional[float] = None    # default 1/D
    # DE operators
    F: float = 0.5
    CR: float = 0.9
    H: int = 6                     # SHADE memory size
    p_best: float = 0.2
    # MOEA/D
    T_neighbours: int = 15
    delta_p: float = 0.9
    nr: int = 2
    # archive
    archive_max: int = 100


@dataclass
class MOResult:
    Y: List[np.ndarray]                       # per instance: (n_i, D) archive designs
    F: List[np.ndarray]                       # per instance: (n_i, 2) objectives
    scenario_evals: int = 0
    candidate_fe: int = 0
    runtime_s: float = 0.0
    history: List[Dict] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------


def _truncate(Fv: np.ndarray, n: int) -> np.ndarray:
    """Indices of the best ``n`` rows by (front rank, then crowding distance)."""
    if Fv.shape[0] <= n:
        return np.arange(Fv.shape[0])
    rank = fast_nondominated_sort(Fv)
    keep: List[int] = []
    for r in range(rank.max() + 1):
        idx = np.flatnonzero(rank == r)
        if len(keep) + idx.size <= n:
            keep.extend(idx.tolist())
            continue
        cd = crowding_distance(Fv[idx])
        order = idx[np.argsort(-cd)]
        keep.extend(order[: n - len(keep)].tolist())
        break
    return np.asarray(keep, dtype=int)


def _archive_update(aY: np.ndarray, aF: np.ndarray, Y: np.ndarray, F: np.ndarray,
                    cap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Merge candidates into a non-dominated archive, truncating by crowding distance."""
    if aY.size == 0:
        allY, allF = Y, F
    else:
        allY = np.concatenate([aY, Y], axis=0)
        allF = np.concatenate([aF, F], axis=0)
    m = nondominated_mask(allF)
    allY, allF = allY[m], allF[m]
    _, uniq = np.unique(np.round(allF, 12), axis=0, return_index=True)
    allY, allF = allY[np.sort(uniq)], allF[np.sort(uniq)]
    if allF.shape[0] > cap:
        k = _truncate(allF, cap)
        allY, allF = allY[k], allF[k]
    return allY, allF


def _sbx(p1: np.ndarray, p2: np.ndarray, lo: np.ndarray, hi: np.ndarray,
         eta_c: float, p_c: float, rng: np.random.Generator) -> np.ndarray:
    """Simulated binary crossover, returning one child per pair."""
    n, D = p1.shape
    u = rng.random((n, D))
    beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (eta_c + 1.0)),
                    (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1.0)))
    sign = np.where(rng.random((n, D)) < 0.5, 1.0, -1.0)
    c = 0.5 * ((p1 + p2) + sign * beta * (p1 - p2))
    do = (rng.random((n, 1)) < p_c) & (rng.random((n, D)) < 0.5)
    c = np.where(do, c, p1)
    return np.clip(c, lo, hi)


def _poly_mutation(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, eta_m: float,
                   p_m: float, rng: np.random.Generator) -> np.ndarray:
    n, D = x.shape
    span = np.maximum(hi - lo, 1e-300)
    d1 = (x - lo) / span
    d2 = (hi - x) / span
    u = rng.random((n, D))
    mut_pow = 1.0 / (eta_m + 1.0)
    deltaq = np.where(
        u <= 0.5,
        (2 * u + (1 - 2 * u) * (1 - d1) ** (eta_m + 1.0)) ** mut_pow - 1.0,
        1.0 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta_m + 1.0)) ** mut_pow,
    )
    do = rng.random((n, D)) < p_m
    return np.clip(np.where(do, x + deltaq * span, x), lo, hi)


def _tournament(Fv: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    rank = fast_nondominated_sort(Fv)
    cd = crowding_distance(Fv)
    a = rng.integers(0, Fv.shape[0], n)
    b = rng.integers(0, Fv.shape[0], n)
    better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] > cd[b]))
    return np.where(better, a, b)


def _fe_for_budget(budget_se: int, S: int) -> int:
    return max(int(budget_se) // max(S, 1), 5)


CHECKPOINTS = tuple(np.round(np.linspace(0.1, 1.0, 10), 3))


def _record(snaps: List[Dict], pending: List[float], se: int, budget: int,
            arcF, Fv: np.ndarray, S_cur: int) -> None:
    """Snapshot the anytime state whenever an SE checkpoint is crossed.

    Both the **archive** front (the reportable one, which by construction only holds designs judged
    at `S >= S_MID`) and the current population front are stored, together with the fidelity in
    force, so the anytime hypervolume can be recomputed offline against a common reference point.
    """
    while pending and se >= pending[0] * budget:
        frac = pending.pop(0)
        snaps.append(dict(
            frac=float(frac), se=int(se), S=int(S_cur),
            F_arch=[np.asarray(a, dtype=FDTYPE).copy() for a in arcF],
            F_pop=[Fv[r][nondominated_mask(Fv[r])].copy() for r in range(Fv.shape[0])],
        ))


def _finish(prob: JointProblem, pop: np.ndarray, Fv: np.ndarray,
            arcY, arcF, cap: int, t0: float, se: int, fe: int,
            history: List[Dict], meta: Dict, merge_pop: bool = True) -> MOResult:
    """``merge_pop=False`` keeps the archive discipline when the final population was scored at a
    fidelity below ``S_MID``: such designs must not enter a reported front."""
    Ys, Fs = [], []
    for r in range(prob.R):
        if merge_pop:
            aY, aF = _archive_update(arcY[r], arcF[r], pop[r], Fv[r], cap)
        else:
            aY, aF = _archive_update(arcY[r], arcF[r],
                                     np.zeros((0, pop.shape[-1])), np.zeros((0, 2)), cap)
        Ys.append(aY)
        Fs.append(aF)
    return MOResult(Y=Ys, F=Fs, scenario_evals=se, candidate_fe=fe,
                    runtime_s=time.time() - t0, history=history, meta=meta)


# --------------------------------------------------------------------------------------
# NSGA-II
# --------------------------------------------------------------------------------------


def nsga2(prob: JointProblem, budget_se: int, seed: int = 0,
          cfg: Optional[MOConfig] = None) -> MOResult:
    t0 = time.time()
    cfg = cfg or MOConfig()
    rng = np.random.default_rng(seed)
    lo, hi = prob.spec.lower(), prob.spec.upper()
    D, R, NP = prob.D, prob.R, cfg.NP
    S = prob.n_scen
    p_m = cfg.p_m if cfg.p_m is not None else 1.0 / D
    budget_fe = _fe_for_budget(budget_se, S)

    pop = prob.random_pop(rng, NP)
    Fv = prob.F(pop)
    fe, se = NP, NP * S
    arcY = [np.zeros((0, D)) for _ in range(R)]
    arcF = [np.zeros((0, 2)) for _ in range(R)]
    hist: List[Dict] = []
    snaps: List[Dict] = []
    pending = list(CHECKPOINTS)
    gen = 0

    while fe < budget_fe:
        gen += 1
        child = np.empty_like(pop)
        for r in range(R):
            i1 = _tournament(Fv[r], rng, NP)
            i2 = _tournament(Fv[r], rng, NP)
            c = _sbx(pop[r][i1], pop[r][i2], lo, hi, cfg.eta_c, cfg.p_c, rng)
            child[r] = _poly_mutation(c, lo, hi, cfg.eta_m, p_m, rng)
        child = prob.repair(child)
        take = min(NP, budget_fe - fe)
        child = child[:, :take]
        Fc = prob.F(child)
        fe += take
        se += take * S

        newpop = np.empty_like(pop)
        newF = np.empty_like(Fv)
        for r in range(R):
            allY = np.concatenate([pop[r], child[r]], axis=0)
            allF = np.concatenate([Fv[r], Fc[r]], axis=0)
            k = _truncate(allF, NP)
            newpop[r], newF[r] = allY[k], allF[k]
            arcY[r], arcF[r] = _archive_update(arcY[r], arcF[r], child[r], Fc[r],
                                               cfg.archive_max)
        pop, Fv = newpop, newF
        hist.append(dict(gen=gen, fe=fe, se=se, S=S))
        _record(snaps, pending, se, int(budget_se), arcF, Fv, S)

    res = _finish(prob, pop, Fv, arcY, arcF, cfg.archive_max, t0, se, fe, hist,
                  dict(optimizer="NSGA-II", NP=NP, S=S, generations=gen,
                       budget_se=int(budget_se)))
    res.meta["snapshots"] = snaps
    return res


# --------------------------------------------------------------------------------------
# MOEA/D-DE
# --------------------------------------------------------------------------------------


def _weights(n: int) -> np.ndarray:
    a = np.linspace(0.0, 1.0, n)
    return np.stack([a, 1.0 - a], axis=1)


def _tchebycheff(F: np.ndarray, W: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Maximisation Tchebycheff: g = max_m w_m (z_m - F_m); smaller is better."""
    return np.max(W * np.maximum(z[None, :] - F, 0.0), axis=1)


def moead_de(prob: JointProblem, budget_se: int, seed: int = 0,
             cfg: Optional[MOConfig] = None) -> MOResult:
    t0 = time.time()
    cfg = cfg or MOConfig()
    rng = np.random.default_rng(seed)
    lo, hi = prob.spec.lower(), prob.spec.upper()
    D, R, NP = prob.D, prob.R, cfg.NP
    S = prob.n_scen
    p_m = cfg.p_m if cfg.p_m is not None else 1.0 / D
    budget_fe = _fe_for_budget(budget_se, S)

    W = _weights(NP)
    Wn = np.maximum(W, 1e-6)
    dist = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=-1)
    B = np.argsort(dist, axis=1)[:, : cfg.T_neighbours]

    pop = prob.random_pop(rng, NP)
    Fv = prob.F(pop)
    z = np.max(Fv, axis=1)                                   # (R,2) ideal point
    fe, se = NP, NP * S
    arcY = [np.zeros((0, D)) for _ in range(R)]
    arcF = [np.zeros((0, 2)) for _ in range(R)]
    hist: List[Dict] = []
    snaps: List[Dict] = []
    pending = list(CHECKPOINTS)
    gen = 0

    while fe < budget_fe:
        gen += 1
        child = np.empty_like(pop)
        pool = np.empty((R, NP, cfg.T_neighbours), dtype=int)
        use_nb = rng.random((R, NP)) < cfg.delta_p
        for r in range(R):
            for i in range(NP):
                P = B[i] if use_nb[r, i] else np.arange(NP)
                pick = rng.choice(P, size=3, replace=False)
                a, b, c = pop[r][pick]
                v = a + cfg.F * (b - c)
                jr = rng.integers(0, D)
                cross = rng.random(D) < cfg.CR
                cross[jr] = True
                u = np.where(cross, v, pop[r, i])
                child[r, i] = np.clip(u, lo, hi)
                pool[r, i, :] = B[i] if use_nb[r, i] else rng.choice(
                    NP, size=cfg.T_neighbours, replace=False)
        for r in range(R):
            child[r] = _poly_mutation(child[r], lo, hi, cfg.eta_m, p_m, rng)
        child = prob.repair(child)
        take = min(NP, budget_fe - fe)
        child = child[:, :take]
        Fc = prob.F(child)
        fe += take
        se += take * S

        for r in range(R):
            z[r] = np.maximum(z[r], np.max(Fc[r], axis=0))
            for i in range(take):
                nrep = 0
                for j in pool[r, i]:
                    if nrep >= cfg.nr:
                        break
                    g_old = _tchebycheff(Fv[r, j][None, :], Wn[j][None, :], z[r])[0]
                    g_new = _tchebycheff(Fc[r, i][None, :], Wn[j][None, :], z[r])[0]
                    if g_new <= g_old:
                        pop[r, j] = child[r, i]
                        Fv[r, j] = Fc[r, i]
                        nrep += 1
            arcY[r], arcF[r] = _archive_update(arcY[r], arcF[r], child[r], Fc[r],
                                               cfg.archive_max)
        hist.append(dict(gen=gen, fe=fe, se=se, S=S))
        _record(snaps, pending, se, int(budget_se), arcF, Fv, S)

    res = _finish(prob, pop, Fv, arcY, arcF, cfg.archive_max, t0, se, fe, hist,
                  dict(optimizer="MOEA/D-DE", NP=NP, S=S, generations=gen,
                       T=cfg.T_neighbours, nr=cfg.nr, budget_se=int(budget_se)))
    res.meta["snapshots"] = snaps
    return res


# --------------------------------------------------------------------------------------
# SHADE-MOO  (GDE3 selection + SHADE success-history adaptation)
# --------------------------------------------------------------------------------------


def _dominates(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.all(a >= b, axis=-1) & np.any(a > b, axis=-1)


def _shade_moo_core(prob: JointProblem, budget_se: int, seed: int, cfg: MOConfig,
                    schedule: Optional[Sequence[Tuple[float, int]]] = None,
                    name: str = "SHADE-MOO") -> MOResult:
    """Shared core.  ``schedule`` maps budget fraction -> scenario count (adaptive fidelity)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    lo, hi = prob.spec.lower(), prob.spec.upper()
    D, R, NP = prob.D, prob.R, cfg.NP

    def S_at(frac: float) -> int:
        if schedule is None:
            return prob.n_scen
        s = schedule[0][1]
        for f, v in schedule:
            if frac >= f:
                s = v
        return s

    budget = int(budget_se)
    S0 = S_at(0.0)
    pop = prob.random_pop(rng, NP)
    Fv = prob.F(pop, n_scen=S0)
    se = NP * S0
    fe = NP
    S_cur = S0

    MF = np.full((R, cfg.H), 0.5)
    MCR = np.full((R, cfg.H), 0.5)
    kmem = np.zeros(R, dtype=int)
    arcY = [np.zeros((0, D)) for _ in range(R)]
    arcF = [np.zeros((0, 2)) for _ in range(R)]
    hist: List[Dict] = []
    snaps: List[Dict] = []
    pending = list(CHECKPOINTS)
    gen = 0

    while se < budget:
        gen += 1
        frac = se / budget
        S_new = S_at(frac)
        if S_new != S_cur:
            # refine the incumbent population at the higher fidelity before using it
            Fv = prob.F(pop, n_scen=S_new)
            se += NP * S_new
            fe += NP
            S_cur = S_new
            if se >= budget:
                break

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
        Fw = np.clip(np.where(draw <= 0.0, 0.05, draw), 1e-6, 1.0)

        child = np.empty_like(pop)
        for r in range(R):
            rank = fast_nondominated_sort(Fv[r])
            cd = crowding_distance(Fv[r])
            order = np.lexsort((-cd, rank))                    # best front, least crowded first
            p_num = max(2, int(round(cfg.p_best * NP)))
            pb = order[rng.integers(0, p_num, NP)]
            r1 = rng.integers(0, NP, NP)
            r2 = rng.integers(0, NP, NP)
            v = pop[r] + Fw[r][:, None] * (pop[r][pb] - pop[r]) \
                + Fw[r][:, None] * (pop[r][r1] - pop[r][r2])
            v = np.clip(np.where(v < lo, 0.5 * (lo + pop[r]), v), lo, hi)
            v = np.clip(np.where(v > hi, 0.5 * (hi + pop[r]), v), lo, hi)
            jr = rng.integers(0, D, NP)
            cross = rng.random((NP, D)) < CR[r][:, None]
            cross[np.arange(NP), jr] = True
            child[r] = np.where(cross, v, pop[r])
        child = prob.repair(child)

        room = (budget - se) // max(S_cur, 1)
        take = int(min(NP, room))
        if take <= 0:
            break
        child = child[:, :take]
        Fc = prob.F(child, n_scen=S_cur)
        fe += take
        se += take * S_cur

        for r in range(R):
            win = _dominates(Fc[r], Fv[r, :take])
            lose = _dominates(Fv[r, :take], Fc[r])
            tie = ~win & ~lose
            if np.any(win):
                idx = np.flatnonzero(win)
                wts = np.sum(np.maximum(Fc[r, idx] - Fv[r, idx], 0.0), axis=1)
                if wts.sum() > 1e-300:
                    k = kmem[r]
                    MF[r, k] = _weighted_lehmer(Fw[r, idx], wts)
                    sCR = CR[r, idx]
                    if MCR[r, k] < 0 or np.max(sCR) < 1e-12:
                        MCR[r, k] = -1.0
                    else:
                        MCR[r, k] = _weighted_lehmer(sCR, wts)
                    kmem[r] = (k + 1) % cfg.H
                pop[r, idx] = child[r, idx]
                Fv[r, idx] = Fc[r, idx]
            if np.any(tie):
                # GDE3: keep both, then truncate the union back to NP
                idx = np.flatnonzero(tie)
                allY = np.concatenate([pop[r], child[r][idx]], axis=0)
                allF = np.concatenate([Fv[r], Fc[r][idx]], axis=0)
                k = _truncate(allF, NP)
                pop[r], Fv[r] = allY[k], allF[k]
            # archive discipline: only designs judged at S >= S_MID may enter
            if S_cur >= S_MID:
                arcY[r], arcF[r] = _archive_update(arcY[r], arcF[r], child[r], Fc[r],
                                                   cfg.archive_max)
        hist.append(dict(gen=gen, fe=fe, se=se, S=S_cur))
        _record(snaps, pending, se, budget, arcF, Fv, S_cur)

    meta = dict(optimizer=name, NP=NP, S=prob.n_scen, generations=gen,
                budget_se=int(budget_se), schedule=None if schedule is None else
                [list(x) for x in schedule], S_final=S_cur)
    # never report a front resting on a low-fidelity estimate: if the run ended while the
    # population was still being scored at S < S_MID, only the archive (which by construction
    # only accepts designs judged at S >= S_MID) is returned.
    res = _finish(prob, pop, Fv, arcY, arcF, cfg.archive_max, t0, se, fe, hist, meta,
                  merge_pop=(S_cur >= S_MID))
    res.meta["snapshots"] = snaps
    return res


def shade_moo(prob: JointProblem, budget_se: int, seed: int = 0,
              cfg: Optional[MOConfig] = None) -> MOResult:
    return _shade_moo_core(prob, budget_se, seed, cfg or MOConfig(), None, "SHADE-MOO")


def sa_rmoea(prob: JointProblem, budget_se: int, seed: int = 0,
             cfg: Optional[MOConfig] = None,
             schedule: Sequence[Tuple[float, int]] = ((0.0, S_LOW), (0.35, S_MID),
                                                      (0.75, S_HIGH))) -> MOResult:
    """Preliminary scenario-adaptive robust MOEA.

    Same operators as `shade_moo`; the only difference is that the number of Eve-CSI realisations
    used to score a candidate follows `schedule` (budget fraction -> S) instead of being fixed.
    Because the banks are nested, raising S refines the estimate of the same quantity rather than
    replacing it, and the incumbent population is re-scored at the new fidelity whenever S changes
    (that re-scoring is charged to the SE budget like any other evaluation).
    """
    return _shade_moo_core(prob, budget_se, seed, cfg or MOConfig(), schedule, "SA-RMOEA")


OPTIMIZERS = {"NSGA-II": nsga2, "MOEA/D-DE": moead_de, "SHADE-MOO": shade_moo,
              "SA-RMOEA": sa_rmoea}


# --------------------------------------------------------------------------------------
# GDE3 (Kukkonen & Lampinen, 2005): DE/rand/1/bin + dominance-based replacement + truncation
# --------------------------------------------------------------------------------------


def gde3(prob: JointProblem, budget_se: int, seed: int = 0,
         cfg: Optional[MOConfig] = None) -> MOResult:
    t0 = time.time()
    cfg = cfg or MOConfig()
    rng = np.random.default_rng(seed)
    lo, hi = prob.spec.lower(), prob.spec.upper()
    D, R, NP = prob.D, prob.R, cfg.NP
    S = prob.n_scen
    budget_fe = _fe_for_budget(budget_se, S)
    pop = prob.random_pop(rng, NP)
    Fv = prob.F(pop)
    fe, se = NP, NP * S
    arcY = [np.zeros((0, D)) for _ in range(R)]
    arcF = [np.zeros((0, 2)) for _ in range(R)]
    hist: List[Dict] = []
    snaps: List[Dict] = []
    pending = list(CHECKPOINTS)
    gen = 0
    while fe < budget_fe:
        gen += 1
        child = np.empty_like(pop)
        for r in range(R):
            idx = np.stack([rng.permutation(NP)[:3] for _ in range(NP)])
            v = pop[r][idx[:, 0]] + cfg.F * (pop[r][idx[:, 1]] - pop[r][idx[:, 2]])
            jr = rng.integers(0, D, NP)
            cross = rng.random((NP, D)) < cfg.CR
            cross[np.arange(NP), jr] = True
            child[r] = np.clip(np.where(cross, v, pop[r]), lo, hi)
        child = prob.repair(child)
        take = min(NP, budget_fe - fe)
        child = child[:, :take]
        Fc = prob.F(child)
        fe += take
        se += take * S
        for r in range(R):
            win = _dominates(Fc[r], Fv[r, :take])
            lose = _dominates(Fv[r, :take], Fc[r])
            tie = ~win & ~lose
            iw = np.flatnonzero(win)
            pop[r, iw], Fv[r, iw] = child[r, iw], Fc[r, iw]
            it = np.flatnonzero(tie)
            if it.size:
                allY = np.concatenate([pop[r], child[r][it]], axis=0)
                allF = np.concatenate([Fv[r], Fc[r][it]], axis=0)
                k = _truncate(allF, NP)
                pop[r], Fv[r] = allY[k], allF[k]
            arcY[r], arcF[r] = _archive_update(arcY[r], arcF[r], child[r], Fc[r],
                                               cfg.archive_max)
        hist.append(dict(gen=gen, fe=fe, se=se, S=S))
        _record(snaps, pending, se, int(budget_se), arcF, Fv, S)
    res = _finish(prob, pop, Fv, arcY, arcF, cfg.archive_max, t0, se, fe, hist,
                  dict(optimizer="GDE3", NP=NP, S=S, generations=gen, budget_se=int(budget_se)))
    res.meta["snapshots"] = snaps
    return res


OPTIMIZERS["GDE3"] = gde3
