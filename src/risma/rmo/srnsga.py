"""SR-NSGA-II -- Structure-aware, Risk-adaptive NSGA-II for the robust joint MA-RIS problem.

Phase R4 of the RMO branch (reports/R4_plan.md).  Built on the NSGA-II backbone that won the R3
pilot, with four switchable components, each aimed at a failure mode measured in R2/R3:

  C1  risk gene + closed-form robust beamformer (Lamarckian).
      Every genotype carries an extra gene ``rho in [0, 1]``.  After variation the beamformer genes
      are *overwritten* by the generalised-eigenvector solution of Algorithm 1 (Eq. 10) applied to a
      risk-weighted scenario-averaged leakage matrix

          B_rho = (1 - rho) * mean_s B_s  +  rho * mean_{s in tail} B_s ,

      where the tail is the ``K = ceil(alpha S)`` worst design scenarios of the individual's first
      parent (cached, so identifying it is free).  rho = 0 is the mean-leakage design, rho = 1 a
      CVaR-flavoured design; the population is free to spread rho along the front.
  C2  memetic robust RIS step.  For a random fraction ``p_phi`` of the offspring, one Dinkelbach/MM
      update of the RIS vector (Eqs. 12-18, unchanged code) on the same risk-weighted virtual-Eve
      set, followed by a fresh C1 beamformer.
  C3a structured initialisation: a fraction of the initial population is produced by a short
      risk-weighted robust AO (C1+C2 iterated) from random feasible (T, Phi).
  C3b plateau escape: when every design-scenario secrecy rate of a design is clipped at 0 the two
      (true) objectives are flat; *selection only* then uses the unclipped log-ratio mean / LCVaR,
      which are <= 0 and therefore never rank such a design above a positive one.  Reported and
      archived objectives are always the true clipped ones.
  C4  dominance-racing scenario fidelity.  Offspring are first scored on the nested prefix
      ``Xi[:S_low]``; only those whose optimistic estimate is non-dominated by the better half of
      the population are topped up (incrementally -- nesting means only the missing scenarios are
      evaluated) to ``S_mid``.  Archive entrants are topped up to ``S_high``.  Every comparison
      inside environmental selection is made on the same ``S_mid`` prefix (common random numbers).

With every switch off the algorithm is plain NSGA-II (SBX + polynomial mutation, rank + crowding),
identical in operators to `moea.nsga2`.

Budget accounting (SE, brief section 22): evaluating a design on s scenarios costs s SE; building
a scenario-averaged leakage matrix over S scenarios (one C1 beamformer or one C2 Gram matrix) is
charged S SE.  Wall-clock time is recorded for every run and reported next to SE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..ao_solver import update_v, update_w
from ..config import AOConfig
from .encoding import JointSpec, decode, encode, random_feasible_joint, repair_vector
from .evaluator import JointRobustEvaluator
from .indicators import crowding_distance, fast_nondominated_sort
from .moea import CHECKPOINTS, MOResult, _archive_update, _poly_mutation, _sbx, _truncate
from .objectives import lcvar, nondominated_mask

FDTYPE = np.float64
CDTYPE = np.complex128
S_LOW, S_MID, S_HIGH = 20, 50, 100


@dataclass
class SRConfig:
    NP: int = 60
    eta_c: float = 20.0
    eta_m: float = 20.0
    p_c: float = 0.9
    p_m: Optional[float] = None          # default 1/(D+1)
    archive_max: int = 100
    alpha: float = 0.10
    # ---- components
    risk_bf: bool = False                # C1
    rho_fixed: Optional[float] = None    # C1 control: freeze the risk gene (e.g. 0.0)
    phi_step: bool = False               # C2
    p_phi: float = 1.0                   # locked by R4 tuning (configs/rmo_r4_locked.yaml)
    seed_init: bool = False              # C3a
    seed_frac: float = 0.5
    seed_iters: int = 3
    raw_select: bool = False             # C3b
    adaptive: bool = False               # C4
    S_low: int = S_LOW
    S_mid: int = S_MID
    S_high: int = S_HIGH
    z_race: float = 1.0
    # fixed-fidelity arms use S_fixed for everything
    S_fixed: int = S_MID
    name: str = "SR-NSGA-II"
    power_ball: bool = False             # R6 Check A: w = 0 when Lambda_max <= 1
    # variable-group study (RQ2): optimised blocks; frozen blocks are held at `ref_y`
    groups: tuple = ("T", "theta", "w")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


class _RateCache:
    """Per-individual raw (unclipped) design-scenario rates with incremental nested fidelity."""

    def __init__(self, R: int, n: int, S_max: int):
        self.raw = np.full((R, n, S_max), np.nan, dtype=FDTYPE)
        self.k = np.zeros((R, n), dtype=int)          # number of scenarios evaluated


def _raw_rates(ev: JointRobustEvaluator, y: np.ndarray, a: int, b: int) -> np.ndarray:
    """Unclipped log-ratio secrecy rates on scenarios ``[a, b)`` of the design bank: (b-a, R, P)."""
    u, gb = ev._prep(y)
    eve = ev.bank.eve[a:b]                                         # (s,R,J,N)
    ye = np.einsum("rpn,srjn->srpj", u, np.conj(eve), optimize=True)
    ge = np.abs(ye) ** 2 / ev.ch.sigma_e2[None, :, None, :]
    return np.log2(1.0 + gb)[None] - np.log2(1.0 + ge.sum(axis=-1))


def _objectives_from_raw(raw: np.ndarray, S: int, alpha: float, raw_select: bool):
    """raw: (..., S_max) with at least S valid entries -> (F_true (...,2), F_sel (...,2))."""
    r = raw[..., :S]
    rc = np.maximum(r, 0.0)
    F = np.stack([rc.mean(axis=-1), lcvar(rc, alpha, axis=-1)], axis=-1)
    if not raw_select:
        return F, F
    Fr = np.stack([r.mean(axis=-1), lcvar(r, alpha, axis=-1)], axis=-1)
    flat = (F[..., 0] <= 0.0)[..., None]
    return F, np.where(flat, Fr, F)


def _order(Fsel: np.ndarray) -> np.ndarray:
    """Indices sorted best-first by (rank, -crowding)."""
    rank = fast_nondominated_sort(Fsel)
    cd = np.zeros(Fsel.shape[0])
    for q in np.unique(rank):
        idx = np.flatnonzero(rank == q)
        cd[idx] = crowding_distance(Fsel[idx])
    return np.lexsort((-cd, rank)), rank, cd


def _tournament(rank, cd, rng, n):
    P = rank.shape[0]
    a = rng.integers(0, P, n)
    b = rng.integers(0, P, n)
    better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] > cd[b]))
    return np.where(better, a, b)


# --------------------------------------------------------------------------------------
# structure operators (C1 / C2)
# --------------------------------------------------------------------------------------


class StructureOps:
    """Risk-weighted robust beamformer / RIS updates on a virtual-Eve set built from the bank.

    Implements the weighted version of the P1 "virtual Eve" construction: scenario ``s`` with weight
    ``omega_s`` (sum 1) contributes Eves with noise power ``sigma_j^2 / omega_s``, so that the
    leakage matrix handed to the *unchanged* Algorithm-1 updates is ``sum_s omega_s B_s``.
    """

    def __init__(self, ev: JointRobustEvaluator, spec: JointSpec, S: int, alpha: float,
                 aocfg: Optional[AOConfig] = None):
        ch = ev.ch
        self.ev, self.spec, self.S, self.alpha = ev, spec, int(S), float(alpha)
        self.K = max(1, int(np.ceil(alpha * S)))
        self.aocfg = aocfg or AOConfig()
        eve = ev.bank.eve[: self.S]                                      # (S,R,J,N)
        R, J, N = ch.R, ch.J, ch.N
        self.he_v = np.transpose(eve, (1, 0, 2, 3)).reshape(R, 1, self.S * J, N)
        self.se2 = ch.sigma_e2                                           # (R,J)
        self.hb = ch.h_b[:, None, :]
        self.J = J
        self.power_ball = False
        self.n_calls = 0
        self.n_low = 0
        self.last_low = None

    def weights(self, rho: np.ndarray, tail_idx: np.ndarray) -> np.ndarray:
        """rho (R,P), tail_idx (R,P,K) -> omega (R,P,S)."""
        R, P = rho.shape
        om = np.broadcast_to(((1.0 - rho) / self.S)[..., None], (R, P, self.S)).copy()
        add = (rho / self.K)[..., None] * np.ones((R, P, self.K))
        np.add.at(om, (np.arange(R)[:, None, None], np.arange(P)[None, :, None], tail_idx), add)
        return om

    def _sig(self, omega: np.ndarray) -> np.ndarray:
        """Virtual-Eve noise powers (R,P,S*J)."""
        om = np.maximum(omega, 1e-12)
        return (self.se2[:, None, None, :] / om[..., None]).reshape(
            omega.shape[0], omega.shape[1], self.S * self.J)

    def beamformer(self, H, v, omega):
        """Surrogate-optimal beamformer over the power BALL (revision R6, Check A).

        On the sphere ||w||^2 = P the maximiser of Gamma = (1 + w^H A w) / (1 + w^H B w) is the
        generalised principal eigenvector (Algorithm-1 update_w) with value Lambda_max.  Over the
        ball, dGamma/dp has the sign of a(u) - b(u), so the optimum is full power if
        Lambda_max > 1 and w = 0 (Gamma = 1) otherwise.  Every call records Lambda_max; the
        zero-beamformer branch is applied only when ``self.power_ball`` is True.
        """
        w = update_w(H, self.hb, self.he_v, v, self.ev.ch.sigma_b2, self._sig(omega),
                     self.spec.Pmax, self.aocfg.nullspace_guard)
        lam = self.surrogate(H, v, w, omega)
        low = lam <= 1.0
        self.n_calls += int(low.size)
        self.n_low += int(np.count_nonzero(low))
        self.last_low = low
        if self.power_ball and np.any(low):
            w = np.where(low[..., None], 0.0, w)
        return w

    def surrogate(self, H, v, w, omega):
        """Gamma_omega(Phi, w) for every individual: (R,P)."""
        Hw = np.matmul(H, w[..., None])[..., 0]
        u = np.conj(v) * Hw
        gb = np.abs(np.sum(u * np.conj(self.hb), axis=-1)) ** 2 / self.ev.ch.sigma_b2
        eve = self.ev.bank.eve[: self.S]                                   # (S,R,J,N)
        ye = np.einsum("rpn,srjn->srpj", u, np.conj(eve), optimize=True)
        ge = (np.abs(ye) ** 2 / self.se2[None, :, None, :]).sum(-1)        # (S,R,P)
        return (1.0 + gb) / (1.0 + np.einsum("srp,rps->rp", ge, omega))

    def ris(self, H, v, w, omega):
        v_new, _ = update_v(H, self.hb, self.he_v, w, v, self.ev.ch.sigma_b2, self._sig(omega),
                            self.aocfg)
        return v_new

    def tail_of(self, raw: np.ndarray) -> np.ndarray:
        """raw (R,P,>=S) -> indices of the K smallest rates among the first S (R,P,K)."""
        return np.argpartition(raw[..., : self.S], self.K - 1, axis=-1)[..., : self.K]

    def apply(self, y: np.ndarray, rho: np.ndarray, tail_idx: np.ndarray,
              phi_mask: Optional[np.ndarray] = None, iters: int = 1, free_w: bool = True):
        """Lamarckian refinement.  Returns (y_new, SE charged per individual (R,P)).

        ``free_w=False`` (variable-group study only) keeps the encoded beamformer fixed and applies
        the RIS step alone."""
        T, v, w_enc = decode(y, self.spec)
        H = self.ev.ch.H_BR(T)
        if tail_idx is None:                       # no tail information yet: mean leakage
            om = np.full(rho.shape + (self.S,), 1.0 / self.S)
        else:
            om = self.weights(rho, tail_idx)
        cost = np.zeros(rho.shape, dtype=np.int64)
        low_final = None
        if free_w:
            w = self.beamformer(H, v, om)
            low_final = self.last_low.copy()
            cost += self.S
        else:
            w = w_enc
        for it in range(iters):
            if phi_mask is not None and np.any(phi_mask):
                v_new = self.ris(H, v, w, om)
                v = np.where(phi_mask[..., None], v_new, v)
                cost += self.S * phi_mask.astype(np.int64)
                if free_w:
                    w2 = self.beamformer(H, v, om)
                    w = np.where(phi_mask[..., None], w2, w)
                    low_final = np.where(phi_mask, self.last_low, low_final)
                    cost += self.S * phi_mask.astype(np.int64)
            if it + 1 < iters:
                # re-identify the tail under the refined design before the next round
                yy = encode(T, np.mod(-np.angle(v), 2 * np.pi), w, self.spec)
                raw = np.moveaxis(_raw_rates(self.ev, yy, 0, self.S), 0, -1)
                tail_idx = self.tail_of(raw)
                om = self.weights(rho, tail_idx)
                cost += self.S
        theta = np.mod(-np.angle(v), 2.0 * np.pi)
        y_new = repair_vector(encode(T, theta, w, self.spec), self.spec)
        self.last_final_low = low_final
        if self.power_ball and free_w and low_final is not None and np.any(low_final):
            # the surrogate optimum over the ball is w = 0: write it back AFTER repair (whose
            # zero-norm fallback exists only for numerically degenerate genotypes)
            zero = low_final
            y_new[..., self.spec.sl_wr] = np.where(zero[..., None], 0.0, y_new[..., self.spec.sl_wr])
            y_new[..., self.spec.sl_wi] = np.where(zero[..., None], 0.0, y_new[..., self.spec.sl_wi])
        return y_new, cost


# --------------------------------------------------------------------------------------
# the algorithm
# --------------------------------------------------------------------------------------


def sr_nsga2(ev: JointRobustEvaluator, spec: JointSpec, budget_se: int, seed: int = 0,
             cfg: Optional[SRConfig] = None, ref_y: Optional[np.ndarray] = None) -> MOResult:
    t0 = time.time()
    cfg = cfg or SRConfig()
    from .joint_opt import group_mask
    gmask = group_mask(spec, cfg.groups)
    free_w, free_th = "w" in cfg.groups, "theta" in cfg.groups
    if not gmask.all() and ref_y is None:
        raise ValueError("a reference design is required when a block is frozen")

    def freeze(yy):
        if gmask.all():
            return yy
        ref = ref_y[:, None, :] if yy.ndim == 3 else ref_y
        return np.where(gmask, yy, ref)
    rng = np.random.default_rng(seed)
    R, D, NP = ev.ch.R, spec.D, cfg.NP
    lo = np.concatenate([spec.lower(), [0.0]])
    hi = np.concatenate([spec.upper(), [1.0]])
    p_m = cfg.p_m if cfg.p_m is not None else 1.0 / (D + 1)
    S_sel = cfg.S_mid if cfg.adaptive else cfg.S_fixed          # selection fidelity
    S_arch = cfg.S_high if cfg.adaptive else cfg.S_fixed        # archive fidelity
    S_max = max(S_sel, S_arch)
    if ev.bank.S < S_max:
        raise ValueError(f"bank has {ev.bank.S} scenarios, need {S_max}")
    use_struct = cfg.risk_bf or cfg.phi_step or cfg.seed_init
    ops = StructureOps(ev, spec, S_sel, cfg.alpha) if use_struct else None
    if ops is not None:
        ops.power_ball = bool(cfg.power_ball)
    budget = int(budget_se)
    se = 0
    fe = 0
    stats = dict(promoted=0, raced=0, topped_high=0, struct_se=0, phi_steps=0)

    def eval_to(yy: np.ndarray, cache_raw: np.ndarray, k: np.ndarray, S_target: int,
                mask: Optional[np.ndarray] = None) -> int:
        """Top up cached raw rates of the individuals in ``mask`` to S_target.  Returns SE."""
        nonlocal fe
        need = (k < S_target) if mask is None else ((k < S_target) & mask)
        if not np.any(need):
            return 0
        a = int(np.min(np.where(need, k, S_target)))
        raw = np.moveaxis(_raw_rates(ev, yy, a, S_target), 0, -1)    # (R,P,S_target-a)
        cost = 0
        for r in range(R):
            for i in np.flatnonzero(need[r]):
                ki = k[r, i]
                cache_raw[r, i, ki:S_target] = raw[r, i, ki - a:]
                cost += S_target - ki
                k[r, i] = S_target
        fe += int(np.count_nonzero(need & (k == S_target)))
        return cost

    # ---------------- initial population
    pop = freeze(random_feasible_joint(rng, (R, NP), spec))
    rho = rng.random((R, NP)) if cfg.rho_fixed is None else np.full((R, NP), cfg.rho_fixed)
    if cfg.seed_init:
        n_seed = int(round(cfg.seed_frac * NP))
        ysub, cost = ops.apply(pop[:, :n_seed], rho[:, :n_seed], None,
                               phi_mask=np.full((R, n_seed), free_th), iters=cfg.seed_iters,
                               free_w=free_w)
        pop[:, :n_seed] = ysub
        se += int(cost.sum() // R)
        stats["struct_se"] += int(cost.sum() // R)
    elif cfg.risk_bf:
        pop, cost = ops.apply(pop, rho, None, free_w=free_w)
        se += int(cost.sum() // R)
        stats["struct_se"] += int(cost.sum() // R)
    pop = freeze(pop)
    if ref_y is not None:                 # the shared reference design seeds every group (D-2026-09-11-02)
        pop[:, 0, :] = ref_y
    cache = np.full((R, NP, S_max), np.nan)
    kk = np.zeros((R, NP), dtype=int)
    se += eval_to(pop, cache, kk, S_sel) // R
    F, Fs = _objectives_from_raw(cache, S_sel, cfg.alpha, cfg.raw_select)

    arcY = [np.zeros((0, D)) for _ in range(R)]
    arcF = [np.zeros((0, 2)) for _ in range(R)]
    arcRho = [np.zeros(0) for _ in range(R)]
    snaps: List[Dict] = []
    pending = list(CHECKPOINTS)
    hist: List[Dict] = []
    gen = 0

    def archive_update(r, Yc, rawc, kc, rhoc):
        """Archive entrants are judged at S_arch (topped up if needed)."""
        nonlocal se
        Fc_sel, _ = _objectives_from_raw(rawc, S_sel, cfg.alpha, False)
        # cheap pre-filter at selection fidelity: only candidates non-dominated w.r.t. the archive
        allF = np.concatenate([arcF[r], Fc_sel]) if arcF[r].size else Fc_sel
        nd = nondominated_mask(allF)[-Fc_sel.shape[0]:] & (Fc_sel[:, 0] > 0)
        if not np.any(nd):
            return
        idx = np.flatnonzero(nd)
        if S_arch > S_sel:
            yy = np.zeros((R, idx.size, D))
            yy[r] = Yc[idx]
            raw_sub = np.full((R, idx.size, S_max), np.nan)
            raw_sub[r] = rawc[idx]
            k_sub = np.full((R, idx.size), S_arch)
            k_sub[r] = kc[idx]
            cost = eval_to(yy, raw_sub, k_sub, S_arch)
            se += cost / R
            stats["topped_high"] += idx.size
            rawc = rawc.copy()
            rawc[idx] = raw_sub[r]
            kc = kc.copy()
            kc[idx] = S_arch
        Fa, _ = _objectives_from_raw(rawc[idx], S_arch, cfg.alpha, False)
        Y_ext = np.concatenate([Yc[idx], rhoc[idx, None]], axis=1)
        A_ext = np.concatenate([arcY[r], arcRho[r][:, None]], axis=1) if arcF[r].size else \
            np.zeros((0, D + 1))
        aY, aF = _archive_update(A_ext, arcF[r], Y_ext, Fa, cfg.archive_max)
        arcY[r], arcRho[r], arcF[r] = aY[:, :D], aY[:, D], aF

    for r in range(R):
        archive_update(r, pop[r], cache[r], kk[r], rho[r])

    # ---------------- main loop
    per_child = float(S_sel * (1 + (2 if cfg.risk_bf else 0)))   # initial cost guess per child
    while se < budget:
        gen += 1
        # mating selection
        ext = np.concatenate([pop, rho[..., None]], axis=-1)
        child = np.empty_like(ext)
        par1 = np.empty((R, NP), dtype=int)
        for r in range(R):
            _, rank, cd = _order(Fs[r])
            i1 = _tournament(rank, cd, rng, NP)
            i2 = _tournament(rank, cd, rng, NP)
            par1[r] = i1
            c = _sbx(ext[r][i1], ext[r][i2], lo, hi, cfg.eta_c, cfg.p_c, rng)
            child[r] = _poly_mutation(c, lo, hi, cfg.eta_m, p_m, rng)
        take = int(min(NP, max(0.0, budget - se) // max(per_child, 1.0)))
        if take < 1:
            break
        child, par1 = child[:, :take], par1[:, :take]
        cy = freeze(repair_vector(freeze(child[..., :D]), spec))
        crho = child[..., D] if cfg.rho_fixed is None else np.full((R, take), cfg.rho_fixed)

        gen_se = 0
        if cfg.risk_bf or cfg.phi_step:
            # parent-inherited tail (free: cached)
            praw = np.take_along_axis(cache, par1[..., None], axis=1)
            tail = ops.tail_of(praw)
            phi_mask = ((rng.random((R, take)) < cfg.p_phi) & free_th) if cfg.phi_step else None
            if free_w or (phi_mask is not None and phi_mask.any()):
                cy, cost = ops.apply(cy, crho, tail, phi_mask=phi_mask, free_w=free_w)
                cy = freeze(cy)
            else:
                cost = np.zeros((R, take), dtype=np.int64)
            c_se = int(cost.sum() // R)
            gen_se += c_se
            stats["struct_se"] += c_se
            if phi_mask is not None:
                stats["phi_steps"] += int(phi_mask.sum())

        craw = np.full((R, take, S_max), np.nan)
        ck = np.zeros((R, take), dtype=int)
        if cfg.adaptive:
            gen_se += eval_to(cy, craw, ck, cfg.S_low) // R
            Fl, Fls = _objectives_from_raw(craw, cfg.S_low, cfg.alpha, cfg.raw_select)
            rc = np.maximum(craw[..., : cfg.S_low], 0.0)
            sd = rc.std(axis=-1)
            K_low = max(1, int(np.ceil(cfg.alpha * cfg.S_low)))
            marg = cfg.z_race * np.stack([sd / np.sqrt(cfg.S_low), sd / np.sqrt(K_low)], -1)
            opt = Fls + marg
            promote = np.zeros((R, take), dtype=bool)
            for r in range(R):
                order, _, _ = _order(Fs[r])
                elite = Fs[r][order[: max(1, NP // 2)]]
                dom = np.all(elite[None, :, :] >= opt[r][:, None, :], -1) & \
                    np.any(elite[None, :, :] > opt[r][:, None, :], -1)
                promote[r] = ~np.any(dom, axis=1)
            stats["raced"] += R * take
            stats["promoted"] += int(promote.sum())
            gen_se += eval_to(cy, craw, ck, S_sel, mask=promote) // R
        else:
            promote = np.ones((R, take), dtype=bool)
            gen_se += eval_to(cy, craw, ck, S_sel) // R
        se += gen_se
        per_child = gen_se / take if gen == 1 else 0.5 * per_child + 0.5 * gen_se / take

        Fc, Fcs = _objectives_from_raw(np.where(np.isnan(craw), 0.0, craw), S_sel, cfg.alpha,
                                       cfg.raw_select)
        # environmental selection over pop U promoted offspring
        newpop = np.empty_like(pop)
        newrho = np.empty_like(rho)
        newcache = np.empty_like(cache)
        newk = np.empty_like(kk)
        for r in range(R):
            idx = np.flatnonzero(promote[r])
            allY = np.concatenate([pop[r], cy[r][idx]])
            allRho = np.concatenate([rho[r], crho[r][idx]])
            allC = np.concatenate([cache[r], craw[r][idx]])
            allK = np.concatenate([kk[r], ck[r][idx]])
            allFs = np.concatenate([Fs[r], Fcs[r][idx]])
            keep = _truncate(allFs, NP)
            newpop[r], newrho[r], newcache[r], newk[r] = allY[keep], allRho[keep], allC[keep], \
                allK[keep]
            if idx.size:
                archive_update(r, cy[r][idx], craw[r][idx], ck[r][idx], crho[r][idx])
        pop, rho, cache, kk = newpop, newrho, newcache, newk
        F, Fs = _objectives_from_raw(cache, S_sel, cfg.alpha, cfg.raw_select)
        hist.append(dict(gen=gen, se=int(se), fe=int(fe // R)))
        while pending and se >= pending[0] * budget:
            frac = pending.pop(0)
            snaps.append(dict(frac=float(frac), se=int(se), S=int(S_sel),
                              F_arch=[a.copy() for a in arcF],
                              F_pop=[F[r][nondominated_mask(F[r])].copy() for r in range(R)]))

    meta = dict(optimizer=cfg.name, NP=NP, S_sel=S_sel, S_arch=S_arch, generations=gen,
                budget_se=budget, cfg={k: v for k, v in cfg.__dict__.items()},
                stats={**stats, "bf_calls": ops.n_calls if ops else 0,
                       "bf_lambda_le_1": ops.n_low if ops else 0}, snapshots=snaps,
                arch_rho=[a.copy() for a in arcRho])
    return MOResult(Y=arcY, F=arcF, scenario_evals=int(se), candidate_fe=int(fe // R),
                    runtime_s=time.time() - t0, history=hist, meta=meta)


# --------------------------------------------------------------------------------------
# named arms of the R4 ablation
# --------------------------------------------------------------------------------------

ARMS: Dict[str, Dict] = {
    "NSGA-II*":        dict(),                                                   # A0 (re-impl.)
    "+plateau":        dict(raw_select=True),                                    # A1
    "+riskBF":         dict(raw_select=True, risk_bf=True),                      # A2
    "+riskBF(rho=0)":  dict(raw_select=True, risk_bf=True, rho_fixed=0.0),       # control
    "+phiMM":          dict(raw_select=True, risk_bf=True, phi_step=True),       # A3
    "+seedInit":       dict(raw_select=True, risk_bf=True, phi_step=True, seed_init=True),  # A4
    "SR-NSGA-II":      dict(raw_select=True, risk_bf=True, phi_step=True, seed_init=True,
                            adaptive=True),                                      # A5 = full
    "+seedInit(S=100)": dict(raw_select=True, risk_bf=True, phi_step=True, seed_init=True,
                             S_fixed=100),                                       # RQ4 control
    "NSGA-II+seed":    dict(seed_init=True),                                     # seeding control
    # ---- R5 final: the method retained after the R4 stop rules (C1 + C2 on plain NSGA-II)
    "Proposed":        dict(risk_bf=True, phi_step=True),
    "Proposed(rho=0)": dict(risk_bf=True, phi_step=True, rho_fixed=0.0),
    "Proposed+C4":     dict(risk_bf=True, phi_step=True, adaptive=True),
    "+wOp":            dict(risk_bf=True),
}


def make_cfg(arm: str, NP: int = 60, **over) -> SRConfig:
    kw = dict(ARMS[arm])
    kw.update(over)
    return SRConfig(NP=NP, name=arm, **kw)
