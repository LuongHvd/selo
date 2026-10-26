"""Multitask PSO for Phase P4.

One implementation covers every transfer-based method in the P4 benchmark.  **Independent PSO is
the same code path with ``lambda_max = 0``**: the transfer randomness is drawn unconditionally, so
switching transfer off leaves the random stream untouched and reproduces independent PSO
*bit-identically* (asserted by `tests/test_mtpso.py`).

Encoding, constraint handling and objective are inherited unchanged from P0-P3: the decision vector
is ``x = vec(T) in [-half, half]^{2M}``, the box is enforced by clipping, the minimum-spacing
constraint by the paper's Eq.-20 penalty inside the objective, and the reported solution is repaired
before evaluation.  Every candidate is scored by the frozen Algorithm-1 inner solver.

Budget accounting: with ``S`` tasks, ``R`` instances and swarm size ``NP``, one iteration consumes
``NP`` candidate evaluations **per task per instance**; ``iters = B // NP`` iterations therefore
spend exactly ``B`` per task, i.e. ``S*B`` in total, matching every other method.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..ma_constraints import random_feasible, repair
from ..multitask import TaskSet
from ..problem import MAPositionProblem

FDTYPE = np.float64
_NEG = -1e30


# --------------------------------------------------------------------------------------


@dataclass
class PSOConfig:
    """Locked P3 single-task PSO configuration (constraint C3 of reports/P4_plan.md)."""

    NP: int = 20
    w: float = 0.72
    c1: float = 1.49
    c2: float = 1.49
    v_clip: float = 0.5          # velocity clip as a fraction of the region half-width
    v_init: float = 0.1


@dataclass
class TransferConfig:
    """Everything that distinguishes the P4 methods from Independent PSO."""

    lambda_max: float = 0.0          # 0 => independent PSO, exactly
    similarity: str = "hybrid"       # none | random | channel | success | hybrid | oracle
    donor_form: str = "gbest"        # gbest | displacement | inject
    # 'inject' is the PSO analogue of the P2b warm-start mechanism: with probability lambda the
    # worst particle of the target swarm is replaced by the donor task's incumbent (velocity reset).
    beta: float = 6.0                # donor-selection temperature
    tau: float = 1.0                 # channel-similarity kernel width
    alpha: float = 0.5               # hybrid weight on the channel term
    succ_a: float = 1.0              # Beta prior on the success rate
    succ_b: float = 1.0
    forget: float = 1.0              # exponential forgetting of the success counts (1 = none)
    use_success_strength: bool = True   # multiply lambda by the reliability factor A
    a_floor: float = 0.0             # floor on A: > 0 disables the fallback-to-independent
    channel_mode: str = "eve"        # eve | full

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MTResult:
    best_obj: np.ndarray             # (S, R) final best objective per task and instance
    best_T: np.ndarray               # (S, R, M, 2)
    anytime: np.ndarray              # (iters, S, R) best-so-far
    anytime_fe: np.ndarray           # (iters,) candidate FE per task
    lambda_trace: np.ndarray         # (iters,) mean applied transfer strength
    donor_entropy: np.ndarray        # (iters,) mean entropy of P(q|s) in nats
    n_transfer: float                # attempted transfer steps
    n_transfer_success: float
    neg_transfer_rate: float         # P(a transfer step fails to improve the particle)
    improve_rate: float              # P(any particle improves its pbest) -- the matched control
    counters: Dict
    meta: Dict


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / np.maximum(np.sum(e, axis=-1, keepdims=True), 1e-300)


# --------------------------------------------------------------------------------------


def multitask_pso(
    ts: TaskSet,
    cfg,
    budget: int,
    seed: int = 0,
    pso: PSOConfig = PSOConfig(),
    tr: TransferConfig = TransferConfig(),
    oracle_sim: Optional[np.ndarray] = None,
) -> MTResult:
    """Run the (multi)task PSO.  ``budget`` is candidate evaluations **per task**."""
    t0 = time.time()
    S, R, M = ts.S, ts.R, ts.M
    D = 2 * M
    NP = pso.NP
    half = cfg.system.region_half
    rng = np.random.default_rng(seed)
    prob = MAPositionProblem(ts.stacked, cfg)

    # ---- FREE similarity input: arithmetic on channels the BS already holds -------------
    S_ch = np.exp(-ts.channel_distance(tr.channel_mode) / max(tr.tau ** 2, 1e-12))   # (R,S,S)
    eye = np.eye(S, dtype=bool)
    n_trans = np.zeros((R, S, S), dtype=FDTYPE)
    n_succ = np.zeros((R, S, S), dtype=FDTYPE)

    def similarity() -> Tuple[np.ndarray, np.ndarray]:
        A = (n_succ + tr.succ_a) / (n_trans + tr.succ_a + tr.succ_b)      # (R,S,S) in (0,1)
        if tr.similarity in ("none", "random"):
            s = np.zeros((R, S, S), dtype=FDTYPE)          # uniform donor probabilities
        elif tr.similarity == "channel":
            s = S_ch
        elif tr.similarity == "success":
            s = A
        elif tr.similarity == "hybrid":
            s = tr.alpha * S_ch + (1.0 - tr.alpha) * A
        elif tr.similarity == "oracle":
            if oracle_sim is None:
                raise ValueError("oracle similarity requested but not provided")
            s = oracle_sim
        else:
            raise ValueError(f"unknown similarity {tr.similarity!r}")
        return s, A

    # ---- initialisation ----------------------------------------------------------------
    X = random_feasible(rng, (S, R, NP), M, cfg.system.min_spacing, half).reshape(S, R, NP, D)
    V = rng.uniform(-pso.v_init * half, pso.v_init * half, (S, R, NP, D))

    def evaluate(x: np.ndarray) -> np.ndarray:
        T = np.clip(x, -half, half).reshape(S * R, NP, M, 2)
        return prob.evaluate(T, keep_wv=False).obj.reshape(S, R, NP)

    fit = evaluate(X)
    pbest_x, pbest_f = X.copy(), fit.copy()
    iters = max(budget // NP, 1)

    any_list = [pbest_f.max(axis=-1).copy()]
    fe_list = [NP]
    lam_list: List[float] = [0.0]
    ent_list: List[float] = [float(np.log(max(S - 1, 1)))]
    tot_trans = 0.0
    tot_succ = 0.0
    tot_particles = 0.0
    tot_improved = 0.0
    r_idx = np.arange(R)[None, :, None]
    s_idx_full = np.broadcast_to(np.arange(S)[:, None, None], (S, R, NP))

    for it in range(1, iters):
        g = np.argmax(pbest_f, axis=-1)                                    # (S,R)
        gbest_x = np.take_along_axis(pbest_x, g[..., None, None], axis=2)  # (S,R,1,D)

        r1 = rng.random((S, R, NP, D))
        r2 = rng.random((S, R, NP, D))
        # ---- transfer randomness is ALWAYS drawn => lambda_max=0 is bit-identical -------
        r3 = rng.random((S, R, NP, D))
        gum = -np.log(-np.log(rng.random((S, R, NP, S)) + 1e-300) + 1e-300)
        pair = rng.integers(0, NP, (2, S, R, NP))
        u_inj = rng.random((S, R))

        sim, A = similarity()
        logits = tr.beta * np.transpose(sim, (1, 0, 2))                    # (S,R,S)
        logits = np.where(eye[:, None, :], _NEG, logits)
        P = _softmax_rows(logits)
        q = np.argmax(np.log(P + 1e-300)[:, :, None, :] + gum, axis=-1)    # (S,R,NP)

        Pn = P / np.maximum(P.max(axis=-1, keepdims=True), 1e-300)   # normalised to the best donor
        Pq = np.take_along_axis(Pn, q, axis=-1)
        Aq = np.take_along_axis(np.transpose(A, (1, 0, 2)), q, axis=-1)
        if not tr.use_success_strength:
            Aq = np.ones_like(Aq)
        elif tr.a_floor > 0.0:
            Aq = np.maximum(Aq, tr.a_floor)
        lam = tr.lambda_max * Pq * Aq                                      # (S,R,NP)

        rr = np.broadcast_to(r_idx, q.shape)
        inject = tr.donor_form == "inject"
        if tr.donor_form == "gbest":
            Dv = gbest_x[:, :, 0, :][q, rr] - X
        elif tr.donor_form == "displacement":
            Dv = pbest_x[q, rr, pair[0]] - pbest_x[q, rr, pair[1]]
        elif inject:
            Dv = np.zeros_like(X)
        else:
            raise ValueError(f"unknown donor_form {tr.donor_form!r}")

        V = (pso.w * V
             + pso.c1 * r1 * (pbest_x - X)
             + pso.c2 * r2 * (gbest_x - X)
             + (0.0 if inject else 1.0) * lam[..., None] * r3 * Dv)
        V = np.clip(V, -pso.v_clip * half, pso.v_clip * half)
        X = np.clip(X + V, -half, half)

        if inject:
            # elite injection: replace the worst particle by the donor incumbent
            worst = np.argmin(pbest_f, axis=-1)                            # (S,R)
            lam_w = np.take_along_axis(lam, worst[..., None], axis=-1)[..., 0]
            q_w = np.take_along_axis(q, worst[..., None], axis=-1)[..., 0]
            do = u_inj < lam_w                                             # (S,R)
            src = gbest_x[:, :, 0, :][q_w, np.broadcast_to(np.arange(R)[None, :], q_w.shape)]
            sel = np.zeros((S, R, NP), dtype=bool)
            np.put_along_axis(sel, worst[..., None], do[..., None], axis=-1)
            X = np.where(sel[..., None], src[:, :, None, :], X)
            V = np.where(sel[..., None], 0.0, V)
            lam = np.where(sel, lam, 0.0)      # only the injected particle counts as a transfer

        prev_pf = pbest_f.copy()
        fit = evaluate(X)
        improved = fit > prev_pf
        pbest_x = np.where(improved[..., None], X, pbest_x)
        pbest_f = np.where(improved, fit, pbest_f)

        # ---- transfer bookkeeping (FREE: reuses evaluations already made) ---------------
        active = lam > 0.0
        if np.any(active):
            if tr.forget < 1.0:
                n_trans *= tr.forget
                n_succ *= tr.forget
            np.add.at(n_trans, (rr[active], s_idx_full[active], q[active]), 1.0)
            np.add.at(n_succ, (rr[active], s_idx_full[active], q[active]),
                      improved[active].astype(FDTYPE))
            tot_trans += float(active.sum())
            tot_succ += float((active & improved).sum())

        tot_particles += float(improved.size)
        tot_improved += float(improved.sum())
        any_list.append(pbest_f.max(axis=-1).copy())
        fe_list.append((it + 1) * NP)
        lam_list.append(float(lam.mean()))
        ent_list.append(float(-(P * np.log(P + 1e-300)).sum(axis=-1).mean()))

    g = np.argmax(pbest_f, axis=-1)
    best_x = np.take_along_axis(pbest_x, g[..., None, None], axis=2)[:, :, 0, :]
    counters = prob.counters.as_dict()
    counters["wall_time"] = time.time() - t0
    counters["fe_per_task"] = iters * NP
    counters["similarity_probe_fe"] = 0        # channel + success estimators are FREE
    return MTResult(
        best_obj=np.take_along_axis(pbest_f, g[..., None], axis=-1)[..., 0],
        best_T=repair(best_x.reshape(S, R, M, 2), cfg.system.min_spacing, half,
                      cfg.penalty.repair_iters),
        anytime=np.stack(any_list), anytime_fe=np.asarray(fe_list),
        lambda_trace=np.asarray(lam_list), donor_entropy=np.asarray(ent_list),
        n_transfer=tot_trans, n_transfer_success=tot_succ,
        neg_transfer_rate=float(1.0 - tot_succ / tot_trans) if tot_trans > 0 else float("nan"),
        improve_rate=float(tot_improved / max(tot_particles, 1.0)),
        counters=counters,
        meta={"pso": asdict(pso), "transfer": tr.to_dict(), "iters": iters,
              "S": S, "R": R, "budget_per_task": budget, "seed": seed},
    )


# --------------------------------------------------------------------------------------
# naive pooled / shared-population PSO
# --------------------------------------------------------------------------------------


def pooled_pso(ts: TaskSet, cfg, budget: int, seed: int = 0,
               pso: PSOConfig = PSOConfig()) -> MTResult:
    """One swarm shared by all tasks; each particle is scored on a rotating task assignment.

    Fitness values coming from different tasks are compared directly in the pbest/gbest updates --
    that naivety is exactly what this baseline is meant to test.  Budget is matched: each iteration
    spends ``NP`` evaluations spread over the tasks, so ``S*B/NP`` iterations spend ``S*B`` in total.
    """
    t0 = time.time()
    S, R, M = ts.S, ts.R, ts.M
    D = 2 * M
    NP = pso.NP
    half = cfg.system.region_half
    rng = np.random.default_rng(seed)
    probs = [MAPositionProblem(b, cfg) for b in ts.per_task]

    X = random_feasible(rng, (R, NP), M, cfg.system.min_spacing, half).reshape(R, NP, D)
    V = rng.uniform(-pso.v_init * half, pso.v_init * half, (R, NP, D))
    assign = np.arange(NP) % S

    best_per_task = np.full((S, R), -np.inf)
    best_T_task = np.zeros((S, R, M, 2))

    def evaluate(x: np.ndarray, asg: np.ndarray) -> np.ndarray:
        """Score each particle only on its assigned task: NP candidate FE per instance."""
        xc = np.clip(x, -half, half)
        out = np.empty((R, NP), dtype=FDTYPE)
        for s in range(S):
            sel = np.flatnonzero(asg == s)
            if sel.size == 0:
                continue
            sub = xc[:, sel].reshape(R, sel.size, M, 2)
            o = probs[s].evaluate(sub, keep_wv=False).obj
            out[:, sel] = o
            k = np.argmax(o, axis=-1)
            v = np.take_along_axis(o, k[:, None], axis=1)[:, 0]
            better = v > best_per_task[s]
            best_per_task[s] = np.where(better, v, best_per_task[s])
            xs = np.take_along_axis(sub, k[:, None, None, None], axis=1)[:, 0]
            best_T_task[s] = np.where(better[:, None, None], xs, best_T_task[s])
        return out

    fit = evaluate(X, assign)
    pbest_x, pbest_f = X.copy(), fit.copy()
    iters = max((S * budget) // NP, 1)
    per_iter_fe = NP / S                       # candidate FE per task per iteration
    any_list = [best_per_task.copy()]
    fe_list = [per_iter_fe]

    for _ in range(1, iters):
        assign = (assign + 1) % S
        g = np.argmax(pbest_f, axis=-1)
        gbest_x = np.take_along_axis(pbest_x, g[:, None, None], axis=1)
        r1 = rng.random((R, NP, D))
        r2 = rng.random((R, NP, D))
        V = pso.w * V + pso.c1 * r1 * (pbest_x - X) + pso.c2 * r2 * (gbest_x - X)
        V = np.clip(V, -pso.v_clip * half, pso.v_clip * half)
        X = np.clip(X + V, -half, half)
        fit = evaluate(X, assign)
        improved = fit > pbest_f
        pbest_x = np.where(improved[..., None], X, pbest_x)
        pbest_f = np.where(improved, fit, pbest_f)
        any_list.append(best_per_task.copy())
        fe_list.append(fe_list[-1] + per_iter_fe)

    counters = {"candidate_fe": 0, "secrecy_evals": 0, "ao_calls": 0, "ao_iters": 0}
    for p in probs:
        c = p.counters.as_dict()
        for k in counters:
            counters[k] += c[k]
    counters["wall_time"] = time.time() - t0
    counters["fe_per_task"] = fe_list[-1]
    counters["similarity_probe_fe"] = 0
    return MTResult(
        best_obj=best_per_task,
        best_T=repair(best_T_task, cfg.system.min_spacing, half, cfg.penalty.repair_iters),
        anytime=np.stack(any_list), anytime_fe=np.asarray(fe_list),
        lambda_trace=np.zeros(len(any_list)), donor_entropy=np.zeros(len(any_list)),
        n_transfer=0.0, n_transfer_success=0.0, neg_transfer_rate=float("nan"),
        improve_rate=float("nan"), counters=counters,
        meta={"pso": asdict(pso), "transfer": {"method": "pooled"}, "iters": iters,
              "S": S, "R": R, "budget_per_task": budget, "seed": seed},
    )
