"""Single-task derivative-free optimisers for Phase P3.

DE, PSO and CMA-ES, all
  * operating on the box-constrained vector `x = vec(T) in [-half, half]^{2M}`,
  * scoring candidates through the *same* ``MAPositionProblem`` (hence the same inner AO solver
    and the same budget counters) as every other method in the study,
  * vectorised over the instance axis, so one call evaluates ``R x K`` candidates at once,
  * counting exactly ``budget`` candidate evaluations per instance, so budgets are matched.

Constraint handling is identical for all of them: the box by clipping (C4), the minimum spacing
by the paper's penalty (Eq. 20) inside the objective, and a final repair before reporting (D11).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..problem import MAPositionProblem
from .baselines import OuterResult, _finish

FDTYPE = np.float64


# --------------------------------------------------------------------------------------


def _as_T(x: np.ndarray, M: int) -> np.ndarray:
    """(R, K, 2M) -> (R, K, M, 2)."""
    return x.reshape(x.shape[:-1] + (M, 2))


def _anytime_points(budget: int, n: int = 25) -> np.ndarray:
    return np.unique(np.round(np.geomspace(1, max(budget, 2), n)).astype(int))


class _Tracker:
    """Best-so-far bookkeeping + anytime trace, shared by the three optimisers."""

    def __init__(self, R: int, budget: int, n_anytime: int = 25):
        self.best_obj = np.full(R, -np.inf)
        self.best_x: Optional[np.ndarray] = None
        self.used = 0
        self.points = set(_anytime_points(budget, n_anytime).tolist())
        self.anytime: List[np.ndarray] = []
        self.anytime_fe: List[int] = []

    def update(self, x: np.ndarray, obj: np.ndarray) -> None:
        """``x`` (R, K, D), ``obj`` (R, K); candidates are consumed in column order."""
        K = obj.shape[1]
        for k in range(K):
            better = obj[:, k] > self.best_obj
            self.best_obj = np.where(better, obj[:, k], self.best_obj)
            if self.best_x is None:
                self.best_x = x[:, k].copy()
            else:
                self.best_x = np.where(better[:, None], x[:, k], self.best_x)
            self.used += 1
            if self.used in self.points:
                self.anytime.append(self.best_obj.copy())
                self.anytime_fe.append(self.used)


def _eval(prob: MAPositionProblem, x: np.ndarray, M: int, half: float) -> np.ndarray:
    x = np.clip(x, -half, half)
    return prob.evaluate(_as_T(x, M)).obj


# --------------------------------------------------------------------------------------
# Differential Evolution (rand/1/bin)
# --------------------------------------------------------------------------------------


def de(prob: MAPositionProblem, budget: int, seed: int = 0, NP: int = 20,
      F: float = 0.5, CR: float = 0.9) -> OuterResult:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    R, M, half = prob.R, prob.M, prob.half
    D = 2 * M
    tr = _Tracker(R, budget)

    X = prob.random_T(rng, k=NP).reshape(R, NP, D)
    fit = _eval(prob, X, M, half)
    tr.update(X, fit)

    while tr.used < budget:
        idx = np.stack([rng.permutation(NP) for _ in range(3 * R)]).reshape(R, 3, NP)
        r1, r2, r3 = idx[:, 0], idx[:, 1], idx[:, 2]
        a = np.take_along_axis(X, r1[..., None], axis=1)
        b = np.take_along_axis(X, r2[..., None], axis=1)
        c = np.take_along_axis(X, r3[..., None], axis=1)
        V = np.clip(a + F * (b - c), -half, half)
        cross = rng.random((R, NP, D)) < CR
        jrand = rng.integers(0, D, (R, NP))
        cross[np.arange(R)[:, None], np.arange(NP)[None, :], jrand] = True
        U = np.where(cross, V, X)

        take = min(NP, budget - tr.used)
        U = U[:, :take]
        fu = _eval(prob, U, M, half)
        tr.update(U, fu)
        better = fu > fit[:, :take]
        X[:, :take] = np.where(better[..., None], U, X[:, :take])
        fit[:, :take] = np.where(better, fu, fit[:, :take])

    return _finish(prob, _as_T(tr.best_x, M), tr.best_obj,
                   anytime=np.array(tr.anytime), anytime_fe=np.array(tr.anytime_fe),
                   meta={"optimizer": "DE", "NP": NP, "F": F, "CR": CR}, t0=t0)


# --------------------------------------------------------------------------------------
# Particle Swarm Optimisation
# --------------------------------------------------------------------------------------


def pso(prob: MAPositionProblem, budget: int, seed: int = 0, NP: int = 20,
        w: float = 0.72, c1: float = 1.49, c2: float = 1.49) -> OuterResult:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    R, M, half = prob.R, prob.M, prob.half
    D = 2 * M
    tr = _Tracker(R, budget)

    X = prob.random_T(rng, k=NP).reshape(R, NP, D)
    V = rng.uniform(-0.1 * half, 0.1 * half, (R, NP, D))
    fit = _eval(prob, X, M, half)
    tr.update(X, fit)
    pbest_x, pbest_f = X.copy(), fit.copy()

    while tr.used < budget:
        g = np.argmax(pbest_f, axis=1)
        gbest = np.take_along_axis(pbest_x, g[:, None, None], axis=1)      # (R,1,D)
        r1 = rng.random((R, NP, D))
        r2 = rng.random((R, NP, D))
        V = w * V + c1 * r1 * (pbest_x - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -0.5 * half, 0.5 * half)
        X = np.clip(X + V, -half, half)

        take = min(NP, budget - tr.used)
        Xt = X[:, :take]
        ft = _eval(prob, Xt, M, half)
        tr.update(Xt, ft)
        better = ft > pbest_f[:, :take]
        pbest_x[:, :take] = np.where(better[..., None], Xt, pbest_x[:, :take])
        pbest_f[:, :take] = np.where(better, ft, pbest_f[:, :take])

    return _finish(prob, _as_T(tr.best_x, M), tr.best_obj,
                   anytime=np.array(tr.anytime), anytime_fe=np.array(tr.anytime_fe),
                   meta={"optimizer": "PSO", "NP": NP, "w": w, "c1": c1, "c2": c2}, t0=t0)


# --------------------------------------------------------------------------------------
# CMA-ES  (standard (mu/mu_w, lambda), full covariance, batched over instances)
# --------------------------------------------------------------------------------------


def cmaes(prob: MAPositionProblem, budget: int, seed: int = 0,
          lam: Optional[int] = None, sigma0: float = 0.3) -> OuterResult:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    R, M, half = prob.R, prob.M, prob.half
    D = 2 * M
    lam = lam or (4 + int(3 * np.log(D)))
    mu = lam // 2

    wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    wts /= wts.sum()
    mueff = 1.0 / np.sum(wts ** 2)
    cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
    cs = (mueff + 2) / (D + mueff + 5)
    c1 = 2 / ((D + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs
    chiN = np.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))

    tr = _Tracker(R, budget)
    m = prob.random_T(rng).reshape(R, D)
    sigma = np.full(R, sigma0 * half)
    C = np.broadcast_to(np.eye(D), (R, D, D)).copy()
    pc = np.zeros((R, D))
    ps = np.zeros((R, D))
    gen = 0

    while tr.used < budget:
        gen += 1
        C = 0.5 * (C + np.swapaxes(C, -1, -2))
        ev, B = np.linalg.eigh(C)
        ev = np.maximum(ev, 1e-20)
        Ahalf = B * np.sqrt(ev)[:, None, :]                      # B * diag(sqrt(ev))
        z = rng.standard_normal((R, lam, D))
        y = np.einsum("rij,rkj->rki", Ahalf, z)
        X = m[:, None, :] + sigma[:, None, None] * y

        take = min(lam, budget - tr.used)
        Xt = np.clip(X[:, :take], -half, half)
        f = _eval(prob, Xt, M, half)
        tr.update(Xt, f)
        if take < mu:
            break

        order = np.argsort(-f, axis=1)[:, :mu]
        y_sel = np.take_along_axis(y[:, :take], order[..., None], axis=1)   # (R, mu, D)
        yw = np.einsum("k,rkd->rd", wts, y_sel)
        m = np.clip(m + sigma[:, None] * yw, -half, half)

        Cinv_half = B * (1.0 / np.sqrt(ev))[:, None, :]
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.einsum(
            "rij,rj->ri", np.einsum("rij,rkj->rik", Cinv_half, B), yw)
        ps_norm = np.linalg.norm(ps, axis=-1)
        hsig = (ps_norm / np.sqrt(1 - (1 - cs) ** (2 * gen)) / chiN
                < 1.4 + 2 / (D + 1)).astype(FDTYPE)
        pc = (1 - cc) * pc + hsig[:, None] * np.sqrt(cc * (2 - cc) * mueff) * yw

        rank1 = pc[:, :, None] * pc[:, None, :]
        rankmu = np.einsum("k,rkd,rke->rde", wts, y_sel, y_sel)
        delta = (1 - hsig) * cc * (2 - cc)
        C = ((1 - c1 - cmu) * C
             + c1 * (rank1 + delta[:, None, None] * C)
             + cmu * rankmu)
        sigma = sigma * np.exp((cs / damps) * (ps_norm / chiN - 1))
        sigma = np.clip(sigma, 1e-6 * half, half)

    return _finish(prob, _as_T(tr.best_x, M), tr.best_obj,
                   anytime=np.array(tr.anytime), anytime_fe=np.array(tr.anytime_fe),
                   meta={"optimizer": "CMA-ES", "lambda": lam, "sigma0": sigma0}, t0=t0)


OPTIMIZERS: Dict[str, Callable] = {"DE": de, "PSO": pso, "CMA-ES": cmaes}
