"""The outer optimisation problem  F(T; H) = R_s(w*(T,H), Phi*(T,H), T; H).

This module is the **fairness contract** of the whole study: every outer optimiser
(PPO, TD3, DDPG, random search, DE, PSO, CMA-ES, the proposed multitask method, ...)
must obtain its objective values through :class:`MAPositionProblem`, so that they all
share the *identical* inner AO solver and the *identical* evaluation budget accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .ao_solver import AOResult, ao_solve
from .channel import ChannelBatch
from .config import ExperimentConfig
from .ma_constraints import repair
from .secrecy import penalised_objective, secrecy_rate, spacing_violation

FDTYPE = np.float64
CDTYPE = np.complex128


@dataclass
class BudgetCounters:
    """Separate accounting of every notion of computational cost (project spec section 9)."""

    candidate_fe: int = 0        # number of candidate MA configurations evaluated
    secrecy_evals: int = 0       # number of R_s(.) evaluations (candidates x scenarios)
    ao_calls: int = 0            # number of Algorithm-1 invocations
    ao_iters: int = 0            # number of Algorithm-1 *iterations* (w+Phi updates)
    wall_time: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "candidate_fe": self.candidate_fe,
            "secrecy_evals": self.secrecy_evals,
            "ao_calls": self.ao_calls,
            "ao_iters": self.ao_iters,
            "wall_time": self.wall_time,
        }

    def __iadd__(self, other: "BudgetCounters") -> "BudgetCounters":
        self.candidate_fe += other.candidate_fe
        self.secrecy_evals += other.secrecy_evals
        self.ao_calls += other.ao_calls
        self.ao_iters += other.ao_iters
        self.wall_time += other.wall_time
        return self


@dataclass
class EvalResult:
    rs: np.ndarray            # (...) secrecy rate after [.]^+ at the *given* T
    obj: np.ndarray           # (...) penalised objective used by the optimisers
    w: np.ndarray             # (..., M)
    v: np.ndarray             # (..., N)
    ao_iters: int = 0
    rs_raw: np.ndarray = None  # (...) log-ratio *before* [.]^+; used for landscape analysis
                               # (the [.]^+ clip creates ties at 0 that destroy rank statistics)


class MAPositionProblem:
    """Outer problem over MA positions for a *batch of independent instances*.

    ``ch`` holds ``R`` independent channel realisations; every optimiser is run on all of
    them simultaneously (vectorised), which is mathematically identical to ``R`` separate
    runs but ~2 orders of magnitude faster.
    """

    def __init__(
        self,
        ch: ChannelBatch,
        cfg: ExperimentConfig,
        optimize_phase: bool = True,
        v_fixed: Optional[np.ndarray] = None,
        chunk: int = 4096,
    ) -> None:
        self.ch = ch
        self.cfg = cfg
        self.sys = cfg.system
        self.optimize_phase = optimize_phase
        self.v_fixed = v_fixed
        self.chunk = chunk
        self.counters = BudgetCounters()

    # ------------------------------------------------------------------ helpers
    @property
    def R(self) -> int:
        return self.ch.R

    @property
    def M(self) -> int:
        return self.sys.M

    @property
    def half(self) -> float:
        return self.sys.region_half

    @property
    def min_spacing(self) -> float:
        return self.sys.min_spacing

    def random_T(self, rng: np.random.Generator, k: int | None = None) -> np.ndarray:
        lead = (self.R,) if k is None else (self.R, k)
        from .ma_constraints import random_feasible

        return random_feasible(rng, lead, self.M, self.min_spacing, self.half)

    def repair(self, T: np.ndarray) -> np.ndarray:
        return repair(T, self.min_spacing, self.half, self.cfg.penalty.repair_iters)

    # ------------------------------------------------------------------ evaluation
    def evaluate(
        self,
        T: np.ndarray,
        count: bool = True,
        keep_wv: bool = True,
        ch: Optional[ChannelBatch] = None,
    ) -> EvalResult:
        """Run Algorithm 1 for every MA configuration in ``T``.

        Parameters
        ----------
        T : (R, ..., M, 2)  MA positions.  Leading axis must be the instance axis.
        """
        ch = ch if ch is not None else self.ch
        T = np.asarray(T, dtype=FDTYPE)
        lead = T.shape[:-2]
        assert lead[0] == ch.R, f"leading axis of T must be the instance axis ({ch.R})"

        H = ch.H_BR(T)                                     # (R, ..., N, M)
        extra = T.ndim - 3
        hb = ch.h_b.reshape((ch.R,) + (1,) * extra + (ch.N,))
        he = ch.h_e.reshape((ch.R,) + (1,) * extra + (ch.J, ch.N))
        se2 = ch.sigma_e2.reshape((ch.R,) + (1,) * extra + (ch.J,))

        v_init = None
        if self.v_fixed is not None:
            v_init = self.v_fixed.reshape((ch.R,) + (1,) * extra + (ch.N,))

        res: AOResult = ao_solve(
            H, hb, he, ch.sigma_b2, se2, self.sys.Pmax, self.cfg.ao,
            v_init=v_init, optimize_phase=self.optimize_phase,
        )
        rs = res.rs
        obj = penalised_objective(rs, T, self.min_spacing, self.cfg.penalty.chi)

        if count:
            n = int(np.prod(lead))
            self.counters.candidate_fe += n
            self.counters.secrecy_evals += n
            self.counters.ao_calls += n
            self.counters.ao_iters += n * res.iters

        return EvalResult(
            rs=rs, obj=obj,
            w=res.w if keep_wv else None,
            v=res.v if keep_wv else None,
            ao_iters=res.iters,
            rs_raw=res.rs_raw,
        )

    def evaluate_fixed_design(
        self, T: np.ndarray, w: np.ndarray, v: np.ndarray, ch: ChannelBatch
    ) -> np.ndarray:
        """R_s of an already designed (T, w, Phi) triple under *another* channel batch.

        Used for out-of-sample robustness evaluation (Phase P1+): the design is frozen and
        only the (uncertain) Eve channels change.
        """
        H = ch.H_BR(T)
        extra = T.ndim - 3
        hb = ch.h_b.reshape((ch.R,) + (1,) * extra + (ch.N,))
        he = ch.h_e.reshape((ch.R,) + (1,) * extra + (ch.J, ch.N))
        se2 = ch.sigma_e2.reshape((ch.R,) + (1,) * extra + (ch.J,))
        return secrecy_rate(H, hb, he, v, w, ch.sigma_b2, se2, clip=True)

    # ------------------------------------------------------------------ reporting
    def report_solution(self, T: np.ndarray) -> Dict[str, np.ndarray]:
        """Final, paper-reportable numbers for a candidate solution.

        The reported secrecy rate is always evaluated at a **repaired feasible** ``T``
        (project decision D11); the feasibility of the raw solution is reported alongside.
        """
        T = np.asarray(T, dtype=FDTYPE)
        raw_viol = spacing_violation(T, self.min_spacing)
        T_rep = self.repair(T) if self.cfg.penalty.repair_on_report else T
        ev = self.evaluate(T_rep, count=False)
        return {
            "rs": ev.rs,
            "T": T_rep,
            "w": ev.w,
            "v": ev.v,
            "raw_spacing_violation": raw_viol,
            "raw_feasible": raw_viol <= 1e-9,
        }
