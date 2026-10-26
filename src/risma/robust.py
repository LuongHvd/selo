"""Robust MA-position optimisation under imperfect Eve CSI (Phase P1+).

Structure (documented in reports/P1_robustness.md):

1. **Inner solver** -- for a candidate ``T`` the BS must commit to *one* pair ``(w, Phi)``.  We obtain it
   by running the *unchanged* Algorithm 1 on an **extended Eve set**: the ``S_design`` uncertainty
   realisations are stacked into ``S*J`` "virtual Eves" whose noise powers are scaled by ``S``.  Because
   ``B = sum_{s,j} c c^H / (S*sigma^2) = (1/S) sum_s B_s``, this is exactly the *scenario-averaged*
   leakage design, and it keeps both closed forms (Eq. 10 and Eqs. 12-18) intact -- no new algorithm, and
   the identical inner solver for every outer optimiser.
   For ``S_design = 1`` with the nominal channel this reduces to the perfect-CSI solver of P0.

2. **Outer objective** -- the resulting ``(T, w, Phi)`` is scored on the *design* scenarios with a robust
   functional: ``nominal`` / ``mean`` / ``worst`` / ``cvar_alpha`` / ``quantile_q``.

3. **Reporting** -- the returned design is evaluated on ``S_test`` **unseen, locked** scenarios;
   mean / median / 5th percentile / CVaR / secrecy-outage probability all come from that set only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .ao_solver import ao_solve
from .channel import ChannelBatch
from .config import ExperimentConfig
from .problem import BudgetCounters, EvalResult, MAPositionProblem
from .secrecy import penalised_objective, secrecy_rate, spacing_violation
from .uncertainty import EveUncertainty

FDTYPE = np.float64
CDTYPE = np.complex128


# --------------------------------------------------------------------------------------
# robust functionals
# --------------------------------------------------------------------------------------


def aggregate_robust(rs: np.ndarray, objective: str, alpha: float = 0.10,
                     axis: int = -1) -> np.ndarray:
    """Reduce per-scenario secrecy rates to a scalar robust score.

    ``rs`` has the scenario axis last by default.
    """
    if objective in ("nominal", "mean"):
        return np.mean(rs, axis=axis)
    if objective == "worst":
        return np.min(rs, axis=axis)
    if objective.startswith("cvar"):
        # mean of the worst alpha-fraction (lower tail)
        S = rs.shape[axis]
        k = max(int(np.ceil(alpha * S)), 1)
        part = np.sort(rs, axis=axis)
        idx = [slice(None)] * rs.ndim
        idx[axis] = slice(0, k)
        return np.mean(part[tuple(idx)], axis=axis)
    if objective.startswith("quantile"):
        return np.quantile(rs, alpha, axis=axis)
    raise ValueError(f"unknown robust objective {objective!r}")


def outage_probability(rs: np.ndarray, threshold: float, axis: int = -1) -> np.ndarray:
    return np.mean(rs < threshold, axis=axis)


# --------------------------------------------------------------------------------------


@dataclass
class RobustReport:
    mean: np.ndarray
    median: np.ndarray
    q05: np.ndarray
    cvar10: np.ndarray
    sop: Dict[float, np.ndarray]
    rs_test: Optional[np.ndarray] = None      # (R, S_test) if kept


class RobustMAProblem:
    """Outer problem ``F_robust(T)`` under Eve-CSI uncertainty.

    Parameters
    ----------
    ch_hat : ChannelBatch with the **nominal (estimated)** Eve channels
    design_eve : (S_design, R, J, N) perturbed Eve channels used for design
    objective : 'nominal' | 'mean' | 'worst' | 'cvar' | 'quantile'
    alpha : tail level for cvar / quantile
    inner_scenarios : how many design scenarios the *inner* AO solver averages over.
        ``None`` -> all of them; ``0``/``1`` -> nominal inner design (paper-style).
    """

    def __init__(
        self,
        ch_hat: ChannelBatch,
        cfg: ExperimentConfig,
        design_eve: np.ndarray,
        objective: str = "cvar",
        alpha: float = 0.10,
        inner_scenarios: Optional[int] = None,
        score_on_nominal: bool = False,
    ) -> None:
        self.ch = ch_hat
        self.cfg = cfg
        self.sys = cfg.system
        self.design_eve = np.asarray(design_eve, dtype=CDTYPE)
        self.objective = objective
        self.alpha = alpha
        self.score_on_nominal = score_on_nominal or objective == "nominal"
        self.counters = BudgetCounters()

        S, R, J, N = self.design_eve.shape
        assert R == ch_hat.R and J == ch_hat.J and N == ch_hat.N
        self.S_design = S
        n_inner = S if inner_scenarios is None else int(inner_scenarios)
        n_inner = max(min(n_inner, S), 0)
        self.n_inner = n_inner

        if n_inner <= 1 and objective == "nominal":
            self._he_inner = ch_hat.h_e                                   # (R, J, N)
            self._se2_inner = ch_hat.sigma_e2
        elif n_inner <= 1:
            self._he_inner = ch_hat.h_e
            self._se2_inner = ch_hat.sigma_e2
        else:
            # stack the first n_inner scenarios as S*J "virtual Eves" with noise power S*sigma^2
            he = np.transpose(self.design_eve[:n_inner], (1, 0, 2, 3)).reshape(R, n_inner * J, N)
            self._he_inner = he
            self._se2_inner = np.repeat(ch_hat.sigma_e2, n_inner, axis=1) * float(n_inner)

    # ------------------------------------------------------------------ helpers
    @property
    def R(self) -> int:
        return self.ch.R

    @property
    def half(self) -> float:
        return self.sys.region_half

    @property
    def min_spacing(self) -> float:
        return self.sys.min_spacing

    def random_T(self, rng, k=None):
        from .ma_constraints import random_feasible
        lead = (self.R,) if k is None else (self.R, k)
        return random_feasible(rng, lead, self.sys.M, self.min_spacing, self.half)

    def repair(self, T):
        from .ma_constraints import repair
        return repair(T, self.min_spacing, self.half, self.cfg.penalty.repair_iters)

    # ------------------------------------------------------------------ core
    def design(self, T: np.ndarray) -> EvalResult:
        """Run the inner solver: returns (w, Phi) plus the *design-set* robust score."""
        T = np.asarray(T, dtype=FDTYPE)
        lead = T.shape[:-2]
        extra = T.ndim - 3
        ch = self.ch
        H = ch.H_BR(T)
        hb = ch.h_b.reshape((ch.R,) + (1,) * extra + (ch.N,))
        he = self._he_inner.reshape((ch.R,) + (1,) * extra + self._he_inner.shape[1:])
        se2 = self._se2_inner.reshape((ch.R,) + (1,) * extra + (self._se2_inner.shape[1],))

        res = ao_solve(H, hb, he, ch.sigma_b2, se2, self.sys.Pmax, self.cfg.ao)

        if self.score_on_nominal:
            rs_scen = secrecy_rate(H, hb, ch.h_e.reshape((ch.R,) + (1,) * extra + (ch.J, ch.N)),
                                   res.v, res.w, ch.sigma_b2,
                                   ch.sigma_e2.reshape((ch.R,) + (1,) * extra + (ch.J,)))[..., None]
        else:
            rs_scen = self._scen_rs(H, hb, res.v, res.w, self.design_eve, extra)

        score = aggregate_robust(rs_scen, self.objective, self.alpha, axis=-1)
        obj = penalised_objective(score, T, self.min_spacing, self.cfg.penalty.chi)

        n = int(np.prod(lead))
        self.counters.candidate_fe += n
        self.counters.ao_calls += n
        self.counters.ao_iters += n * res.iters
        self.counters.secrecy_evals += n * max(rs_scen.shape[-1], 1)
        return EvalResult(rs=score, obj=obj, w=res.w, v=res.v, ao_iters=res.iters)

    def _scen_rs(self, H, hb, v, w, eve: np.ndarray, extra: int) -> np.ndarray:
        """Per-scenario secrecy rate.  ``eve`` is (S, R, J, N); result (..., S)."""
        S = eve.shape[0]
        ch = self.ch
        out = []
        # loop over scenarios keeps peak memory bounded; each step is fully vectorised
        for s in range(S):
            he = eve[s].reshape((ch.R,) + (1,) * extra + (ch.J, ch.N))
            se2 = ch.sigma_e2.reshape((ch.R,) + (1,) * extra + (ch.J,))
            out.append(secrecy_rate(H, hb, he, v, w, ch.sigma_b2, se2))
        return np.stack(out, axis=-1)

    # convenience alias so that the outer optimisers can be reused unchanged
    def evaluate(self, T, count: bool = True, **kw) -> EvalResult:
        return self.design(T)

    # ------------------------------------------------------------------ reporting
    def report_on_test(self, T: np.ndarray, test_eve: np.ndarray,
                       thresholds: Sequence[float] = (0.5, 1.0, 2.0, 3.0),
                       keep_rs: bool = False, chunk: int = 100) -> RobustReport:
        """Out-of-sample evaluation of a finished design on locked test scenarios."""
        T = self.repair(np.asarray(T, dtype=FDTYPE))
        ev = self.design(T)                     # (w, Phi) for the design; not counted as budget
        self.counters.candidate_fe -= T.shape[0]
        self.counters.ao_calls -= T.shape[0]

        ch = self.ch
        H = ch.H_BR(T)
        hb = ch.h_b
        se2 = ch.sigma_e2
        S = test_eve.shape[0]
        parts: List[np.ndarray] = []
        for a in range(0, S, chunk):
            blk = test_eve[a: a + chunk]
            rs = np.stack([
                secrecy_rate(H, hb, blk[s], ev.v, ev.w, ch.sigma_b2, se2) for s in range(blk.shape[0])
            ], axis=-1)
            parts.append(rs)
        rs_test = np.concatenate(parts, axis=-1)          # (R, S_test)

        return RobustReport(
            mean=rs_test.mean(axis=-1),
            median=np.median(rs_test, axis=-1),
            q05=np.quantile(rs_test, 0.05, axis=-1),
            cvar10=aggregate_robust(rs_test, "cvar", 0.10, axis=-1),
            sop={float(t): outage_probability(rs_test, t, axis=-1) for t in thresholds},
            rs_test=rs_test if keep_rs else None,
        )
