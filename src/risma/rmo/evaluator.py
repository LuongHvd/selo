"""Direct joint robust evaluation: no inner AO.

A candidate already carries ``(T, Phi, w)``, so the secrecy rate of Eq. (9) is computed directly
for every uncertainty realisation.  The evaluator exploits the fact that **only the eavesdropper
term depends on the scenario**:

    per candidate      H = C_BR G(T)  ->  Hw  ->  u = conj(v) * Hw  ->  gamma_b
    per (candidate, xi) y_e,j = sum_n u_n conj(h_e,j,n(xi))

so the per-scenario work is O(J N) instead of a full channel reconstruction.  Measured cost is
0.13-0.60 us per scenario evaluation (SE) versus 0.44-4.5 ms per AO call in the legacy T-only
pipeline.

Budget accounting (brief section 22):
    1 SE = one complete secrecy evaluation of one candidate under one Eve-CSI realisation.
    candidate_fe counts candidates; scenario_evals counts SE = sum_i S_i.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..channel import ChannelBatch
from .encoding import JointSpec, decode
from .scenarios import ScenarioBank

FDTYPE = np.float64
CDTYPE = np.complex128


@dataclass
class SECounters:
    candidate_fe: int = 0
    scenario_evals: int = 0
    ao_calls: int = 0            # always 0 for the joint evaluator; kept for a uniform interface
    wall_time: float = 0.0
    repair: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {"candidate_fe": self.candidate_fe, "scenario_evals": self.scenario_evals,
                "ao_calls": self.ao_calls, "wall_time": self.wall_time,
                **{f"repair_{k}": v for k, v in self.repair.items()}}


class JointRobustEvaluator:
    """Scores encoded joint designs against a (nested) scenario bank.

    ``ch``  holds ``R`` channel instances; ``bank.eve`` has shape ``(S, R, J, N)``.
    All methods accept ``y`` of shape ``(R, P, D)`` (a population per instance) or ``(R, D)``.
    """

    def __init__(self, ch: ChannelBatch, spec: JointSpec, bank: ScenarioBank,
                 chunk_scen: int = 250):
        if bank.R != ch.R:
            raise ValueError(f"bank has R={bank.R} but the channel batch has R={ch.R}")
        self.ch = ch
        self.spec = spec
        self.bank = bank
        self.chunk_scen = chunk_scen
        self.counters = SECounters()

    # ------------------------------------------------------------------ core
    def _prep(self, y: np.ndarray):
        """Scenario-independent part: returns (u (R,P,N), gamma_b (R,P))."""
        spec = self.spec
        T, v, w = decode(y, spec)
        H = self.ch.H_BR(T)                                        # (R,P,N,M)
        Hw = np.matmul(H, w[..., None])[..., 0]                    # (R,P,N)
        u = np.conj(v) * Hw                                        # (R,P,N)
        hb = self.ch.h_b[:, None, :] if u.ndim == 3 else self.ch.h_b
        yb = np.sum(u * np.conj(hb), axis=-1)
        gb = np.abs(yb) ** 2 / self.ch.sigma_b2
        return u, gb

    def rates(self, y: np.ndarray, n_scen: Optional[int] = None,
              bank: Optional[ScenarioBank] = None, count: bool = True) -> np.ndarray:
        """Secrecy rates, shape ``(S_use, R, P)`` (or ``(S_use, R)`` if ``y`` is ``(R, D)``).

        ``n_scen`` selects the nested prefix ``Xi[:n_scen]`` of the bank.
        """
        bank = bank or self.bank
        S_use = bank.S if n_scen is None else min(int(n_scen), bank.S)
        squeeze = (y.ndim == 2)
        if squeeze:
            y = y[:, None, :]
        R, P, _ = y.shape
        u, gb = self._prep(y)                                      # (R,P,N), (R,P)
        se2 = self.ch.sigma_e2                                     # (R,J)

        out = np.empty((S_use, R, P), dtype=FDTYPE)
        cu = np.conj(u)
        for a in range(0, S_use, self.chunk_scen):
            b = min(a + self.chunk_scen, S_use)
            eve = bank.eve[a:b]                                    # (s,R,J,N)
            ye = np.einsum("rpn,srjn->srpj", u, np.conj(eve), optimize=True)
            ge = np.abs(ye) ** 2 / se2[None, :, None, :]
            out[a:b] = np.log2(1.0 + gb)[None] - np.log2(1.0 + ge.sum(axis=-1))
        np.maximum(out, 0.0, out)                                  # the [.]^+ of Eq. (9)

        if count:
            self.counters.candidate_fe += R * P
            self.counters.scenario_evals += R * P * S_use
        return out[:, :, 0] if squeeze else out

    # ------------------------------------------------------------------ nominal (eta = 0)
    def nominal(self, y: np.ndarray, count: bool = False) -> np.ndarray:
        """Secrecy rate at the *estimated* Eve channel, i.e. the uncertainty-blind value."""
        squeeze = (y.ndim == 2)
        if squeeze:
            y = y[:, None, :]
        R, P, _ = y.shape
        u, gb = self._prep(y)
        he = self.ch.h_e                                           # (R,J,N)
        ye = np.einsum("rpn,rjn->rpj", u, np.conj(he), optimize=True)
        ge = np.abs(ye) ** 2 / self.ch.sigma_e2[:, None, :]
        rs = np.maximum(np.log2(1.0 + gb) - np.log2(1.0 + ge.sum(axis=-1)), 0.0)
        if count:
            self.counters.candidate_fe += R * P
            self.counters.scenario_evals += R * P
        return rs[:, 0] if squeeze else rs

    # ------------------------------------------------------------------ reference cross-check
    def rates_reference(self, y: np.ndarray, n_scen: Optional[int] = None,
                        bank: Optional[ScenarioBank] = None) -> np.ndarray:
        """Same quantity computed through the validated `risma.secrecy.secrecy_rate`.

        Used only by the R1 validation tests (TEST J5) -- it re-forms the channel for every
        scenario and is therefore far slower.
        """
        from ..secrecy import secrecy_rate

        bank = bank or self.bank
        S_use = bank.S if n_scen is None else min(int(n_scen), bank.S)
        squeeze = (y.ndim == 2)
        if squeeze:
            y = y[:, None, :]
        R, P, _ = y.shape
        T, v, w = decode(y, self.spec)
        H = self.ch.H_BR(T)
        hb = self.ch.h_b[:, None, :]
        se2 = self.ch.sigma_e2[:, None, :]
        out = np.empty((S_use, R, P), dtype=FDTYPE)
        for s in range(S_use):
            he = bank.eve[s][:, None, :, :]                        # (R,1,J,N)
            out[s] = secrecy_rate(H, hb, he, v, w, self.ch.sigma_b2, se2, clip=True)
        return out[:, :, 0] if squeeze else out
