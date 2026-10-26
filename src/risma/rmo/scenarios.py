"""Nested Eve-CSI scenario banks with common random numbers and a strict DEV/TEST split.

Nesting is by construction: one draw of ``S_high`` scenarios is made per (instance set, eta, model,
split); ``Xi_low = Xi[:20] subset Xi_mid = Xi[:50] subset Xi_high = Xi[:100]``.  Every candidate in a
comparison is scored against the *same* bank -- Eve errors are never resampled per candidate.

Seed spaces (RMO plan section 7); all disjoint, and the locked test bank depends only on the
configuration key, never on an optimiser seed:

    RMO_DEV_BANK  = 730_000    design banks used inside DEV optimisation
    RMO_DEVVAL    = 740_000    1000-sample DEV validation (unseen by DEV optimisation)
    RMO_TEST_BANK = 750_000    design banks for the final TEST grid
    RMO_TEST1000  = 760_000    the locked S_test = 1000 evaluation bank
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..channel import ChannelBatch
from ..config import SystemConfig
from ..uncertainty import EveUncertainty

CDTYPE = np.complex128

RMO_DEV_BANK = 730_000
RMO_DEVVAL = 740_000
RMO_TEST_BANK = 750_000
RMO_TEST1000 = 760_000

S_LOW, S_MID, S_HIGH = 20, 50, 100


def bank_key(sysc: SystemConfig, eta: float, model: str, channel_seed: int) -> int:
    """Deterministic, process-independent key of a (system, uncertainty, instance-set) combination."""
    s = (f"{sysc.M}|{sysc.N}|{sysc.J}|{sysc.L}|{sysc.A:.4f}|{sysc.Pmax_dbm:.4f}|"
         f"{sysc.noise_dbm:.4f}|{sysc.wavelength:.6f}|{eta:.6f}|{model}|{channel_seed}")
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 1_000_000


@dataclass
class ScenarioBank:
    """``eve`` has shape (S, R, J, N): perturbed Eve channels, scenario-major."""

    eve: np.ndarray
    eta: float
    model: str
    split: str          # "dev-design" | "dev-val" | "test-design" | "test-1000"
    seed: int

    @property
    def S(self) -> int:
        return self.eve.shape[0]

    @property
    def R(self) -> int:
        return self.eve.shape[1]

    def subset(self, s: int) -> "ScenarioBank":
        """The nested prefix of size ``s`` -- guarantees Xi_low subset Xi_mid subset Xi_high."""
        if s > self.S:
            raise ValueError(f"requested {s} scenarios from a bank of {self.S}")
        return ScenarioBank(self.eve[:s], self.eta, self.model, self.split, self.seed)

    def as_dict(self) -> Dict:
        return {"S": self.S, "R": self.R, "eta": self.eta, "model": self.model,
                "split": self.split, "seed": self.seed}


def _make(ch: ChannelBatch, eta: float, model: str, n: int, seed: int,
          split: str) -> ScenarioBank:
    unc = EveUncertainty(model, eta)
    rng = np.random.default_rng(seed)
    return ScenarioBank(unc.perturb(ch.h_e, n, rng).astype(CDTYPE), eta, model, split, seed)


# --------------------------------------------------------------------------------------


def dev_design_bank(ch: ChannelBatch, sysc: SystemConfig, eta: float, model: str,
                    channel_seed: int, opt_seed: int = 0,
                    n: int = S_HIGH) -> ScenarioBank:
    """Nested 20/50/100 design bank for DEV optimisation (may depend on the optimiser seed)."""
    key = bank_key(sysc, eta, model, channel_seed)
    return _make(ch, eta, model, n, RMO_DEV_BANK + (key * 1_000 + opt_seed) % 900_000,
                 "dev-design")


def dev_validation_bank(ch: ChannelBatch, sysc: SystemConfig, eta: float, model: str,
                        channel_seed: int, n: int = 1000) -> ScenarioBank:
    """Unseen DEV validation set; independent of any optimiser seed."""
    key = bank_key(sysc, eta, model, channel_seed)
    return _make(ch, eta, model, n, RMO_DEVVAL + key % 900_000, "dev-val")


def test_design_bank(ch: ChannelBatch, sysc: SystemConfig, eta: float, model: str,
                     channel_seed: int, opt_seed: int = 0, n: int = S_HIGH) -> ScenarioBank:
    key = bank_key(sysc, eta, model, channel_seed)
    return _make(ch, eta, model, n, RMO_TEST_BANK + (key * 1_000 + opt_seed) % 900_000,
                 "test-design")


def locked_test_bank(ch: ChannelBatch, sysc: SystemConfig, eta: float, model: str,
                     channel_seed: int, n: int = 1000) -> ScenarioBank:
    """The locked S_test = 1000 bank every reported solution is evaluated on."""
    key = bank_key(sysc, eta, model, channel_seed)
    return _make(ch, eta, model, n, RMO_TEST1000 + key % 900_000, "test-1000")
