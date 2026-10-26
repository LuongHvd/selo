"""Configuration objects for the RIS-MA-PLS simulator.

Every scientifically relevant parameter lives here (or in a YAML/JSON file that is
deserialised into these dataclasses).  Nothing important may be hard-coded in scripts.

Reference: docs/reference_paper_summary.md, section 5 (Simulation setup).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# unit helpers
# --------------------------------------------------------------------------------------


def dbm_to_watt(x_dbm: float) -> float:
    return 10.0 ** ((x_dbm - 30.0) / 10.0)


def watt_to_dbm(x_w: float) -> float:
    return 10.0 * math.log10(x_w) + 30.0


def db_to_lin(x_db: float) -> float:
    return 10.0 ** (x_db / 10.0)


# --------------------------------------------------------------------------------------
# system configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemConfig:
    """Physical / system-model parameters (paper Table + Section IV)."""

    # --- array sizes -------------------------------------------------------------
    M: int = 4                      # number of movable antennas at the BS
    N: int = 16                     # number of RIS reflecting elements
    J: int = 3                      # number of colluding eavesdroppers
    L: int = 4                      # number of propagation paths (Lt = Lr = L)

    # --- MA region ---------------------------------------------------------------
    A: float = 4.0                  # normalised region size (region is A*lambda square)
    wavelength: float = 0.125       # lambda [m]

    # --- power / noise -----------------------------------------------------------
    Pmax_dbm: float = 30.0          # BS transmit power budget
    noise_dbm: float = -85.0        # noise power at Bob and every Eve

    # --- path loss ---------------------------------------------------------------
    pathloss_exponent: float = 2.5
    pathloss_ref_db: float = 30.0   # reference path LOSS at 1 m (so gain = -30 dB)

    # --- geometry [m] ------------------------------------------------------------
    bs_pos: Tuple[float, float] = (0.0, 0.0)
    user_pos: Tuple[float, float] = (45.0, 0.0)
    ris_pos: Tuple[float, float] = (25.0, 5.0)
    eve_center: Tuple[float, float] = (50.0, 5.0)
    eve_radius: float = 2.0

    # --- modelling switches (see docs/reference_paper_summary.md section 9) -------
    angle_support: str = "2pi"      # "2pi" (simulation section) | "pi" (model section)
    prm_normalization: str = "total"
    # "total"    : sigma_ij ~ CN(0, PL/(Lr*Lt))  -> E|H_BR[n,m]|^2 = PL, total power conserved,
    #              so the number of paths L changes only the spatial structure  [ASSUMPTION D2]
    # "per_path" : sigma_ij ~ CN(0, PL/L)        -> E|H_BR[n,m]|^2 = L*PL, the gain grows with L.
    #              Not power-conserving, but it is what reproduces Fig. 9 of the paper (SR
    #              increasing with L); kept as a diagnostic, never as the default.
    ris_element_spacing_factor: float = 0.5   # RIS UPA spacing = factor * lambda

    # --- derived -----------------------------------------------------------------
    @property
    def Pmax(self) -> float:
        return dbm_to_watt(self.Pmax_dbm)

    @property
    def sigma2(self) -> float:
        return dbm_to_watt(self.noise_dbm)

    @property
    def region_size(self) -> float:
        """Side length of the square MA region in metres."""
        return self.A * self.wavelength

    @property
    def region_half(self) -> float:
        return 0.5 * self.A * self.wavelength

    @property
    def min_spacing(self) -> float:
        """C3: minimum inter-MA distance = lambda / 2."""
        return 0.5 * self.wavelength

    @property
    def Lt(self) -> int:
        return self.L

    @property
    def Lr(self) -> int:
        return self.L

    def pathloss_linear(self, distance_m: float) -> float:
        """Large-scale channel *gain* (linear) at the given distance."""
        d = max(distance_m, 1.0)
        return db_to_lin(-self.pathloss_ref_db) * d ** (-self.pathloss_exponent)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AOConfig:
    """Inner alternating-optimisation solver (Algorithm 1)."""

    max_iters: int = 10             # I_AO  (paper: "AO method is limited to 10 iterations")
    tol: float = 1e-3               # eps_AO (paper: stopping criterion 1e-3)
    mm_iters: int = 10              # inner MM/Dinkelbach iterations  [ASSUMPTION D10]
    mm_tol: float = 1e-3
    lambda_max_mode: str = "auto"      # "auto" | "lowrank" (exact) | "direct" | "bound" | "gersh"
    init_phase: str = "identity"    # "identity" | "random"          [ASSUMPTION D7]
    nullspace_guard: bool = True    # avoid the rank(H)<M deadlock (D-2026-09-08-06)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PenaltyConfig:
    """Constraint handling shared by every outer optimiser (paper Eq. 20)."""

    chi: float = 10.0               # penalty factor (paper Table I)
    repair_on_report: bool = True   # report R_s only at a repaired feasible T (D11)
    repair_iters: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level container written next to every result file."""

    name: str
    phase: str
    system: SystemConfig = field(default_factory=SystemConfig)
    ao: AOConfig = field(default_factory=AOConfig)
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    n_mc: int = 200                 # Monte-Carlo channel realisations
    base_seed: int = 20260101
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "phase": self.phase,
            "system": self.system.to_dict(),
            "ao": self.ao.to_dict(),
            "penalty": self.penalty.to_dict(),
            "n_mc": self.n_mc,
            "base_seed": self.base_seed,
            "extra": self.extra,
        }

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def with_system(self, **kw: Any) -> "ExperimentConfig":
        return replace(self, system=replace(self.system, **kw))


def load_config(path: str) -> ExperimentConfig:
    """Load an ExperimentConfig from JSON or YAML."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read YAML configs") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return config_from_dict(raw)


def config_from_dict(raw: Dict[str, Any]) -> ExperimentConfig:
    sys_kw = dict(raw.get("system", {}))
    for key in ("bs_pos", "user_pos", "ris_pos", "eve_center"):
        if key in sys_kw and sys_kw[key] is not None:
            sys_kw[key] = tuple(sys_kw[key])
    return ExperimentConfig(
        name=raw["name"],
        phase=raw["phase"],
        system=SystemConfig(**sys_kw),
        ao=AOConfig(**raw.get("ao", {})),
        penalty=PenaltyConfig(**raw.get("penalty", {})),
        n_mc=raw.get("n_mc", 200),
        base_seed=raw.get("base_seed", 20260101),
        extra=raw.get("extra", {}),
    )


REFERENCE_CONFIG = ExperimentConfig(
    name="reference_default",
    phase="P0",
    system=SystemConfig(),
    ao=AOConfig(),
    penalty=PenaltyConfig(),
    n_mc=200,
)
"""The paper's default configuration: M=4, N=16, J=3, L=4, A=4, Pmax=30 dBm, sigma2=-85 dBm."""
