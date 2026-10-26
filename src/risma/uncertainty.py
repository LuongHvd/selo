"""Eavesdropper-CSI uncertainty models (Phase P1).

    h_{e,j} = hhat_{e,j} + Delta h_{e,j}

Two families, both parameterised by a *normalised* uncertainty level ``eta``:

* ``bounded_sphere`` / ``bounded_ball`` -- deterministic norm-bounded set
  ``||Delta h_{e,j}||_2 <= eta * ||hhat_{e,j}||_2``.  ``sphere`` samples the boundary of the set
  (the worst case of a norm-bounded set always lies on the boundary, so this is the right
  discretisation for a worst-case objective); ``ball`` samples the interior uniformly.
* ``gaussian`` -- stochastic ``Delta h_{e,j} ~ CN(0, sigma_j^2 I_N)`` with
  ``sigma_j^2 = eta^2 ||hhat_{e,j}||^2 / N``, i.e. ``E||Delta h_{e,j}||^2 = eta^2 ||hhat_{e,j}||^2``,
  matching the deterministic level in the mean-square sense.

**Design/test separation** is structural: :func:`design_scenarios` and :func:`locked_test_scenarios` draw from
different, independently seeded ``numpy.random.Generator`` streams built from disjoint seed spaces, and the
test stream depends only on a locked configuration key -- never on the optimiser seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from .channel import ChannelBatch

CDTYPE = np.complex128
FDTYPE = np.float64

DESIGN_SEED_SPACE = 10_000_000
TEST_SEED_SPACE = 70_000_000


@dataclass(frozen=True)
class EveUncertainty:
    """Uncertainty model for the RIS->Eve channels."""

    model: str = "bounded_sphere"     # bounded_sphere | bounded_ball | gaussian
    eta: float = 0.10

    def to_dict(self) -> Dict:
        return asdict(self)

    # ------------------------------------------------------------------ sampling
    def perturb(self, h_hat_e: np.ndarray, n_scen: int, rng: np.random.Generator) -> np.ndarray:
        """Return ``n_scen`` perturbed Eve channel sets.

        Parameters
        ----------
        h_hat_e : (R, J, N) complex   nominal (estimated) Eve channels
        Returns
        -------
        (n_scen, R, J, N) complex     perturbed channels ``hhat + Delta h``
        """
        R, J, N = h_hat_e.shape
        if self.eta <= 0.0 or n_scen <= 0:
            return np.broadcast_to(h_hat_e, (max(n_scen, 1), R, J, N)).copy()

        norm = np.linalg.norm(h_hat_e, axis=-1)                    # (R, J)
        shape = (n_scen, R, J, N)
        z = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(CDTYPE)

        if self.model == "gaussian":
            # E||dh||^2 = eta^2 ||hhat||^2  =>  per-entry variance eta^2||hhat||^2 / N
            scale = self.eta * norm / np.sqrt(N)                   # (R, J)
            delta = z * (scale[None, :, :, None] / np.sqrt(2.0))
        elif self.model in ("bounded_sphere", "bounded_ball"):
            u = z / np.linalg.norm(z, axis=-1, keepdims=True)       # uniform on the complex sphere
            radius = self.eta * norm[None, :, :]                    # (1, R, J)
            if self.model == "bounded_ball":
                # uniform in the 2N-real-dimensional ball
                v = rng.random((n_scen, R, J))
                radius = radius * v ** (1.0 / (2 * N))
            delta = u * radius[..., None]
        else:
            raise ValueError(f"unknown uncertainty model {self.model!r}")

        return h_hat_e[None, ...] + delta

    def realised_level(self, h_hat_e: np.ndarray, h_e: np.ndarray) -> np.ndarray:
        """Empirical ``||dh|| / ||hhat||`` -- used by the unit tests and the report."""
        d = h_e - h_hat_e[None, ...] if h_e.ndim == h_hat_e.ndim + 1 else h_e - h_hat_e
        return np.linalg.norm(d, axis=-1) / np.linalg.norm(h_hat_e, axis=-1)


# --------------------------------------------------------------------------------------
# scenario factories with structurally separated streams
# --------------------------------------------------------------------------------------


def design_scenarios(ch: ChannelBatch, unc: EveUncertainty, n_scen: int,
                     config_key: int, opt_seed: int) -> np.ndarray:
    """Design-time uncertainty realisations (may depend on the optimiser seed)."""
    rng = np.random.default_rng(DESIGN_SEED_SPACE + (config_key * 1_000 + opt_seed) % 9_000_000)
    return unc.perturb(ch.h_e, n_scen, rng)


def locked_test_scenarios(ch: ChannelBatch, unc: EveUncertainty, n_scen: int,
                   config_key: int) -> np.ndarray:
    """Locked out-of-sample uncertainty realisations.

    Depends **only** on the configuration key and the uncertainty model, never on an optimiser
    seed, so the same 1000 test scenarios are shared by every method and every seed.
    """
    tag = int(1e6 * unc.eta) + {"bounded_sphere": 0, "bounded_ball": 1, "gaussian": 2}[unc.model]
    rng = np.random.default_rng(TEST_SEED_SPACE + (config_key * 97 + tag) % 20_000_000)
    return unc.perturb(ch.h_e, n_scen, rng)
