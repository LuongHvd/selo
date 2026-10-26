"""Joint real-valued encoding  y = [vec(T), theta, Re w, Im w]  with  D = 4M + N.

Constraint handling (RMO plan section 5), Lamarckian: `repair_vector` writes the repaired design
back into the genotype, so the search never operates on a plateau where many vectors decode to the
same physical design.

  * `T`      clipped to the square region C, then the **existing validated** `ma_constraints.repair`
             enforces the lambda/2 minimum spacing (deterministic, bounded iterations).
  * `theta`  wrapped modulo 2*pi; the unit-modulus constraint |phi_n| = 1 is exact by construction.
  * `w`      projected to `sqrt(Pmax) w/||w||` **only if** ||w||^2 > Pmax; a beamformer already
             inside the power ball is left untouched.  A numerically zero beamformer is replaced by
             a deterministic fallback and counted.

The simulator's RIS vector satisfies ``v_n = conj(phi_n)`` (see `risma.secrecy.snrs`), therefore
``decode`` returns ``v = exp(-1j*theta)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..config import SystemConfig
from ..ma_constraints import fpa_layout, random_feasible, repair
from ..secrecy import spacing_violation

FDTYPE = np.float64
CDTYPE = np.complex128
_ZERO_W = 1e-12

# C3 feasibility tolerance, matching `ma_constraints.feasibility_report`.
FEAS_TOL = 1e-9
# Boundary-deadlock fallback (see docs/protocol_decisions.md, D-2026-09-11-01).
_SHRINK_ROUNDS = 4
_SHRINK = 0.85


def repair_positions(T: np.ndarray, spec: "JointSpec",
                     stats: Optional[Dict] = None) -> np.ndarray:
    """Enforce C4 (region) and C3 (minimum spacing) on MA positions.

    `ma_constraints.repair` is the frozen, validated operator used by P0-P4 and is applied
    first, unchanged.  It can however deadlock: antennas pinned to a corner of the region are
    pushed apart along a direction that points outwards, and the subsequent clip returns them
    to the same point.  Measured incidence in a realistic joint DE loop is 0 / 7200 designs
    under the midpoint bound rule but 0.17-0.83 % under the clip rule, where the violation is a
    full lambda/2 (two antennas exactly coincident).

    Two deterministic fallbacks are therefore applied to the violating designs only:
      1. shrink towards the region centre by 0.85 and re-repair (up to 4 rounds), which frees
         the points from the boundary;
      2. if anything still violates C3, substitute the FPA lambda/2 grid, which is feasible by
         construction.
    Both are counted in ``stats``.  Feasible designs are returned untouched, so behaviour on
    the population produced by the canonical midpoint rule is identical to `repair` alone.
    """
    T = np.clip(np.asarray(T, dtype=FDTYPE), -spec.half, spec.half)
    Tr = repair(T, spec.min_spacing, spec.half, spec.repair_iters)

    n_shrink = 0
    for _ in range(_SHRINK_ROUNDS):
        bad = spacing_violation(Tr, spec.min_spacing) > FEAS_TOL
        if not np.any(bad):
            break
        n_shrink += int(np.count_nonzero(bad))
        cand = repair(Tr * _SHRINK, spec.min_spacing, spec.half, spec.repair_iters)
        Tr = np.where(bad[..., None, None], cand, Tr)

    n_fpa = 0
    bad = spacing_violation(Tr, spec.min_spacing) > FEAS_TOL
    if np.any(bad):
        n_fpa = int(np.count_nonzero(bad))
        fb = fpa_layout(spec.M, 2.0 * spec.min_spacing, spec.half)
        Tr = np.where(bad[..., None, None], fb, Tr)

    if stats is not None:
        stats["spacing_shrink"] = stats.get("spacing_shrink", 0) + n_shrink
        stats["spacing_fpa_fallback"] = stats.get("spacing_fpa_fallback", 0) + n_fpa
        stats["residual_infeasible"] = stats.get("residual_infeasible", 0) + int(
            np.count_nonzero(spacing_violation(Tr, spec.min_spacing) > FEAS_TOL))
    return Tr


@dataclass(frozen=True)
class JointSpec:
    """Everything the encoding needs, derived from the validated `SystemConfig`."""

    M: int
    N: int
    half: float            # MA region half-width  (A*lambda/2)
    min_spacing: float     # lambda/2
    Pmax: float
    repair_iters: int = 200

    @staticmethod
    def from_system(sysc: SystemConfig, repair_iters: int = 200) -> "JointSpec":
        return JointSpec(M=sysc.M, N=sysc.N, half=sysc.region_half,
                         min_spacing=sysc.min_spacing, Pmax=sysc.Pmax,
                         repair_iters=repair_iters)

    # ---------------------------------------------------------------- layout
    @property
    def D(self) -> int:
        return 4 * self.M + self.N

    @property
    def sl_T(self) -> slice:
        return slice(0, 2 * self.M)

    @property
    def sl_theta(self) -> slice:
        return slice(2 * self.M, 2 * self.M + self.N)

    @property
    def sl_wr(self) -> slice:
        a = 2 * self.M + self.N
        return slice(a, a + self.M)

    @property
    def sl_wi(self) -> slice:
        a = 3 * self.M + self.N
        return slice(a, a + self.M)

    # ---------------------------------------------------------------- box
    def lower(self) -> np.ndarray:
        lo = np.empty(self.D, dtype=FDTYPE)
        lo[self.sl_T] = -self.half
        lo[self.sl_theta] = 0.0
        lo[self.sl_wr] = -np.sqrt(self.Pmax)
        lo[self.sl_wi] = -np.sqrt(self.Pmax)
        return lo

    def upper(self) -> np.ndarray:
        hi = np.empty(self.D, dtype=FDTYPE)
        hi[self.sl_T] = self.half
        hi[self.sl_theta] = 2.0 * np.pi
        hi[self.sl_wr] = np.sqrt(self.Pmax)
        hi[self.sl_wi] = np.sqrt(self.Pmax)
        return hi


# --------------------------------------------------------------------------------------
# encode / decode / repair
# --------------------------------------------------------------------------------------


def encode(T: np.ndarray, theta: np.ndarray, w: np.ndarray, spec: JointSpec) -> np.ndarray:
    """(..., M, 2), (..., N), (..., M) complex  ->  (..., D)."""
    lead = T.shape[:-2]
    y = np.empty(lead + (spec.D,), dtype=FDTYPE)
    y[..., spec.sl_T] = T.reshape(lead + (2 * spec.M,))
    y[..., spec.sl_theta] = theta
    y[..., spec.sl_wr] = np.real(w)
    y[..., spec.sl_wi] = np.imag(w)
    return y


def decode(y: np.ndarray, spec: JointSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(..., D) -> (T (...,M,2), v (...,N) complex, w (...,M) complex).

    ``v`` is the simulator's RIS vector ``v_n = conj(phi_n) = exp(-j theta_n)``.
    The input is assumed already canonical (see :func:`repair_vector`); only cheap, idempotent
    operations are applied here so that decode is safe to call inside the evaluator.
    """
    lead = y.shape[:-1]
    T = y[..., spec.sl_T].reshape(lead + (spec.M, 2))
    theta = y[..., spec.sl_theta]
    v = np.exp(-1j * theta).astype(CDTYPE)
    w = (y[..., spec.sl_wr] + 1j * y[..., spec.sl_wi]).astype(CDTYPE)
    return T, v, w


def repair_vector(y: np.ndarray, spec: JointSpec,
                  stats: Optional[Dict] = None) -> np.ndarray:
    """Canonicalise a genotype in place-safe fashion; returns a new array.

    Records, when ``stats`` is given: how many designs needed a spacing repair, the mean MA
    displacement caused by it, and how many beamformers were power-projected or were degenerate.
    """
    y = np.array(y, dtype=FDTYPE, copy=True)
    lead = y.shape[:-1]

    # ---- MA positions ------------------------------------------------------------------
    T = np.clip(y[..., spec.sl_T].reshape(lead + (spec.M, 2)), -spec.half, spec.half)
    viol = spacing_violation(T, spec.min_spacing)
    need = viol > 1e-12
    Tr = repair_positions(T, spec, stats)
    y[..., spec.sl_T] = Tr.reshape(lead + (2 * spec.M,))

    # ---- RIS phases --------------------------------------------------------------------
    y[..., spec.sl_theta] = np.mod(y[..., spec.sl_theta], 2.0 * np.pi)

    # ---- beamformer --------------------------------------------------------------------
    wr = y[..., spec.sl_wr]
    wi = y[..., spec.sl_wi]
    nrm = np.sqrt(np.sum(wr * wr + wi * wi, axis=-1))                     # (...)
    degenerate = nrm < _ZERO_W
    if np.any(degenerate):
        # deterministic fallback: unit power on the first antenna
        fb_r = np.zeros(lead + (spec.M,), dtype=FDTYPE)
        fb_r[..., 0] = np.sqrt(spec.Pmax)
        wr = np.where(degenerate[..., None], fb_r, wr)
        wi = np.where(degenerate[..., None], 0.0, wi)
        nrm = np.sqrt(np.sum(wr * wr + wi * wi, axis=-1))
    over = nrm > np.sqrt(spec.Pmax)
    scale = np.where(over, np.sqrt(spec.Pmax) / np.maximum(nrm, _ZERO_W), 1.0)
    y[..., spec.sl_wr] = wr * scale[..., None]
    y[..., spec.sl_wi] = wi * scale[..., None]

    if stats is not None:
        stats["n"] = stats.get("n", 0) + int(np.prod(lead)) if lead else stats.get("n", 0) + 1
        stats["repair_needed"] = stats.get("repair_needed", 0) + int(np.count_nonzero(need))
        disp = np.linalg.norm((Tr - T).reshape(lead + (spec.M * 2,)), axis=-1)
        stats["repair_disp_sum"] = stats.get("repair_disp_sum", 0.0) + float(disp.sum())
        stats["power_projected"] = stats.get("power_projected", 0) + int(np.count_nonzero(over))
        stats["w_degenerate"] = stats.get("w_degenerate", 0) + int(np.count_nonzero(degenerate))
    return y


def random_feasible_joint(rng: np.random.Generator, shape_lead, spec: JointSpec) -> np.ndarray:
    """Uniform feasible initial population, using the validated MA sampler for the T block."""
    lead = tuple(shape_lead)
    T = random_feasible(rng, lead, spec.M, spec.min_spacing, spec.half)
    theta = rng.uniform(0.0, 2.0 * np.pi, lead + (spec.N,))
    # uniform direction, uniform power magnitude in [0, Pmax]
    z = rng.standard_normal(lead + (spec.M,)) + 1j * rng.standard_normal(lead + (spec.M,))
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), _ZERO_W)
    amp = np.sqrt(spec.Pmax * rng.random(lead + (1,)))
    w = (z * amp).astype(CDTYPE)
    return repair_vector(encode(T, theta, w, spec), spec)


def power(y: np.ndarray, spec: JointSpec) -> np.ndarray:
    """||w||^2 of each encoded design."""
    return np.sum(y[..., spec.sl_wr] ** 2 + y[..., spec.sl_wi] ** 2, axis=-1)
