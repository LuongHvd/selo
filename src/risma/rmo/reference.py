"""The common nominal reference design used by the R2 variable-group ablation.

Every ablation group optimises its own block and freezes the rest at **one** design, computed once
per channel instance and shared by all groups, so that the comparison isolates the effect of the
variable group and not of the starting point:

    T_0        = the validated FPA lambda/2 layout (the P0 FPA-RIS reference)
    (Phi_0, w_0) = Algorithm 1 (the frozen inner AO) applied to T_0 on the **estimated** Eve
                   channel h_e-hat, i.e. the uncertainty-blind design the reference paper produces.

Freezing at an uncertainty-blind design is deliberate: it is exactly the design a non-robust
pipeline would deploy, so group A ("optimise T only") reproduces the P0-P3 setting up to the inner
solver, and the ablation measures what each additional block buys *on top of* that.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..ao_solver import ao_solve
from ..channel import ChannelBatch
from ..config import AOConfig, SystemConfig
from ..ma_constraints import fpa_layout
from ..secrecy import secrecy_rate
from .encoding import JointSpec, encode, repair_vector

FDTYPE = np.float64


def nominal_reference(ch: ChannelBatch, sysc: SystemConfig, spec: JointSpec,
                      aocfg: AOConfig = None) -> Tuple[np.ndarray, Dict]:
    """Returns ``(y0 (R, D), info)`` -- the shared frozen design, already canonicalised."""
    aocfg = aocfg or AOConfig()
    R = ch.R
    T0 = np.broadcast_to(fpa_layout(spec.M, sysc.wavelength, spec.half),
                         (R, spec.M, 2)).astype(FDTYPE).copy()
    H = ch.H_BR(T0)
    res = ao_solve(H, ch.h_b, ch.h_e, ch.sigma_b2, ch.sigma_e2, spec.Pmax, aocfg)
    v0, w0 = res.v, res.w
    # the encoding stores phases with v = exp(-j theta)
    theta0 = np.mod(-np.angle(v0), 2.0 * np.pi)
    stats: Dict = {}
    y0 = repair_vector(encode(T0, theta0, w0, spec), spec, stats)

    rs_nom = secrecy_rate(H, ch.h_b, ch.h_e, v0, w0, ch.sigma_b2, ch.sigma_e2, clip=True)
    info = dict(iters=int(getattr(res, "iters", -1)), repair=stats,
                nominal_rs_mean=float(np.mean(rs_nom)),
                nominal_rs=np.asarray(rs_nom, dtype=FDTYPE),
                power=float(np.max(np.sum(np.abs(w0) ** 2, axis=-1))))
    return y0, info
