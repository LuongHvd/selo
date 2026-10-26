"""Task-set construction and channel stacking for Phase P4.

A **task** is one possible security state of the same cell: a particular estimate of the
eavesdropper channels, at a particular estimation quality.  A **task set** is the collection of
states a base station must prepare MA configurations for.

All tasks of a set share the MA region and the problem dimension, so they can be stacked along the
instance axis of a single :class:`~risma.channel.ChannelBatch` (task-major, ``index = s*R + r``) and
scored by one batched Algorithm-1 call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .channel import ChannelBatch, sample_channels
from .config import ExperimentConfig
from .uncertainty import EveUncertainty

CDTYPE = np.complex128
FDTYPE = np.float64


# --------------------------------------------------------------------------------------
# stacking
# --------------------------------------------------------------------------------------


def stack_batches(batches: Sequence[ChannelBatch]) -> ChannelBatch:
    """Concatenate per-task ChannelBatch objects along the instance axis (task-major)."""
    b0 = batches[0]
    for b in batches:
        if b.N != b0.N or b.Lt != b0.Lt or b.J != b0.J or b.R != b0.R:
            raise ValueError("all tasks must share (R, N, Lt, J) to be stacked")
    return ChannelBatch(
        C_BR=np.concatenate([b.C_BR for b in batches], axis=0),
        alpha=np.concatenate([b.alpha for b in batches], axis=0),
        beta=np.concatenate([b.beta for b in batches], axis=0),
        h_b=np.concatenate([b.h_b for b in batches], axis=0),
        h_e=np.concatenate([b.h_e for b in batches], axis=0),
        sigma_b2=b0.sigma_b2,
        sigma_e2=np.concatenate([b.sigma_e2 for b in batches], axis=0),
        wavelength=b0.wavelength,
        meta={"stacked": len(batches), "R_per_task": b0.R},
    )


# --------------------------------------------------------------------------------------
# task sets
# --------------------------------------------------------------------------------------


@dataclass
class TaskSet:
    """``S`` tasks x ``R`` instances, stacked task-major into one ChannelBatch."""

    stacked: ChannelBatch
    S: int
    R: int
    labels: List[str]
    per_task: List[ChannelBatch]
    family: str
    eta: float
    meta: Dict

    @property
    def M(self) -> int:
        return int(self.meta["M"])

    def channel_distance(self, mode: str = "eve") -> np.ndarray:
        """FREE similarity input: normalised squared distance between task channels.

        Returns ``(R, S, S)``.  ``mode='eve'`` uses only the Eve channel estimates (what the brief
        prescribes); ``mode='full'`` additionally uses Bob's channel, which the BS also knows and
        which distinguishes tasks whose whole realisation differs.
        """
        S, R = self.S, self.R
        he = np.stack([b.h_e.reshape(R, -1) for b in self.per_task])          # (S, R, J*N)
        parts = [he]
        if mode == "full":
            parts.append(np.stack([b.h_b.reshape(R, -1) for b in self.per_task]))
        d2 = np.zeros((R, S, S), dtype=FDTYPE)
        for arr in parts:
            diff = arr[:, None, :, :] - arr[None, :, :, :]                   # (S, S, R, F)
            num = np.sum(np.abs(diff) ** 2, axis=-1)                          # (S, S, R)
            pw = np.sum(np.abs(arr) ** 2, axis=-1)                            # (S, R)
            den = 0.5 * (pw[:, None, :] + pw[None, :, :]) + 1e-30
            d2 += np.transpose(num / den, (2, 0, 1))
        return d2 / len(parts)


def _eve_geom(cfg: ExperimentConfig, R: int, J: int, N: int,
              rng: np.random.Generator) -> np.ndarray:
    """Resampled Eve positions inside the Eve region + fresh Rayleigh channels."""
    sysc = cfg.system
    u = rng.random((R, J))
    ang = rng.random((R, J)) * 2 * np.pi
    rad = sysc.eve_radius * np.sqrt(u)
    xy = np.stack([sysc.eve_center[0] + rad * np.cos(ang),
                   sysc.eve_center[1] + rad * np.sin(ang)], axis=-1)
    d = np.linalg.norm(xy - np.asarray(sysc.ris_pos)[None, None, :], axis=-1)
    pl = 10.0 ** (-sysc.pathloss_ref_db / 10.0) * np.maximum(d, 1.0) ** (-sysc.pathloss_exponent)
    z = rng.standard_normal((R, J, N)) + 1j * rng.standard_normal((R, J, N))
    return (np.sqrt(pl[..., None] / 2.0) * z).astype(CDTYPE)


def build_task_set(cfg: ExperimentConfig, family: str, eta: float, n_tasks: int,
                   n_inst: int, seed: int, base_seed: int) -> TaskSet:
    """Construct one of the three P4 task families.

    ``base_seed`` fixes the *instances* (shared by every method and seed); ``seed`` fixes the
    task-specific draws, so tasks differ between optimiser seeds while instances do not.
    """
    sysc = cfg.system
    base = sample_channels(sysc, n_inst, base_seed)
    rng = np.random.default_rng(1_000_003 + seed)
    S = n_tasks
    per_task: List[ChannelBatch] = []
    labels: List[str] = []

    if family == "heterogeneous":
        # A base station preparing for several plausible security states at once:
        #   * the estimate is fresh but noisy               -> csi draws at the working eta
        #   * the Eves have moved inside their region       -> eve_geom
        #   * the estimation quality differs between states -> a coarser/finer eta
        n1 = S // 3
        n2 = S // 3
        n3 = S - n1 - n2
        unc = EveUncertainty("gaussian", eta)
        he1 = unc.perturb(base.h_e, n1, rng)
        for k in range(n1):
            per_task.append(base.with_eve(he1[k]))
            labels.append(f"csi@{eta:g}")
        for _ in range(n2):
            per_task.append(base.with_eve(_eve_geom(cfg, n_inst, sysc.J, sysc.N, rng)))
            labels.append("eve_geom")
        eta2 = min(4.0 * eta, 0.8)
        unc2 = EveUncertainty("gaussian", eta2)
        he3 = unc2.perturb(base.h_e, n3, rng)
        for k in range(n3):
            per_task.append(base.with_eve(he3[k]))
            labels.append(f"csi@{eta2:g}")

    elif family == "homogeneous":
        unc = EveUncertainty("gaussian", eta)
        he = unc.perturb(base.h_e, S, rng)
        for k in range(S):
            per_task.append(base.with_eve(he[k]))
            labels.append(f"csi@{eta:g}")

    elif family == "unrelated":
        for k in range(S):
            per_task.append(sample_channels(sysc, n_inst, 4_000_000 + seed * 100 + k))
            labels.append("independent")

    else:
        raise ValueError(f"unknown task family {family!r}")

    return TaskSet(
        stacked=stack_batches(per_task), S=S, R=n_inst, labels=labels,
        per_task=per_task, family=family, eta=eta,
        meta={"M": sysc.M, "N": sysc.N, "J": sysc.J, "L": sysc.L,
              "base_seed": base_seed, "task_seed": seed},
    )
