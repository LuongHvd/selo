"""Phase P2 -- does a *multitask* structure exist at all?  (Hypothesis H1)

For a common candidate set ``X = {T_1, ..., T_K}`` and a family of "tasks" (each task is one
eavesdropper-channel situation) we build the fitness vectors

    F_s = [ f_s(T_1), ..., f_s(T_K) ],   f_s(T) = R_s(w*(T,H_s), Phi*(T,H_s), T; H_s)

and measure the pairwise Spearman / Kendall / Pearson correlation between tasks together with a
channel distance ``d_ij = ||H_e^(i) - H_e^(j)||_F / ||H_e||_F``.

Four task families are compared, because "related uncertainty scenario" can mean very different
things and the answer to H1 depends on which one is meant:

| family | what varies between tasks | what stays fixed |
|---|---|---|
| `csi_error`  | the CSI error realisation `Delta h_e` at a fixed `eta` | everything else |
| `eta_level`  | the uncertainty level `eta` | the error *direction* |
| `eve_geom`   | the Eve positions and their Rayleigh channels | BS/RIS/Bob channel |
| `instance`   | the whole channel realisation | nothing (negative control) |

`instance` is the negative control: those tasks are statistically independent, so whatever
correlation they show is the *floor* produced by the shared structure of the problem
(feasibility, geometry of the MA region), not by task relatedness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from ..channel import ChannelBatch, sample_channels
from ..config import ExperimentConfig
from ..ma_constraints import random_feasible
from ..problem import MAPositionProblem
from ..uncertainty import EveUncertainty
from .p0 import TEST_SEED_BASE, config_key

FDTYPE = np.float64
CDTYPE = np.complex128


# --------------------------------------------------------------------------------------
# candidate set
# --------------------------------------------------------------------------------------


def build_candidates(cfg: ExperimentConfig, ch: ChannelBatch, K_random: int,
                     K_traj: int, seed: int) -> np.ndarray:
    """Common candidate set X, shape (R, K, M, 2).

    ``K_random`` uniform feasible configurations + ``K_traj`` configurations harvested from a
    (1+1) hill-climber trajectory on the *nominal* problem (project spec section 8 asks for both).
    """
    rng = np.random.default_rng(seed)
    sysc = cfg.system
    Xr = random_feasible(rng, (ch.R, K_random), sysc.M, sysc.min_spacing, sysc.region_half)
    if K_traj <= 0:
        return Xr

    prob = MAPositionProblem(ch, cfg)
    half = sysc.region_half
    T = prob.random_T(rng)
    best = prob.evaluate(T).obj
    bestT = T.copy()
    sigma = np.full(ch.R, 0.25 * half)
    traj = []
    for _ in range(K_traj):
        cand = np.clip(bestT + sigma[:, None, None] * rng.standard_normal(bestT.shape),
                       -half, half)
        obj = prob.evaluate(cand).obj
        better = obj > best
        best = np.where(better, obj, best)
        bestT = np.where(better[:, None, None], cand, bestT)
        sigma = np.clip(np.where(better, sigma * 1.15, sigma * 0.97), 1e-4 * half, half)
        traj.append(bestT.copy())
    Xt = np.stack(traj, axis=1)                       # (R, K_traj, M, 2)
    return np.concatenate([Xr, Xt], axis=1)


# --------------------------------------------------------------------------------------
# task families
# --------------------------------------------------------------------------------------


def make_tasks(family: str, cfg: ExperimentConfig, ch: ChannelBatch, n_tasks: int,
               eta: float, seed: int) -> Tuple[np.ndarray, List[float]]:
    """Return (eve_channels, task_labels).

    ``eve_channels`` has shape (n_tasks, R, J, N); ``ch`` is the nominal batch.
    """
    rng = np.random.default_rng(seed)
    if family == "csi_error":
        unc = EveUncertainty("gaussian", eta)
        return unc.perturb(ch.h_e, n_tasks, rng), [eta] * n_tasks
    if family == "eta_level":
        etas = list(np.linspace(0.0, max(eta, 0.8), n_tasks))
        # same random direction, scaled: isolates the *level* from the direction
        z = rng.standard_normal((1,) + ch.h_e.shape) + 1j * rng.standard_normal((1,) + ch.h_e.shape)
        u = z / np.linalg.norm(z, axis=-1, keepdims=True)
        nrm = np.linalg.norm(ch.h_e, axis=-1)
        out = np.stack([ch.h_e + (e * nrm)[..., None] * u[0] for e in etas])
        return out.astype(CDTYPE), etas
    if family == "eve_geom":
        # resample Eve positions (hence path loss) and their Rayleigh channels
        sysc = cfg.system
        out = []
        for _ in range(n_tasks):
            u = rng.random((ch.R, ch.J))
            ang = rng.random((ch.R, ch.J)) * 2 * np.pi
            rad = sysc.eve_radius * np.sqrt(u)
            xy = np.stack([sysc.eve_center[0] + rad * np.cos(ang),
                           sysc.eve_center[1] + rad * np.sin(ang)], axis=-1)
            d = np.linalg.norm(xy - np.asarray(sysc.ris_pos)[None, None, :], axis=-1)
            pl = 10.0 ** (-sysc.pathloss_ref_db / 10.0) * np.maximum(d, 1.0) ** (
                -sysc.pathloss_exponent)
            z = rng.standard_normal((ch.R, ch.J, ch.N)) + 1j * rng.standard_normal(
                (ch.R, ch.J, ch.N))
            out.append((np.sqrt(pl[..., None] / 2.0) * z).astype(CDTYPE))
        return np.stack(out), list(range(n_tasks))
    if family == "instance":
        # negative control: statistically independent Eve channels drawn like the nominal ones
        out = []
        for t in range(n_tasks):
            other = sample_channels(cfg.system, ch.R, 500_000 + seed * 1000 + t)
            out.append(other.h_e)
        return np.stack(out), list(range(n_tasks))
    if family == "full_channel":
        # second negative control: *everything* differs (BS-RIS link, Bob, Eves).  Handled by
        # make_task_channels; only the Eve part is returned here so that the distance metric and
        # the array shapes stay uniform.
        out = []
        for t in range(n_tasks):
            other = sample_channels(cfg.system, ch.R, 600_000 + seed * 1000 + t)
            out.append(other.h_e)
        return np.stack(out), list(range(n_tasks))
    if family == "mixed":
        # Heterogeneous task set: a third of the tasks are CSI-error draws of the *same*
        # nominal Eve channel (high similarity), a third have resampled Eve geometry
        # (medium), and a third come from completely independent channel realisations
        # (similarity ~ 0).  This is the only family in which pairwise task similarity has
        # enough spread for a *similarity-aware* transfer rule to be distinguishable from a
        # random one, i.e. the family in which H3 can actually be tested.
        n1 = n_tasks // 3
        n2 = n_tasks // 3
        n3 = n_tasks - n1 - n2
        a, _ = make_tasks("csi_error", cfg, ch, n1, eta, seed)
        b, _ = make_tasks("eve_geom", cfg, ch, n2, eta, seed + 11)
        c, _ = make_tasks("full_channel", cfg, ch, n3, eta, seed + 22)
        labels = ["csi"] * n1 + ["geom"] * n2 + ["alien"] * n3
        return np.concatenate([a, b, c], axis=0), labels
    raise ValueError(f"unknown task family {family!r}")


def make_task_channels(family: str, cfg: ExperimentConfig, ch: ChannelBatch,
                       eve: np.ndarray, seed: int) -> List[ChannelBatch]:
    """Turn the task Eve channels into full per-task ChannelBatch objects.

    For every family except ``full_channel`` the BS-RIS link and Bob's channel are shared, which
    is exactly what makes those tasks "related".  ``full_channel`` replaces the whole realisation.
    """
    S = eve.shape[0]
    if family == "full_channel":
        return [sample_channels(cfg.system, ch.R, 600_000 + seed * 1000 + t) for t in range(S)]
    if family == "mixed":
        n1 = S // 3
        n2 = S // 3
        out = [ch.with_eve(eve[s]) for s in range(n1 + n2)]
        out += [sample_channels(cfg.system, ch.R, 600_000 + (seed + 22) * 1000 + t)
                for t in range(S - n1 - n2)]
        return out
    return [ch.with_eve(eve[s]) for s in range(S)]


# --------------------------------------------------------------------------------------
# landscape evaluation
# --------------------------------------------------------------------------------------


def evaluate_landscapes(cfg: ExperimentConfig, chans: List[ChannelBatch], X: np.ndarray,
                        chunk: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Landscapes ``F[s, r, k] = f_s(X[r, k])``.

    Returns ``(F_clip, F_raw)``.  ``F_clip`` is the objective the optimisers actually see
    (``[.]^+`` applied); ``F_raw`` is the un-clipped log-ratio.  Rank statistics are computed on
    ``F_raw`` because in leakage-limited regimes a large fraction of candidates is clipped to
    exactly 0, which creates massive ties and makes Spearman/Kendall undefined or meaningless.
    """
    S = len(chans)
    R, K = X.shape[0], X.shape[1]
    F = np.empty((S, R, K), dtype=FDTYPE)
    Fr = np.empty((S, R, K), dtype=FDTYPE)
    for s in range(S):
        chs = chans[s]
        prob = MAPositionProblem(chs, cfg)
        for a in range(0, K, chunk):
            ev = prob.evaluate(X[:, a: a + chunk], count=False, keep_wv=False)
            F[s, :, a: a + chunk] = ev.rs
            Fr[s, :, a: a + chunk] = ev.rs_raw
    return F, Fr


def landscape_decomposition(cfg: ExperimentConfig, chans: List[ChannelBatch], X: np.ndarray,
                            chunk: int = 40) -> Dict[str, float]:
    """How much of the landscape's variation across candidates comes from the *Eve* term?

    ``R_s = log2(1+gamma_b) - log2(1+sum_j gamma_{e,j})``.  If the second term barely varies with
    ``T``, then tasks that differ only in the Eve channels cannot induce different landscapes --
    which is the mechanistic explanation for whatever H1 finds.
    """
    from ..secrecy import snrs
    S, R = len(chans), chans[0].R
    K = X.shape[1]
    vb, ve, vt = [], [], []
    for s in range(S):
        chs = chans[s]
        prob = MAPositionProblem(chs, cfg)
        bob = np.empty((R, K)); eav = np.empty((R, K))
        for a in range(0, K, chunk):
            sub = X[:, a: a + chunk]
            ev = prob.evaluate(sub, count=False)
            H = chs.H_BR(sub)
            hb = chs.h_b[:, None, :]
            he = chs.h_e[:, None, :, :]
            se2 = chs.sigma_e2[:, None, :]
            gb, ge = snrs(H, hb, he, ev.v, ev.w, chs.sigma_b2, se2)
            bob[:, a: a + chunk] = np.log2(1.0 + gb)
            eav[:, a: a + chunk] = np.log2(1.0 + ge.sum(axis=-1))
        vb.append(bob.var(axis=-1)); ve.append(eav.var(axis=-1))
        vt.append((bob - eav).var(axis=-1))
    vb = np.mean(vb); ve = np.mean(ve); vt = np.mean(vt)
    return {
        "var_bob_term": float(vb), "var_eve_term": float(ve), "var_total": float(vt),
        "eve_share_of_variance": float(ve / max(vb + ve, 1e-30)),
    }


def pairwise_correlations(F: np.ndarray) -> Dict[str, np.ndarray]:
    """Task-task correlation matrices, per instance.  ``F`` is (S, R, K)."""
    S, R, K = F.shape
    sp = np.ones((R, S, S))
    ke = np.ones((R, S, S))
    pe = np.ones((R, S, S))
    for r in range(R):
        ranks = np.apply_along_axis(stats.rankdata, 1, F[:, r, :])   # (S, K)
        c = np.corrcoef(ranks)
        sp[r] = c
        pe[r] = np.corrcoef(F[:, r, :])
        for i in range(S):
            for j in range(i + 1, S):
                t = stats.kendalltau(F[i, r], F[j, r]).statistic
                ke[r, i, j] = ke[r, j, i] = t
    return {"spearman": sp, "kendall": ke, "pearson": pe}


def channel_distances(eve: np.ndarray) -> np.ndarray:
    """Normalised Frobenius distance between task Eve-channel matrices.  (R, S, S)."""
    S, R, J, N = eve.shape
    flat = eve.reshape(S, R, J * N)
    scale = np.linalg.norm(flat, axis=-1).mean(axis=0)               # (R,)
    d = np.linalg.norm(flat[:, None, :, :] - flat[None, :, :, :], axis=-1)   # (S, S, R)
    return np.transpose(d, (2, 0, 1)) / scale[:, None, None]


# --------------------------------------------------------------------------------------


@dataclass
class P2Result:
    family: str
    eta: float
    F: np.ndarray
    F_raw: np.ndarray
    corr: Dict[str, np.ndarray]
    dist: np.ndarray
    meta: Dict


def run_p2(family: str, cfg: ExperimentConfig, eta: float = 0.10, n_tasks: int = 12,
           n_inst: int = 20, K_random: int = 200, K_traj: int = 50,
           seed: int = 0, decompose: bool = True) -> P2Result:
    t0 = time.time()
    key = config_key(cfg.system)
    ch = sample_channels(cfg.system, n_inst, TEST_SEED_BASE + key + 31)
    X = build_candidates(cfg, ch, K_random, K_traj, seed)
    eve, labels = make_tasks(family, cfg, ch, n_tasks, eta, seed + 1)
    chans = make_task_channels(family, cfg, ch, eve, seed + 1)
    F, F_raw = evaluate_landscapes(cfg, chans, X)
    corr = pairwise_correlations(F_raw)
    dist = channel_distances(eve)
    iu = np.triu_indices(n_tasks, k=1)
    meta = {
        "labels": labels, "n_tasks": n_tasks, "n_inst": n_inst,
        "K": X.shape[1], "K_random": K_random, "K_traj": K_traj,
        "runtime_s": time.time() - t0,
        "mean_spearman": float(np.mean(corr["spearman"][:, iu[0], iu[1]])),
        "std_spearman": float(np.std(corr["spearman"][:, iu[0], iu[1]])),
        "mean_kendall": float(np.mean(corr["kendall"][:, iu[0], iu[1]])),
        "mean_dist": float(np.mean(dist[:, iu[0], iu[1]])),
        "fitness_spread": float(np.mean(F.std(axis=-1))),
        "fitness_spread_raw": float(np.mean(F_raw.std(axis=-1))),
        "clipped_fraction": float(np.mean(F <= 1e-12)),
    }
    if decompose:
        meta.update(landscape_decomposition(cfg, chans[:min(4, len(chans))], X))
    # correlation between task distance and landscape correlation, pooled over instances
    dd = dist[:, iu[0], iu[1]].ravel()
    ss = corr["spearman"][:, iu[0], iu[1]].ravel()
    if np.std(dd) > 1e-12 and np.std(ss) > 1e-12:
        r, p = stats.spearmanr(dd, ss)
        meta["dist_vs_corr_spearman"] = float(r)
        meta["dist_vs_corr_p"] = float(p)
    else:
        meta["dist_vs_corr_spearman"] = float("nan")
        meta["dist_vs_corr_p"] = float("nan")
    return P2Result(family=family, eta=eta, F=F, F_raw=F_raw, corr=corr, dist=dist, meta=meta)
