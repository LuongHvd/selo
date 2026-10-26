"""Phase P0 -- reproduction of the reference paper (perfect CSI).

Protocol (documented in reports/P0_reproduction.md):

* **Locked evaluation sets.**  For every system configuration a *test* set of ``n_mc``
  channel realisations is generated from a fixed seed and never used for tuning;
  a disjoint *validation* set drives the training curves.
* **DRL protocol A (paper-faithful).**  One policy per (method, configuration) is trained
  on freshly sampled channel realisations (the state contains channel information, so the
  policy is expected to generalise), then evaluated on the locked test set by rolling out
  the deterministic policy for ``episode_len`` steps.
* **Reference points.**  ``MA-Random`` (one random feasible layout), ``AO-RS`` (random
  search) and ``AO-LS`` ((1+1) hill climber) run *per instance* with an explicit budget.
* Every scheme obtains its objective values through ``MAPositionProblem`` -> identical
  inner AO solver.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..channel import sample_channels
from ..config import AOConfig, ExperimentConfig, PenaltyConfig, SystemConfig
from ..env import VecMAEnv, VecMAEnvAll
from ..ma_constraints import fpa_layout
from ..optimizers.baselines import (
    OuterResult, fpa_ris, local_search, ma_random, ma_rps_random, random_search,
)
from ..optimizers.drl import DRLConfig, train_ppo, train_td3
from ..problem import MAPositionProblem
from ..secrecy import spacing_violation
from ..utils.registry import Registry, save_run

FDTYPE = np.float64

TEST_SEED_BASE = 900_000
VAL_SEED_BASE = 800_000


def config_key(sysc: SystemConfig) -> int:
    """Deterministic (process-independent) integer key of a system configuration.

    The *test* and *validation* channel sets of a configuration are locked to this key, so
    every method sees exactly the same instances and the sets never change between runs.
    """
    import hashlib
    s = f"{sysc.M}|{sysc.N}|{sysc.J}|{sysc.L}|{sysc.A:.4f}|{sysc.Pmax_dbm:.4f}|{sysc.noise_dbm:.4f}|{sysc.wavelength:.6f}"
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 100_000


# --------------------------------------------------------------------------------------
# policy roll-out evaluation
# --------------------------------------------------------------------------------------


def rollout_policy(
    cfg: ExperimentConfig, ch, policy_fn: Callable, steps: int, seed: int,
    env_cls=VecMAEnv, **env_kw,
) -> Dict[str, np.ndarray]:
    """Deterministic roll-out of a trained policy on a *fixed* channel batch."""
    env = env_cls(cfg, ch.R, seed=seed, episode_len=steps + 1,
                  resample_channels=False, channels=ch, **env_kw)
    obs = env.obs
    for _ in range(steps):
        u = policy_fn(obs, True)
        obs, rew, done, info = env.step(u)
    prob = env.prob
    rep = prob.report_solution(env.best_T)
    return {
        "rs": rep["rs"], "T": rep["T"], "obj": env.best_rs,
        "counters": env.counters, "raw_feasible": rep["raw_feasible"],
    }


def rollout_policy_all(cfg: ExperimentConfig, ch, policy_fn, steps: int, seed: int):
    """PPO-ALL roll-out: the reported R_s uses the agent's own (w, Phi), no AO."""
    from ..secrecy import secrecy_rate
    env = VecMAEnvAll(cfg, ch.R, seed=seed, episode_len=steps + 1,
                      resample_channels=False, channels=ch)
    obs = env.obs
    for _ in range(steps):
        u = policy_fn(obs, True)
        obs, rew, done, info = env.step(u)
    from ..ma_constraints import repair
    T = repair(env.best_T, cfg.system.min_spacing, cfg.system.region_half)
    H = ch.H_BR(T)
    rs = secrecy_rate(H, ch.h_b, ch.h_e, env.best_v, env.best_w,
                      ch.sigma_b2, ch.sigma_e2, clip=True)
    return {"rs": rs, "T": T, "obj": env.best_rs, "counters": env.counters,
            "raw_feasible": spacing_violation(env.best_T, cfg.system.min_spacing) <= 1e-9}


# --------------------------------------------------------------------------------------
# individual methods
# --------------------------------------------------------------------------------------


def run_method(
    method: str, cfg: ExperimentConfig, seed: int, train_steps: int, budget: int,
    n_env: int = 204, eval_every: int = 50_000, drl: DRLConfig = DRLConfig(),
    log: Optional[Callable[[str], None]] = None,
) -> Dict:
    """Run one outer optimiser on one system configuration and return raw results."""
    t0 = time.time()
    sysc = cfg.system
    key = config_key(sysc)
    ch_test = sample_channels(sysc, cfg.n_mc, TEST_SEED_BASE + key)
    ch_val = sample_channels(sysc, min(cfg.n_mc, 200), VAL_SEED_BASE + key)

    out: Dict = {"method": method, "history": None, "anytime": None, "anytime_fe": None}

    if method == "FPA-RIS":
        res = fpa_ris(ch_test, cfg)
        out.update(rs=res.rs, T=res.T, counters=res.counters.as_dict(),
                   raw_feasible_frac=1.0)

    elif method == "MA-Random":
        prob = MAPositionProblem(ch_test, cfg)
        res = ma_random(prob, seed=seed)
        out.update(rs=res.rs, T=res.T, counters=res.counters.as_dict(),
                   raw_feasible_frac=res.meta["raw_feasible_frac"])

    elif method.startswith("AO-RS"):
        prob = MAPositionProblem(ch_test, cfg)
        res = random_search(prob, budget, seed=seed)
        out.update(rs=res.rs, T=res.T, counters=res.counters.as_dict(),
                   anytime=res.anytime, anytime_fe=res.anytime_fe,
                   raw_feasible_frac=res.meta["raw_feasible_frac"])

    elif method.startswith("AO-LS"):
        prob = MAPositionProblem(ch_test, cfg)
        res = local_search(prob, budget, seed=seed)
        out.update(rs=res.rs, T=res.T, counters=res.counters.as_dict(),
                   anytime=res.anytime, anytime_fe=res.anytime_fe,
                   raw_feasible_frac=res.meta["raw_feasible_frac"])

    elif method == "MA-RPS":
        res = ma_rps_random(ch_test, cfg, budget, seed=seed)
        out.update(rs=res.rs, T=res.T, counters=res.counters.as_dict(),
                   anytime=res.anytime, anytime_fe=res.anytime_fe,
                   raw_feasible_frac=res.meta["raw_feasible_frac"])

    elif method in ("AO-PPO", "AO-TD3", "AO-DDPG", "PPO-ALL"):
        env_cls = VecMAEnvAll if method == "PPO-ALL" else VecMAEnv
        env = env_cls(cfg, n_env, seed=seed, episode_len=drl.episode_len,
                      resample_channels=True)

        def eval_fn(pf):
            fn = rollout_policy_all if method == "PPO-ALL" else rollout_policy
            kw = {} if method == "PPO-ALL" else {"env_cls": VecMAEnv}
            r = fn(cfg, ch_val, pf, drl.episode_len, seed=seed + 5_000, **kw)
            return float(np.mean(r["rs"]))

        if method in ("AO-PPO", "PPO-ALL"):
            net, hist = train_ppo(env, train_steps, drl, seed=seed, eval_fn=eval_fn,
                                  eval_every=eval_every, log=log)
        else:
            net, hist = train_td3(env, train_steps, drl, seed=seed,
                                  twin=(method == "AO-TD3"), eval_fn=eval_fn,
                                  eval_every=eval_every, log=log)

        import torch

        def policy_fn(o, deterministic=True):
            with torch.no_grad():
                t = torch.as_tensor(o, dtype=torch.float32)
                if hasattr(net, "dist"):
                    u = net.dist(t).mean
                else:
                    u = net(t)
            return u.numpy().astype(FDTYPE)

        train_counters = env.counters.as_dict()
        if method == "PPO-ALL":
            r = rollout_policy_all(cfg, ch_test, policy_fn, drl.episode_len, seed=seed + 9_000)
        else:
            r = rollout_policy(cfg, ch_test, policy_fn, drl.episode_len, seed=seed + 9_000)
        ec = r["counters"].as_dict()
        merged = {k: train_counters.get(k, 0) + ec.get(k, 0) for k in ec}
        merged["train_" + "candidate_fe"] = train_counters["candidate_fe"]
        merged["eval_candidate_fe"] = ec["candidate_fe"]
        out.update(rs=r["rs"], T=r["T"], counters=merged,
                   raw_feasible_frac=float(np.mean(r["raw_feasible"])),
                   history={"steps": hist.steps, "eval_rs": hist.eval_rs,
                            "train_reward": hist.train_reward, "wall": hist.wall})
    else:
        raise ValueError(f"unknown method {method!r}")

    out["runtime_s"] = time.time() - t0
    out["n_mc"] = cfg.n_mc
    return out
