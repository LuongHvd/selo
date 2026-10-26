"""DRL outer optimisers: PPO (proposed in the paper), TD3 and DDPG baselines.

All three drive the *same* :class:`~risma.env.VecMAEnv`, hence the *same* inner AO solver,
so the comparison is fair by construction.

Hyper-parameters follow Table I of the reference paper where given; every value the paper
does not specify is listed in ``DRLConfig`` with a comment and reported in the P0 report.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FDTYPE = np.float64


# --------------------------------------------------------------------------------------
# hyper-parameters
# --------------------------------------------------------------------------------------


@dataclass
class DRLConfig:
    # ---- from the paper's Table I -------------------------------------------
    hidden: int = 256
    n_layers: int = 4               # 4 weight layers  => 3 hidden layers of `hidden`
    lr_policy: float = 1e-4
    lr_value: float = 2e-4
    gamma: float = 0.97
    buffer: int = 2048
    batch: int = 128
    clip: float = 0.2
    episode_len: int = 10           # "time steps per episode"
    # ---- not specified by the paper (documented assumptions) ----------------
    gae_lambda: float = 0.95
    ppo_epochs: int = 10
    entropy_coef: float = 0.0
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    init_log_std: float = -0.5
    # off-policy (TD3/DDPG); the paper gives no table for them
    off_buffer: int = 100_000
    tau: float = 0.005
    expl_noise: float = 0.2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    start_steps: int = 2048
    updates_per_step: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------------------


def mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int, out_act=None) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(max(n_layers - 1, 1)):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    if out_act is not None:
        layers += [out_act]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: DRLConfig):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, cfg.hidden, cfg.n_layers)
        self.log_std = nn.Parameter(torch.full((act_dim,), cfg.init_log_std))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu = self.net(obs)
        return torch.distributions.Normal(mu, self.log_std.exp())


class DetPolicy(nn.Module):
    """Deterministic policy producing the *pre-tanh* output u (Eq. 19 applies tanh in env)."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: DRLConfig):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, cfg.hidden, cfg.n_layers)
        self.scale = 3.0             # u range; tanh saturates beyond ~3

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.net(obs))


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: DRLConfig):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, cfg.hidden, cfg.n_layers)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


# --------------------------------------------------------------------------------------
# training-history container
# --------------------------------------------------------------------------------------


@dataclass
class TrainHistory:
    steps: List[int] = field(default_factory=list)
    train_reward: List[float] = field(default_factory=list)
    eval_rs: List[float] = field(default_factory=list)
    wall: List[float] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# PPO
# --------------------------------------------------------------------------------------


def train_ppo(
    env,
    total_steps: int,
    cfg: DRLConfig = DRLConfig(),
    seed: int = 0,
    eval_fn: Optional[Callable[[Callable], float]] = None,
    eval_every: int = 50_000,
    device: str = "cpu",
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[nn.Module, TrainHistory]:
    torch.manual_seed(seed)
    dev = torch.device(device)
    pi = GaussianPolicy(env.obs_dim, env.act_dim, cfg).to(dev)
    vf = mlp(env.obs_dim, 1, cfg.hidden, cfg.n_layers).to(dev)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=cfg.lr_policy)
    opt_vf = torch.optim.Adam(vf.parameters(), lr=cfg.lr_value)

    n_env = env.n_env
    rollout = max(int(round(cfg.buffer / n_env)), 1)
    hist = TrainHistory()
    obs = env.obs.copy()
    t_start = time.time()
    next_eval = 0
    steps_done = 0

    def policy_fn(o: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(o, dtype=torch.float32, device=dev)
            d = pi.dist(t)
            u = d.mean if deterministic else d.sample()
        return u.cpu().numpy().astype(FDTYPE)

    while steps_done < total_steps:
        ob_buf = np.zeros((rollout, n_env, env.obs_dim), dtype=np.float32)
        ac_buf = np.zeros((rollout, n_env, env.act_dim), dtype=np.float32)
        lp_buf = np.zeros((rollout, n_env), dtype=np.float32)
        rw_buf = np.zeros((rollout, n_env), dtype=np.float32)
        dn_buf = np.zeros((rollout, n_env), dtype=np.float32)
        vl_buf = np.zeros((rollout + 1, n_env), dtype=np.float32)

        for t in range(rollout):
            ot = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            with torch.no_grad():
                d = pi.dist(ot)
                u = d.sample()
                lp = d.log_prob(u).sum(-1)
                v = vf(ot).squeeze(-1)
            ob_buf[t] = obs
            ac_buf[t] = u.cpu().numpy()
            lp_buf[t] = lp.cpu().numpy()
            vl_buf[t] = v.cpu().numpy()
            obs, rew, done, _ = env.step(u.cpu().numpy().astype(FDTYPE))
            rw_buf[t] = rew
            dn_buf[t] = done.astype(np.float32)
            steps_done += n_env

        with torch.no_grad():
            vl_buf[rollout] = vf(
                torch.as_tensor(obs, dtype=torch.float32, device=dev)
            ).squeeze(-1).cpu().numpy()

        # GAE
        adv = np.zeros((rollout, n_env), dtype=np.float32)
        last = np.zeros(n_env, dtype=np.float32)
        for t in reversed(range(rollout)):
            nonterm = 1.0 - dn_buf[t]
            delta = rw_buf[t] + cfg.gamma * vl_buf[t + 1] * nonterm - vl_buf[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + vl_buf[:rollout]

        b_ob = torch.as_tensor(ob_buf.reshape(-1, env.obs_dim), device=dev)
        b_ac = torch.as_tensor(ac_buf.reshape(-1, env.act_dim), device=dev)
        b_lp = torch.as_tensor(lp_buf.reshape(-1), device=dev)
        b_ad = torch.as_tensor(adv.reshape(-1), device=dev)
        b_rt = torch.as_tensor(ret.reshape(-1), device=dev)
        b_ad = (b_ad - b_ad.mean()) / (b_ad.std() + 1e-8)
        n = b_ob.shape[0]

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, cfg.batch):
                idx = perm[i: i + cfg.batch]
                d = pi.dist(b_ob[idx])
                lp = d.log_prob(b_ac[idx]).sum(-1)
                ratio = torch.exp(lp - b_lp[idx])
                a = b_ad[idx]
                l1 = ratio * a
                l2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a
                loss_pi = -torch.min(l1, l2).mean() - cfg.entropy_coef * d.entropy().sum(-1).mean()
                opt_pi.zero_grad(set_to_none=True)
                loss_pi.backward()
                nn.utils.clip_grad_norm_(pi.parameters(), cfg.max_grad_norm)
                opt_pi.step()

                v = vf(b_ob[idx]).squeeze(-1)
                loss_v = F.mse_loss(v, b_rt[idx])
                opt_vf.zero_grad(set_to_none=True)
                loss_v.backward()
                nn.utils.clip_grad_norm_(vf.parameters(), cfg.max_grad_norm)
                opt_vf.step()

        if steps_done >= next_eval:
            ev = float(eval_fn(policy_fn)) if eval_fn is not None else float("nan")
            hist.steps.append(steps_done)
            hist.train_reward.append(float(rw_buf.mean()))
            hist.eval_rs.append(ev)
            hist.wall.append(time.time() - t_start)
            if log:
                log(f"    ppo step {steps_done:>9d}  train_r={rw_buf.mean():.3f}  eval={ev:.3f}")
            next_eval += eval_every

    return pi, hist


# --------------------------------------------------------------------------------------
# DDPG / TD3
# --------------------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, cap: int, obs_dim: int, act_dim: int):
        self.cap = cap
        self.obs = np.zeros((cap, obs_dim), dtype=np.float32)
        self.act = np.zeros((cap, act_dim), dtype=np.float32)
        self.rew = np.zeros(cap, dtype=np.float32)
        self.nobs = np.zeros((cap, obs_dim), dtype=np.float32)
        self.done = np.zeros(cap, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, no, d):
        n = o.shape[0]
        idx = (self.ptr + np.arange(n)) % self.cap
        self.obs[idx] = o
        self.act[idx] = a
        self.rew[idx] = r
        self.nobs[idx] = no
        self.done[idx] = d
        self.ptr = int((self.ptr + n) % self.cap)
        self.size = min(self.size + n, self.cap)

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, self.size, batch)
        return self.obs[idx], self.act[idx], self.rew[idx], self.nobs[idx], self.done[idx]


def train_td3(
    env,
    total_steps: int,
    cfg: DRLConfig = DRLConfig(),
    seed: int = 0,
    twin: bool = True,
    eval_fn: Optional[Callable[[Callable], float]] = None,
    eval_every: int = 50_000,
    device: str = "cpu",
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[nn.Module, TrainHistory]:
    """TD3 (``twin=True``) or DDPG (``twin=False``)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    actor = DetPolicy(env.obs_dim, env.act_dim, cfg).to(dev)
    actor_t = copy.deepcopy(actor)
    q1 = QNet(env.obs_dim, env.act_dim, cfg).to(dev)
    q1_t = copy.deepcopy(q1)
    q2 = QNet(env.obs_dim, env.act_dim, cfg).to(dev) if twin else None
    q2_t = copy.deepcopy(q2) if twin else None
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr_policy)
    qparams = list(q1.parameters()) + (list(q2.parameters()) if twin else [])
    opt_q = torch.optim.Adam(qparams, lr=cfg.lr_value)

    buf = ReplayBuffer(cfg.off_buffer, env.obs_dim, env.act_dim)
    hist = TrainHistory()
    obs = env.obs.copy()
    t_start = time.time()
    steps_done = 0
    next_eval = 0
    it = 0
    amax = actor.scale

    def policy_fn(o: np.ndarray, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            u = actor(torch.as_tensor(o, dtype=torch.float32, device=dev))
        u = u.cpu().numpy().astype(FDTYPE)
        if not deterministic:
            u = u + cfg.expl_noise * amax * rng.standard_normal(u.shape)
        return u

    while steps_done < total_steps:
        if steps_done < cfg.start_steps:
            u = rng.uniform(-amax, amax, (env.n_env, env.act_dim))
        else:
            u = policy_fn(obs, deterministic=False)
        u = np.clip(u, -amax, amax)
        nobs, rew, done, info = env.step(u)
        term = info.get("terminal_obs", nobs)
        buf.add(obs, u.astype(np.float32), rew.astype(np.float32),
                term.astype(np.float32), done.astype(np.float32))
        obs = nobs
        steps_done += env.n_env

        n_upd = int(env.n_env * cfg.updates_per_step / max(cfg.batch // 16, 1))
        n_upd = max(n_upd, 1)
        if buf.size >= cfg.batch and steps_done >= cfg.start_steps:
            for _ in range(n_upd):
                it += 1
                o, a, r, no, d = buf.sample(cfg.batch, rng)
                o = torch.as_tensor(o, device=dev)
                a = torch.as_tensor(a, device=dev)
                r = torch.as_tensor(r, device=dev)
                no = torch.as_tensor(no, device=dev)
                d = torch.as_tensor(d, device=dev)
                with torch.no_grad():
                    na = actor_t(no)
                    if twin:
                        noise = (torch.randn_like(na) * cfg.policy_noise * amax).clamp(
                            -cfg.noise_clip * amax, cfg.noise_clip * amax
                        )
                        na = (na + noise).clamp(-amax, amax)
                        tq = torch.min(q1_t(no, na), q2_t(no, na))
                    else:
                        tq = q1_t(no, na)
                    y = r + cfg.gamma * (1.0 - d) * tq
                loss_q = F.mse_loss(q1(o, a), y)
                if twin:
                    loss_q = loss_q + F.mse_loss(q2(o, a), y)
                opt_q.zero_grad(set_to_none=True)
                loss_q.backward()
                nn.utils.clip_grad_norm_(qparams, cfg.max_grad_norm)
                opt_q.step()

                if (not twin) or (it % cfg.policy_delay == 0):
                    loss_a = -q1(o, actor(o)).mean()
                    opt_a.zero_grad(set_to_none=True)
                    loss_a.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                    opt_a.step()
                    with torch.no_grad():
                        for p, pt in zip(actor.parameters(), actor_t.parameters()):
                            pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
                        for p, pt in zip(q1.parameters(), q1_t.parameters()):
                            pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
                        if twin:
                            for p, pt in zip(q2.parameters(), q2_t.parameters()):
                                pt.mul_(1 - cfg.tau).add_(cfg.tau * p)

        if steps_done >= next_eval:
            ev = float(eval_fn(policy_fn)) if eval_fn is not None else float("nan")
            hist.steps.append(steps_done)
            hist.train_reward.append(float(np.mean(rew)))
            hist.eval_rs.append(ev)
            hist.wall.append(time.time() - t_start)
            if log:
                tag = "td3" if twin else "ddpg"
                log(f"    {tag} step {steps_done:>9d}  train_r={np.mean(rew):.3f}  eval={ev:.3f}")
            next_eval += eval_every

    return actor, hist
