"""Vectorised MDP environment for MA-position optimisation (paper Section III-B).

One *env* corresponds to one channel realisation.  ``VecMAEnv`` runs ``n_env`` of them
simultaneously so that a single batched Algorithm-1 call serves the whole vector, which is
what makes 10^6-step DRL training affordable.

State  (Eq. 21):  s = [ a_{t-1}, gamma_b, gamma_e (J), htilde_b (2M), htilde_e (2MJ), r_{t-1} ]
Action (Eq. 19):  a = (A*lambda/2) * tanh(u),  reshaped to the absolute MA coordinates.
Reward (Eq. 20):  r = R_s - chi * sum_{i<j} [lambda/2 - ||t_i - t_j||]^+
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .channel import ChannelBatch, sample_channels
from .config import ExperimentConfig
from .problem import BudgetCounters, MAPositionProblem
from .secrecy import snrs, spacing_violation

FDTYPE = np.float64


def state_dim(M: int, J: int) -> int:
    return 2 * M + 1 + J + 2 * M + 2 * M * J + 1


def action_dim(M: int) -> int:
    return 2 * M


class VecMAEnv:
    """A vector of ``n_env`` independent MA-position MDPs."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        n_env: int,
        seed: int,
        episode_len: int = 10,
        resample_channels: bool = True,
        channels: Optional[ChannelBatch] = None,
        optimize_phase: bool = True,
        v_fixed: Optional[np.ndarray] = None,
    ) -> None:
        self.cfg = cfg
        self.sys = cfg.system
        self.n_env = n_env
        self.episode_len = episode_len
        self.resample_channels = resample_channels
        self.optimize_phase = optimize_phase
        self._v_fixed_given = v_fixed
        self.rng = np.random.default_rng(seed)
        self._chan_seed = int(self.rng.integers(0, 2 ** 62))
        self._fixed_channels = channels
        self.obs_dim = state_dim(self.sys.M, self.sys.J)
        self.act_dim = action_dim(self.sys.M)
        self.total_steps = 0
        self.cum_counters = BudgetCounters()
        self.prob = None
        self.reset()

    # ---------------------------------------------------------------- internals
    def _new_channels(self) -> ChannelBatch:
        if self._fixed_channels is not None:
            return self._fixed_channels
        self._chan_seed += 1
        return sample_channels(self.sys, self.n_env, self._chan_seed)

    def _make_problem(self, ch: ChannelBatch) -> MAPositionProblem:
        v_fixed = self._v_fixed_given
        if not self.optimize_phase and v_fixed is None:
            v_fixed = np.exp(1j * self.rng.uniform(0, 2 * np.pi, (self.n_env, self.sys.N)))
        return MAPositionProblem(
            ch, self.cfg, optimize_phase=self.optimize_phase, v_fixed=v_fixed
        )

    def _observe(self, T, w, v, rs, prev_action, prev_reward) -> np.ndarray:
        ch = self.ch
        H = ch.H_BR(T)
        gb, ge = snrs(H, ch.h_b, ch.h_e, v, w, ch.sigma_b2, ch.sigma_e2)
        # equivalent channels htilde = h^H Phi H_BR   (in C^{1 x M})
        hb_eq = np.einsum("...n,...n,...nm->...m", np.conj(v), np.conj(ch.h_b), H)
        he_eq = np.einsum("...n,...jn,...nm->...jm", np.conj(v), np.conj(ch.h_e), H)
        scale = np.sqrt(np.mean(np.abs(hb_eq) ** 2, axis=-1, keepdims=True)) + 1e-30
        hb_n = hb_eq / scale
        he_n = he_eq / scale[..., None, :]
        parts = [
            prev_action / self.sys.region_half,
            np.log10(1.0 + gb)[..., None],
            np.log10(1.0 + ge),
            hb_n.real, hb_n.imag,
            he_n.real.reshape(self.n_env, -1), he_n.imag.reshape(self.n_env, -1),
            prev_reward[..., None],
        ]
        obs = np.concatenate([np.asarray(p, dtype=FDTYPE) for p in parts], axis=-1)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------------------------------------------------------- api
    def reset(self) -> np.ndarray:
        if self.prob is not None:
            self.cum_counters += self.prob.counters
        self.ch = self._new_channels()
        self.prob = self._make_problem(self.ch)
        self.step_in_ep = 0
        T = self.prob.random_T(self.rng)
        ev = self.prob.evaluate(T)
        self.T = T
        rs = ev.rs
        pen = spacing_violation(T, self.sys.min_spacing) * self.cfg.penalty.chi
        reward = rs - pen
        self.best_rs = reward.copy()
        self.best_T = T.copy()
        self.last_action = T.reshape(self.n_env, -1)
        self.obs = self._observe(T, ev.w, ev.v, rs, self.last_action, reward)
        self.total_steps += self.n_env
        return self.obs

    def step(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """``u`` is the raw policy output of shape (n_env, 2M); Eq. (19) is applied here."""
        a = self.sys.region_half * np.tanh(np.asarray(u, dtype=FDTYPE))
        T = a.reshape(self.n_env, self.sys.M, 2)
        ev = self.prob.evaluate(T)
        rs = ev.rs
        pen = self.cfg.penalty.chi * spacing_violation(T, self.sys.min_spacing)
        reward = rs - pen
        self.T = T
        self.last_action = a
        # track the best *feasible-penalised* configuration seen in this episode
        better = reward > self.best_rs
        self.best_rs = np.where(better, reward, self.best_rs)
        self.best_T = np.where(better[:, None, None], T, self.best_T)

        self.step_in_ep += 1
        self.total_steps += self.n_env
        done = self.step_in_ep >= self.episode_len
        self.obs = self._observe(T, ev.w, ev.v, rs, a, reward)
        info = {"rs": rs, "penalty": pen, "T": T, "w": ev.w, "v": ev.v}
        dones = np.full(self.n_env, done, dtype=bool)
        if done:
            obs_final = self.obs
            if self.resample_channels:
                self.reset()
            else:
                self.step_in_ep = 0
                T0 = self.prob.random_T(self.rng)
                ev0 = self.prob.evaluate(T0)
                pen0 = self.cfg.penalty.chi * spacing_violation(T0, self.sys.min_spacing)
                self.last_action = T0.reshape(self.n_env, -1)
                self.obs = self._observe(T0, ev0.w, ev0.v, ev0.rs, self.last_action,
                                         ev0.rs - pen0)
            info["terminal_obs"] = obs_final
        return self.obs, reward, dones, info

    @property
    def counters(self) -> BudgetCounters:
        c = BudgetCounters(**self.cum_counters.as_dict())
        if self.prob is not None:
            c += self.prob.counters
        return c


class VecMAEnvAll(VecMAEnv):
    """PPO-ALL baseline: the agent controls *all* variables (T, w, Phi); no AO is used.

    Action layout (all pre-tanh):
        u[:, :2M]        -> MA positions via Eq. (19)
        u[:, 2M:4M]      -> real/imag parts of w, renormalised to ||w||^2 = Pmax
        u[:, 4M:4M+N]    -> RIS phases in [-pi, pi]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_dim = 4 * self.sys.M + self.sys.N
        self.obs_dim = self.obs.shape[-1]

    def _split(self, u):
        M, N = self.sys.M, self.sys.N
        a = self.sys.region_half * np.tanh(u[:, : 2 * M])
        wr = np.tanh(u[:, 2 * M: 3 * M])
        wi = np.tanh(u[:, 3 * M: 4 * M])
        w = wr + 1j * wi
        nrm = np.maximum(np.linalg.norm(w, axis=-1, keepdims=True), 1e-12)
        w = np.sqrt(self.sys.Pmax) * w / nrm
        ph = np.pi * np.tanh(u[:, 4 * M: 4 * M + N])
        v = np.exp(-1j * ph)                      # v_n = conj(phi_n)
        return a, w, v

    def _eval_direct(self, T, w, v):
        from .secrecy import secrecy_rate
        ch = self.ch
        H = ch.H_BR(T)
        rs = secrecy_rate(H, ch.h_b, ch.h_e, v, w, ch.sigma_b2, ch.sigma_e2, clip=True)
        self.prob.counters.candidate_fe += self.n_env
        self.prob.counters.secrecy_evals += self.n_env
        return rs

    def reset(self) -> np.ndarray:
        if self.prob is not None:
            self.cum_counters += self.prob.counters
        self.ch = self._new_channels()
        self.prob = self._make_problem(self.ch)
        self.step_in_ep = 0
        T = self.prob.random_T(self.rng)
        w = np.sqrt(self.sys.Pmax) * np.ones((self.n_env, self.sys.M), dtype=complex) / np.sqrt(self.sys.M)
        v = np.ones((self.n_env, self.sys.N), dtype=complex)
        rs = self._eval_direct(T, w, v)
        pen = self.cfg.penalty.chi * spacing_violation(T, self.sys.min_spacing)
        reward = rs - pen
        self.T, self.w, self.v = T, w, v
        self.best_rs = reward.copy()
        self.best_T = T.copy()
        self.best_w = w.copy()
        self.best_v = v.copy()
        self.last_action = np.zeros((self.n_env, 4 * self.sys.M + self.sys.N))
        self.obs = self._observe_all(T, w, v, self.last_action, reward)
        self.total_steps += self.n_env
        return self.obs

    def _observe_all(self, T, w, v, prev_action, prev_reward):
        M, N = self.sys.M, self.sys.N
        pa = prev_action[:, : 2 * M] / self.sys.region_half
        base = self._observe(T, w, v, None, pa, prev_reward)
        extra = np.concatenate(
            [(w / np.sqrt(self.sys.Pmax)).real, (w / np.sqrt(self.sys.Pmax)).imag,
             np.angle(v) / np.pi], axis=-1)
        return np.concatenate([base, extra], axis=-1)

    def step(self, u):
        u = np.asarray(u, dtype=FDTYPE)
        a, w, v = self._split(u)
        T = a.reshape(self.n_env, self.sys.M, 2)
        rs = self._eval_direct(T, w, v)
        pen = self.cfg.penalty.chi * spacing_violation(T, self.sys.min_spacing)
        reward = rs - pen
        self.T, self.w, self.v = T, w, v
        self.last_action = np.concatenate(
            [a, u[:, 2 * self.sys.M:]], axis=-1)
        better = reward > self.best_rs
        self.best_rs = np.where(better, reward, self.best_rs)
        self.best_T = np.where(better[:, None, None], T, self.best_T)
        self.best_w = np.where(better[:, None], w, self.best_w)
        self.best_v = np.where(better[:, None], v, self.best_v)
        self.step_in_ep += 1
        self.total_steps += self.n_env
        done = self.step_in_ep >= self.episode_len
        self.obs = self._observe_all(T, w, v, self.last_action, reward)
        info = {"rs": rs, "penalty": pen, "T": T, "w": w, "v": v}
        dones = np.full(self.n_env, done, dtype=bool)
        if done:
            info["terminal_obs"] = self.obs
            if self.resample_channels:
                self.reset()
            else:
                self.step_in_ep = 0
        return self.obs, reward, dones, info
