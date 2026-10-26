"""Channel model: field-response BS->RIS link + Rayleigh RIS->Bob / RIS->Eve links.

Everything is *batched over Monte-Carlo realisations* in the leading axis so that the
whole experiment can be vectorised.

Reference: docs/reference_paper_summary.md sections 1.1, 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import SystemConfig

CDTYPE = np.complex128
FDTYPE = np.float64


# --------------------------------------------------------------------------------------
# RIS geometry
# --------------------------------------------------------------------------------------


def factor_upa(n: int) -> Tuple[int, int]:
    """Factor N into (rows, cols) as square as possible; rows <= cols."""
    if n <= 0:
        raise ValueError("N must be positive")
    best = 1
    for k in range(1, int(np.floor(np.sqrt(n))) + 1):
        if n % k == 0:
            best = k
    return best, n // best


def ris_element_positions(cfg: SystemConfig) -> np.ndarray:
    """RIS element coordinates R (N, 2) relative to r_0 = [0, 0]. [ASSUMPTION D3]"""
    rows, cols = factor_upa(cfg.N)
    d = cfg.ris_element_spacing_factor * cfg.wavelength
    xs = np.arange(cols, dtype=FDTYPE) * d
    ys = np.arange(rows, dtype=FDTYPE) * d
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


# --------------------------------------------------------------------------------------
# field response
# --------------------------------------------------------------------------------------


def direction_coeffs(theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """rho_i(x, y) = alpha_i * x + beta_i * y  with alpha = sin(theta)cos(phi), beta = cos(theta)."""
    alpha = np.sin(theta) * np.cos(phi)
    beta = np.cos(theta)
    return alpha, beta


def frv(positions: np.ndarray, alpha: np.ndarray, beta: np.ndarray, wavelength: float) -> np.ndarray:
    """Field-response matrix.

    Parameters
    ----------
    positions : (..., P, 2)  antenna/element coordinates in metres
    alpha, beta : (..., L)   direction coefficients
    Returns
    -------
    (..., L, P) complex field-response matrix  [g(t_1), ..., g(t_P)]
    """
    x = positions[..., 0]                       # (..., P)
    y = positions[..., 1]
    rho = alpha[..., :, None] * x[..., None, :] + beta[..., :, None] * y[..., None, :]
    return np.exp(1j * (2.0 * np.pi / wavelength) * rho).astype(CDTYPE)


# --------------------------------------------------------------------------------------
# scenario container
# --------------------------------------------------------------------------------------


@dataclass
class ChannelBatch:
    """A batch of ``R`` independent channel realisations.

    Attributes
    ----------
    C_BR : (R, N, Lt)  = F(R)^H @ Sigma       -- everything of the BS->RIS link except G(T)
    alpha, beta : (R, Lt)                     -- transmit-path direction coefficients
    h_b  : (R, N)                             -- RIS->Bob channel (h_b, *not* conjugated)
    h_e  : (R, J, N)                          -- RIS->Eve channels
    sigma_b2 : float                          -- noise power at Bob
    sigma_e2 : (R, J)                         -- noise power at each Eve
    """

    C_BR: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    h_b: np.ndarray
    h_e: np.ndarray
    sigma_b2: float
    sigma_e2: np.ndarray
    wavelength: float
    meta: dict

    # ---------------------------------------------------------------- properties
    @property
    def R(self) -> int:
        return self.C_BR.shape[0]

    @property
    def N(self) -> int:
        return self.C_BR.shape[1]

    @property
    def Lt(self) -> int:
        return self.C_BR.shape[2]

    @property
    def J(self) -> int:
        return self.h_e.shape[1]

    # ---------------------------------------------------------------- indexing
    def __getitem__(self, idx) -> "ChannelBatch":
        idx = np.atleast_1d(np.asarray(idx))
        return ChannelBatch(
            C_BR=self.C_BR[idx],
            alpha=self.alpha[idx],
            beta=self.beta[idx],
            h_b=self.h_b[idx],
            h_e=self.h_e[idx],
            sigma_b2=self.sigma_b2,
            sigma_e2=self.sigma_e2[idx],
            wavelength=self.wavelength,
            meta=dict(self.meta),
        )

    def with_eve(self, h_e_new: np.ndarray) -> "ChannelBatch":
        """Return a copy with replaced Eve channels (used for CSI-uncertainty phases)."""
        return ChannelBatch(
            C_BR=self.C_BR,
            alpha=self.alpha,
            beta=self.beta,
            h_b=self.h_b,
            h_e=h_e_new,
            sigma_b2=self.sigma_b2,
            sigma_e2=self.sigma_e2 if self.sigma_e2.shape[1] == h_e_new.shape[1]
            else np.full(h_e_new.shape[:2], float(self.sigma_e2.flat[0])),
            wavelength=self.wavelength,
            meta=dict(self.meta),
        )

    # ---------------------------------------------------------------- channel eval
    def H_BR(self, T: np.ndarray) -> np.ndarray:
        """Equivalent BS->RIS channel for MA positions.

        Parameters
        ----------
        T : (..., M, 2) MA positions in metres.  Leading dims must broadcast against (R,).

        Returns
        -------
        (..., N, M) complex
        """
        G = frv(T, self.alpha_b(T.ndim), self.beta_b(T.ndim), self.wavelength)  # (..., Lt, M)
        C = self.C_BR
        # align C_BR leading dims with T's leading dims
        extra = T.ndim - 2 - 1                                # extra dims beyond (R,)
        if extra > 0:
            C = C.reshape((C.shape[0],) + (1,) * extra + C.shape[1:])
        return np.matmul(C, G)

    # broadcast helpers -------------------------------------------------------
    def alpha_b(self, tdim: int) -> np.ndarray:
        extra = tdim - 2 - 1
        a = self.alpha
        if extra > 0:
            a = a.reshape((a.shape[0],) + (1,) * extra + (a.shape[1],))
        return a

    def beta_b(self, tdim: int) -> np.ndarray:
        extra = tdim - 2 - 1
        b = self.beta
        if extra > 0:
            b = b.reshape((b.shape[0],) + (1,) * extra + (b.shape[1],))
        return b


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------


def _crandn(rng: np.random.Generator, shape, var: np.ndarray | float) -> np.ndarray:
    """CN(0, var) samples of the given shape (var broadcastable)."""
    re = rng.standard_normal(shape)
    im = rng.standard_normal(shape)
    return (np.sqrt(np.asarray(var) / 2.0) * (re + 1j * im)).astype(CDTYPE)


def sample_channels(cfg: SystemConfig, n_mc: int, seed: int) -> ChannelBatch:
    """Draw ``n_mc`` independent channel realisations.

    Randomness: Eve positions, path angles (transmit + receive), PRM Sigma,
    RIS->Bob and RIS->Eve Rayleigh vectors.
    """
    rng = np.random.default_rng(seed)
    N, Lt, Lr, J = cfg.N, cfg.Lt, cfg.Lr, cfg.J
    lam = cfg.wavelength

    # ---- geometry / path loss ------------------------------------------------
    bs = np.asarray(cfg.bs_pos, dtype=FDTYPE)
    ris = np.asarray(cfg.ris_pos, dtype=FDTYPE)
    user = np.asarray(cfg.user_pos, dtype=FDTYPE)
    d_br = float(np.linalg.norm(ris - bs))
    d_ru = float(np.linalg.norm(user - ris))
    pl_br = cfg.pathloss_linear(d_br)
    pl_ru = cfg.pathloss_linear(d_ru)

    # Eve positions: uniform in a disc of radius eve_radius around eve_center
    u = rng.random((n_mc, J))
    ang = rng.random((n_mc, J)) * 2.0 * np.pi
    rad = cfg.eve_radius * np.sqrt(u)
    eve_xy = np.stack(
        [cfg.eve_center[0] + rad * np.cos(ang), cfg.eve_center[1] + rad * np.sin(ang)], axis=-1
    )                                                            # (R, J, 2)
    d_re = np.linalg.norm(eve_xy - ris[None, None, :], axis=-1)  # (R, J)
    pl_re = np.where(d_re < 1.0, 1.0, d_re) ** (-cfg.pathloss_exponent)
    pl_re = pl_re * 10.0 ** (-cfg.pathloss_ref_db / 10.0)

    # ---- path angles ---------------------------------------------------------
    hi = 2.0 * np.pi if cfg.angle_support == "2pi" else np.pi
    theta_t = rng.random((n_mc, Lt)) * hi
    phi_t = rng.random((n_mc, Lt)) * hi
    theta_r = rng.random((n_mc, Lr)) * hi
    phi_r = rng.random((n_mc, Lr)) * hi
    alpha_t, beta_t = direction_coeffs(theta_t, phi_t)
    alpha_r, beta_r = direction_coeffs(theta_r, phi_r)

    # ---- RIS FRM  F(R) in C^{Lr x N} ----------------------------------------
    R_pos = ris_element_positions(cfg)                            # (N, 2)
    F = frv(np.broadcast_to(R_pos, (n_mc, N, 2)), alpha_r, beta_r, lam)   # (R, Lr, N)

    # ---- PRM Sigma in C^{Lr x Lt}  [ASSUMPTION D2, see SystemConfig.prm_normalization] ----
    prm_var = pl_br / (Lr * Lt) if cfg.prm_normalization == "total" else pl_br / max(Lt, 1)
    Sigma = _crandn(rng, (n_mc, Lr, Lt), prm_var)

    C_BR = np.matmul(np.conj(np.transpose(F, (0, 2, 1))), Sigma)  # (R, N, Lt)

    # ---- Rayleigh RIS->Bob / RIS->Eve ---------------------------------------
    h_b = _crandn(rng, (n_mc, N), pl_ru)
    h_e = _crandn(rng, (n_mc, J, N), pl_re[..., None])

    sigma2 = cfg.sigma2
    return ChannelBatch(
        C_BR=C_BR,
        alpha=alpha_t,
        beta=beta_t,
        h_b=h_b,
        h_e=h_e,
        sigma_b2=sigma2,
        sigma_e2=np.full((n_mc, J), sigma2, dtype=FDTYPE),
        wavelength=lam,
        meta={
            "seed": seed,
            "n_mc": n_mc,
            "d_br": d_br,
            "d_ru": d_ru,
            "pl_br": pl_br,
            "pl_ru": pl_ru,
            "eve_xy": eve_xy,
            "d_re": d_re,
            "pl_re": pl_re,
            "M": cfg.M,
            "N": N,
            "J": J,
            "L": cfg.L,
        },
    )
