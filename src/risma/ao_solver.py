"""Batched implementation of Algorithm 1 of the reference paper.

Alternating optimisation of
  * the transmit beamformer ``w``  -- closed form, Eq. (10);
  * the RIS reflection vector ``v`` (``v_n = conj(phi_n)``) -- Dinkelbach + MM, Eqs. (12)-(18).

Both sub-problems are solved *exactly* for the other variable fixed, therefore the
secrecy rate is provably non-decreasing over AO iterations (unit-tested).

The whole solver is vectorised over an arbitrary leading batch shape, which is what makes
the Monte-Carlo / population / uncertainty-scenario experiments tractable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import AOConfig

CDTYPE = np.complex128
FDTYPE = np.float64
_EPS = 1e-30


@dataclass
class AOResult:
    w: np.ndarray          # (..., M) beamformer
    v: np.ndarray          # (..., N) RIS vector, v_n = conj(phi_n)
    rs: np.ndarray         # (...) secrecy rate after [.]^+
    rs_raw: np.ndarray     # (...) log-ratio before [.]^+
    iters: int             # AO iterations actually executed
    history: Optional[np.ndarray] = None   # (iters, ...) rs_raw per AO iteration


# --------------------------------------------------------------------------------------
# sub-problem 1: beamforming (Eq. 10)
# --------------------------------------------------------------------------------------


def update_w(
    H: np.ndarray, h_b: np.ndarray, h_e: np.ndarray, v: np.ndarray,
    sigma_b2: float, sigma_e2: np.ndarray, Pmax: float, nullspace_guard: bool = True,
) -> np.ndarray:
    """w = sqrt(Pmax) * principal eigenvector of (B + I/Pmax)^-1 (A + I/Pmax)."""
    M = H.shape[-1]
    Hc = np.conj(H)
    c_b = np.einsum("...nm,...n->...m", Hc, v * h_b)                        # (..., M)
    c_e = np.einsum("...nm,...jn->...jm", Hc, v[..., None, :] * h_e)        # (..., J, M)

    eye = np.eye(M, dtype=CDTYPE) / Pmax
    Amat = (c_b[..., :, None] * np.conj(c_b)[..., None, :]) / sigma_b2 + eye
    Bmat = np.einsum("...jm,...jk->...mk", c_e / sigma_e2[..., None], np.conj(c_e)) + eye

    # Hermitian symmetrisation guards against accumulation of round-off.
    Amat = 0.5 * (Amat + np.conj(np.swapaxes(Amat, -1, -2)))
    Bmat = 0.5 * (Bmat + np.conj(np.swapaxes(Bmat, -1, -2)))

    L = np.linalg.cholesky(Bmat)                                   # B = L L^H
    Z = np.linalg.solve(L, Amat)                                   # L^-1 A
    C = np.conj(np.swapaxes(np.linalg.solve(L, np.conj(np.swapaxes(Z, -1, -2))), -1, -2))
    C = 0.5 * (C + np.conj(np.swapaxes(C, -1, -2)))
    _, U = np.linalg.eigh(C)
    y = U[..., :, -1]                                              # largest eigenvalue
    LH = np.conj(np.swapaxes(L, -1, -2))
    u = np.linalg.solve(LH, y[..., None])[..., 0]
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), _EPS)

    # ---- null-space safeguard (see docs/protocol_decisions.md D-2026-09-08-06) -------------
    # H_BR = F^H Sigma G(T) has rank <= L, so whenever L < M the matrix H has a null space.
    # Every direction in it gives w^H A w = w^H B w = 0, hence a generalised eigenvalue of
    # exactly 1, i.e. R_s = 0.  If the problem is momentarily infeasible (no w achieves positive
    # secrecy for the current Phi) that tie is the maximum, the solver returns a null-space w,
    # and then H w = 0 makes the subsequent RIS update completely blind: b = 0 and E = 0, so
    # psi = 0 and Phi can never move again -- the AO deadlocks at R_s = 0 forever.
    # Falling back to the MRT direction c_b (which lies in range(H^H) by construction) keeps the
    # alternation alive; it changes nothing whenever the eigenvector is already informative.
    Hu = np.linalg.norm(np.matmul(H, u[..., None])[..., 0], axis=-1)
    scale = np.linalg.norm(H, axis=(-2, -1)) / np.sqrt(M)
    degenerate = (Hu <= 1e-8 * np.maximum(scale, _EPS)) & nullspace_guard
    if np.any(degenerate):
        mrt = c_b / np.maximum(np.linalg.norm(c_b, axis=-1, keepdims=True), _EPS)
        u = np.where(degenerate[..., None], mrt, u)
    return np.sqrt(Pmax) * u


# --------------------------------------------------------------------------------------
# sub-problem 2: RIS phases (Eqs. 12-18)
# --------------------------------------------------------------------------------------


def _lambda_max_X(E: np.ndarray, b: np.ndarray, mu: np.ndarray, mode: str) -> np.ndarray:
    """lambda_max of  X = sum_j e_j e_j^H - mu b b^H   (Hermitian, rank <= J+1).

    ``E`` has shape (..., J, N), ``b`` has shape (..., N), ``mu`` has shape (...).
    """
    N = b.shape[-1]
    Jn = E.shape[-2]
    if mode == "auto":
        # the low-rank factorisation only pays off while rank(X) = J+1 < N, which stops being
        # true once many uncertainty scenarios are stacked as virtual Eves (Phase P1+).
        mode = "lowrank" if Jn + 1 < N else "direct"
    if mode == "direct":
        X = np.einsum("...jn,...jk->...nk", E, np.conj(E)) - mu[..., None, None] * (
            b[..., :, None] * np.conj(b)[..., None, :]
        )
        X = 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))
        return np.linalg.eigvalsh(X)[..., -1]
    if mode == "bound":
        # Valid (looser) majorizer: lambda_max(X) <= lambda_max(sum_j e_j e_j^H).
        Gm = np.einsum("...jn,...kn->...jk", E, np.conj(E))
        Gm = 0.5 * (Gm + np.conj(np.swapaxes(Gm, -1, -2)))
        return np.maximum(np.linalg.eigvalsh(Gm)[..., -1], 0.0)
    if mode == "gersh":
        # Cheapest valid majorizer: Gershgorin bound on lambda_max(E E^H) >= lambda_max(X)
        # (X = E E^H - mu b b^H  with mu >= 0, so subtracting a PSD term only lowers it).
        Gm = np.einsum("...jn,...kn->...jk", E, np.conj(E))
        return np.maximum(np.sum(np.abs(Gm), axis=-1).max(axis=-1), 0.0)
    if mode != "lowrank":
        raise ValueError(f"unknown lambda_max_mode {mode!r}")

    # ---- exact, via the rank-(J+1) factorisation X = U D U^H -----------------
    U = np.concatenate([np.swapaxes(E, -1, -2), b[..., :, None]], axis=-1)   # (..., N, J+1)
    Q, Rm = np.linalg.qr(U)                                                  # Rm: (..., r, J+1)
    d = np.concatenate(
        [np.ones(mu.shape + (Jn,), dtype=FDTYPE), -mu[..., None]], axis=-1
    )                                                                        # (..., J+1)
    S = np.einsum("...ip,...p,...jp->...ij", Rm, d, np.conj(Rm))
    S = 0.5 * (S + np.conj(np.swapaxes(S, -1, -2)))
    lam = np.linalg.eigvalsh(S)[..., -1]
    if N > Jn + 1:
        lam = np.maximum(lam, 0.0)
    return lam


def update_v(
    H: np.ndarray, h_b: np.ndarray, h_e: np.ndarray, w: np.ndarray, v: np.ndarray,
    sigma_b2: float, sigma_e2: np.ndarray, cfg: AOConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """MM / Dinkelbach update of the RIS vector.  Returns (v_new, ratio).

    ``E`` and ``b`` depend only on ``(H, w)``, so the Gram matrix ``G = sum_j e_j e_j^H`` is
    constant across the MM iterations and is hoisted out of the loop.  This matters a lot from
    Phase P1 on, where ``S_design`` uncertainty scenarios are stacked into ``S*J`` virtual Eves:
    the per-iteration cost then becomes O(N^2) instead of O(S*J*N).
    Note ``sum_j |v^H e_j|^2 = v^H G v``, so ``E`` itself is not needed inside the loop.
    """
    Hw = np.matmul(H, w[..., None])[..., 0]                        # (..., N)
    b = np.conj(h_b) * Hw / np.sqrt(sigma_b2)                      # (..., N)
    E = np.conj(h_e) * Hw[..., None, :] / np.sqrt(sigma_e2)[..., None]   # (..., J, N)
    G = np.einsum("...jn,...jk->...nk", E, np.conj(E))             # (..., N, N)
    G = 0.5 * (G + np.conj(np.swapaxes(G, -1, -2)))

    N = b.shape[-1]
    Jn = E.shape[-2]
    mode = cfg.lambda_max_mode
    if mode == "auto":
        mode = "lowrank" if Jn + 1 < N else "direct"
    lam_const = None
    if mode in ("bound", "gersh"):
        # lambda_max(X) <= lambda_max(G): independent of mu, so compute it once.
        lam_const = (np.maximum(np.linalg.eigvalsh(G)[..., -1], 0.0) if mode == "bound"
                     else np.maximum(np.sum(np.abs(G), axis=-1).max(axis=-1), 0.0))

    def _ratio(vv):
        vb = np.einsum("...n,...n->...", np.conj(vv), b)
        Gv = np.matmul(G, vv[..., None])[..., 0]
        num = np.real(np.einsum("...n,...n->...", np.conj(vv), Gv)) + 1.0
        den = np.abs(vb) ** 2 + 1.0
        return num / den, vb, Gv

    ratio = None
    for _ in range(cfg.mm_iters):
        mu, vb, Gv = _ratio(v)
        if ratio is not None and np.all(np.abs(ratio - mu) <= cfg.mm_tol * np.abs(ratio)):
            ratio = mu
            break
        ratio = mu

        if mode == "lowrank":
            lam = _lambda_max_X(E, b, mu, "lowrank")
        elif mode == "direct":
            X = G - mu[..., None, None] * (b[..., :, None] * np.conj(b)[..., None, :])
            X = 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))
            lam = np.linalg.eigvalsh(X)[..., -1]
        else:
            lam = lam_const

        Xv = Gv - mu[..., None] * b * np.conj(vb)[..., None]
        psi = lam[..., None] * v - Xv
        mag = np.abs(psi)
        v = np.where(mag > 1e-14, psi / np.maximum(mag, _EPS), v)

    ratio, _, _ = _ratio(v)
    return v, ratio


# --------------------------------------------------------------------------------------
# Algorithm 1
# --------------------------------------------------------------------------------------


def _broadcast_inputs(H, h_b, h_e, sigma_e2):
    bshape = np.broadcast_shapes(
        H.shape[:-2], h_b.shape[:-1], h_e.shape[:-2], sigma_e2.shape[:-1]
    )
    N, M = H.shape[-2], H.shape[-1]
    Jn = h_e.shape[-2]
    H = np.broadcast_to(H, bshape + (N, M))
    h_b = np.broadcast_to(h_b, bshape + (N,))
    h_e = np.broadcast_to(h_e, bshape + (Jn, N))
    sigma_e2 = np.broadcast_to(sigma_e2, bshape + (Jn,))
    return H, h_b, h_e, sigma_e2, bshape


def ao_solve(
    H: np.ndarray,
    h_b: np.ndarray,
    h_e: np.ndarray,
    sigma_b2: float,
    sigma_e2: np.ndarray,
    Pmax: float,
    cfg: AOConfig = AOConfig(),
    v_init: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    keep_history: bool = False,
    optimize_phase: bool = True,
) -> AOResult:
    """Algorithm 1: alternate the exact ``w`` and ``v`` updates.

    Parameters
    ----------
    H : (..., N, M) equivalent BS->RIS channel H_BR(T)
    h_b : (..., N)   RIS->Bob channel
    h_e : (..., J, N) RIS->Eve channels
    sigma_b2, sigma_e2 : noise powers
    Pmax : transmit power budget
    optimize_phase : if False the RIS vector is held at ``v_init`` (MA-RPS baseline).
    """
    H, h_b, h_e, sigma_e2, bshape = _broadcast_inputs(H, h_b, h_e, sigma_e2)
    N = H.shape[-2]

    if v_init is not None:
        v = np.broadcast_to(v_init, bshape + (N,)).astype(CDTYPE).copy()
    elif cfg.init_phase == "random":
        g = rng if rng is not None else np.random.default_rng(0)
        v = np.exp(1j * g.uniform(0, 2 * np.pi, bshape + (N,))).astype(CDTYPE)
    else:
        v = np.ones(bshape + (N,), dtype=CDTYPE)

    prev = None
    hist = [] if keep_history else None
    it = 0
    ratio = None
    w = None
    for it in range(1, cfg.max_iters + 1):
        w = update_w(H, h_b, h_e, v, sigma_b2, sigma_e2, Pmax, cfg.nullspace_guard)
        if optimize_phase:
            v, ratio = update_v(H, h_b, h_e, w, v, sigma_b2, sigma_e2, cfg)
        else:
            Hw = np.matmul(H, w[..., None])[..., 0]
            b = np.conj(h_b) * Hw / np.sqrt(sigma_b2)
            E = np.conj(h_e) * Hw[..., None, :] / np.sqrt(sigma_e2)[..., None]
            vb = np.einsum("...n,...n->...", np.conj(v), b)
            ve = np.einsum("...n,...jn->...j", np.conj(v), E)
            ratio = (np.sum(np.abs(ve) ** 2, axis=-1) + 1.0) / (np.abs(vb) ** 2 + 1.0)
        rs_raw = -np.log2(np.maximum(ratio, _EPS))
        if hist is not None:
            hist.append(rs_raw.copy())
        if prev is not None and np.all(np.abs(rs_raw - prev) <= cfg.tol):
            prev = rs_raw
            break
        prev = rs_raw

    rs_raw = -np.log2(np.maximum(ratio, _EPS))
    return AOResult(
        w=w,
        v=v,
        rs=np.maximum(rs_raw, 0.0),
        rs_raw=rs_raw,
        iters=it,
        history=np.stack(hist) if hist else None,
    )
