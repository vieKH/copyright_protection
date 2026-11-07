"""Extraction: original single-block method and the four multi-block decoders.

For every B x B window t with origin (m_t, n_t) the DFT values at the carrier positions
are phase-compensated:  Z_t = F_t * exp(-j 2 pi (u (m_t mod B) + v (n_t mod B)) / B).
Every decision uses the median threshold over the L*L carriers (balanced object).

  mag_vote    majority vote of per-block decisions on |Z_t|   (plan, variant 1; T = 1 -> original method)
  phase_vote  majority vote of per-block decisions on Re{Z_t e^{-j theta}}
  avg_mag     |mean_t Z_t| , one decision                       (plan, variant 2)
  avg_phase   Re{mean_t Z_t e^{-j theta}} , one decision
"""
from __future__ import annotations

import numpy as np

from .embedding import Layout

DECODERS = ("mag_vote", "phase_vote", "avg_mag", "avg_phase")


def grid(shape, B: int, stride: int, start=(0, 0), rng: np.random.Generator | None = None):
    """Window origins on a regular grid; optional random order (prefixes = random subsets)."""
    r0, c0 = start
    coords = [(r, c) for r in range(r0, shape[0] - B + 1, stride) for c in range(c0, shape[1] - B + 1, stride)]
    if rng is not None:
        coords = [coords[i] for i in rng.permutation(len(coords))]
    return coords


def carriers(image: np.ndarray, lay: Layout, coords, chunk: int = 64) -> np.ndarray:
    """Phase-compensated carrier values, shape (T, L*L); batched FFT."""
    u, v = lay.positions
    B = lay.B
    img = np.asarray(image, dtype=np.float64)
    out = np.empty((len(coords), u.size), dtype=np.complex128)
    for s in range(0, len(coords), chunk):
        part = np.asarray(coords[s:s + chunk])
        stack = np.stack([img[r:r + B, c:c + B] for r, c in part])
        F = np.fft.fft2(stack)[:, u, v]
        out[s:s + len(part)] = F * np.exp(-2j * np.pi * (np.outer(part[:, 0] % B, u) + np.outer(part[:, 1] % B, v)) / B)
    return out


def _above_median(x: np.ndarray) -> np.ndarray:
    return (x > np.median(x, axis=-1, keepdims=True)).astype(np.int32)


def decode(Z: np.ndarray, theta: np.ndarray, counts) -> dict:
    """Decisions of the four decoders for every prefix length T in `counts`."""
    c = np.exp(1j * theta.ravel())
    dm = np.cumsum(_above_median(np.abs(Z)), 0)
    dp = np.cumsum(_above_median(np.real(Z * np.conj(c))), 0)
    cz = np.cumsum(Z, 0)
    res = {}
    for T in counts:
        zbar = cz[T - 1] / T
        res[T] = dict(mag_vote=(2 * dm[T - 1] > T).astype(np.int32),
                      phase_vote=(2 * dp[T - 1] > T).astype(np.int32),
                      avg_mag=_above_median(np.abs(zbar)),
                      avg_phase=_above_median(np.real(zbar * np.conj(c))))
    return res


def ber(bits: np.ndarray, decided: np.ndarray) -> float:
    return float(np.mean(bits.ravel() != np.ravel(decided)))


def single_block_ber(image, lay: Layout, bits, rng: np.random.Generator, trials: int = 16) -> float:
    """Original method: one arbitrary block, modulus, median threshold; mean over random positions."""
    H, W = image.shape
    coords = [(int(rng.integers(0, H - lay.B + 1)), int(rng.integers(0, W - lay.B + 1))) for _ in range(trials)]
    A = np.abs(carriers(image, lay, coords))
    return float(np.mean(_above_median(A) != bits.ravel()[None, :]))


def multi_block_ber(image, lay: Layout, bits, theta, stride: int, start=(0, 0), rng: np.random.Generator | None = None, counts=None):
    """BER of every decoder for prefixes of a random-order window grid; T = all windows is always included."""
    coords = grid(image.shape, lay.B, stride, start, rng)
    n = len(coords)
    counts = sorted({c for c in (counts or []) if c < n} | {n})
    res = decode(carriers(image, lay, coords), theta, counts)
    return {T: {d: ber(bits, res[T][d]) for d in DECODERS} for T in counts}, n
