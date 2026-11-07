"""Embedding of a periodic spectral watermark (DFT, conjugate pairs).

A balanced binary object B[r, s] (L x L, exactly L*L/2 ones) is written into the
spectrum W of one period P = Q = B: every one-bit becomes a unit-modulus coefficient
exp(j*theta_rs) at its carrier position (u, v) and the conjugate value at (-u, -v), so
that w = IDFT(W) is real.  The image is split into non-overlapping B x B blocks and
alpha * w is added to every block (periodic repetition), then rounded to 8 bits.
alpha follows from the prescribed watermark variance D (Parseval):  alpha = B^2 sqrt(D / 2K).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Layout:
    """Carrier layout of an L x L object in a B x B spectrum.

    Rows u = a + step*i; the left half of the columns starts at v = a, the right half
    ends at v = B - a//2 - 1 (negative frequencies).  Same rule as the original code
    (x = y = a, offset = a // 2, gap = step - 1).
    """
    B: int
    L: int
    a: int
    step: int = 2

    @cached_property
    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        B, L, a, step = self.B, self.L, self.a, self.step
        offset = a // 2
        left = (L + 1) // 2
        right = L - left
        if a + (L - 1) * step >= B:
            raise ValueError("object exceeds the row bound")
        left_end = a + (left - 1) * step
        right_base = B - offset - 1 - (right - 1) * step
        if right and (right_base < 0 or left_end >= right_base):
            raise ValueError("left and right halves overlap")
        i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        u = a + step * i
        v = np.where(j < left, a + step * j, right_base + step * (j - left))
        u, v = u.ravel(), v.ravel()
        own = set(zip(u.tolist(), v.tolist()))
        conj = set(zip(((-u) % B).tolist(), ((-v) % B).tolist()))
        if own & conj:
            raise ValueError("carriers collide with conjugate positions")
        return u, v

    def feasible(self) -> bool:
        try:
            self.positions
            return True
        except ValueError:
            return False

    @cached_property
    def rho(self) -> np.ndarray:
        """Normalised radial frequency of every carrier (cycles / sample)."""
        u, v = self.positions
        f = np.fft.fftfreq(self.B)
        return np.hypot(f[u], f[v])


def binary_object(L: int, seed: int) -> np.ndarray:
    """Balanced random L x L object (identical to the original generate_watermark)."""
    rng = np.random.default_rng(seed)
    bits = np.zeros(L * L, dtype=np.uint8)
    bits[rng.choice(L * L, size=L * L // 2, replace=False)] = 1
    return bits.reshape(L, L)


def phase_key(L: int, seed: int) -> np.ndarray:
    """Secret per-bit phases theta_rs, uniform on [-pi, pi)."""
    return np.random.Generator(np.random.PCG64(seed)).uniform(-np.pi, np.pi, size=(L, L))


def alpha_from_variance(B: int, bits: np.ndarray, D: float) -> float:
    return B ** 2 * np.sqrt(D / (2 * int(bits.sum())))


def spectrum(lay: Layout, bits: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Spectrum W of one period: exp(j theta) at the one-bit carriers, conjugates at (-u, -v)."""
    u, v = lay.positions
    b = bits.ravel().astype(bool)
    W = np.zeros((lay.B, lay.B), dtype=np.complex128)
    c = np.exp(1j * theta.ravel()[b])
    W[u[b], v[b]] += c
    W[(-u[b]) % lay.B, (-v[b]) % lay.B] += np.conj(c)
    return W


def period(lay: Layout, bits: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """One period w[p, q] = IDFT(W) (unit-amplitude carriers, real)."""
    return np.real(np.fft.ifft2(spectrum(lay, bits, theta)))


def embed(image: np.ndarray, lay: Layout, bits: np.ndarray, theta: np.ndarray, D: float):
    """Return (watermarked uint8 image, alpha)."""
    img = np.asarray(image, dtype=np.float64)
    H, W_ = img.shape
    if H % lay.B or W_ % lay.B:
        raise ValueError("image size must be a multiple of B")
    alpha = alpha_from_variance(lay.B, bits, D)
    w = np.tile(period(lay, bits, theta), (H // lay.B, W_ // lay.B))
    return np.rint(np.clip(img + alpha * w, 0, 255)).astype(np.uint8), alpha
