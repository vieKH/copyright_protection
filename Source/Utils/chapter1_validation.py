from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from skimage.metrics import structural_similarity

from .utils import qr_to_spectrum_positions


@dataclass(frozen=True)
class SingleBlockResult:
    bits: np.ndarray
    magnitudes: np.ndarray
    threshold: float
    block_origin: tuple[int, int]


def count_ssim(original: np.ndarray, watermarked: np.ndarray) -> float:
    """SSIM for 8-bit grayscale arrays; report alongside PSNR, not instead."""
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(watermarked, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or min(a.shape) < 7:
        raise ValueError('SSIM requires matching grayscale arrays of at least 7x7')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Images must have finite pixel values')
    if np.min(a) < 0 or np.max(a) > 255 or np.min(b) < 0 or np.max(b) > 255:
        raise ValueError('Expected 8-bit intensity scale [0, 255]')
    return float(structural_similarity(a, b, data_range=255))


def predicted_embed_mse(q: float, n: int, ones: int) -> float:
    """Unquantized MSE for 2*ones unique unit-modulus Fourier carriers.

    Uses the repo's unnormalized forward FFT and 1/n^2 normalized inverse FFT.
    Actual MSE differs after rounding or clipping into uint8.
    """
    if not np.isfinite(q) or q < 0 or n <= 0 or ones < 0:
        raise ValueError('Invalid embedding parameters')
    return 2.0 * ones * float(q) ** 2 / float(n) ** 4


def _threshold_scores(scores: np.ndarray, threshold: Optional[float],
                      expected_ones: Optional[int]) -> tuple[np.ndarray, float]:
    flat = np.asarray(scores, dtype=float).ravel()
    if not np.isfinite(flat).all() or not flat.size:
        raise ValueError('Nonfinite or empty detection scores')
    if threshold is not None and expected_ones is not None:
        raise ValueError('Choose a fixed threshold OR expected_ones, not both')
    if expected_ones is not None:
        if isinstance(expected_ones, (bool, np.bool_)) or not isinstance(expected_ones, (int, np.integer)) or not 0 <= expected_ones <= flat.size:
            raise ValueError('expected_ones must be between 0 and L^2')
        rank = np.argsort(-flat, kind='stable')
        decision = np.zeros(flat.size, dtype=np.uint8)
        decision[rank[:expected_ones]] = 1
        if expected_ones == 0:
            boundary = float('inf')
        elif expected_ones == flat.size:
            boundary = float('-inf')
        else:
            boundary = float((flat[rank[expected_ones-1]] + flat[rank[expected_ones]])/2)
        return decision.reshape(scores.shape), boundary
    boundary = float(np.median(flat) if threshold is None else threshold)
    if not np.isfinite(boundary):
        raise ValueError('threshold must be finite')
    return (scores > boundary).astype(np.uint8), boundary


def extract_single_block_magnitude(
    image: np.ndarray, qr_size: int, block_size: int,
    start_row: int = 0, start_col: int = 0, *,
    x: Optional[int] = None, y: Optional[int] = None,
    offset: Optional[int] = None, gap: Optional[int] = None,
    threshold: Optional[float] = None, expected_ones: Optional[int] = None,
) -> SingleBlockResult:
    """Original single-block baseline: FFT -> magnitude -> threshold.

    No phase key and NO multi-block averaging. Assumes a periodic watermark of
    period block_size and the repo's known spectral positions. The median rule
    assumes an approximately balanced payload; expected_ones uses its known
    Hamming weight, not the bit positions. Neither is an image-level detector.
    """
    arr = np.asarray(image)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise ValueError('Expected a finite grayscale image')
    if isinstance(block_size, bool) or block_size < 2 or qr_size < 1:
        raise ValueError('Invalid block or payload dimensions')
    if (start_row < 0 or start_col < 0 or
        start_row + block_size > arr.shape[0] or
        start_col + block_size > arr.shape[1]):
        raise ValueError('The chosen extraction window is outside the image')
    positions = qr_to_spectrum_positions(qr_size, block_size, x, y, offset, gap=gap)
    spectrum = np.fft.fft2(arr[start_row:start_row+block_size,
                               start_col:start_col+block_size].astype(np.float64))
    magnitudes = np.empty((qr_size, qr_size), dtype=np.float64)
    for pos in positions:
        u, v = pos['row'], pos['col']
        # For a real image the two magnitudes are identical up to roundoff.
        magnitudes[pos['qr_i'], pos['qr_j']] = 0.5 * (
            abs(spectrum[u, v]) + abs(spectrum[(-u) % block_size, (-v) % block_size])
        )
    bits, boundary = _threshold_scores(magnitudes, threshold, expected_ones)
    return SingleBlockResult(bits, magnitudes, boundary, (start_row, start_col))


def verify_dft_properties(image: np.ndarray, shift: tuple[int, int] = (3, 5)) -> dict[str, float]:
    """Numerical verification for rectangular M x N arrays (not only N x N)."""
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Expected a finite 2D real array')
    m, n = a.shape
    F = np.fft.fft2(a)
    uu, vv = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    phase = np.exp(-2j*np.pi*(uu*shift[0]/m + vv*shift[1]/n))
    F_shift = np.fft.fft2(np.roll(a, shift, axis=(0, 1)))
    F_conjugate = F[np.ix_((-np.arange(m)) % m, (-np.arange(n)) % n)]
    energy_spatial = np.sum(a*a)
    energy_spectral = np.sum(np.abs(F)**2)/(m*n)
    denom = max(float(np.linalg.norm(F)), 1.0)
    return {
        'roundtrip_max_abs': float(np.max(np.abs(np.fft.ifft2(F).real-a))),
        'parseval_relative': float(abs(energy_spatial-energy_spectral)/max(float(energy_spatial), 1.0)),
        'shift_relative': float(np.linalg.norm(F_shift-F*phase)/denom),
        'hermitian_relative': float(np.linalg.norm(F_conjugate-np.conj(F))/denom),
    }
