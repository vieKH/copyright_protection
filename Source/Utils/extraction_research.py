"""Progressive watermark extraction utilities.

This module keeps the embedding algorithm unchanged and adds a research-oriented
extractor: it averages more and more extraction blocks, then recovers the QR from
the averaged spectrum with a data-driven threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .my_function import my_fft2
from .utils import (
    bit_accuracy,
    design_params,
    iter_offset_blocks,
    qr_to_spectrum_positions,
)

EPS = 1e-6


@dataclass
class ProgressiveExtractionResult:
    blocks_used: int
    recovered_qr: np.ndarray
    score_map: np.ndarray
    threshold: float
    predicted_ones: int
    accuracy: Optional[float] = None


def random_extract_start(
    block_size: int,
    seed: Optional[int] = None,
    avoid_zero_zero: bool = True,
) -> Tuple[int, int]:
    """Pick a random extraction start inside one block.

    start_x/start_y are extraction offsets in the image domain, not the QR
    embedding position in the spectrum. If avoid_zero_zero=True, (0, 0) is
    rejected so the experiment is not the trivial aligned case.
    """
    rng = np.random.default_rng(seed)
    while True:
        start_x = int(rng.integers(0, block_size))
        start_y = int(rng.integers(0, block_size))
        if not avoid_zero_zero or (start_x, start_y) != (0, 0):
            return start_x, start_y


def progressive_block_counts(n_available_blocks: int) -> List[int]:
    """Return 1, 2, 4, 8, ... and finally all available blocks."""
    if n_available_blocks <= 0:
        raise ValueError("n_available_blocks must be positive")

    counts: List[int] = []
    k = 1
    while k <= n_available_blocks:
        counts.append(k)
        k *= 2

    if counts[-1] != n_available_blocks:
        counts.append(n_available_blocks)

    return counts


def _conj_index(u: int, v: int, n: int) -> Tuple[int, int]:
    return (-u) % n, (-v) % n


def _resolve_embedding_params(
    size_region: int,
    x: Optional[int],
    y: Optional[int],
    offset: Optional[int],
) -> Tuple[int, int, int]:
    dx, dy, doffset = design_params(size_region)
    return (
        dx if x is None else int(x),
        dy if y is None else int(y),
        doffset if offset is None else int(offset),
    )


def _protected_spectrum_coordinates(
    qr_size: int,
    size_region: int,
    x: int = 0,
    y: int = 0,
    offset: int = 0,
) -> set[Tuple[int, int]]:
    protected: set[Tuple[int, int]] = set()
    positions = qr_to_spectrum_positions(qr_size, size_region, x, y, offset)
    for pos in positions:
        u = pos["row"]
        v = pos["col"]
        protected.add((u, v))
        protected.add(_conj_index(u, v, size_region))
    return protected


def _local_median_magnitude(
    mag: np.ndarray,
    u: int,
    v: int,
    radius: int,
    protected: set[Tuple[int, int]],
    min_samples: int = 8,
    max_radius: Optional[int] = None,
) -> float:
    if max_radius is None:
        max_radius = max(radius + 1, min(mag.shape) // 4)

    h, w = mag.shape

    for r in range(radius, max_radius + 1):
        u1 = max(0, u - r)
        u2 = min(h, u + r + 1)
        v1 = max(0, v - r)
        v2 = min(w, v + r + 1)

        vals: List[float] = []

        for uu in range(u1, u2):
            for vv in range(v1, v2):
                if (uu, vv) == (u, v):
                    continue

                if (uu, vv) in protected:
                    continue

                val = float(mag[uu, vv])
                if val > EPS:
                    vals.append(val)

        if len(vals) >= min_samples:
            return float(np.median(vals))


    vals = [
        float(mag[uu, vv])
        for uu in range(h)
        for vv in range(w)
        if (uu, vv) not in protected and float(mag[uu, vv]) > EPS
    ]

    if vals:
        return float(np.median(vals))

    return EPS


def _detrend_score_map(score_map: np.ndarray) -> np.ndarray:
    score = np.asarray(score_map, dtype=np.float64).copy()
    score -= np.mean(score, axis=1, keepdims=True)
    score -= np.mean(score, axis=0, keepdims=True)
    return score


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return (values - med) / (1.4826 * mad + EPS)


def score_watermark_map(
    avg_spectrum: np.ndarray,
    qr_size: int,
    size_region: int,
    phi: float,
    x: Optional[int] = None,
    y: Optional[int] = None,
    offset: Optional[int] = None,
    ring_radius: Optional[int] = None,
    detrend: bool = True,
) -> np.ndarray:
    """Build a QR score map from the averaged spectrum.

    A high score means the frequency pair is aligned with the expected watermark
    phase. This is still semi-blind because phi/x/y/offset are detector
    parameters, but the binary decision does not require knowing the true number
    of 1-bits.
    """
    x, y, offset = _resolve_embedding_params(size_region, x, y, offset)
    if ring_radius is None:
        ring_radius = 2

    positions = qr_to_spectrum_positions(qr_size, size_region, x, y, offset)
    protected = _protected_spectrum_coordinates(qr_size, size_region, x, y, offset)
    mag = np.abs(avg_spectrum)
    raw = np.zeros((qr_size, qr_size), dtype=np.float64)

    e_neg = np.exp(-1j * phi)
    e_pos = np.exp(1j * phi)

    for pos in positions:
        qi = pos["qr_i"]
        qj = pos["qr_j"]
        u = pos["row"]
        v = pos["col"]
        uc, vc = _conj_index(u, v, size_region)

        # Paired phase projection: watermark bit 1 should align with +phi at
        # (u, v) and -phi at the conjugate coordinate.
        signal = 0.5 * (
            np.real(avg_spectrum[u, v] * e_neg)
            + np.real(avg_spectrum[uc, vc] * e_pos)
        )

        raw[qi, qj] = signal

    if detrend:
        raw = _detrend_score_map(raw)

    return _robust_zscore(raw)


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu threshold for a small vector, without extra dependencies."""
    flat = np.sort(np.asarray(values, dtype=np.float64).ravel())
    if flat.size < 2:
        return float(flat[0]) if flat.size else 0.0

    unique = np.unique(flat)
    if unique.size < 2:
        return float(unique[0])

    thresholds = (unique[:-1] + unique[1:]) / 2.0
    best_threshold = float(np.median(flat))
    best_score = -np.inf

    for tau in thresholds:
        left = flat[flat <= tau]
        right = flat[flat > tau]
        if left.size == 0 or right.size == 0:
            continue
        w0 = left.size / flat.size
        w1 = right.size / flat.size
        score = w0 * w1 * (float(np.mean(left)) - float(np.mean(right))) ** 2
        if score > best_score:
            best_score = score
            best_threshold = float(tau)

    return best_threshold


def recover_qr_from_score(score_map: np.ndarray) -> Tuple[np.ndarray, float]:
    flat = np.asarray(score_map, dtype=np.float64).ravel()

    threshold = float(np.median(flat))
    recovered = (flat > threshold).astype(np.uint8)

    return recovered.reshape(score_map.shape), threshold


def collect_block_spectra(
    image: np.ndarray,
    block_size: int,
    start_x: int,
    start_y: int,
    phase_sign: int = 1,
    shuffle_blocks: bool = False,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Extract all valid blocks at an offset and return their corrected spectra."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")

    blocks = list(iter_offset_blocks(image, block_size, start_x, start_y))
    if not blocks:
        raise ValueError("No blocks were extracted")

    if shuffle_blocks:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(blocks))
        blocks = [blocks[int(i)] for i in order]

    uu, vv = np.meshgrid(
        np.arange(block_size),
        np.arange(block_size),
        indexing="ij",
    )
    phase_correction = np.exp(
        phase_sign
        * 1j
        * 2.0
        * np.pi
        * ((uu * start_x + vv * start_y) / block_size)
    )

    spectra: List[np.ndarray] = []
    for block, _, _ in blocks:
        spectra.append(my_fft2(block.astype(np.float64)) * phase_correction)
    return spectra


def extract_progressive_by_blocks(
    image: np.ndarray,
    qr_size: int,
    size_region: int,
    phi: float,
    start_x: int,
    start_y: int,
    x: Optional[int] = None,
    y: Optional[int] = None,
    offset: Optional[int] = None,
    block_counts: Optional[Sequence[int]] = None,
    qr_true: Optional[np.ndarray] = None,
    ring_radius: Optional[int] = None,
    phase_sign: int = 1,
    shuffle_blocks: bool = False,
    seed: Optional[int] = None,
    detrend: bool = True,
) -> Tuple[List[ProgressiveExtractionResult], int]:
    """Average 1, 2, 4, ... blocks and recover QR at each step."""
    x, y, offset = _resolve_embedding_params(size_region, x, y, offset)
    spectra = collect_block_spectra(
        image=image,
        block_size=size_region,
        start_x=start_x,
        start_y=start_y,
        phase_sign=phase_sign,
        shuffle_blocks=shuffle_blocks,
        seed=seed,
    )

    n_available = len(spectra)
    if block_counts is None:
        block_counts = progressive_block_counts(n_available)
    else:
        block_counts = sorted({int(v) for v in block_counts if int(v) > 0})
        block_counts = [min(v, n_available) for v in block_counts]
        block_counts = sorted(set(block_counts))
        if not block_counts:
            raise ValueError("block_counts does not contain any positive value")

    results: List[ProgressiveExtractionResult] = []
    running_sum = np.zeros_like(spectra[0], dtype=np.complex128)
    next_count_idx = 0

    for idx, spectrum in enumerate(spectra, start=1):
        running_sum += spectrum
        if idx != block_counts[next_count_idx]:
            continue

        avg_spectrum = running_sum / idx
        score_map = score_watermark_map(
            avg_spectrum=avg_spectrum,
            qr_size=qr_size,
            size_region=size_region,
            phi=phi,
            x=x,
            y=y,
            offset=offset,
            ring_radius=ring_radius,
            detrend=detrend,
        )
        recovered_qr, threshold = recover_qr_from_score(score_map)

        accuracy = None
        if qr_true is not None:
            accuracy = bit_accuracy(qr_true, recovered_qr)

        results.append(
            ProgressiveExtractionResult(
                blocks_used=idx,
                recovered_qr=recovered_qr,
                score_map=score_map,
                threshold=threshold,
                predicted_ones=int(np.sum(recovered_qr)),
                accuracy=accuracy,
            )
        )

        next_count_idx += 1
        if next_count_idx >= len(block_counts):
            break

    return results, n_available
