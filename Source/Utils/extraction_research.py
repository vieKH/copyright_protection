from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .my_function import my_fft2
from .utils import  EPSILON,  bit_accuracy, conjugate_index, iter_offset_blocks, qr_to_spectrum_positions, resolve_embedding_params


@dataclass(frozen=True)
class ProgressiveExtractionResult:
    blocks_used: int
    recovered_qr: np.ndarray
    score_map: np.ndarray
    threshold: float
    predicted_ones: int
    accuracy: Optional[float] = None


def random_extract_start( block_size: int, seed: Optional[int] = None, avoid_zero_zero: bool = True) -> Tuple[int, int]:
    """Pick a reproducible random extraction offset inside one block."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    rng = np.random.default_rng(seed)
    while True:
        start_x = int(rng.integers(0, block_size))
        start_y = int(rng.integers(0, block_size))
        if not avoid_zero_zero or (start_x, start_y) != (0, 0):
            return start_x, start_y


def progressive_block_counts(n_available_blocks: int) -> List[int]:
    """Return [1, 2, 4, 8, ...] and append all blocks if needed."""
    if n_available_blocks <= 0:
        raise ValueError("n_available_blocks must be positive")

    counts: List[int] = []
    value = 1
    while value <= n_available_blocks:
        counts.append(value)
        value *= 2

    if counts[-1] != n_available_blocks:
        counts.append(n_available_blocks)

    return counts


def _phase_correction_matrix(block_size: int, start_x: int, start_y: int, phase_sign: int) -> np.ndarray:
    uu, vv = np.meshgrid(np.arange(block_size), np.arange(block_size), indexing="ij")
    return np.exp( phase_sign * 1j * 2.0 * np.pi * ((uu * start_x + vv * start_y) / block_size))


def _detrend_score_map(score_map: np.ndarray) -> np.ndarray:
    score = np.asarray(score_map, dtype=np.float64).copy()
    score -= np.mean(score, axis=1, keepdims=True)
    score -= np.mean(score, axis=0, keepdims=True)
    return score


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return (arr - median) / (1.4826 * mad + EPSILON)


def recover_qr_from_score(score_map: np.ndarray) -> Tuple[np.ndarray, float]:
    """Recover binary QR bits using the median score as threshold."""
    flat = np.asarray(score_map, dtype=np.float64).ravel()
    threshold = float(np.median(flat))
    recovered = (flat > threshold).astype(np.uint8)
    return recovered.reshape(score_map.shape), threshold


def score_watermark_map(avg_spectrum: np.ndarray, qr_size: int, size_region: int, phi: float, x: Optional[int] = None,
                        y: Optional[int] = None, offset: Optional[int] = None, detrend: bool = True, ) -> np.ndarray:
    """Build a normalized score map from an averaged extraction spectrum."""
    x, y, offset = resolve_embedding_params(size_region, x, y, offset)
    avg = np.asarray(avg_spectrum, dtype=np.complex128)
    if avg.shape != (size_region, size_region):
        raise ValueError("avg_spectrum shape must match (size_region, size_region)")

    raw_score = np.zeros((qr_size, qr_size), dtype=np.float64)
    e_neg = np.exp(-1j * phi)
    e_pos = np.exp(1j * phi)

    positions = qr_to_spectrum_positions( qr_size=qr_size, size_region=size_region, x=x, y=y, offset=offset)

    for pos in positions:
        qi = pos["qr_i"]
        qj = pos["qr_j"]
        u = pos["row"]
        v = pos["col"]
        uc, vc = conjugate_index(u, v, size_region)

        signal = 0.5 * (
            np.real(avg[u, v] * e_neg) + np.real(avg[uc, vc] * e_pos)
        )
        raw_score[qi, qj] = signal

    if detrend:
        raw_score = _detrend_score_map(raw_score)

    return _robust_zscore(raw_score)



blind_score_map = score_watermark_map


def collect_block_spectra(
    image: np.ndarray,
    block_size: int,
    start_x: int,
    start_y: int,
    phase_sign: int = 1,
    shuffle_blocks: bool = False,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Extract all valid offset blocks and return their phase-corrected spectra."""
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("Input image must be grayscale")

    blocks = list(iter_offset_blocks(img, block_size, start_x, start_y))
    if not blocks:
        raise ValueError("No blocks were extracted")

    if shuffle_blocks:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(blocks))
        blocks = [blocks[int(i)] for i in order]

    phase_correction = _phase_correction_matrix(
        block_size=block_size,
        start_x=start_x,
        start_y=start_y,
        phase_sign=phase_sign,
    )

    return [
        my_fft2(block.astype(np.float64)) * phase_correction
        for block, _, _ in blocks
    ]


def average_offset_spectrum(
    image: np.ndarray,
    block_size: int,
    start_x: int,
    start_y: int,
    phase_sign: int = 1,
) -> Tuple[np.ndarray, int]:
    """Average all phase-corrected spectra from one offset extraction grid."""
    spectra = collect_block_spectra(
        image=image,
        block_size=block_size,
        start_x=start_x,
        start_y=start_y,
        phase_sign=phase_sign,
    )
    return np.mean(spectra, axis=0), len(spectra)


def average_offset_spectrum_limited(
    image: np.ndarray,
    block_size: int,
    start_x: int,
    start_y: int,
    max_blocks: Optional[int] = None,
    random_blocks: bool = False,
    seed: Optional[int] = None,
    phase_sign: int = 1,
) -> Tuple[np.ndarray, int]:
    """Average only a fixed number of offset-block spectra."""
    spectra = collect_block_spectra(
        image=image,
        block_size=block_size,
        start_x=start_x,
        start_y=start_y,
        phase_sign=phase_sign,
        shuffle_blocks=random_blocks,
        seed=seed,
    )

    if max_blocks is not None:
        if max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        spectra = spectra[: min(max_blocks, len(spectra))]

    return np.mean(spectra, axis=0), len(spectra)


def extract_watermark( image: np.ndarray, qr_size: int, size_region: int, phi: float, start_x: int = 0, start_y: int = 0,
                       x: Optional[int] = None, y: Optional[int] = None, offset: Optional[int] = None,
                       phase_sign: int = 1, detrend: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Extract a QR watermark from all available offset blocks."""
    avg_spectrum, blocks_used = average_offset_spectrum(
        image=image,
        block_size=size_region,
        start_x=start_x,
        start_y=start_y,
        phase_sign=phase_sign,
    )
    score_map = score_watermark_map(
        avg_spectrum=avg_spectrum,
        qr_size=qr_size,
        size_region=size_region,
        phi=phi,
        x=x,
        y=y,
        offset=offset,
        detrend=detrend,
    )
    recovered, threshold = recover_qr_from_score(score_map)
    return recovered, score_map, avg_spectrum, blocks_used, threshold


def extract_watermark_limited_blocks(image: np.ndarray, qr_size: int, size_region: int, phi: float, start_x: int,start_y: int,
                                     max_blocks: int, random_blocks: bool = False, seed: Optional[int] = None,
                                     x: Optional[int] = None, y: Optional[int] = None, offset: Optional[int] = None,
                                     phase_sign: int = 1, detrend: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Extract a QR watermark using at most max_blocks extraction blocks."""
    avg_spectrum, blocks_used = average_offset_spectrum_limited(
        image=image,
        block_size=size_region,
        start_x=start_x,
        start_y=start_y,
        max_blocks=max_blocks,
        random_blocks=random_blocks,
        seed=seed,
        phase_sign=phase_sign,
    )
    score_map = score_watermark_map(
        avg_spectrum=avg_spectrum,
        qr_size=qr_size,
        size_region=size_region,
        phi=phi,
        x=x,
        y=y,
        offset=offset,
        detrend=detrend,
    )
    recovered, threshold = recover_qr_from_score(score_map)
    return recovered, score_map, avg_spectrum, blocks_used, threshold


def extract_watermark_search_offsets( image: np.ndarray, qr_size: int, size_region: int, phi: float,
                                      offset_candidates: Sequence[Tuple[int, int]],  x: Optional[int] = None, y: Optional[int] = None,
                                      offset: Optional[int] = None, phase_sign_candidates: Tuple[int, ...] = (1, -1), detrend: bool = True, ) -> Dict[str, object]:
    """Try several extraction offsets and return the best separated score map."""
    best: Optional[Dict[str, object]] = None
    diagnostics: List[Dict[str, object]] = []

    for phase_sign in phase_sign_candidates:
        for start_x, start_y in offset_candidates:
            recovered, score_map, avg_spectrum, blocks_used, threshold = extract_watermark(
                image=image,
                qr_size=qr_size,
                size_region=size_region,
                phi=phi,
                start_x=start_x,
                start_y=start_y,
                x=x,
                y=y,
                offset=offset,
                phase_sign=phase_sign,
                detrend=detrend,
            )

            flat = score_map.ravel()
            mask_one = recovered.ravel() == 1
            mask_zero = recovered.ravel() == 0
            if np.any(mask_one) and np.any(mask_zero):
                metric = float(
                    (np.mean(flat[mask_one]) - np.mean(flat[mask_zero]))
                    / (np.std(flat) + EPSILON)
                )
            else:
                metric = float("-inf")

            item: Dict[str, object] = {
                "start_x": start_x,
                "start_y": start_y,
                "phase_sign": phase_sign,
                "recovered_qr": recovered,
                "score_map": score_map,
                "avg_spectrum": avg_spectrum,
                "blocks_used": blocks_used,
                "threshold": threshold,
                "predicted_ones": int(np.sum(recovered)),
                "metric": metric,
            }
            diagnostics.append(
                {
                    "start_x": start_x,
                    "start_y": start_y,
                    "phase_sign": phase_sign,
                    "metric": metric,
                    "blocks_used": blocks_used,
                    "predicted_ones": int(np.sum(recovered)),
                    "threshold": threshold,
                }
            )

            if best is None or metric > float(best["metric"]):
                best = item

    if best is None:
        raise ValueError("No offset candidate was evaluated")

    best["diagnostics"] = diagnostics
    return best


def extract_progressive_by_blocks( image: np.ndarray, qr_size: int, size_region: int, phi: float, start_x: int, start_y: int,
                                   x: Optional[int] = None, y: Optional[int] = None, offset: Optional[int] = None,
                                   block_counts: Optional[Sequence[int]] = None, qr_true: Optional[np.ndarray] = None,
                                   phase_sign: int = 1, shuffle_blocks: bool = False, seed: Optional[int] = None,
                                   detrend: bool = True) -> Tuple[List[ProgressiveExtractionResult], int]:
    """Average 1, 2, 4, ... blocks and recover QR at each step."""
    x, y, offset = resolve_embedding_params(size_region, x, y, offset)
    spectra = collect_block_spectra( image=image, block_size=size_region, start_x=start_x, start_y=start_y,
                                     phase_sign=phase_sign, shuffle_blocks=shuffle_blocks, seed=seed)

    n_available = len(spectra)
    if block_counts is None:
        counts = progressive_block_counts(n_available)
    else:
        counts = sorted({min(int(v), n_available) for v in block_counts if int(v) > 0})

    if not counts:
        raise ValueError("block_counts does not contain any positive value")

    results: List[ProgressiveExtractionResult] = []
    running_sum = np.zeros_like(spectra[0], dtype=np.complex128)
    target_idx = 0

    for idx, spectrum in enumerate(spectra, start=1):
        running_sum += spectrum
        if idx != counts[target_idx]:
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
            detrend=detrend,
        )
        recovered_qr, threshold = recover_qr_from_score(score_map)
        accuracy = None if qr_true is None else bit_accuracy(qr_true, recovered_qr)

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

        target_idx += 1
        if target_idx >= len(counts):
            break

    return results, n_available
