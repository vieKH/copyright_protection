"""Comparable single-window, magnitude voting, phase voting and coherent averaging.

The same FFT windows are used for every detector. The accumulated complex score
matches BatchDecoder's phase correction and phase-projection definitions.
No detector uses ground-truth bit positions; they are only used by the evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .extraction_research import recover_qr_from_score
from .phases import resolve_phase_map
from .utils import qr_to_spectrum_positions


@dataclass(frozen=True)
class Comparison:
    blocks: int
    magnitude_vote: np.ndarray
    phase_vote: np.ndarray
    complex_average: np.ndarray
    magnitude_vote_ties: int
    phase_vote_ties: int


def _reference_bins(n: int, qr_size: int, phase, x: int, y: int, offset: int, gap: int):
    positions = qr_to_spectrum_positions(qr_size, n, x, y, offset, gap=gap)
    u = np.asarray([p['row'] for p in positions], dtype=np.intp)
    v = np.asarray([p['col'] for p in positions], dtype=np.intp)
    phi = resolve_phase_map(phase, qr_size)
    carrier = np.exp(1j * np.asarray(phi).reshape(-1))
    return u, v, carrier


def compare_methods(image: np.ndarray, qr_size: int, block_size: int, phase,
                    x: int, y: int, offset: int, gap: int, start_row: int,
                    start_col: int, stride: int, counts=(1, 4, 16, 64),
                    *, shuffle_seed: int | None = None) -> list[Comparison]:
    """Compare three detectors on identical spatial windows and thresholds.

    The two voting decoders use per-window median thresholds. For an even number
    of windows, ties are classified as zero. Complex averaging uses one median
    threshold AFTER averaging phase-compensated Fourier coefficients.

    `shuffle_seed=None` keeps deterministic row-major window order. A fixed
    integer shuffles the complete list before taking the requested prefix.
    Each returned `blocks` value is an actual count, never an invented K.
    """
    img = np.asarray(image)
    if img.ndim != 2 or not np.isfinite(img).all():
        raise ValueError('Expected a finite grayscale image')
    n = int(block_size)
    if n < 2 or stride < 1 or not (0 <= start_row < n and 0 <= start_col < n):
        raise ValueError('Invalid block size, stride, or window origin')
    requested = tuple(sorted(set(int(z) for z in counts)))
    if not requested or requested[0] < 1:
        raise ValueError('counts must contain positive integers')
    u, v, carrier = _reference_bins(n, qr_size, phase, x, y, offset, gap)
    ur, vr = (-u) % n, (-v) % n
    # Keep a symmetric pair to match the repository's BatchDecoder exactly.
    coords = [(r, c)
              for r in range(start_row, img.shape[0] - n + 1, stride)
              for c in range(start_col, img.shape[1] - n + 1, stride)]
    if not coords:
        raise ValueError('No complete FFT window available')
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        coords = [coords[i] for i in rng.permutation(len(coords))]
    available = min(len(coords), requested[-1])
    counts_used = sorted({min(c, available) for c in requested})
    targets = set(counts_used)
    pixel_count = qr_size * qr_size
    magnitude_votes = np.zeros(pixel_count, dtype=np.int32)
    phase_votes = np.zeros(pixel_count, dtype=np.int32)
    complex_sum = np.zeros(pixel_count, dtype=np.complex128)
    complex_sum_conj = np.zeros(pixel_count, dtype=np.complex128)
    out = []
    for k, (r, c) in enumerate(coords[:available], start=1):
        fft = np.fft.fft2(img[r:r+n, c:c+n].astype(np.float64))
        # Magnitudes are invariant under unit-modulus phase correction.
        mag = 0.5 * (np.abs(fft[u, v]) + np.abs(fft[ur, vr]))
        decisions_mag, _ = recover_qr_from_score(mag.reshape(qr_size, qr_size))
        magnitude_votes += decisions_mag.reshape(-1)

        z = fft[u, v] * np.exp(-2j*np.pi*(u*(r%n) + v*(c%n))/n)
        z_conj = fft[ur, vr] * np.exp(-2j*np.pi*(ur*(r%n) + vr*(c%n))/n)
        # Same projection as score_watermark_map / BatchDecoder, prior to z-score.
        phase_scores = 0.5 * (
            np.real(z * np.conj(carrier)) + np.real(z_conj * carrier))
        decisions_phase, _ = recover_qr_from_score(phase_scores.reshape(qr_size, qr_size))
        phase_votes += decisions_phase.reshape(-1)
        complex_sum += z
        complex_sum_conj += z_conj

        if k in targets:
            # The float scaling 1/k and robust z-score are monotone; they do
            # not change a global median decision.
            average_score = 0.5 * (
                np.real(complex_sum * np.conj(carrier)) +
                np.real(complex_sum_conj * carrier)) / k
            decision_avg, _ = recover_qr_from_score(average_score.reshape(qr_size, qr_size))
            decision_mag = (magnitude_votes > k / 2).astype(np.uint8).reshape(qr_size, qr_size)
            decision_phase = (phase_votes > k / 2).astype(np.uint8).reshape(qr_size, qr_size)
            out.append(Comparison(
                blocks=k,
                magnitude_vote=decision_mag,
                phase_vote=decision_phase,
                complex_average=decision_avg,
                magnitude_vote_ties=int(np.count_nonzero(magnitude_votes * 2 == k)),
                phase_vote_ties=int(np.count_nonzero(phase_votes * 2 == k)),
            ))
    return out
