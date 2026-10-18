from dataclasses import dataclass
import math

import numpy as np

from .batch_research import BatchDecoder


@dataclass(frozen=True)
class DetectionResult:
    detected: bool
    score: float
    threshold: float
    blocks_used: int


def validate_reference(reference):
    bits = np.asarray(reference)
    if (bits.ndim != 2 or bits.shape[0] != bits.shape[1] or bits.size < 4
            or not np.all((bits == 0) | (bits == 1))
            or np.all(bits == bits.flat[0])):
        raise ValueError('reference must be a square binary matrix containing both 0 and 1')
    return bits.astype(np.float64)


def reference_score(score_map, reference):
    """Pearson correlation of soft projection scores with the reference bits.

    Unlike the median bit decoder, a binary output is not evidence of presence.
    A constant score map has no evidence and returns zero, not NaN.
    """
    bits = validate_reference(reference)
    scores = np.asarray(score_map, dtype=np.float64)
    if scores.shape != bits.shape or not np.all(np.isfinite(scores)):
        raise ValueError('score_map must be finite and have the reference shape')
    # Scale before centering to avoid overflow on large finite inputs.
    scale = float(np.max(np.abs(scores)))
    if scale == 0:
        return 0.0
    centered = scores / scale
    centered = centered - np.mean(centered)
    target = bits - np.mean(bits)
    norm = np.linalg.norm(centered) * np.linalg.norm(target)
    if norm == 0:
        return 0.0
    return float(np.clip(np.sum(centered * target) / norm, -1.0, 1.0))


def calibrate_threshold(negative_scores, alpha=0.01):
    """Conservative upper order statistic; decision is STRICTLY score > threshold.

    Rank ceil((n+1)*(1-alpha)) is a split-conformal negative quantile.
    The nominal false-positive bound requires exchangeable calibration/test
    negatives and a fixed scoring protocol. It is not a per-image confidence.
    Insufficient calibration data raises instead of silently inventing a gate.
    """
    scores = np.asarray(negative_scores, dtype=np.float64)
    if (scores.ndim != 1 or not scores.size or not np.all(np.isfinite(scores))
            or np.any(np.abs(scores) > 1)):
        raise ValueError('negative_scores must be a nonempty finite vector in [-1, 1]')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError('Not enough independent negative images for the requested alpha')
    return float(np.sort(scores)[rank - 1])


def classify_score(score, threshold, blocks_used=0):
    if (not np.isfinite(score) or not -1 <= score <= 1
            or not np.isfinite(threshold) or not -1 <= threshold <= 1):
        raise ValueError('score and threshold must be finite values in [-1, 1]')
    return DetectionResult(bool(score > threshold), float(score), float(threshold), int(blocks_used))


class WatermarkDetector:
    """Reusable detector for a known reference and fixed extraction settings.

    No threshold is supplied by default: measure()/score_map() can collect
    calibration data, but detect() requires an explicitly calibrated threshold.
    Changing settings or searching multiple keys requires recalibration.
    """
    def __init__(self, reference, size_region, phi, *, x=None, y=None, offset=None,
                 gap=1, start_x=0, start_y=0, stride=None):
        self.reference = validate_reference(reference).copy()
        if not isinstance(size_region, (int, np.integer)) or size_region <= 0:
            raise ValueError('size_region must be a positive integer')
        stride = size_region if stride is None else stride
        if (not isinstance(stride, (int, np.integer)) or stride <= 0
                or any(not isinstance(v, (int, np.integer)) or not 0 <= v < size_region
                       for v in (start_x, start_y))):
            raise ValueError('stride must be positive; extraction starts must be in the block')
        self.decoder = BatchDecoder(size_region, self.reference.shape[0], phi,
                                    x, y, offset, gap, start_x, start_y, stride)

    def measure(self, image):
        score_map, used = self.decoder.score_map(image)
        return reference_score(score_map, self.reference), used

    def detect(self, image, *, threshold):
        score, used = self.measure(image)
        return classify_score(score, threshold, used)
