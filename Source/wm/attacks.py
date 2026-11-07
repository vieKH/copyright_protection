"""Attacks of chapter 4 (geometric attacks are not considered).  Output: rounded uint8."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy import ndimage


def _gauss_1d(sigma: float) -> np.ndarray:
    half = int(3 * sigma)
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def _u8(x: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(x, 0, 255)).astype(np.uint8)


def contrast(img, k: float):
    """J = k I (linear contrast change; the constant term only affects the zero frequency)."""
    return _u8(k * np.asarray(img, dtype=np.float64))


def noise(img, variance: float, rng: np.random.Generator):
    """Additive white Gaussian noise N(0, variance)."""
    f = np.asarray(img, dtype=np.float64)
    return _u8(f + rng.normal(0.0, np.sqrt(variance), f.shape))


def blur(img, sigma: float):
    """Separable Gaussian kernel, window 2*int(3 sigma)+1, symmetric boundary."""
    k = _gauss_1d(sigma)
    f = np.asarray(img, dtype=np.float64)
    return _u8(ndimage.correlate1d(ndimage.correlate1d(f, k, 0, mode="reflect"), k, 1, mode="reflect"))


def sharpen(img, M: int, gain: float = 5.0):
    """Unsharp masking J = I + gain (I - box_M * I)."""
    f = np.asarray(img, dtype=np.float64)
    return _u8(f + gain * (f - ndimage.uniform_filter(f, size=int(M) | 1, mode="reflect")))


def jpeg(img, quality: int):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf), dtype=np.uint8)


def apply(img, name: str, value, rng: np.random.Generator | None = None):
    if name == "none":
        return np.asarray(img)
    if name == "noise":
        return noise(img, value, rng)
    return {"contrast": contrast, "blur": blur, "sharpen": sharpen, "jpeg": jpeg}[name](img, value)
