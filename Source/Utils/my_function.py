from __future__ import annotations
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=32)
def dft_matrix(size: int, inverse: bool = False) -> np.ndarray:
    """Return a cached 1D DFT or inverse-DFT matrix."""
    if size <= 0:
        raise ValueError("size must be positive")

    n = np.arange(size)
    k = n.reshape(size, 1)
    sign = 1 if inverse else -1
    matrix = np.exp(sign * 2j * np.pi * k * n / size)

    if inverse:
        matrix = matrix / size

    return matrix


def my_fft2(img: np.ndarray) -> np.ndarray:
    """2D FFT implemented with DFT matrices."""
    arr = np.asarray(img, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError("img must be a 2D array")

    h, w = arr.shape
    return dft_matrix(h) @ arr @ dft_matrix(w)


def my_ifft2(spectrum: np.ndarray) -> np.ndarray:
    """2D inverse FFT implemented with inverse-DFT matrices."""
    spec = np.asarray(spectrum, dtype=np.complex128)
    if spec.ndim != 2:
        raise ValueError("spectrum must be a 2D array")

    h, w = spec.shape
    return dft_matrix(h, inverse=True) @ spec @ dft_matrix(w, inverse=True)
