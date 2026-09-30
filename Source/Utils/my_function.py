import numpy as np


def my_fft2(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError("img must be a 2D array")

    return np.fft.fft2(arr)


def my_ifft2(spectrum: np.ndarray) -> np.ndarray:
    spec = np.asarray(spectrum, dtype=np.complex128)

    if spec.ndim != 2:
        raise ValueError("spectrum must be a 2D array")

    return np.fft.ifft2(spec)