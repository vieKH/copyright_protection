import numpy as np

def dft_matrix(size: int, inverse: bool = False) -> np.ndarray:
    n = np.arange(size)
    k = n.reshape(size, 1)
    sign = 1 if inverse else -1
    w = np.exp(sign * 2j * np.pi * k * n / size)
    if inverse:
        w = w / size
    return w

def my_fft2(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.complex128)
    h, w = img.shape

    w_h = dft_matrix(h)
    w_w = dft_matrix(w)

    return w_h @ img @ w_w

def my_ifft2(spectrum: np.ndarray) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    h, w = spectrum.shape

    w_h_inv = dft_matrix(h, inverse=True)
    w_w_inv = dft_matrix(w, inverse=True)

    return w_h_inv @ spectrum @ w_w_inv