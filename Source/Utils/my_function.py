import numpy as np

def dft_matrix(N: int, inverse: bool = False) -> np.ndarray:
    n = np.arange(N)
    k = n.reshape(N, 1)
    sign = 1 if inverse else -1
    W = np.exp(sign * 2j * np.pi * k * n / N)
    if inverse:
        W = W / N
    return W


def my_fft2(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.complex128)
    N, M = img.shape
    if N != M:
        raise ValueError("Only square image NxN is supported.")
    W = dft_matrix(N)
    return W @ img @ W


def my_ifft2(spectrum: np.ndarray) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    N, M = spectrum.shape
    if N != M:
        raise ValueError("Only square spectrum NxN is supported.")
    W_inv = dft_matrix(N, inverse=True)
    return W_inv @ spectrum @ W_inv



