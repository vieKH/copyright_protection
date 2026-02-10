import numpy as np
import matplotlib.pyplot as plt


def compression_spectrum(spectrum: np.ndarray):
    """
    Optimize value of spectrum for showing
    :param spectrum: spectrum in from np.ndarray
    :return: data of spectrum after optimizing
    """
    #spectrum_shift = np.fft.fftshift(spectrum)
    return np.log(1 + np.abs(spectrum))

def add_qr_to_spectrum(qr: np.ndarray, spectrum: np.ndarray, x: int, y: int ):
    """

    :param qr: QR code in from np.ndarray
    :param spectrum: spectrum in from np.ndarray
    :param x: position X in spectrum for adding
    :param y: position Y in spectrum for adding
    :return:
    """
    N = spectrum.shape[0]
    L = qr.shape[0]

    if x > N//2 - L or y > N//2 - L:
        assert "fix x or y please!"

    for i in range(L):
        for j in range(L//2):
            spectrum[i+x][j+y] = qr[i][j]
            spectrum[N-i-x][N-j-y] = qr[i][j]
            spectrum[i+x][-1- j] = qr[i][-1-j]
            spectrum[N - i - x][j+1] = qr[i][-1-j]
        if L % 2 == 1:
            spectrum[i+x][L//2 + y + 1] = qr[i][L//2 + 1]
    return spectrum


