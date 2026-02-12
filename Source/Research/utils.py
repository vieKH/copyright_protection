import numpy as np
import matplotlib.pyplot as plt
epsilon = 10**-5

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
    Add QR into spectrum by position (x, y), but QR will be divided to 2 part (left-right)
    :param qr: QR code in from np.ndarray
    :param spectrum: spectrum in from np.ndarray
    :param x: position X in spectrum for adding
    :param y: position Y in spectrum for adding
    :return:
    """
    N = spectrum.shape[0]
    L = qr.shape[0]

    if x > N//2 - L or y > N//2 - L:
        assert "fix x, y please!"

    for i in range(L):
        for j in range(L//2):
            spectrum[i+x][j+y] += qr[i][j] * np.exp(1j * np.pi/5)
            spectrum[N-i-x][N-j-y] += qr[i][j] * np.exp(-1j * np.pi/5)
            spectrum[i+x][-1- j] += qr[i][-1-j] * np.exp(1j * np.pi/5)
            spectrum[N - i - x][j+1] += qr[i][-1-j] * np.exp(-1j * np.pi/5)

        spectrum[i+x][L//2 + y] += qr[i][L//2] * np.exp(1j * np.pi/5)
        spectrum[N - i - x][N - L//2 - y] += qr[i][L//2] * np.exp(-1j * np.pi/5)
    return spectrum


def extract_qr_from_image(spectrum_qr: np.ndarray, L):
    N = spectrum_qr.shape[0]
    sum = np.zeros((N//2 - L , N//2 - L ))
    #print(spectrum_qr)
    for i in range(N // 2 - L):
        for j in range(N//2 - L):
            sum[i][j] = energy_region_qr(i, j, L, spectrum_qr)

    x, y = np.unravel_index(np.argmax(sum), sum.shape)
    qr = np.real(extract_qr(int(x), int(y), spectrum_qr, L))
    qr_bits = (np.abs(qr) > epsilon).astype(np.uint8)
    return qr_bits

def energy_region_qr(x: int, y: int, L: int, spectrum: np.ndarray):
    sum = 0
    for i in range(L):
        for j in range(L//2):
            sum += np.abs(np.real(spectrum[i + x][j + y]))
            sum += np.abs(np.real(spectrum[i + x][-1 - j]))

        sum += np.abs(np.real(spectrum[i + x][L // 2 + y]))
    return sum


def extract_qr(x: int, y: int, spectrum_qr: np.ndarray, L: int):
    qr = np.zeros((L, L), dtype=np.complex128)
    for i in range(L):
        for j in range(L//2):
            qr[i][j] = spectrum_qr[i+x][j+y]
            qr[i][-1-j] = spectrum_qr[i+x][-1- j]

        qr[i][L//2] = spectrum_qr[i+x][L//2 + y]
    return qr


def bit_error(qr: np.ndarray, qr_extracted: np.ndarray):
    L = qr.shape[0]
    return (qr ^ qr_extracted).sum() // L**2
