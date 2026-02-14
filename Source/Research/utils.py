import numpy as np

epsilon = 1e-8


def compression_spectrum(spectrum: np.ndarray):
    """
    Optimize value of spectrum for showing
    :param spectrum: spectrum in from np.ndarray
    :return: data of spectrum after optimizing
    """
    return np.log(1 + np.abs(spectrum))


def check_error_data(image_research: np.ndarray):
    image_imag = np.imag(image_research)
    rows, columns = np.where(np.abs(image_imag) > epsilon)
    for i, j in zip(rows, columns):
        print(i, j, image_imag[i, j])
    return None


def add_qr_to_spectrum(qr: np.ndarray, spectrum: np.ndarray, x: int, y: int, phase: float = np.pi/5):
    """
    Add QR into spectrum by position (x, y), but QR will be divided to 2 part (left-right)
    :param qr: QR code in from np.ndarray
    :param spectrum: spectrum in from np.ndarray
    :param phase: phi in e^i*phi
    :param x: position X in spectrum for adding
    :param y: position Y in spectrum for adding
    :return:
    """
    N = spectrum.shape[0]
    L = qr.shape[0]
    L_mid = L//2

    if x > N//2 - L or y > N//2 - L:
        assert "fix x, y please!"

    e_pos = np.exp(1j * phase)
    e_neg = np.exp(-1j * phase)

    # rows = slice(x, x+L)
    # left_columns = slice(y, y + L_mid+1)
    #
    # spectrum[rows, left_columns] += qr[:, :L_mid+1] * e_pos
    # spectrum[N - x - L: N-x, N - (y + L_mid): N - y] += qr[:, :L_mid+1] * e_neg
    #
    # right_columns = slice(N - L_mid, N)
    # spectrum[rows, right_columns] += qr[:, L_mid+1:] * e_pos

    for i in range(L):
        for j in range(L//2):
            spectrum[i+x][j+y] += qr[i][j] * e_pos
            spectrum[N-i-x][N-j-y] += qr[i][j] * e_neg
            spectrum[i+x][-1- j] += qr[i][-1-j] * e_pos
            spectrum[N - i - x][j+1] += qr[i][-1-j] * e_neg

        spectrum[i+x][L//2 + y] += qr[i][L//2] * e_pos
        spectrum[N - i - x][N - L//2 - y] += qr[i][L//2] * e_neg
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
    L_mid = L // 2
    left = spectrum[x:x + L, y:y + L_mid]
    right = spectrum[x:x + L, -L_mid:]
    mid = spectrum[x:x + L, y + L_mid]
    return np.abs(np.real(left)).sum() + np.abs(np.real(right)).sum() + np.abs(np.real(mid)).sum()


def extract_qr(x: int, y: int, spectrum_qr: np.ndarray, L: int):
    L_mid = L // 2
    qr = np.zeros((L, L), dtype=np.complex128)

    qr[:, :L_mid] = spectrum_qr[x:x+L, y:y+L_mid]
    qr[:, -L_mid:] = spectrum_qr[x:x+L, -L_mid:]
    qr[:, L_mid] = spectrum_qr[x:x+L, y+L_mid]

    return qr


def bit_error(qr: np.ndarray, qr_extracted: np.ndarray):
    L = qr.shape[0]
    return (qr ^ qr_extracted).sum() / (L**2)
