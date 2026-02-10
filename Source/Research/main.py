import numpy as np
import matplotlib.pyplot as plt
from utils import compression_spectrum, add_qr_to_spectrum


def couple_of_points(image: np.ndarray):
    """
    Showing analyse what was happened when changed 1 couple of points in spectrum
    :param image: image in form np.ndarray
    :return: None
    """
    plt.figure(figsize=(15, 8))
    N = image.shape[0]

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title("Original image")

    plt.subplot(1, 3, 2)
    spectrum = np.fft.fft2(image)
    spectrum[2][N-2] = 1
    spectrum[N-2][2] = 1
    spectrumCompression = compression_spectrum(spectrum)
    plt.imshow(spectrumCompression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum after changing couple of points")

    plt.subplot(1, 3, 3)
    print("-------Data image after changing a couple of point-------")
    print(np.fft.ifft2(spectrum))
    image_trans = np.real(np.fft.ifft2(spectrum))
    plt.imshow(image_trans, cmap="gray")
    plt.axis('off')
    plt.title("Image after changing couple of points")

    plt.show()


def many_couple_of_points(image: np.ndarray):
    """
    Showing analyse what has happened when changed a little couple of points in spectrum
    :param image: image in form np.ndarray
    :return: None
    """
    plt.figure(figsize=(15, 8))
    N = image.shape[0]

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title("Original image")

    plt.subplot(1, 3, 2)
    spectrum = np.fft.fft2(image)
    m = [2, 4, 14]
    n = [2, 0, 2]

    for i in range(3):
        spectrum[m[i], n[i]] = 1
        spectrum[(N-m[i]) % N, (N-n[i]) % N] = 1

    spectrumCompression = compression_spectrum(spectrum)
    plt.imshow(spectrumCompression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum after changing couple of points")

    plt.subplot(1, 3, 3)
    print("-------Data image after changing a few couples of point-------")
    print(np.fft.ifft2(spectrum))
    image_trans = np.real(np.fft.ifft2(spectrum))
    plt.imshow(image_trans, cmap="gray")
    plt.axis('off')
    plt.title("Image after changing couple of points")

    plt.show()


def phase(image: np.ndarray):
    """
    Showing analyse what was happened when add phase e^(i*phi)
    :param image: image in form np.ndarray
    :return: None
    """
    plt.figure(figsize=(15, 8))
    N = image.shape[0]

    plt.subplot(1, 3, 1)
    plt.imshow(blackImage, cmap="gray")
    plt.axis('off')
    plt.title("Original image")

    plt.subplot(1, 3, 2)
    spectrum = np.fft.fft2(blackImage)
    spectrum[2, 2] = np.exp(1j * np.pi)
    spectrum[N-2, N-2] = np.exp(-1j * np.pi)
    spectrum[2, 14] = np.exp(1j * np.pi)
    spectrum[N-2, N-14] = np.exp(-1j * np.pi)
    spectrum[3, 6] = np.exp(1j * np.pi)
    spectrum[N-3, N-6] = np.exp(-1j * np.pi)
    spectrumCompression = compression_spectrum(spectrum)
    plt.imshow(spectrumCompression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum after changing phase")

    plt.subplot(1, 3, 3)
    print("-------Data image after changing phase-------")
    print(np.fft.ifft2(spectrum))
    image = np.real(np.fft.ifft2(spectrum))
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title("Image after changing phase")

    plt.show()


def high_frequency():
    """
    For watching frenquency
    :return: None
    """
    imageTest = plt.imread("C://Users//Hoang//OneDrive//Desktop//Study//copyright_protection//Image//goldhilf.tif")

    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(imageTest, cmap="gray")
    plt.axis('off')
    plt.title("Original image")

    plt.subplot(1, 2, 2)
    spectrum = np.fft.fft2(imageTest)
    spectrumCompression = compression_spectrum(spectrum)
    plt.imshow(spectrumCompression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum")
    plt.show()


def research_qr(image: np.ndarray, L):
    """
    Show analyse when adding QR code in spectrum
    :param image: image in form np.ndarray
    :param L: Size of QR code
    :return: None
    """
    rng = np.random.default_rng(0)
    qr = (rng.random((L, L)) > 0.5).astype(np.uint8)

    plt.figure(figsize=(16,8))
    plt.subplot(1,4,1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(qr, cmap="gray")
    plt.title("QR code")
    plt.axis("off")

    plt.subplot(1,4,3)
    spectrum = add_qr_to_spectrum(qr, np.fft.fft2(image), 3, 4)
    spectrumCompression = compression_spectrum(spectrum)
    plt.imshow(spectrumCompression, cmap="gray")
    plt.title("Spectrum after adding QR code")
    plt.axis("off")

    plt.subplot(1,4,4)
    image_trans = np.real(np.fft.ifft2(spectrum))
    print(image_trans)
    plt.imshow(image_trans, cmap="gray")
    plt.title("Image after adding QR code")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    N = 32
    blackImage = np.zeros((N, N))

    # couple_of_points(blackImage)
    # many_couple_of_points(blackImage)
    # phase(blackImage)
    # high_frequency()
    research_qr(blackImage, 8)