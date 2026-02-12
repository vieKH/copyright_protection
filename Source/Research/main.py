import numpy as np
import matplotlib.pyplot as plt

from Source.Research.utils import bit_error
from utils import compression_spectrum, add_qr_to_spectrum, extract_qr_from_image


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
    spectrum_compression = compression_spectrum(spectrum)
    plt.imshow(spectrum_compression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum after changing couple of points")

    plt.subplot(1, 3, 3)
    print("-------Data image after changing a couple of point-------")
    print(np.fft.ifft2(spectrum))
    image_with_watermark = np.real(np.fft.ifft2(spectrum))
    plt.imshow(image_with_watermark, cmap="gray")
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

    spectrum_compression = compression_spectrum(spectrum)
    plt.imshow(spectrum_compression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum after changing couple of points")

    plt.subplot(1, 3, 3)
    print("-------Data image after changing a few couples of point-------")
    print(np.fft.ifft2(spectrum))
    image_with_watermark = np.real(np.fft.ifft2(spectrum))
    plt.imshow(image_with_watermark, cmap="gray")
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
    spectrum_compression = compression_spectrum(spectrum)
    plt.imshow(spectrum_compression, cmap="gray")
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
    image_test = plt.imread("C://Users//Hoang//OneDrive//Desktop//Study//copyright_protection//Image//goldhilf.tif")

    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(image_test, cmap="gray")
    plt.axis('off')
    plt.title("Original image")

    plt.subplot(1, 2, 2)
    spectrum = np.fft.fft2(image_test)
    spectrum_compression = compression_spectrum(spectrum)
    plt.imshow(spectrum_compression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum")
    plt.show()


def research_qr(image: np.ndarray, L, x, y):
    """
    Show analyse when adding QR code in spectrum
    :param image: image in form np.ndarray
    :param L: Shape of QR  L x L
    :return: None
    """
    rng = np.random.default_rng(0)
    qr = (rng.random((L, L)) > 0.5).astype(np.uint8)

    plt.figure(figsize=(16,8))
    plt.subplot(2, 4, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 4,2)
    spectrum_original_image = np.fft.fft2(image)
    spectrum_compression = compression_spectrum(spectrum_original_image)
    plt.imshow(spectrum_compression, cmap="gray")
    plt.title("Spectrum original image")
    plt.axis("off")

    plt.subplot(2, 4,3)
    plt.imshow(qr, cmap="gray")
    plt.title("QR code")
    plt.axis("off")

    plt.subplot(2, 4,4)
    spectrum_embed_qr = add_qr_to_spectrum(qr, spectrum_original_image, x, y)
    spectrum_compression = compression_spectrum(spectrum_embed_qr)
    plt.imshow(spectrum_compression, cmap="gray")
    plt.title("Spectrum after adding QR code")
    plt.axis("off")

    plt.subplot(2, 4,5)
    image_with_watermark = np.real(np.fft.ifft2(spectrum_embed_qr))
    plt.imshow(image_with_watermark, cmap="gray")
    plt.title("Image after adding QR code")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    spectrum_with_watermark = np.fft.fft2(image_with_watermark)
    original_spectrum = np.fft.fft2(image)
    spectrum_image_only_qr = spectrum_with_watermark - original_spectrum
    spectrum_compression = compression_spectrum(spectrum_image_only_qr)
    plt.imshow(spectrum_compression, cmap="gray")
    plt.title("Spectrum after removing original image")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    qr_extracted = extract_qr_from_image(spectrum_image_only_qr, L)
    print(qr_extracted)
    plt.imshow(qr_extracted, cmap="gray")
    plt.axis("off")
    plt.title("QR was be extracted")

    print(f"Error extract  = {bit_error(qr, qr_extracted)}")
    plt.show()



if __name__ == "__main__":
    N = 32
    blackImage = np.zeros((N, N))
    image = plt.imread("C://Users//Hoang//OneDrive//Desktop//Study//copyright_protection//Image//goldhilf.tif")
    # couple_of_points(blackImage)
    # many_couple_of_points(blackImage)
    # phase(blackImage)
    # high_frequency()
    #research_qr(blackImage, 9, 3, 4)
    research_qr(image, 41, 40, 50)