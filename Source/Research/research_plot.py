import matplotlib.pyplot as plt
from Source.Research.utils import *


def couple_of_points(image: np.ndarray, x: int, y: int):
    """
    Showing analyse what was happened when changed 1 couple of points in spectrum
    :param image: image in form np.ndarray
    :param x: position x for adding QR
    :param y: position y for adding QR
    :return: None
    """
    print("Research about changing 1 couple of point")
    plt.figure(figsize=(15, 8))
    N = image.shape[0]

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title("Original image")

    plt.subplot(1, 3, 2)
    spectrum = np.fft.fft2(image)

    spectrum[x][y] = 1
    spectrum[(N-x) % N][(N -y) % N] = 1

    spectrum_compression = compression_spectrum(spectrum)
    plt.imshow(spectrum_compression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum after changing couple of points")

    plt.subplot(1, 3, 3)

    image_after_research = np.fft.ifft2(spectrum)
    print(f"The wrong pixels:")
    check_error_data(image_after_research)
    image_with_watermark = np.real(image_after_research)
    plt.imshow(image_with_watermark, cmap="gray")
    plt.axis('off')
    plt.title("Image after changing couple of points")

    plt.show()


def many_couple_of_points(image: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Showing analyse what has happened when changed a little couple of points in spectrum
    :param x: list position x for adding QR
    :param y: list position y for adding QR
    :param image: image in form np.ndarray
    :return: None
    """
    print("Research about changing many couples of point")
    plt.figure(figsize=(15, 8))
    N = image.shape[0]

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title("Original image")

    plt.subplot(1, 3, 2)
    spectrum = np.fft.fft2(image)

    for i in range(x.shape[0]):
        spectrum[x[i], y[i]] = 1
        spectrum[(N-x[i]) % N, (N-y[i]) % N] = 1

    spectrum_compression = compression_spectrum(spectrum)
    plt.imshow(spectrum_compression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum after changing couple of points")

    plt.subplot(1, 3, 3)
    image_after_research = np.fft.ifft2(spectrum)
    print(f"The wrong pixels:")
    check_error_data(image_after_research)
    image_with_watermark = np.real(image_after_research)
    plt.imshow(image_with_watermark, cmap="gray")
    plt.axis('off')
    plt.title("Image after changing couple of points")

    plt.show()


def phase_research(image: np.ndarray, x: np.ndarray, y: np.ndarray, phase: float = np.pi):
    """
    Showing analyse what was happened when add phase e^(i*phi)
    :param image: image in form np.ndarray
    :param x: list position x for adding QR
    :param y: list position y for adding QR
    :param phase: phase using in adding (e^i*phi)
    :return: None
    """
    print("Research about changing phase many couple of point")
    plt.figure(figsize=(15, 8))
    N = image.shape[0]

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title("Original image")

    plt.subplot(1, 3, 2)
    spectrum = np.fft.fft2(image)

    e_pos = np.exp(1j * phase)
    e_neg = np.exp(-1j * phase)

    for i in range(x.shape[0]):
        spectrum[x[i], y[i]] += e_pos
        spectrum[(N - x[i]) % N, (N - y[i]) % N] = e_neg

    spectrum_compression = compression_spectrum(spectrum)
    plt.imshow(spectrum_compression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum after changing phase")

    plt.subplot(1, 3, 3)
    image_after_research = np.fft.ifft2(spectrum)
    print(f"The wrong pixels:")
    check_error_data(image_after_research)
    image_with_watermark = np.real(image_after_research)
    plt.imshow(image_with_watermark, cmap="gray")
    plt.axis('off')
    plt.title("Image after changing phase")

    plt.show()


def high_frequency():
    """
    For watching frequency
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


def research_qr(image: np.ndarray, L: int, x: int, y: int):
    """
    Show analyse when adding QR code in spectrum
    :param image: image in form np.ndarray
    :param L: Shape of QR  L x L
    :return: None
    """
    print("Research about marking QR")
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
    image_after_research = np.fft.ifft2(spectrum_embed_qr)
    print(f"The wrong pixels:")
    check_error_data(image_after_research)
    image_with_watermark = np.real(image_after_research)
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
    plt.imshow(qr_extracted, cmap="gray")
    plt.axis("off")
    plt.title("QR was be extracted")

    print(f"Error extraction  = {bit_error(qr, qr_extracted)}")
    plt.show()


