import numpy as np
import matplotlib.pyplot as plt
from .my_function import my_ifft2, my_fft2
from .utils import compression_spectrum, generate_watermark,embed_watermark_into_image

def couple_of_points(image: np.ndarray, x: int, y: int, save_path: str):
    """
    Showing analyse what was happened when changed 1 couple of points in spectrum
    :param image: image in form np.ndarray
    :param x: position x for adding QR
    :param y: position y for adding QR
    :param save_path: path for saving
    :return: None
    """
    print("Research about changing 1 couple of point")

    plt.figure(figsize=(15, 8))
    N = image.shape[0]

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image, cmap="gray")
    ax1.axis('off')
    ax1.set_title("Original image")

    ax2 = plt.subplot(1, 3, 2)
    spectrum = my_fft2(image)

    spectrum[x][y] = 1
    spectrum[(N-x) % N][(N -y) % N] = 1

    spectrum_compression = compression_spectrum(spectrum)
    ax2.imshow(spectrum_compression, cmap="gray")
    ax2.axis('off')
    ax2.set_title("Spectrum after changing couple of points")

    ax3 = plt.subplot(1, 3, 3)

    image_after_research = my_ifft2(spectrum)
    image_with_watermark = np.real(image_after_research)
    ax3.imshow(image_with_watermark, cmap="gray")
    ax3.axis('off')
    ax3.set_title("Image after changing couple of points")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def many_couple_of_points(image: np.ndarray, x: np.ndarray, y: np.ndarray, save_path: str):
    """
    Showing analyse what has happened when changed a little couple of points in spectrum
    :param x: list position x for adding QR
    :param y: list position y for adding QR
    :param image: image in form np.ndarray
    :param save_path: path for saving
    :return: None
    """
    print("Research about changing many couples of point")
    plt.figure(figsize=(15, 8))
    N = image.shape[0]

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image, cmap="gray")
    ax1.axis('off')
    ax1.set_title("Original image")

    ax2 = plt.subplot(1, 3, 2)
    spectrum = my_fft2(image)

    for i in range(x.shape[0]):
        spectrum[x[i], y[i]] = 1
        spectrum[(N-x[i]) % N, (N-y[i]) % N] = 1

    spectrum_compression = compression_spectrum(spectrum)
    ax2.imshow(spectrum_compression, cmap="gray")
    ax2.axis('off')
    ax2.set_title("Spectrum after changing couples of points")

    ax3 = plt.subplot(1, 3, 3)
    image_after_research = my_ifft2(spectrum)
    image_with_watermark = np.real(image_after_research)
    ax3.imshow(image_with_watermark, cmap="gray")
    ax3.axis('off')
    ax3.set_title("Image after changing couples of points")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def phase_research(image: np.ndarray, x: np.ndarray, y: np.ndarray,save_path: str, phase: float = np.pi):
    """
    Showing analyze what was happened when add phase e^(i*phi)
    :param image: image in form np.ndarray
    :param x: list position x for adding QR
    :param y: list position y for adding QR
    :param phase: phase using in adding (e^i*phi)
    :param save_path: path for saving
    :return: None
    """
    print("Research about changing phase many couple of point")
    plt.figure(figsize=(15, 8))
    N = image.shape[0]

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image, cmap="gray")
    ax1.axis('off')
    ax1.set_title("Original image")

    ax2 = plt.subplot(1, 3, 2)
    spectrum = my_fft2(image)

    e_pos = np.exp(1j * phase)
    e_neg = np.exp(-1j * phase)

    for i in range(x.shape[0]):
        spectrum[x[i], y[i]] = e_pos
        spectrum[(N - x[i]) % N, (N - y[i]) % N] = e_neg

    spectrum_compression = compression_spectrum(spectrum)
    ax2.imshow(spectrum_compression, cmap="gray")
    ax2.axis('off')
    ax2.set_title("Spectrum after changing phase")

    ax3 = plt.subplot(1, 3, 3)
    image_after_research = my_ifft2(spectrum)
    image_with_watermark = np.real(image_after_research)
    ax3.imshow(image_with_watermark, cmap="gray")
    ax3.axis('off')
    ax3.set_title("Image after changing phase")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def show_frequency(image_path: str, save_path: str):
    """
    :param image_path: Path to image
    :param save_path: path for saving
    For watching frequency
    :return: None
    """
    image = plt.imread(image_path)

    plt.figure(figsize=(15, 8))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image, cmap="gray")
    ax1.axis('off')
    ax1.set_title("Original image")

    ax2 = plt.subplot(1, 2, 2)
    spectrum = my_fft2(image)
    spectrum_compression = compression_spectrum(spectrum)
    ax2.imshow(spectrum_compression, cmap="gray")
    ax2.axis('off')
    ax2.set_title("Spectrum")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def research_qr(image: np.ndarray, size_qr: int, x: int, y: int, save_path: str, S: int=60, phase: float=np.pi/2):
    """
    Show analyze when adding QR code in spectrum
    :param image: image in form np.ndarray
    :param size_qr: Shape of QR  L x L
    :param x: list position x for adding QR
    :param y: list position y for adding QR
    :param save_path: path for saving
    :param S: signal/ noise
    :param phase: phase using in adding (e^i*phi)
    :return: None
    """
    qr = generate_watermark(size_qr, seed=42)

    plt.figure(figsize=(16,8))
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    spectrum_original_image = my_fft2(image)
    spectrum_compression = compression_spectrum(spectrum_original_image)
    ax2.imshow(spectrum_compression, cmap="gray")
    ax2.set_title("Spectrum original image")
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(qr, cmap="gray")
    ax3.set_title("QR code")
    ax3.axis("off")

    ax4 = plt.subplot(2, 3, 4)
    image_after_embedding = embed_watermark_into_image(image, qr, phase, 1)
    ax4.imshow(image_after_embedding, cmap="gray")
    ax4.set_title("Image after adding QR code")
    ax4.axis("off")


    ax5 = plt.subplot(2, 3, 5)
    spectrum_image_after_embedding = my_fft2(image_after_embedding)
    spectrum_compression = compression_spectrum(spectrum_image_after_embedding)
    ax5.imshow(spectrum_compression, cmap="gray")
    ax5.set_title("Spectrum after adding QR code")
    ax5.axis("off")


    # plt.subplot(2, 3, 6)
    # spectrum_with_watermark = np.fft.fft2(image_with_watermark)
    # qr_extracted = extract_qr_from_watermarked_image(spectrum_with_watermark)
    #
    # plt.imshow(qr_extracted, cmap="gray")
    # plt.set_title("Extract QR")
    # plt.axis("off")


    # print(f"Error extraction  = {bit_error(qr, qr_extracted)}")
    plt.show()



