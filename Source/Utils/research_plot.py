import numpy as np
import matplotlib.pyplot as plt

from . import split_into_blocks
from .my_function import my_ifft2, my_fft2
from .utils import compression_spectrum, generate_watermark,embed_watermark_into_image, merge_blocks, split_into_blocks

def couple_of_points(image: np.ndarray, x: int, y: int, save_path: str):
    """
    Showing analyze what was happened when changed 1 couple of points in spectrum
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
    Showing analyze what has happened when changed a little couple of points in spectrum
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

def spectrum_image_draw_and_compare(image: np.ndarray, size_region: int):

    blocks_image = split_into_blocks(image, size_region)
    blocks_spectrum = np.zeros_like(blocks_image, dtype=np.complex128)
    blocks_spectrum_compression = np.zeros_like(blocks_image, dtype=np.float64)
    rows, cols = blocks_spectrum.shape[:2]
    for row in range(rows):
        for col in range(cols):
            blocks_spectrum[row, col] = my_fft2(blocks_image[row, col])
            blocks_spectrum_compression[row, col] = compression_spectrum(blocks_spectrum[row, col])

    spectrum = merge_blocks(blocks_spectrum)
    spectrum_compression = merge_blocks(blocks_spectrum_compression)
    return spectrum, spectrum_compression

def diff_spectrum(spectrum1: np.ndarray, spectrum2: np.ndarray, size_region: int):
    blocks_spectrum1 = split_into_blocks(spectrum1, size_region)
    blocks_spectrum2 = split_into_blocks(spectrum2, size_region)
    diff_spectrum_compression = np.zeros_like(blocks_spectrum1, dtype=np.float64)
    rows, cols = blocks_spectrum1.shape[:2]

    for row in range(rows):
        for col in range(cols):
            diff_spectrum_compression[row, col] = compression_spectrum(blocks_spectrum1[row, col] - blocks_spectrum2[row, col])

    return merge_blocks(diff_spectrum_compression)


def research_qr(image: np.ndarray, size_qr: int, size_region: int, x: int, y: int, offset: int, q: float, phase: float, save_path: str):
    """
    Show analyze when adding QR code in spectrum
    :param image: image in form np.ndarray
    :param size_qr: Shape of QR  L x L
    :param size_region: shape black region for embedding qr
    :param x: list position x for adding QR
    :param y: list position y for adding QR
    :param offset: offset of position embedding QR
    :param save_path: path for saving
    :param q: strength
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
    spectrum_original_image, spectrum_original_compression_image = spectrum_image_draw_and_compare(image, size_region)
    ax2.imshow(spectrum_original_compression_image, cmap="gray")
    ax2.set_title("Spectrum original image")
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(qr, cmap="gray")
    ax3.set_title("QR code")
    ax3.axis("off")

    ax4 = plt.subplot(2, 3, 4)
    image_after_embedding = embed_watermark_into_image(image, qr, size_region, x, y, offset, phase, q)
    ax4.imshow(image_after_embedding, cmap="gray")
    ax4.set_title("Image after adding QR code")
    ax4.axis("off")


    ax5 = plt.subplot(2, 3, 5)
    spectrum_after_embedding, spectrum_after_embedding_compression_image = spectrum_image_draw_and_compare(image_after_embedding, size_region)
    ax5.imshow(spectrum_after_embedding_compression_image, cmap="gray")
    ax5.set_title("Spectrum after adding QR code")
    ax5.axis("off")


    ax6 = plt.subplot(2, 3, 6)
    diff_spectrum_compress = diff_spectrum(spectrum_after_embedding, spectrum_original_image, size_region)
    ax6.imshow(diff_spectrum_compress, cmap="gray")
    ax6.axis("off")
    ax6.set_title("Difference spectrum")

    # plt.imshow(qr_extracted, cmap="gray")
    # plt.set_title("Extract QR")
    # plt.axis("off")


    # print(f"Error extraction  = {bit_error(qr, qr_extracted)}")
    # plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



