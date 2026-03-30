import numpy as np
from .my_function import my_fft2, my_ifft2

epsilon = 1e-6


def compression_spectrum(spectrum: np.ndarray):
    """
    Optimize value of spectrum for showing
    :param spectrum: spectrum in from np.ndarray
    :return: data of spectrum after optimizing
    """
    return np.log(1 + np.abs(spectrum))


def generate_watermark(size: int, seed: int = 42):
    """
    Generate watermark image
    :param size: Size of watermark
    :param seed: Seed
    :return: Watermark
    """
    rng = np.random.default_rng(seed)
    n = size * size
    k = n // 2
    arr = np.zeros(n, dtype=np.uint8)
    idx = rng.choice(n, size=k, replace=False)
    arr[idx] = 1
    return arr.reshape(size, size)


def embed_wm_to_black_region(watermark: np.ndarray, size_region: int, x: int, y: int, offset: int = 0, phase: float = np.pi / 2, q: float = 1):
    """
    Embed watermark into black region
    :param watermark: watermark image
    :param size_region: size of region
    :param x: position of x for embed watermark
    :param y: position of y for embed watermark
    :param offset: parameter for offset
    :param phase: parameter for phase when embed watermark
    :param q: strength of phase when embed watermark
    :return: spectrum after embed watermark
    """
    l = watermark.shape[0]
    l_mid = (l + 1) // 2

    spectrum_qr = np.zeros((size_region, size_region), dtype=np.complex128)

    e_pos = np.exp(1j * phase)
    e_neg = np.exp(-1j * phase)

    left = watermark[:, :l_mid]
    spectrum_qr[x:x + l, y:y + l_mid] = q * left * e_pos
    spectrum_qr[size_region - x - l + 1: size_region - x + 1,
                size_region - y - l_mid + 1: size_region - y + 1] = q * left[::-1, ::-1] * e_neg

    right = watermark[:, l_mid:]
    spectrum_qr[x:x + l, size_region - l + l_mid - offset: size_region - offset] = q * right * e_pos
    spectrum_qr[size_region - x - l + 1: size_region - x + 1, 1 + offset : 1 + l - l_mid + offset] = q * right[::-1, ::-1] * e_neg

    return spectrum_qr


def split_into_blocks(img: np.ndarray, block_size: int = 64):
    """"""
    height, weight = img.shape
    if height % block_size != 0 or weight % block_size != 0:
        raise ValueError("Sth wrong in split_into_blocks")

    block_height = height // block_size
    block_weight = weight // block_size

    blocks = img.reshape(block_height, block_size, block_weight, block_size).transpose(0, 2, 1, 3)
    return blocks


def merge_blocks(blocks: np.ndarray) -> np.ndarray:
    """"""
    block_height, block_weight, number_block_h, number_block_w = blocks.shape
    img = blocks.transpose(0, 2, 1, 3).reshape(block_height * number_block_h, block_weight * number_block_w)
    return img


def embed_watermark_into_image(image: np.ndarray, qr: np.ndarray, phase: float, q: float):
    watermark = embed_wm_to_black_region(qr, 64, 4, 4, 8, phase, q)

    blocks = split_into_blocks(image, block_size=64)
    block_height, block_weight = blocks.shape[:2]

    watermarked_blocks = np.zeros_like(blocks, dtype=np.float64)

    for i in range(block_height):
        for j in range(block_weight):
            block = blocks[i, j]

            x = my_fft2(block)

            y = x + q * watermark

            y = my_ifft2(y)
            y = np.real(y)

            watermarked_blocks[i, j] = y

    watermarked_image = merge_blocks(watermarked_blocks)
    watermarked_image = np.clip(watermarked_image, 0, 255)

    return watermarked_image.astype(np.uint8)