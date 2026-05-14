from __future__ import annotations
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from .my_function import my_fft2, my_ifft2
EPSILON = 1e-6


def count_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute PSNR between two equally shaped images."""
    if image1.shape != image2.shape:
        raise ValueError("Two images must have the same shape")

    img1 = np.asarray(image1, dtype=np.float64)
    img2 = np.asarray(image2, dtype=np.float64)
    mse = np.mean((img1 - img2) ** 2)

    if mse < EPSILON:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compression_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Log-compress a Fourier spectrum for visualization."""
    return np.log1p(np.abs(spectrum))


def generate_watermark(size: int, seed: int = 42) -> np.ndarray:
    """Generate a reproducible binary QR-like watermark with 50% ones."""
    if size <= 0:
        raise ValueError("size must be positive")

    rng = np.random.default_rng(seed)
    n = size * size
    arr = np.zeros(n, dtype=np.uint8)
    arr[rng.choice(n, size=n // 2, replace=False)] = 1
    return arr.reshape(size, size)


def bit_accuracy(qr_true: np.ndarray, qr_pred: np.ndarray) -> float:
    """Return the bitwise accuracy between two binary QR matrices."""
    true = np.asarray(qr_true, dtype=np.uint8)
    pred = np.asarray(qr_pred, dtype=np.uint8)

    if true.shape != pred.shape:
        raise ValueError("qr_true and qr_pred must have the same shape")
    return float(np.mean(true == pred))


def calculate_q(s_param: float, qr_size: int, region_size: int) -> float:
    """Compute the embedding strength Q used in experiments."""
    if s_param == 0:
        raise ValueError("s_param must be non-zero")
    if qr_size <= 0 or region_size <= 0:
        raise ValueError("qr_size and region_size must be positive")
    return float((255 * region_size * region_size) / (s_param * qr_size))


def design_params(block_size: int) -> Tuple[int, int, int]:
    """Default spectral embedding parameters for one block size."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    x = block_size // 8
    y = block_size // 8
    offset = block_size // 16
    return x, y, offset


def resolve_embedding_params(size_region: int, x: Optional[int] = None, y: Optional[int] = None,
                             offset: Optional[int] = None) -> Tuple[int, int, int]:
    """Use explicit embedding parameters or fall back to design_params()."""
    dx, dy, doffset = design_params(size_region)
    return (
        dx if x is None else int(x),
        dy if y is None else int(y),
        doffset if offset is None else int(offset),
    )


def _conj_index(u: int, v: int, n: int) -> Tuple[int, int]:
    """Coordinate of the conjugate Fourier coefficient."""
    return (-u) % n, (-v) % n


def max_qr_size_for_block(block_size: int) -> int:
    """Return the largest QR size supported by the current spectral layout."""
    x, y, offset = design_params(block_size)
    best = 0

    for qr_size in range(1, block_size + 1):
        try:
            qr_to_spectrum_positions(qr_size, block_size, x, y, offset)
        except ValueError:
            break
        best = qr_size

    return best


def qr_to_spectrum_positions(qr_size: int, size_region: int, x: Optional[int] = None, y: Optional[int] = None,
                             offset: Optional[int] = None) -> List[Dict[str, int]]:
    """Map QR bit coordinates to Fourier-spectrum coordinates
    The QR is split into a low-frequency left part and a high-frequency right
    part. The conjugate coordinates are handled later by ``build_wm_spectrum``.
    """
    if qr_size <= 0:
        raise ValueError("qr_size must be positive")
    if size_region <= 0:
        raise ValueError("size_region must be positive")

    x, y, offset = resolve_embedding_params(size_region, x, y, offset)
    positions: List[Dict[str, int]] = []

    left_width = (qr_size + 1) // 2
    right_width = qr_size - left_width

    if x < 0 or y < 0 or offset < 0:
        raise ValueError("x, y, and offset must be non-negative")
    if x + qr_size > size_region:
        raise ValueError("QR exceeds row bound")

    left_end = y + left_width - 1
    if left_end >= size_region:
        raise ValueError("Left half exceeds column bound")

    right_base_v = size_region - offset - right_width
    if right_width > 0:
        if right_base_v < 0:
            raise ValueError("Right half start is negative")
        if left_end >= right_base_v:
            raise ValueError("Left and right QR halves overlap")

    for i in range(qr_size):
        for j in range(qr_size):
            u = x + i
            if j < left_width:
                v = y + j
            else:
                v = right_base_v + (j - left_width)

            if not (0 <= u < size_region and 0 <= v < size_region):
                raise ValueError("QR mapping index is out of bounds")
            positions.append({"qr_i": i, "qr_j": j, "row": u, "col": v})
    return positions


def split_into_blocks(img: np.ndarray, block_size: int) -> np.ndarray:
    """Split a 2D image into non-overlapping square blocks."""
    image = np.asarray(img)
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    height, width = image.shape
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError("Image size must be divisible by block_size")

    n_rows = height // block_size
    n_cols = width // block_size
    return image.reshape(n_rows, block_size, n_cols, block_size).transpose(0, 2, 1, 3)


def merge_blocks(blocks: np.ndarray) -> np.ndarray:
    """Merge blocks produced by split_into_blocks()."""
    arr = np.asarray(blocks)
    if arr.ndim != 4:
        raise ValueError("blocks must have shape (n_rows, n_cols, h, w)")

    n_rows, n_cols, h, w = arr.shape
    return arr.transpose(0, 2, 1, 3).reshape(n_rows * h, n_cols * w)


def iter_offset_blocks(image: np.ndarray, block_size: int, start_x: int,  start_y: int) -> Iterator[Tuple[np.ndarray, int, int]]:
    """Yield square blocks from an offset extraction grid."""
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("Input image must be grayscale")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not (0 <= start_x < block_size and 0 <= start_y < block_size):
        raise ValueError("start_x and start_y must satisfy 0 <= offset < block_size")

    height, width = img.shape
    for r in range(start_x, height - block_size + 1, block_size):
        for c in range(start_y, width - block_size + 1, block_size):
            yield img[r : r + block_size, c : c + block_size], r, c


def build_wm_spectrum(qr: np.ndarray, size_region: int, phi: float, x: Optional[int] = None, y: Optional[int] = None,
                      offset: Optional[int] = None) -> np.ndarray:
    """Build the watermark spectrum for one image block."""
    qr_bits = np.asarray(qr, dtype=np.uint8)
    if qr_bits.ndim != 2 or qr_bits.shape[0] != qr_bits.shape[1]:
        raise ValueError("qr must be a square 2D array")

    x, y, offset = resolve_embedding_params(size_region, x, y, offset)
    spectrum_qr = np.zeros((size_region, size_region), dtype=np.complex128)
    e_pos = np.exp(1j * phi)
    e_neg = np.exp(-1j * phi)

    positions = qr_to_spectrum_positions(qr_size=qr_bits.shape[0], size_region=size_region, x=x, y=y, offset=offset)

    for pos in positions:
        i = pos["qr_i"]
        j = pos["qr_j"]
        if qr_bits[i, j] != 1:
            continue

        u = pos["row"]
        v = pos["col"]
        spectrum_qr[u, v] += e_pos

        uc, vc = _conj_index(u, v, size_region)
        spectrum_qr[uc, vc] += e_neg

    return spectrum_qr


def embed_watermark_into_image(image: np.ndarray, qr: np.ndarray,size_region: int, q: float,  phi: float,
                               x: Optional[int] = None, y: Optional[int] = None, offset: Optional[int] = None) -> np.ndarray:
    """Embed a binary QR watermark into every non-overlapping image block."""
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("Input image must be grayscale")

    x, y, offset = resolve_embedding_params(size_region, x, y, offset)
    watermark_spectrum = build_wm_spectrum(qr=qr, size_region=size_region, phi=phi, x=x, y=y, offset=offset)

    blocks = split_into_blocks(img, block_size=size_region)
    watermarked_blocks = np.empty_like(blocks, dtype=np.float64)

    for row in range(blocks.shape[0]):
        for col in range(blocks.shape[1]):
            block = blocks[row, col].astype(np.float64)
            spectrum = my_fft2(block) + q * watermark_spectrum
            watermarked_blocks[row, col] = np.real(my_ifft2(spectrum))

    watermarked = merge_blocks(watermarked_blocks)
    return np.clip(watermarked, 0, 255).astype(np.uint8)
