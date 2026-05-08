import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from .my_function import my_fft2, my_ifft2

epsilon = 1e-6


def count_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    if image1.shape != image2.shape:
        raise ValueError("Two images must have the same shape")
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    mse = np.mean((image1 - image2) ** 2)
    if mse < epsilon:
        return np.inf
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compression_spectrum(spectrum: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(spectrum))


def generate_watermark(size: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = size * size
    k = n // 2
    arr = np.zeros(n, dtype=np.uint8)
    idx = rng.choice(n, size=k, replace=False)
    arr[idx] = 1
    return arr.reshape(size, size)


def bit_accuracy(qr_true: np.ndarray, qr_pred: np.ndarray) -> float:
    qr_true = np.asarray(qr_true, dtype=np.uint8)
    qr_pred = np.asarray(qr_pred, dtype=np.uint8)
    if qr_true.shape != qr_pred.shape:
        raise ValueError("qr_true and qr_pred must have the same shape")
    return float(np.mean(qr_true == qr_pred))


def design_params(block_size: int) -> Tuple[int, int, int]:
    x = block_size // 8
    y = block_size // 8
    offset = block_size // 16
    return x, y, offset


def _conj_index(u: int, v: int, n: int) -> Tuple[int, int]:
    return (-u) % n, (-v) % n


def max_qr_size_for_block(block_size: int, pitch: int = 1) -> int:
    x, y, offset = design_params(block_size)

    best = 0
    for L in range(1, 10000):
        l_mid = (L + 1) // 2
        right_w = L - l_mid

        if x + (L - 1) * pitch >= block_size:
            break

        left_end = y + (l_mid - 1) * pitch
        if left_end >= block_size:
            break

        if right_w == 0:
            best = L
            continue

        base_v = block_size - 1 - offset - (right_w - 1) * pitch
        if base_v < 0:
            break

        if left_end >= base_v:
            break

        best = L

    return best


def recommended_qr_range(block_size: int, pitch: int = 1) -> Tuple[int, int]:
    """"""
    geom_max = max_qr_size_for_block(block_size, pitch)
    low = max(5, geom_max // 6)
    high = max(low, geom_max // 3)
    return low, high


def qr_to_spectrum_positions(qr_size: int, size_region: int, pitch: int, x: int, y: int , offset: int) -> List[Dict[str, int]]:
    """"""
    positions: List[Dict[str, int]] = []
    l_mid = (qr_size + 1) // 2
    right_w = qr_size - l_mid

    if x + (qr_size - 1) * pitch >= size_region:
        raise ValueError("QR exceeds row bound")

    left_end = y + (l_mid - 1) * pitch

    if left_end >= size_region:
        raise ValueError("Left half exceeds col bound")

    base_v = size_region - 1 - offset - (right_w - 1) * pitch

    if base_v < 0:
        raise ValueError("Right half start negative")

    if right_w > 0 and left_end >= base_v:
        raise ValueError("Left and right halves overlap")

    for i in range(qr_size):
        for j in range(qr_size):
            u = x + i * pitch
            if j < l_mid:
                v = y + j * pitch
            else:
                v = base_v + (j - l_mid) * pitch

            if not (0 <= u < size_region and 0 <= v < size_region):
                raise ValueError("QR mapping index out of bounds")

            positions.append({
                "qr_i": i,
                "qr_j": j,
                "row": u,
                "col": v
            })
    return positions


def build_wm_spectrum(qr: np.ndarray,  size_region: int, phi: float, pitch: int = 1, x: int = None, y: int = None, offset: int = None) -> np.ndarray:

    qr = np.asarray(qr, dtype=np.uint8)
    spectrum_qr = np.zeros((size_region, size_region), dtype=np.complex128)
    positions = qr_to_spectrum_positions(qr.shape[0], size_region, pitch, x, y, offset)

    for pos in positions:
        i = pos["qr_i"]
        j = pos["qr_j"]
        u = pos["row"]
        v = pos["col"]

        if qr[i, j] != 1:
            continue

        spectrum_qr[u, v] += np.exp(1j * phi)
        uc, vc = _conj_index(u, v, size_region)
        spectrum_qr[uc, vc] += np.exp(-1j * phi)

    return spectrum_qr



def split_into_blocks(img: np.ndarray, block_size: int) -> np.ndarray:
    height, width = img.shape
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError("Image size must be divisible by block_size")
    block_height = height // block_size
    block_width = width // block_size
    return img.reshape(block_height, block_size, block_width, block_size).transpose(0, 2, 1, 3)


def merge_blocks(blocks: np.ndarray) -> np.ndarray:
    block_height, block_width, h, w = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(block_height * h, block_width * w)


def embed_watermark_into_image(image: np.ndarray, qr: np.ndarray, size_region: int, q: float, phi: float, pitch: int = 1, x: int = None, y: int = None, offset: int = None) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")

    spectrum_qr = build_wm_spectrum(qr, size_region, phi, pitch, x, y, offset)
    blocks = split_into_blocks(image, block_size=size_region)
    watermarked_blocks = np.zeros_like(blocks, dtype=np.float64)

    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            block = blocks[i, j].astype(np.float64)
            wm_spectrum = my_fft2(block) + q * spectrum_qr
            watermarked_blocks[i, j] = np.real(my_ifft2(wm_spectrum))

    watermarked_image = merge_blocks(watermarked_blocks)
    watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

    return watermarked_image


def iter_offset_blocks(image: np.ndarray, block_size: int, start_x: int, start_y: int):
    h, w = image.shape
    if not (0 <= start_x < block_size and 0 <= start_y < block_size):
        raise ValueError("row0 and col0 must satisfy 0 <= offset < block_size")

    for r in range(start_x, h - block_size + 1, block_size):
        for c in range(start_y, w - block_size + 1, block_size):
            yield image[r:r + block_size, c:c + block_size], r, c


def average_offset_spectrum(image: np.ndarray, block_size: int, start_x: int, start_y: int, phase_sign: int = 1) -> Tuple[np.ndarray, int]:
    acc = np.zeros((block_size, block_size), dtype=np.complex128)
    count = 0

    uu, vv = np.meshgrid(np.arange(block_size), np.arange(block_size), indexing="ij")
    base_phase = np.exp(phase_sign * 1j * 2.0 * np.pi * ((uu * start_x + vv * start_y) / block_size))

    for block, _, _ in iter_offset_blocks(image, block_size, start_x, start_y):
        f = my_fft2(block.astype(np.float64))
        acc += f * base_phase
        count += 1

    if count == 0:
        raise ValueError("No blocks were extracted")

    return acc / count, count


def _build_neighbor_exclusion_set(qr_size: int, size_region: int, pitch: int = 1, x: int = None, y: int = None, offset: int = None) -> Set[Tuple[int, int]]:
    """"""
    protected: Set[Tuple[int, int]] = set()
    positions = qr_to_spectrum_positions(qr_size, size_region, pitch, x, y, offset)
    for pos in positions:
        u = pos["row"]
        v = pos["col"]
        protected.add((u, v))
        protected.add(_conj_index(u, v, size_region))
    return protected


def _ring_background( mag: np.ndarray, u: int, v: int, radius: int, exclude_coords: Optional[Set[Tuple[int, int]]] = None) -> float:
    u1 = max(0, u - radius)
    u2 = min(mag.shape[0], u + radius + 1)
    v1 = max(0, v - radius)
    v2 = min(mag.shape[1], v + radius + 1)

    vals = []
    for uu in range(u1, u2):
        for vv in range(v1, v2):
            if uu == u and vv == v:
                continue
            if exclude_coords is not None and (uu, vv) in exclude_coords:
                continue
            val = float(mag[uu, vv])
            if val > epsilon:
                vals.append(val)

    if len(vals) == 0:
        return 0.0
    return float(np.median(vals))


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-6
    return (vals - med) / (1.4826 * mad + 1e-6)


def _detrend_score_map(score_map: np.ndarray) -> np.ndarray:
    s = np.asarray(score_map, dtype=np.float64).copy()
    s -= np.mean(s, axis=1, keepdims=True)
    s -= np.mean(s, axis=0, keepdims=True)
    return s


def _recover_topk(score_map: np.ndarray, ones_ratio: float = 0.5) -> Tuple[np.ndarray, float]:
    flat = np.asarray(score_map, dtype=np.float64).ravel()
    n = flat.size
    k = max(0, min(int(round(n * ones_ratio)), n))

    recovered = np.zeros(n, dtype=np.uint8)
    if k > 0:
        idx = np.argsort(flat)[-k:]
        recovered[idx] = 1
        tau = float(np.min(flat[idx]))
    else:
        tau = float(np.max(flat) + 1.0)

    return recovered.reshape(score_map.shape), tau


def default_ring_radius(pitch: int) -> int:
    return max(2, 2 * pitch)


def blind_score_map(avg_spectrum: np.ndarray, qr_size: int, size_region: int, phi: float, pitch: int, x: int , y: int, offset: int, ring_radius: Optional[int] = None) -> np.ndarray:
    """"""
    if ring_radius is None:
        ring_radius = default_ring_radius(pitch)

    raw_score = np.zeros((qr_size, qr_size), dtype=np.float64)
    positions = qr_to_spectrum_positions(qr_size, size_region, pitch, x, y, offset)
    mag = np.abs(avg_spectrum)
    exclude_coords = _build_neighbor_exclusion_set(qr_size, size_region, pitch, x, y, offset)

    for pos in positions:
        qi = pos["qr_i"]
        qj = pos["qr_j"]
        u = pos["row"]
        v = pos["col"]
        uc, vc = _conj_index(u, v, size_region)

        signal = np.real(avg_spectrum[u, v] * np.exp(-1j * phi))
        signal += np.real(avg_spectrum[uc, vc] * np.exp(1j * phi))

        bg1 = _ring_background(mag, u, v, ring_radius, exclude_coords)
        bg2 = _ring_background(mag, uc, vc, ring_radius, exclude_coords)
        raw_score[qi, qj] = signal / (bg1 + bg2 + 1e-6)

    return _robust_normalize(_detrend_score_map(raw_score))


def extract_watermark(image: np.ndarray, qr_size: int, size_region: int, phi: float, ones_ratio: float, start_x: int = 0,  start_y: int = 0,
                      pitch: int = 1, x: int = None, y: int = None, offset: int = None, ring_radius: Optional[int] = None, phase_sign: int = 1):
    """ """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")

    avg_spectrum, blocks_used = average_offset_spectrum(image, size_region, start_x, start_y, phase_sign)

    score_map = blind_score_map(avg_spectrum, qr_size, size_region, phi, pitch, x, y, offset, ring_radius)

    recovered, tau = _recover_topk(score_map, ones_ratio)
    return recovered, score_map, avg_spectrum, blocks_used, tau


def extract_watermark_search_offsets(image: np.ndarray, qr_size: int, size_region: int, phi: float, ones_ratio: float,
        offset_candidates: List[Tuple[int, int]], pitch: int = 1, x: int = None, y: int = None, offset: int = None,
        ring_radius: Optional[int] = None, phase_sign_candidates: Tuple[int, ...] = (1, -1)):

    best = None
    diagnostics = []

    for phase_sign in phase_sign_candidates:
        for start_x, start_y in offset_candidates:
            recovered, score_map, avg_spectrum, blocks_used, tau = extract_watermark(
                image=image,
                qr_size=qr_size,
                size_region=size_region,
                phi=phi,
                ones_ratio=ones_ratio,
                start_x=start_x,
                start_y=start_y,
                pitch=pitch,
                x=x,
                y=y,
                offset=offset,
                ring_radius=ring_radius,
                phase_sign=phase_sign
            )

            flat = score_map.ravel()
            idx1 = recovered.ravel() == 1
            idx0 = recovered.ravel() == 0

            if np.any(idx1) and np.any(idx0):
                metric = float((np.mean(flat[idx1]) - np.mean(flat[idx0])) / (np.std(flat) + 1e-6))
            else:
                metric = -np.inf

            item = {
                "start_x": start_x,
                "start_y": start_y,
                "phase_sign": phase_sign,
                "recovered_qr": recovered,
                "score_map": score_map,
                "avg_spectrum": avg_spectrum,
                "blocks_used": blocks_used,
                "threshold": tau,
                "metric": metric,
            }
            diagnostics.append({
                "start_x": start_x,
                "start_y": start_y,
                "phase_sign": phase_sign,
                "metric": metric,
                "blocks_used": blocks_used,
            })

            if best is None or item["metric"] > best["metric"]:
                best = item

    best["diagnostics"] = diagnostics
    return best

def average_offset_spectrum_limited(
    image: np.ndarray,
    block_size: int,
    start_x: int,
    start_y: int,
    max_blocks: int | None = None,
    random_blocks: bool = False,
    seed: int | None = None,
    phase_sign: int = 1,
) -> Tuple[np.ndarray, int]:
    blocks = list(iter_offset_blocks(image, block_size, start_x, start_y))

    if len(blocks) == 0:
        raise ValueError("No blocks were extracted")

    if max_blocks is not None:
        max_blocks = min(max_blocks, len(blocks))
        if random_blocks:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(blocks), size=max_blocks, replace=False)
            blocks = [blocks[i] for i in idx]
        else:
            blocks = blocks[:max_blocks]

    acc = np.zeros((block_size, block_size), dtype=np.complex128)

    uu, vv = np.meshgrid(
        np.arange(block_size),
        np.arange(block_size),
        indexing="ij"
    )
    base_phase = np.exp(
        phase_sign * 1j * 2.0 * np.pi * ((uu * start_x + vv * start_y) / block_size)
    )

    for block, _, _ in blocks:
        acc += my_fft2(block.astype(np.float64)) * base_phase

    return acc / len(blocks), len(blocks)


def extract_watermark_limited_blocks(
    image: np.ndarray,
    qr_size: int,
    size_region: int,
    phi: float,
    ones_ratio: float,
    start_x: int,
    start_y: int,
    max_blocks: int,
    random_blocks: bool = False,
    seed: int | None = None,
    pitch: int = 1,
    x: int = None,
    y: int = None,
    offset: int = None,
    ring_radius: Optional[int] = None,
    phase_sign: int = 1,
):
    avg_spectrum, blocks_used = average_offset_spectrum_limited(
        image=image,
        block_size=size_region,
        start_x=start_x,
        start_y=start_y,
        max_blocks=max_blocks,
        random_blocks=random_blocks,
        seed=seed,
        phase_sign=phase_sign,
    )

    score_map = blind_score_map(
        avg_spectrum, qr_size, size_region, phi,
        pitch, x, y, offset, ring_radius
    )

    recovered, tau = _recover_topk(score_map, ones_ratio)

    return recovered, score_map, avg_spectrum, blocks_used, tau


def random_nonzero_start(block_size: int, seed: int | None = None) -> Tuple[int, int]:
    rng = np.random.default_rng(seed)

    while True:
        start_x = int(rng.integers(0, block_size / 4))
        start_y = int(rng.integers(0, block_size / 4))
        if (start_x, start_y) != (0, 0):
            return start_x, start_y