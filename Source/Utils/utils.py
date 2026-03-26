import numpy as np
import cv2

epsilon = 1e-6

def make_balanced_qr(L: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = L * L
    k = n // 2
    arr = np.zeros(n, dtype=np.uint8)
    idx = rng.choice(n, size=k, replace=False)
    arr[idx] = 1
    return arr.reshape(L, L)


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


def add_pattern_all_blocks(img: np.ndarray,
                           pattern: np.ndarray,
                           alpha: float,
                           bs: int = 64,
                           clip: bool = True) -> np.ndarray:
    """
    Cộng cùng 1 pattern (bs×bs) vào TẤT CẢ block bs×bs của ảnh.
    img: (H,W) grayscale
    pattern: (bs,bs) watermark pattern (ví dụ block_fft sau IFFT)
    alpha: hệ số nhúng
    bs: block size (64)
    clip: nếu ảnh là 0..255 thì clip lại
    """
    if img.ndim != 2:
        raise ValueError("img must be 2D grayscale")
    H, W = img.shape
    if H % bs != 0 or W % bs != 0:
        raise ValueError(f"Image size must be multiple of bs={bs}")
    if pattern.shape != (bs, bs):
        raise ValueError(f"pattern must have shape {(bs, bs)}")

    # float để cộng không tràn số
    out = img.astype(np.float32, copy=True)

    # (Hb, Wb, bs, bs)
    Hb, Wb = H // bs, W // bs
    blocks = out.reshape(Hb, bs, Wb, bs).transpose(0, 2, 1, 3)

    # chuẩn hoá pattern để alpha có ý nghĩa (khuyến nghị)
    p = pattern.astype(np.float32)
    p = p / (np.max(np.abs(p)) + 1e-9)

    # broadcast add to all blocks
    blocks += alpha * p[None, None, :, :]

    # ghép lại ảnh
    out = blocks.transpose(0, 2, 1, 3).reshape(H, W)

    if clip:
        # nếu ảnh gốc là 8-bit
        out = np.clip(out, 0, 255).astype(np.uint8)

    return out

def embed_qr_into_black_image(qr: np.ndarray, img: np.ndarray, N: int, x: int, y: int, phase: float = np.pi / 5):
    """
    Add QR into spectrum by position (x, y), but QR will be divided to 2 part (left-right)
    :param qr: QR code in from np.ndarray
    :param spectrum: spectrum in from np.ndarray
    :param phase: phi in e^i*phi
    :param x: position X in spectrum for adding
    :param y: position Y in spectrum for adding
    :param phase: phase using in adding (e^i*phi)
    :param N: Size FFT
    :return: spectrum of image with qr
    """
    black_image = np.zeros((N, N))

    spectrum_in_block_fft = np.fft.fft2(black_image)
    L = qr.shape[0]

    L_mid = (L + 1) // 2

    if x > N//2 - L or y > N//2 - L:
        assert ValueError("fix x, y please!")

    e_pos = np.exp(1j * phase)
    e_neg = np.exp(-1j * phase)

    left = qr[:, :L_mid]
    spectrum_in_block_fft[x:x + L, y:y + L_mid] += left * e_pos
    spectrum_in_block_fft[N - x - L + 1:N - x + 1, N - y - L_mid + 1:N - y + 1] += left[::-1, ::-1] * e_neg

    right = qr[:, L_mid:]
    spectrum_in_block_fft[x:x + L, N - L + L_mid:] += right * e_pos
    spectrum_in_block_fft[N - x - L + 1:N - x + 1, 1:1 + L - L_mid] += right[::-1, ::-1] * e_neg

    block_fft = np.real(np.fft.ifft2(spectrum_in_block_fft))

    return add_pattern_all_blocks(img, block_fft, 0.1, N)


def extract_qr(x: int, y: int, spectrum_qr: np.ndarray, L: int):
    L_mid = (L + 1) // 2
    qr = np.zeros((L, L), dtype=np.complex128)

    qr[:, :L_mid] = spectrum_qr[x:x+L, y:y+L_mid]
    qr[:, - L + L_mid:] = spectrum_qr[x:x+L,   - L + L_mid:]

    return qr

def extract_qr_from_watermarked_image(spectrum_watermarked_image: np.ndarray, L_max: int = 41, x: int = 1, y: int = 1):
    N = spectrum_watermarked_image.shape[0]
    if L_max > N//4:
        assert ValueError("def extracted qr from watermarked image: choose another L_max")

    real = np.real(spectrum_watermarked_image)
    imag = np.imag(spectrum_watermarked_image)

    real_bg = cv2.GaussianBlur(real, (0, 0), 2)
    imag_bg = cv2.GaussianBlur(imag, (0, 0), 2)

    spectrum_qr_in_image_estimated = (real - real_bg) + 1j * (imag - imag_bg)

    qr_extracted = []
    best_check = 0.5


    for i in range(3,L_max):
        spectrum_qr_estimated = extract_qr(x, y, spectrum_qr_in_image_estimated, i)
        m = np.log1p(np.abs(spectrum_qr_estimated)).astype(np.float32)

        m = m - m.min()
        m = m / (m.max() + epsilon)
        m8 = (m*255).astype(np.uint8)

        t, bw = cv2.threshold(m8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        qr_bits = (bw > 0).astype(np.uint8)

        # ---- TIÊU CHÍ MỚI: between-class variance (Otsu separability) ----
        # t đang ở thang 0..255 của m8, nên tách lớp theo m8 cho đồng nhất
        c0 = m8[m8 <= t]
        c1 = m8[m8 > t]

        if c0.size == 0 or c1.size == 0:
            continue

        w0 = c0.size / m8.size
        w1 = c1.size / m8.size
        mu0 = float(c0.mean())
        mu1 = float(c1.mean())

        # độ tách lớp: càng lớn càng tốt
        check = w0 * w1 * (mu0 - mu1) ** 2

        # (tuỳ chọn) phạt L quá lớn một chút để tránh “ăn gian” theo diện tích
        check = check / (i + 1e-9)

        # thử đảo màu nếu cần (không ảnh hưởng score nhiều, chủ yếu để output đẹp)
        inv = 1 - qr_bits

        # chọn bản nào có ít nhiễu hơn theo “độ mượt” đơn giản: số lần đổi bit theo hàng/cột
        def roughness(b):
            return np.mean(np.abs(np.diff(b, axis=0))) + np.mean(np.abs(np.diff(b, axis=1)))

        if roughness(inv) < roughness(qr_bits):
            qr_bits = inv

        if check > best_check:
            best_check = check
            print(i)
            print(qr_bits)
            qr_extracted = qr_bits.copy()

    return qr_extracted



def bit_error(qr: np.ndarray, qr_extracted: np.ndarray):
    L = qr.shape[0]
    return (qr ^ qr_extracted).sum() / (L**2)
