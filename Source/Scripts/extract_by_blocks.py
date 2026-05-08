import os
import numpy as np
import matplotlib.pyplot as plt

from Source.Utils.utils import (
    generate_watermark,
    embed_watermark_into_image,
    extract_watermark_limited_blocks,
    random_nonzero_start,
    bit_accuracy,
)

SIZE_QR = 6
SIZE_REGION = 64
PHASE = np.pi / 3
PITCH = 2
S = 140
X = 8
Y = 8
OFFSET = 5
SEED = 42

IMAGE_PATH = os.path.join("Image", "lena.tif")

def calculate_q(snr: float, l: int, n: int) -> float:
    return (255 * n * n) / (snr * l)

if __name__ == "__main__":
    image = plt.imread(IMAGE_PATH)

    if image.ndim == 3:
        image = image[:, :, 0]

    qr_true = generate_watermark(SIZE_QR, seed=SEED)
    q = calculate_q(S, SIZE_QR, SIZE_REGION)

    watermarked = embed_watermark_into_image(
        image=image,
        qr=qr_true,
        size_region=SIZE_REGION,
        q=q,
        phi=PHASE,
        pitch=PITCH,
        x=X,
        y=Y,
        offset=OFFSET,
    )

    start_x, start_y = random_nonzero_start(SIZE_REGION, seed=38)
    print(f"Random start offset: ({start_x}, {start_y})")

    total_blocks = len(list(range(start_x, image.shape[0] - SIZE_REGION + 1, SIZE_REGION))) * \
                   len(list(range(start_y, image.shape[1] - SIZE_REGION + 1, SIZE_REGION)))

    block_counts = []
    k = 1
    while k <= total_blocks:
        block_counts.append(k)
        k *= 2

    if block_counts[-1] != total_blocks:
        block_counts.append(total_blocks)

    for n_blocks in block_counts:
        qr_pred, score_map, avg_spectrum, used, tau = extract_watermark_limited_blocks(
            image=watermarked,
            qr_size=SIZE_QR,
            size_region=SIZE_REGION,
            phi=PHASE,
            ones_ratio=0.5,
            start_x=start_x,
            start_y=start_y,
            max_blocks=n_blocks,
            random_blocks=True,
            seed=SEED,
            pitch=PITCH,
            x=X,
            y=Y,
            offset=OFFSET,
        )

        acc = bit_accuracy(qr_true, qr_pred)
        print(f"blocks={used:>3} | bit_accuracy={acc:.4f} | threshold={tau:.4f}")