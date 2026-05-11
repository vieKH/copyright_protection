import math
import os

import matplotlib.pyplot as plt
import numpy as np

from Source.Utils import extract_progressive_by_blocks
from Source.Utils import count_psnr, embed_watermark_into_image, generate_watermark


IMAGE_PATH = os.path.join("Image","lena.tif")
OUTPUT_DIR = os.path.join("Results", "Extraction_Research")

REGION_SIZE = 64
QR_SIZE = 14
PHI = np.pi / 3
EMBED_X = 8
EMBED_Y = 8
EMBED_OFFSET = 4
S_PARAM = 300
QR_SEED = 42
EXTRACT_START_SEED = 6513
BLOCK_ORDER_SEED = 2026
AVOID_ZERO_ZERO_START = True
SHUFFLE_BLOCK_ORDER = True
PHASE_SIGN = -1
DETREND = QR_SIZE <= 8


def calculate_q(s_param: float, qr_size: int, region_size: int) -> float:
    return (255 * region_size * region_size) / (s_param * qr_size)


def plot_figure_1(image: np.ndarray, qr_true: np.ndarray, watermarked: np.ndarray, results, save_path: str):
    """ Plot for extracted blocks vs number of extracted blocks """
    n_items = 3 + len(results)
    n_cols = 4
    n_rows = math.ceil(n_items / n_cols)

    plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    ax = plt.subplot(n_rows, n_cols, 1)
    ax.imshow(image, cmap="gray")
    ax.set_title("Original image")
    ax.axis("off")

    ax = plt.subplot(n_rows, n_cols, 2)
    ax.imshow(qr_true, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Original QR")
    ax.axis("off")

    ax = plt.subplot(n_rows, n_cols, 3)
    ax.imshow(watermarked, cmap="gray")
    ax.set_title("Watermarked image")
    ax.axis("off")

    for plot_idx, result in enumerate(results, start=4):
        ax = plt.subplot(n_rows, n_cols, plot_idx)
        ax.imshow(result.recovered_qr, cmap="gray", vmin=0, vmax=1)
        acc_text = "" if result.accuracy is None else f", acc={result.accuracy:.3f}"
        ax.set_title(f"{result.blocks_used} blocks{acc_text}")
        ax.axis("off")

    plt.suptitle("Progressive extraction results")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_figure_2(results, save_path: str):
    """Plot for accuracy vs number of extracted blocks"""
    blocks = [r.blocks_used for r in results]
    accuracies = [r.accuracy for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(blocks, accuracies, marker="o")
    plt.xscale("log", base=2)
    plt.xticks(blocks, [str(b) for b in blocks])
    plt.ylim(0, 1.05)
    plt.xlabel("Number of blocks used for extraction")
    plt.ylabel("Bit accuracy")
    plt.title("Accuracy vs number of extracted blocks")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    image = plt.imread(IMAGE_PATH)
    qr_true = generate_watermark(QR_SIZE, seed=QR_SEED)
    q = calculate_q(S_PARAM, QR_SIZE, REGION_SIZE)

    watermarked = embed_watermark_into_image(image=image, qr=qr_true, size_region=REGION_SIZE, q=q,
                                             phi=PHI, x=EMBED_X, y=EMBED_Y, offset=EMBED_OFFSET)

    start_x = 5
    start_y = 8

    results, n_available = extract_progressive_by_blocks(
        image=watermarked,
        qr_size=QR_SIZE,
        size_region=REGION_SIZE,
        phi=PHI,
        start_x=start_x,
        start_y=start_y,
        x=EMBED_X,
        y=EMBED_Y,
        offset=EMBED_OFFSET,
        qr_true=qr_true,
        phase_sign=PHASE_SIGN,
        shuffle_blocks=SHUFFLE_BLOCK_ORDER,
        seed=BLOCK_ORDER_SEED,
        detrend=(QR_SIZE<=8),
    )

    print("Experiment config")
    print("- image_size:", image.shape)
    print("- region_size:", REGION_SIZE)
    print("- qr_size:", QR_SIZE)
    print("- q:", q)
    print("- PSNR:", count_psnr(image, watermarked))
    print("- embed x/y/offset:", (EMBED_X, EMBED_Y, EMBED_OFFSET))
    print("- extract start_x/start_y:", (start_x, start_y))
    print("- available blocks:", n_available)
    print("- decision rule: median threshold on phase-projection score")
    print()

    for r in results:
        print(
            f"blocks={r.blocks_used:>3} | "
            f"accuracy={r.accuracy:.4f} | "
            f"predicted_ones={r.predicted_ones:>2} | "
            f"threshold={r.threshold:.4f}"
        )

    fig1_path = os.path.join(OUTPUT_DIR, "figure_1_extract_results_by_blocks.png")
    fig2_path = os.path.join(OUTPUT_DIR, "figure_2_accuracy_vs_blocks.png")


    plot_figure_1(image, qr_true, watermarked, results, fig1_path)
    plot_figure_2(results, fig2_path)

    print()
    print("Saved:")
    print("-", fig1_path)
    print("-", fig2_path)