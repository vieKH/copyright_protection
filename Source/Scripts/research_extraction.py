import math
import os
from pathlib import Path
from Source.Utils.run_output import create_run_directory

import matplotlib.pyplot as plt
import numpy as np
from Source import config
from Source.Utils import calculate_q

from Source.Utils import extract_progressive_by_blocks
from Source.Utils import count_psnr, embed_watermark_into_image, generate_watermark


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = str(PROJECT_ROOT / "Image" / "vn_beach.tif")
OUTPUT_DIR = str(PROJECT_ROOT / "Results" / "Extraction_Research")

REGION_SIZE = config.REGION_SIZE
EXTRACT_STRIDE = config.EXTRACT_STRIDE
QR_SIZE = 14
PHI = np.pi / 3
EMBED_X = config.EMBED_X
EMBED_Y = config.EMBED_Y
EMBED_OFFSET = config.EMBED_OFFSET
S_PARAM = 300
QR_SEED = 42
EXTRACT_START_SEED = 6513
BLOCK_ORDER_SEED = 2026
AVOID_ZERO_ZERO_START = True
SHUFFLE_BLOCK_ORDER = True
PHASE_SIGN = -1
DETREND = False


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
    print("LEGACY baseline without synchronization. For sync use Source.Scripts.research_synchronization.")
    q = calculate_q(S_PARAM, QR_SIZE, REGION_SIZE)

    RUN_DIR = create_run_directory(
        base=OUTPUT_DIR,
        image_path=IMAGE_PATH,
        qr_size=QR_SIZE,
        block_size=REGION_SIZE,
        gap_size=config.EMBED_GAP,
        param_name="q",
        param_value=q,
    )

    image = plt.imread(IMAGE_PATH)
    qr_true = generate_watermark(QR_SIZE, seed=QR_SEED)


    watermarked = embed_watermark_into_image(image=image, qr=qr_true, size_region=REGION_SIZE, q=q,
                                             phi=PHI, x=EMBED_X, y=EMBED_Y, offset=EMBED_OFFSET, gap=config.EMBED_GAP)

    start_x = 5
    start_y = 8

    results, n_available = extract_progressive_by_blocks(
        image=watermarked,
        qr_size=QR_SIZE,
        size_region=REGION_SIZE,
        phi=PHI,
        start_x=start_x,
        start_y=start_y,
        stride=EXTRACT_STRIDE,
        x=EMBED_X,
        y=EMBED_Y,
        offset=EMBED_OFFSET,
        gap=config.EMBED_GAP,
        qr_true=qr_true,
        phase_sign=PHASE_SIGN,
        shuffle_blocks=SHUFFLE_BLOCK_ORDER,
        seed=BLOCK_ORDER_SEED,
        detrend=DETREND,
    )

    print("Experiment config")
    print("- image_size:", image.shape)
    print("- region_size:", REGION_SIZE)
    print("- extract_stride:", EXTRACT_STRIDE)
    print("- embed_gap (empty spectral bins):", config.EMBED_GAP)
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

    fig1_path = os.path.join(RUN_DIR, "figure_1_extract_results_by_blocks.png")

    fig2_path = os.path.join(RUN_DIR,"figure_2_accuracy_vs_blocks.png")

    plot_figure_1(image, qr_true, watermarked, results, fig1_path)
    plot_figure_2(results, fig2_path)

    print()
    print("Saved:")
    print("-", fig1_path)
    print("-", fig2_path)