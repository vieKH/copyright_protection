from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Source.Utils.distorter import ImageDistorter
from Source.Utils.extraction_research import extract_progressive_by_blocks
from Source.Utils.utils import (
    bit_accuracy,
    count_psnr,
    embed_watermark_into_image,
    generate_watermark,
)

IMAGE_PATH = os.path.join("Image", "lena.tif")
OUTPUT_DIR = os.path.join("Results", "Distortion_Research")

REGION_SIZE = 64
QR_SIZE = 14
PHI = np.pi / 3
EMBED_X = REGION_SIZE // 8
EMBED_Y = REGION_SIZE // 8
EMBED_OFFSET = REGION_SIZE // 16
S_PARAM = 200
QR_SEED = 42

# Baseline extraction offset used in your current experiments.
BASE_START_X = 5
BASE_START_Y = 8
PHASE_SIGN_CANDIDATES = (-1, 1)

# For geometric attacks, search block alignment again.
GEOMETRIC_OFFSET_SEARCH_STEP = 8

# Figure layouts requested by the user.
GRID_ROWS = 3
GRID_COLS = 6


@dataclass(frozen=True)
class AttackCase:
    name: str
    group: str
    parameter: str
    apply: Callable[[np.ndarray], np.ndarray]
    search_offsets: bool


def calculate_q(s_param: float, qr_size: int, region_size: int) -> float:
    return (255 * region_size * region_size) / (s_param * qr_size)


def load_grayscale(path: str) -> np.ndarray:
    image = plt.imread(path)
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.dtype.kind == "f" and image.max() <= 1.0:
        image = (255 * image).astype(np.uint8)
    return image.astype(np.uint8)


def separation_metric(score_map: np.ndarray, recovered_qr: np.ndarray) -> float:
    flat = np.asarray(score_map, dtype=np.float64).ravel()
    pred = np.asarray(recovered_qr, dtype=np.uint8).ravel()
    idx1 = pred == 1
    idx0 = pred == 0

    if not np.any(idx1) or not np.any(idx0):
        return -np.inf

    return float((np.mean(flat[idx1]) - np.mean(flat[idx0])) / (np.std(flat) + 1e-6))


def offset_grid(block_size: int, step: int) -> List[Tuple[int, int]]:
    values = list(range(0, block_size, step))
    return [(sx, sy) for sx in values for sy in values]


def extract_last_result(
    image: np.ndarray,
    qr_true: np.ndarray,
    start_x: int,
    start_y: int,
    phase_sign: int,
):
    results, n_available = extract_progressive_by_blocks(
        image=image,
        qr_size=QR_SIZE,
        size_region=REGION_SIZE,
        phi=PHI,
        start_x=start_x,
        start_y=start_y,
        x=EMBED_X,
        y=EMBED_Y,
        offset=EMBED_OFFSET,
        block_counts=[10**9],
        qr_true=qr_true,
        phase_sign=phase_sign,
        shuffle_blocks=False,
        seed=None,
        detrend=(QR_SIZE <= 8),
    )
    return results[-1], n_available


def extract_with_search(image: np.ndarray, qr_true: np.ndarray, search_offsets: bool):
    if search_offsets:
        candidates = offset_grid(REGION_SIZE, GEOMETRIC_OFFSET_SEARCH_STEP)
    else:
        candidates = [(BASE_START_X, BASE_START_Y)]

    best = None

    for phase_sign in PHASE_SIGN_CANDIDATES:
        for sx, sy in candidates:
            try:
                result, n_available = extract_last_result(
                    image=image,
                    qr_true=qr_true,
                    start_x=sx,
                    start_y=sy,
                    phase_sign=phase_sign,
                )
            except Exception:
                continue

            metric = separation_metric(result.score_map, result.recovered_qr)
            item = {
                "result": result,
                "metric": metric,
                "start_x": sx,
                "start_y": sy,
                "phase_sign": phase_sign,
                "available_blocks": n_available,
            }

            if best is None or metric > best["metric"]:
                best = item

    if best is None:
        raise RuntimeError("No valid extraction candidate found")

    return best


def build_attack_cases() -> List[AttackCase]:
    return [
        AttackCase("none", "baseline", "-", lambda img: img, False),
        AttackCase("gaussian_noise", "non_geometric", "variance=500",
                   lambda img: ImageDistorter(img).white_noise(variance=500, seed=1).image, False),
        AttackCase("gaussian_noise", "non_geometric", "variance=800",
                   lambda img: ImageDistorter(img).white_noise(variance=800, seed=1).image, False),
        AttackCase("salt_pepper", "non_geometric", "fraction=0.02",
                   lambda img: ImageDistorter(img).salt_pepper(noise_fraction=0.02, seed=2).image, False),
        AttackCase("contrast", "non_geometric", "factor=0.75",
                   lambda img: ImageDistorter(img).contrast(0.75).image, False),
        AttackCase("contrast", "non_geometric", "factor=1.25",
                   lambda img: ImageDistorter(img).contrast(1.25).image, False),
        AttackCase("jpeg", "non_geometric", "quality=80",
                   lambda img: ImageDistorter(img).jpeg(80).image, False),
        AttackCase("jpeg", "non_geometric", "quality=50",
                   lambda img: ImageDistorter(img).jpeg(50).image, False),
        AttackCase("gaussian_blur", "non_geometric", "sigma=2.0",
                   lambda img: ImageDistorter(img).gauss_blur(2.0).image, False),
        AttackCase("median_filter", "non_geometric", "window=5",
                   lambda img: ImageDistorter(img).median(5).image, False),
        AttackCase("crop_restore", "geometric", "retained_fraction=0.90 center",
                   lambda img: ImageDistorter(img).crop_restore(0.90, position="center", fill_value=0).image, False),
        AttackCase("cutout", "geometric", "area_fraction=0.10 center",
                   lambda img: ImageDistorter(img).cutout(0.10, position="center", fill_value=0).image, False),
        AttackCase("resampling", "geometric", "factor=0.75 down-up",
                   lambda img: ImageDistorter(img).resampling(0.75, interpolation="bilinear").image, False),
        AttackCase("scale_rest", "geometric", "factor=1.0",
                   lambda img: ImageDistorter(img).scale_rest(1.0, interpolation="bilinear").image, False),
        AttackCase("rotation", "geometric", "angle=65.13 deg",
                   lambda img: ImageDistorter(img).rotation(65.13, interpolation="bilinear", fill_value=0).image, False),
        AttackCase("rotation_rest", "geometric", "angle=30.0 deg",
                   lambda img: ImageDistorter(img).rotation_rest(30.0, interpolation="bilinear", fill_value=0).image, False),
        AttackCase("cyclic_shift", "geometric", "fraction=0.35",
                   lambda img: ImageDistorter(img).cyclic_shift(0.35).image, False),
    ]


def save_accuracy_plot(rows: Sequence[dict], save_path: str) -> None:
    labels = [f"{r['attack']}\n{r['parameter']}" for r in rows]
    values = [r["accuracy"] for r in rows]

    plt.figure(figsize=(max(10, 0.8 * len(rows)), 5))
    plt.plot(range(len(rows)), values, marker="o")
    plt.xticks(range(len(rows)), labels, rotation=60, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Bit accuracy")
    plt.title("Watermark robustness under distortions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _draw_grid(images: Sequence[np.ndarray], titles: Sequence[str], save_path: str,
               suptitle: str, vmin=None, vmax=None) -> None:
    n_slots = GRID_ROWS * GRID_COLS
    if len(images) > n_slots:
        raise ValueError(f"Too many images: got {len(images)}, but grid has only {n_slots} slots")

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(3.6 * GRID_COLS, 3.6 * GRID_ROWS))
    axes = np.asarray(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for idx, (img, title) in enumerate(zip(images, titles)):
        ax = axes[idx]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_distorted_images_figure(watermarked: np.ndarray, rows: Sequence[dict], save_path: str) -> None:
    images = [watermarked]
    titles = ["Watermarked image"]

    for row in rows:
        images.append(row["distorted_image"])
        titles.append(f"{row['attack']}\n{row['parameter']}")

    _draw_grid(
        images=images,
        titles=titles,
        save_path=save_path,
        suptitle="Watermarked image and attacked images",
    )


def save_extracted_qr_figure(qr_true: np.ndarray, rows: Sequence[dict], save_path: str) -> None:
    images = [qr_true]
    titles = ["Original QR"]

    for row in rows:
        images.append(row["recovered_qr"])
        titles.append(
            f"{row['attack']}\nacc={row['accuracy']:.3f}"
        )

    _draw_grid(
        images=images,
        titles=titles,
        save_path=save_path,
        suptitle="Original QR and extracted QRs after attacks",
        vmin=0,
        vmax=1,
    )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = load_grayscale(IMAGE_PATH)
    qr_true = generate_watermark(QR_SIZE, seed=QR_SEED)
    q = calculate_q(S_PARAM, QR_SIZE, REGION_SIZE)

    watermarked = embed_watermark_into_image(
        image=image,
        qr=qr_true,
        size_region=REGION_SIZE,
        q=q,
        phi=PHI,
        x=EMBED_X,
        y=EMBED_Y,
        offset=EMBED_OFFSET,
    )

    rows = []

    for case in build_attack_cases():
        distorted = case.apply(watermarked)
        distorted = np.clip(distorted, 0, 255).astype(np.uint8)

        best = extract_with_search(
            image=distorted,
            qr_true=qr_true,
            search_offsets=case.search_offsets,
        )
        result = best["result"]

        row = {
            "attack": case.name,
            "group": case.group,
            "parameter": case.parameter,
            "psnr_watermarked_vs_distorted": count_psnr(watermarked, distorted),
            "accuracy": bit_accuracy(qr_true, result.recovered_qr),
            "predicted_ones": int(result.predicted_ones),
            "threshold": float(result.threshold),
            "blocks_used": int(result.blocks_used),
            "available_blocks": int(best["available_blocks"]),
            "metric": float(best["metric"]),
            "start_x": int(best["start_x"]),
            "start_y": int(best["start_y"]),
            "phase_sign": int(best["phase_sign"]),
            "search_offsets": bool(case.search_offsets),
            # only for figures, not for CSV
            "distorted_image": distorted.copy(),
            "recovered_qr": result.recovered_qr.copy(),
        }
        rows.append(row)

        psnr_value = row['psnr_watermarked_vs_distorted']
        psnr_text = "inf" if np.isinf(psnr_value) else f"{psnr_value:.2f}"

        print(
            f"{case.name:16s} | {case.parameter:28s} | "
            f"acc={row['accuracy']:.4f} | psnr={psnr_text} | "
            f"start=({row['start_x']},{row['start_y']}) | phase={row['phase_sign']}"
        )

    # Save CSV without image arrays.
    csv_rows = []
    for row in rows:
        clean_row = {k: v for k, v in row.items() if k not in ("distorted_image", "recovered_qr")}
        csv_rows.append(clean_row)

    csv_path = os.path.join(OUTPUT_DIR, "distortion_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    plot_path = os.path.join(OUTPUT_DIR, "distortion_accuracy.png")
    save_accuracy_plot(csv_rows, plot_path)

    attacked_grid_path = os.path.join(OUTPUT_DIR, "distortion_attacked_images_grid.png")
    save_distorted_images_figure(watermarked, rows, attacked_grid_path)

    extracted_qr_grid_path = os.path.join(OUTPUT_DIR, "distortion_extracted_qr_grid.png")
    save_extracted_qr_figure(qr_true, rows, extracted_qr_grid_path)

    print()
    print("Saved:")
    print("-", csv_path)
    print("-", plot_path)
    print("-", attacked_grid_path)
    print("-", extracted_qr_grid_path)


if __name__ == "__main__":
    main()
