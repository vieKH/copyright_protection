from __future__ import annotations

import csv
import math
import os
import shutil
from dataclasses import dataclass
from typing import Callable, List, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Source.Utils.distorter import ImageDistorter
from Source.Utils.extraction_research import extract_progressive_by_blocks
from Source.Utils.utils import (
    bit_accuracy,
    count_psnr,
    embed_watermark_into_image,
    generate_watermark,
)

Number = Union[int, float]

# ============================================================
# Main experiment configuration
# ============================================================
IMAGE_PATH = os.path.join("Image", "lena.tif")
OUTPUT_DIR = os.path.join("Results", "Distortion_Research")

REGION_SIZE = 64
QR_SIZE = 14
PHI = np.pi / 3

# Embedding position in every Fourier block.
EMBED_X = 8
EMBED_Y = 8
EMBED_OFFSET = 4

# Watermark strength parameter.
S_PARAM = 300
QR_SEED = 42


START_X = 5
START_Y = 8
PHASE_SIGN = -1

# Only small QR sizes usually need detrending.
DETREND = QR_SIZE <= 8

# Figure layout. The number of rows is computed automatically.
MAX_GRID_COLS = 5


@dataclass(frozen=True)
class AttackSpec:
    name: str
    group: str
    param_name: str
    param_symbol: str
    values: Sequence[Number]
    apply: Callable[[np.ndarray, Number], np.ndarray]


def calculate_q(s_param: float, qr_size: int, region_size: int) -> float:
    return (255 * region_size * region_size) / (s_param * qr_size)


def load_grayscale(path: str) -> np.ndarray:
    image = plt.imread(path)
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.dtype.kind == "f" and image.max() <= 1.0:
        image = (255 * image).astype(np.uint8)
    return np.clip(image, 0, 255).astype(np.uint8)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def remove_path(path: str) -> None:
    """Remove a file or directory if it already exists."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def reset_dir(path: str) -> None:
    """Delete and recreate a directory to avoid stale files from previous runs."""
    remove_path(path)
    ensure_dir(path)


def prepare_attack_dir(attack_dir: str) -> str:
    """Keep only the figures folder inside every attack directory.

    This also removes legacy outputs from older runs:
    - attacked/
    - extracted_qr/
    - metrics.csv
    """
    ensure_dir(attack_dir)

    remove_path(os.path.join(attack_dir, "attacked"))
    remove_path(os.path.join(attack_dir, "extracted_qr"))
    remove_path(os.path.join(attack_dir, "metrics.csv"))

    figures_dir = os.path.join(attack_dir, "figures")
    remove_path(figures_dir)
    ensure_dir(figures_dir)
    return figures_dir


def auto_metric_upper(values: Sequence[float], min_upper: float = 1e-3, margin: float = 0.15) -> float:
    """Return a compact upper y-limit for small-valued metrics such as BER."""
    finite_values = [float(v) for v in values if np.isfinite(float(v))]
    if not finite_values:
        return min_upper

    max_value = max(finite_values)
    if max_value <= 0:
        return min_upper

    return max(min_upper, max_value * (1.0 + margin))


def frange(start: float, stop: float, step: float, decimals: int = 6) -> List[float]:
    """Inclusive numeric range for distortion parameter sweeps."""
    if step <= 0:
        raise ValueError("step must be positive")

    values: List[float] = []
    current = float(start)
    eps = abs(step) * 1e-6
    while current <= float(stop) + eps:
        values.append(round(current, decimals))
        current += step
    return values


def irange(start: int, stop: int, step: int) -> List[int]:
    """Inclusive integer range."""
    if step <= 0:
        raise ValueError("step must be positive")
    return list(range(int(start), int(stop) + 1, int(step)))


def value_to_text(value: Number) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text


def value_to_filename(value: Number) -> str:
    return value_to_text(value).replace("-", "m").replace(".", "p")


def save_gray_image(path: str, image: np.ndarray, vmin=None, vmax=None) -> None:
    ensure_dir(os.path.dirname(path))
    arr = np.asarray(image)

    if arr.ndim != 2:
        raise ValueError("save_gray_image expects a grayscale image")

    if vmin is not None or vmax is not None:
        lo = 0 if vmin is None else vmin
        hi = 255 if vmax is None else vmax
        arr = np.clip((arr.astype(np.float64) - lo) / max(hi - lo, 1e-12), 0, 1)
        arr = (255 * arr).astype(np.uint8)
    else:
        if arr.dtype.kind == "f":
            max_value = 1.0 if arr.max() <= 1.0 else 255.0
            arr = np.clip(arr / max_value, 0, 1)
            arr = (255 * arr).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    Image.fromarray(arr, mode="L").save(path)


def save_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    save_path: str,
    suptitle: str,
    vmin=None,
    vmax=None,
) -> None:
    if len(images) != len(titles):
        raise ValueError("images and titles must have the same length")
    if not images:
        return

    n_items = len(images)
    n_cols = min(MAX_GRID_COLS, n_items)
    n_rows = int(math.ceil(n_items / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.3 * n_rows))
    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(
    rows: Sequence[dict],
    save_path: str,
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    ylim=None,
) -> None:
    if not rows:
        return

    x = [float(r[x_key]) for r in rows]
    y = [float(r[y_key]) for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_attack_specs() -> List[AttackSpec]:
    """Attack parameter table.

    Each attack is swept from p_min to p_max with step delta_p.
    One attack corresponds to one output folder.
    """
    return [
        AttackSpec(
            name="contrast",
            group="pixel_value",
            param_name="alpha",
            param_symbol="α",
            values=frange(0.7, 1.3, 0.1),
            apply=lambda img, v: ImageDistorter(img).contrast(float(v)).image,
        ),
        AttackSpec(
            name="rot_rest",
            group="geometric",
            param_name="angle_deg",
            param_symbol="φ",
            values=frange(0.0, 42.0, 7.0),
            apply=lambda img, v: ImageDistorter(img).rotation_rest(
                float(v), interpolation="bilinear", fill_value=0
            ).image,
        ),
        AttackSpec(
            name="rotation",
            group="geometric",
            param_name="angle_deg",
            param_symbol="φ",
            values=frange(1.0, 90.0, 8.9),
            apply=lambda img, v: ImageDistorter(img).rotation(
                float(v), interpolation="bilinear", fill_value=0
            ).image,
        ),
        AttackSpec(
            name="scale_rest",
            group="geometric",
            param_name="scale_factor",
            param_symbol="k",
            values=frange(0.55, 1.45, 0.15),
            apply=lambda img, v: ImageDistorter(img).scale_rest(
                float(v), interpolation="bilinear", fill_value=0
            ).image,
        ),
        AttackSpec(
            name="scale",
            group="geometric",
            param_name="scale_factor",
            param_symbol="k",
            values=frange(0.55, 1.45, 0.15),
            apply=lambda img, v: ImageDistorter(img).scale(
                float(v), interpolation="bilinear", fill_value=0
            ).image,
        ),
        AttackSpec(
            name="cut",
            group="geometric",
            param_name="area_fraction",
            param_symbol="ϑ",
            values=frange(0.2, 0.9, 0.1),
            apply=lambda img, v: ImageDistorter(img).cutout(
                float(v), position="center", fill_value=0
            ).image,
        ),
        AttackSpec(
            name="cyclic_shift",
            group="geometric",
            param_name="shift_fraction",
            param_symbol="r",
            values=frange(0.1, 0.9, 0.1),
            apply=lambda img, v: ImageDistorter(img).cyclic_shift(float(v)).image,
        ),
        AttackSpec(
            name="smooth",
            group="filtering",
            param_name="window_size",
            param_symbol="M",
            values=irange(3, 15, 2),
            apply=lambda img, v: ImageDistorter(img).smooth(int(v)).image,
        ),
        AttackSpec(
            name="gauss_blur",
            group="filtering",
            param_name="sigma",
            param_symbol="σ",
            values=frange(1.0, 4.0, 0.5),
            apply=lambda img, v: ImageDistorter(img).gauss_blur(float(v)).image,
        ),
        AttackSpec(
            name="sharpen",
            group="filtering",
            param_name="window_size",
            param_symbol="M",
            values=irange(3, 15, 2),
            apply=lambda img, v: ImageDistorter(img).sharpen(int(v)).image,
        ),
        AttackSpec(
            name="median",
            group="filtering",
            param_name="window_size",
            param_symbol="M",
            values=irange(3, 15, 2),
            apply=lambda img, v: ImageDistorter(img).median(int(v)).image,
        ),
        AttackSpec(
            name="wh_noise",
            group="noise",
            param_name="variance",
            param_symbol="Dξ",
            values=irange(400, 1000, 100),
            apply=lambda img, v: ImageDistorter(img).white_noise(
                variance=float(v), seed=1
            ).image,
        ),
        AttackSpec(
            name="salt_pepper",
            group="noise",
            param_name="noise_fraction",
            param_symbol="q",
            values=frange(0.05, 0.5, 0.05),
            apply=lambda img, v: ImageDistorter(img).salt_pepper(
                noise_fraction=float(v), seed=2
            ).image,
        ),
        AttackSpec(
            name="jpeg",
            group="compression",
            param_name="quality_factor",
            param_symbol="F",
            values=irange(30, 90, 10),
            apply=lambda img, v: ImageDistorter(img).jpeg(int(v)).image,
        ),
    ]


def extract_last_result(image: np.ndarray, qr_true: np.ndarray):
    """Extract QR using fixed image-domain start offset for all cases."""
    results, n_available = extract_progressive_by_blocks(
        image=image,
        qr_size=QR_SIZE,
        size_region=REGION_SIZE,
        phi=PHI,
        start_x=START_X,
        start_y=START_Y,
        x=EMBED_X,
        y=EMBED_Y,
        offset=EMBED_OFFSET,
        block_counts=[10**9],
        qr_true=qr_true,
        phase_sign=PHASE_SIGN,
        shuffle_blocks=False,
        seed=None,
        detrend=DETREND,
    )
    return results[-1], n_available


def run_one_attack(
    spec: AttackSpec,
    watermarked: np.ndarray,
    qr_true: np.ndarray,
    attack_dir: str,
) -> List[dict]:
    figures_dir = prepare_attack_dir(attack_dir)

    rows: List[dict] = []
    attacked_images: List[np.ndarray] = []
    attacked_titles: List[str] = []
    qr_images: List[np.ndarray] = [qr_true]
    qr_titles: List[str] = ["Original QR"]

    for value in spec.values:
        value_text = value_to_text(value)

        attacked = spec.apply(watermarked, value)
        attacked = np.clip(attacked, 0, 255).astype(np.uint8)

        result, n_available = extract_last_result(attacked, qr_true)
        accuracy = float(bit_accuracy(qr_true, result.recovered_qr))
        ber = 1.0 - accuracy
        psnr_wm = float(count_psnr(watermarked, attacked))

        row = {
            "attack": spec.name,
            "group": spec.group,
            "parameter_name": spec.param_name,
            "parameter_symbol": spec.param_symbol,
            "parameter_value": value_text,
            "psnr_watermarked_vs_attacked": psnr_wm,
            "accuracy": accuracy,
            "ber": ber,
            "predicted_ones": int(result.predicted_ones),
            "threshold": float(result.threshold),
            "blocks_used": int(result.blocks_used),
            "available_blocks": int(n_available),
            "start_x": int(START_X),
            "start_y": int(START_Y),
            "phase_sign": int(PHASE_SIGN),
        }
        rows.append(row)

        attacked_images.append(attacked)
        attacked_titles.append(f"{spec.param_symbol}={value_text}\nacc={accuracy:.3f}")
        qr_images.append(result.recovered_qr)
        qr_titles.append(f"{spec.param_symbol}={value_text}\nacc={accuracy:.3f}")

        psnr_text = "inf" if np.isinf(psnr_wm) else f"{psnr_wm:.2f}"
        print(
            f"{spec.name:14s} | {spec.param_name}={value_text:>7s} | "
            f"acc={accuracy:.4f} | ber={ber:.4f} | psnr={psnr_text} | "
            f"start=({START_X},{START_Y}) | phase={PHASE_SIGN}"
        )

    save_image_grid(
        images=attacked_images,
        titles=attacked_titles,
        save_path=os.path.join(figures_dir, f"{spec.name}_attacked_images.png"),
        suptitle=f"{spec.name}: attacked images",
    )

    save_image_grid(
        images=qr_images,
        titles=qr_titles,
        save_path=os.path.join(figures_dir, f"{spec.name}_extracted_qrs.png"),
        suptitle=f"{spec.name}: extracted QRs",
        vmin=0,
        vmax=1,
    )

    save_metric_plot(
        rows=rows,
        save_path=os.path.join(figures_dir, f"{spec.name}_accuracy.png"),
        x_key="parameter_value",
        y_key="accuracy",
        title=f"{spec.name}: accuracy vs {spec.param_symbol}",
        xlabel=f"{spec.param_name} ({spec.param_symbol})",
        ylabel="Bit accuracy",
        ylim=(0, 1.05),
    )

    save_metric_plot(
        rows=rows,
        save_path=os.path.join(figures_dir, f"{spec.name}_ber.png"),
        x_key="parameter_value",
        y_key="ber",
        title=f"{spec.name}: BER vs {spec.param_symbol}",
        xlabel=f"{spec.param_name} ({spec.param_symbol})",
        ylabel="Bit error rate",
        ylim=(0.0, auto_metric_upper([float(r["ber"]) for r in rows])),
    )

    return rows


def save_summary_accuracy_plot(rows: Sequence[dict], save_path: str) -> None:
    if not rows:
        return

    labels = [f"{r['attack']}\n{r['parameter_value']}" for r in rows]
    values = [float(r["accuracy"]) for r in rows]

    plt.figure(figsize=(max(14, 0.28 * len(rows)), 5))
    plt.plot(range(len(rows)), values, marker="o", linewidth=1)
    plt.xticks(range(len(rows)), labels, rotation=90, fontsize=6)
    plt.ylim(0, 1.05)
    plt.ylabel("Bit accuracy")
    plt.title("Watermark robustness summary over all distortion parameters")
    plt.grid(True)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary_ber_plot(rows: Sequence[dict], save_path: str) -> None:
    if not rows:
        return

    labels = [f"{r['attack']}\n{r['parameter_value']}" for r in rows]
    values = [float(r["ber"]) for r in rows]

    plt.figure(figsize=(max(14, 0.28 * len(rows)), 5))
    plt.plot(range(len(rows)), values, marker="o", linewidth=1)
    plt.xticks(range(len(rows)), labels, rotation=90, fontsize=6)
    plt.ylim(0.0, auto_metric_upper(values))
    plt.ylabel("Bit error rate")
    plt.title("BER summary over all distortion parameters")
    plt.grid(True)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    # Start from a clean output directory so old attacked/ and extracted_qr/
    # folders from previous versions cannot remain visible.
    reset_dir(OUTPUT_DIR)
    summary_dir = os.path.join(OUTPUT_DIR, "summary")
    ensure_dir(summary_dir)

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
    watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)

    save_gray_image(os.path.join(OUTPUT_DIR, "watermarked.png"), watermarked)
    save_gray_image(os.path.join(OUTPUT_DIR, "original_qr.png"), qr_true, vmin=0, vmax=1)

    print("Experiment config")
    print("- image_size:", image.shape)
    print("- region_size:", REGION_SIZE)
    print("- qr_size:", QR_SIZE)
    print("- q:", q)
    print("- PSNR original vs watermarked:", count_psnr(image, watermarked))
    print("- embed x/y/offset:", (EMBED_X, EMBED_Y, EMBED_OFFSET))
    print("- fixed extraction start_x/start_y:", (START_X, START_Y))
    print("- fixed phase_sign:", PHASE_SIGN)
    print()

    all_rows: List[dict] = []
    for spec in build_attack_specs():
        print(f"\n=== Running attack: {spec.name} ===")
        attack_dir = os.path.join(OUTPUT_DIR, spec.name)
        attack_rows = run_one_attack(spec, watermarked, qr_true, attack_dir)
        all_rows.extend(attack_rows)

    all_csv_path = os.path.join(summary_dir, "all_results.csv")
    save_csv(all_csv_path, all_rows)
    save_summary_accuracy_plot(
        all_rows,
        os.path.join(summary_dir, "accuracy_summary_all_attacks.png"),
    )
    save_summary_ber_plot(
        all_rows,
        os.path.join(summary_dir, "ber_summary_all_attacks.png"),
    )

    print("\nSaved summary:")
    print("-", all_csv_path)
    print("-", os.path.join(summary_dir, "accuracy_summary_all_attacks.png"))
    print("-", os.path.join(summary_dir, "ber_summary_all_attacks.png"))
    print("\nEach attack folder contains only:")
    print("- figures/")
    print("\nCSV output is stored only in:")
    print("-", all_csv_path)


if __name__ == "__main__":
    main()
