"""Helpers shared by the experiment scripts: seeds, image loading, matplotlib setup."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiments import config as C

FONTS = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]


def seed(*parts) -> int:
    """Deterministic 32-bit seed from any parts, e.g. seed("boat", 3, "bits", 21)."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def load_image(path: Path, check: bool = False) -> np.ndarray:
    """8-bit grayscale image as an array; `check` verifies the size is a multiple of B0."""
    img = np.asarray(Image.open(path))
    if check and (img.ndim != 2 or img.dtype != np.uint8 or img.shape[0] % C.B0 or img.shape[1] % C.B0):
        raise ValueError(f"{path.name}: expected 8-bit grayscale image, size multiple of {C.B0}")
    return img


def setup_plots(font_size: int, **rc) -> None:
    """Serif fonts (Russian labels) and quiet font-manager warnings."""
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({"font.family": FONTS, "font.size": font_size, **rc})


def save_figure(fig, path: Path, dpi: int, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
