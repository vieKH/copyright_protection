"""Basic studies of the 2-D DFT on a black image (chapter 1, figures 5-7).

    python experiments/basic.py

1_pair.png               one conjugate pair of coefficients -> a single 2-D cosine wave
2_pairs_one_quadrant.png four pairs in one quadrant -> waves of one orientation      (figure 6)
3_pairs_two_quadrants.png the same pairs split between two quadrants -> isotropic pattern (figure 7)
4_phase.png              the same carriers with phase pi/3 -> shifted waves
5_object.png             balanced binary object 14x14                                   (figure 5)
6_object_one_quadrant.png / 7_object_halves.png
                         a 14x14 object written into one quadrant vs. split into two halves
                         (the layout used by the method, wm.Layout)
8_spectrum.png           a test image and its log-magnitude spectrum
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Source/

import matplotlib.pyplot as plt
import numpy as np

import wm
from experiments.common import config as C
from experiments.common import load_image, save_figure, setup_plots

OUT = C.RESULTS / "basic"
N = 64                                   # size of the black image in the point studies


def log_spectrum(F):
    """Log-compressed modulus for display (zero frequency in the top-left corner, as in the DFT definition)."""
    return np.log1p(np.abs(F))


def set_pairs(F, u, v, phase=0.0):
    """Write unit coefficients exp(j phase) at (u, v) and the conjugate values at (-u, -v)."""
    M, K = F.shape
    for a, b in zip(np.atleast_1d(u), np.atleast_1d(v)):
        F[a, b] = np.exp(1j * phase)
        F[(-a) % M, (-b) % K] = np.exp(-1j * phase)
    return F


def panels(images, titles, name, cmaps=None):
    fig, axs = plt.subplots(1, len(images), figsize=(5 * len(images), 5.4))
    for ax, im, t, cm in zip(np.atleast_1d(axs), images, titles, cmaps or ["gray"] * len(images)):
        ax.imshow(im, cmap=cm, interpolation="nearest"); ax.set_title(t); ax.set_axis_off()
    save_figure(fig, OUT / name, dpi=300)


def point_study(u, v, name, what, phase=0.0):
    """Black image -> change coefficients -> inverse DFT."""
    black = np.zeros((N, N))
    F = set_pairs(np.fft.fft2(black), u, v, phase)
    w = np.real(np.fft.ifft2(F))
    panels([black, log_spectrum(F), w],
           ["Исходное\nизображение", f"Спектр после\nизменения {what}", f"Изображение после\nизменения {what}"], name)


def object_study(lay: wm.Layout, bits, name, title):
    """Black B x B block with a whole binary object in the spectrum (unit modulus, phase key)."""
    theta = wm.phase_key(lay.L, C.PHASE_SEED)
    panels([np.abs(wm.spectrum(lay, bits, theta)) > 0, wm.period(lay, bits, theta)],
           [f"Спектр: {title}", "Пространственный ЦВЗ (один период)"], name)


class OneQuadrant(wm.Layout):
    """Whole object in one quadrant: rows and columns both start at a (no split into halves)."""

    @property
    def positions(self):
        i, j = np.meshgrid(np.arange(self.L), np.arange(self.L), indexing="ij")
        return (self.a + self.step * i).ravel(), (self.a + self.step * j).ravel()


def main():
    setup_plots(20)
    OUT.mkdir(parents=True, exist_ok=True)
    point_study(3, 4, "1_pair.png", "пары коэффициентов")
    point_study([2, 4, 6, 10], [2, 8, 6, 10], "2_pairs_one_quadrant.png", "пар коэффициентов")
    point_study([2, 4, 50, 55], [2, 8, 6, 10], "3_pairs_two_quadrants.png", "пар коэффициентов")
    point_study([10, 50], [4, 50], "4_phase.png", "фазы (φ = π/3)", np.pi / 3)

    bits = wm.binary_object(14, 42)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(bits, cmap="gray", interpolation="nearest"); ax.set_axis_off()
    save_figure(fig, OUT / "5_object.png", dpi=300, tight=False)

    object_study(OneQuadrant(128, 14, 16, 2), bits, "6_object_one_quadrant.png", "объект в одном квадранте")
    object_study(wm.Layout(128, 14, 16, 2), bits, "7_object_halves.png", "две половины объекта")

    img = load_image(C.IMAGE_DIR / "boat.tiff").astype(float)
    panels([img, log_spectrum(np.fft.fft2(img))], ["Исходное изображение", "Спектр (логарифм модуля)"], "8_spectrum.png")
    print("basic studies:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
