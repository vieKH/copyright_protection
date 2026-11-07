"""Figures (Results/figures/*.png) and summary tables (Results/tables/*.csv) from Results/raw.csv.

    python experiments/make_figures.py        (from Source/)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Source/

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

import wm
from experiments import config as C
from experiments.common import load_image, save_figure, setup_plots

FIG, TAB = C.RESULTS / "figures", C.RESULTS / "tables"
BLUE = "#1f5fa8"
BER_LABEL = "Средняя BER, %"
DEC = {"mag_vote": "голосование, модуль", "phase_vote": "голосование, фазовая проекция",
       "avg_mag": "усреднение, модуль", "avg_phase": "усреднение, фазовая проекция"}
MARK = {"mag_vote": "o--", "phase_vote": "s-", "avg_mag": "^--", "avg_phase": "D-"}
COMMA = FuncFormatter(lambda v, p: f"{v:g}".replace(".", ","))


def ru(s: str) -> str:
    """Decimal comma."""
    return s.replace(".", ",")


def save(fig, name):
    """Plot with decimal commas on both axes."""
    for ax in fig.axes:
        ax.xaxis.set_major_formatter(COMMA); ax.yaxis.set_major_formatter(COMMA)
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
    save_figure(fig, FIG / f"{name}.png", dpi=220)


def per_image_mean(df, keys, col):
    """Mean over trials of every image, then over images (every image has the same weight)."""
    return df.groupby(keys + ["image"])[col].mean().groupby(keys).mean()


def line(x, y, name, xlabel, ylabel=BER_LABEL, logx=False, xticks=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(x, y, "o-", color=BLUE)
    if logx: ax.set_xscale("log")
    if xticks is not None: ax.set_xticks(xticks)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_ylim(bottom=0)
    save(fig, name)


# ---------------------------------------------------------------- dataset

def fig_dataset(files):
    n = len(files); cols = 5; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(11, 2.35 * rows))
    for ax in axs.ravel(): ax.set_axis_off()
    for ax, p in zip(axs.ravel(), files):
        ax.imshow(load_image(p), cmap="gray", vmin=0, vmax=255)
        ax.set_title(C.NAMES_RU.get(p.stem, p.stem), fontsize=11)
    save_figure(fig, FIG / "dataset.png", dpi=200)


def table_stats(d):
    st = d[d.exp == "stats"].set_index("image").dropna(axis=1, how="all").drop(columns=["trial", "exp"])
    st.insert(0, "name_ru", [C.NAMES_RU.get(k, k) for k in st.index])
    st.round(4).to_csv(TAB / "dataset_stats.csv")
    return st


def fig_smooth(files, st):
    """Smooth fragment of the smoothest image: original / D0 / D = 16 / difference x20."""
    name = st.smooth_frac.idxmax()
    img = load_image(next(p for p in files if p.stem == name))
    lay = wm.Layout(**C.default_layout_args(C.B0, C.L0))
    bits, th = wm.binary_object(C.L0, C.DEMO_BITS_SEED), wm.phase_key(C.L0, C.PHASE_SEED)
    y1, _ = wm.embed(img, lay, bits, th, C.D0); y2, _ = wm.embed(img, lay, bits, th, 16.0)
    # 3x3 group of 64x64 blocks with the smallest standard deviation
    h, w = img.shape[0] // 64, img.shape[1] // 64
    blk = img.astype(float)[:h * 64, :w * 64].reshape(h, 64, -1, 64).std(axis=(1, 3))
    r, c = np.unravel_index(np.argmin(blk[:-3, :-3] + blk[1:-2, 1:-2] + blk[3:, 3:]), blk[:-3, :-3].shape)
    sl = (slice(64 * r, 64 * r + 192), slice(64 * c, 64 * c + 192))
    titles = ["Исходный фрагмент", ru(f"D = {C.D0:g} (PSNR {wm.metrics.psnr(img, y1):.1f} дБ)"),
              ru(f"D = 16 (PSNR {wm.metrics.psnr(img, y2):.1f} дБ)"), ru(f"Разность при D = {C.D0:g} (×20)")]
    ims = [img[sl], y1[sl], y2[sl], np.clip(128 + 20 * (y1[sl].astype(int) - img[sl]), 0, 255)]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, im, t in zip(axs, ims, titles):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.set_axis_off(); ax.set_title(t, fontsize=11)
    save_figure(fig, FIG / "smooth.png", dpi=200)


# ---------------------------------------------------------------- chapter 2

def fig_variance(d):
    """2.4.1"""
    D = d[d.exp == "D"]
    t = pd.DataFrame({k: per_image_mean(D, ["D"], k) for k in ["psnr", "mse", "ssim", "ber"]})
    t.to_csv(TAB / "D.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(t.index, t.psnr, "o-", color=BLUE); ax[1].plot(t.index, t.ssim, "o-", color=BLUE)
    for a in ax: a.set_xscale("log"); a.set_xlabel("Дисперсия ЦВЗ D"); a.axvline(C.D0, color="gray", ls=":", lw=1)
    ax[0].axhline(40, color="gray", ls="--", lw=1); ax[0].set_ylabel("PSNR, дБ"); ax[1].set_ylabel("SSIM")
    save(fig, "fig_D")


def fig_block_object(d):
    """2.4.2-2.4.3"""
    t = per_image_mean(d[d.exp == "BL"], ["B", "L"], "ber").unstack("B") * 100
    t.to_csv(TAB / "BL.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for B, mk in zip(C.B_GRID, ["o-", "s-", "^-"]):
        ax.plot(t.index, t[B], mk, label=f"B = {B}")
    ax.set_xticks(C.L_GRID); ax.set_xlabel("Сторона бинарного объекта L"); ax.set_ylabel(BER_LABEL); ax.legend()
    save(fig, "fig_BL")
    line(C.B_GRID, [t.loc[C.L0, B] for B in C.B_GRID], "fig_B", "Размер блока ДПФ B", xticks=C.B_GRID)
    tt = t.stack().reset_index(); tt.columns = ["L", "B", "ber"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(tt.B / tt.L, tt.ber, color=BLUE, s=28); ax.set_xscale("log")
    ax.set_xlabel("Отношение B / L"); ax.set_ylabel(BER_LABEL); ax.set_ylim(bottom=0)
    save(fig, "fig_BL_ratio")


def fig_position(d):
    """2.4.4"""
    t = pd.DataFrame({k: per_image_mean(d[d.exp == "pos"], ["anchor"], k) for k in ["rmin", "rho", "rmax", "ber"]})
    t.to_csv(TAB / "pos.csv")
    line(t.rho, t.ber * 100, "fig_pos", r"Средняя нормированная частота носителей $\bar\rho$")


# ---------------------------------------------------------------- chapter 3

def fig_multi_block(d):
    """3.2  BER versus the number of blocks T; the recommended grid (step B/2, all blocks)."""
    cols = [f"ber_{k}" for k in DEC]
    t = pd.concat({c: per_image_mean(d[d.exp == "multi"], ["T"], c) for c in cols}, axis=1) * 100
    rec = pd.concat({c: per_image_mean(d[d.exp == "multi_rec"], ["T"], c) for c in cols}, axis=1) * 100
    t.to_csv(TAB / "multi.csv"); rec.to_csv(TAB / "multi_rec.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for k, lab in DEC.items():
        ax.plot(t.index, t[f"ber_{k}"], MARK[k], label=lab)
    ax.set_xscale("log", base=2); ax.set_xticks(C.COUNTS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_xlabel("Число блоков T"); ax.set_ylabel(BER_LABEL); ax.set_ylim(bottom=0); ax.legend(fontsize=10)
    save(fig, "fig_T")


# ---------------------------------------------------------------- chapter 4

def fig_attacks(d):
    t = pd.DataFrame({k: per_image_mean(d[d.exp == "attack"], ["attack", "value"], k)
                      for k in ["ber_phase_vote", "psnr_attack"]})
    t.to_csv(TAB / "attacks.csv")
    line(t.loc["noise"].index, t.loc["noise"].ber_phase_vote * 100, "fig_noise", "Дисперсия шума σ²", logx=True)
    line(t.loc["sharpen"].index, t.loc["sharpen"].ber_phase_vote * 100, "fig_sharpen", "Размер окна M",
         xticks=C.ATTACKS["sharpen"])


def fig_position_attacks(d):
    t = per_image_mean(d[d.exp == "pos_attack"], ["attack", "value", "anchor"], "ber_phase_vote").unstack("anchor") * 100
    t.to_csv(TAB / "pos_attacks.csv")
    rho = per_image_mean(d[d.exp == "pos"], ["anchor"], "rho")
    for att, xl in [("blur", "Параметр размытия σ"), ("jpeg", "Параметр качества JPEG Q")]:
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for a in C.ANCHORS:
            ax.plot(t.loc[att].index, t.loc[att][a], "o-", label=ru(rf"$\bar\rho$ = {rho[a]:.2f}"))
        ax.axhline(50, color="gray", ls=":", lw=1); ax.set_xlabel(xl); ax.set_ylabel(BER_LABEL)
        ax.legend(fontsize=9, ncol=2)
        save(fig, f"fig_{att}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=C.RESULTS / "raw.csv")
    args = ap.parse_args()
    setup_plots(12, **{"axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 1.6, "lines.markersize": 5})
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.raw)
    files = C.image_files()

    fig_dataset(files)
    fig_smooth(files, table_stats(d))
    fig_variance(d)
    fig_block_object(d)
    fig_position(d)
    fig_multi_block(d)
    fig_attacks(d)
    fig_position_attacks(d)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
