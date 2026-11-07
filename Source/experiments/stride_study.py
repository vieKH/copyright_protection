"""How many blocks are needed: window step B, B/2, ..., B/6 with the grid starting at (0, 0).

    python experiments/stride_study.py --workers 4      ->  Results/stride/stride.csv, stride_summary.csv, fig_stride.png

Method fixed by chapters 2-3: B = 128, L = 21, D = 0.72, a = 28 (rho ~ 0.45), phase voting over all blocks.
Besides the attack-free case, two moderate attacks from chapter 4 are included (noise sigma^2 = 50, JPEG Q = 90).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Source/

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

import wm
from wm import attacks
from experiments import config as C
from experiments.run_all import seed

DIVISORS = [1, 2, 3, 4, 5, 6]                       # step = B / k
CASES = [("none", 0), ("noise", 50), ("jpeg", 90)]
OUT = C.RESULTS / "stride"


def run_trial(job):
    path, trial = job
    name = path.stem
    img = np.asarray(Image.open(path))
    rng = lambda *p: np.random.default_rng(seed(name, trial, *p))
    bits, theta = wm.binary_object(C.L0, seed(name, trial, "bits", C.L0)), wm.phase_key(C.L0, C.PHASE_SEED)
    lay = wm.Layout(C.B0, C.L0, C.A0, C.STEP)
    y, _ = wm.embed(img, lay, bits, theta, C.D0)
    rows = []
    for att, v in CASES:
        z = attacks.apply(y, att, v, rng("noise", v))
        for k in DIVISORS:
            step = int(round(C.B0 / k))
            t0 = time.perf_counter()
            res, n = wm.multi_block_ber(z, lay, bits, theta, step, (0, 0), rng("win", step))
            dt = time.perf_counter() - t0
            rows.append(dict(image=name, trial=trial, attack=att, value=v, k=k, step=step, nblocks=n,
                             ber=res[n]["phase_vote"], seconds=dt))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--trials", type=int, default=C.TRIALS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(p, t) for p in C.image_files() for t in range(args.trials)]
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(run_trial, jobs):
            rows.extend(r)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "stride.csv", index=False)
    per_img = d.groupby(["attack", "k", "step", "nblocks", "image"])[["ber", "seconds"]].mean()
    s = per_img.groupby(["attack", "k", "step", "nblocks"]).mean()
    s["ber"] *= 100
    s["ms"] = s.pop("seconds") * 1000
    s.round(3).to_csv(OUT / "stride_summary.csv")
    print(s.round(2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"font.family": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 12,
                         "axes.grid": True, "grid.alpha": 0.3})
    names = {"none": "без атаки", "noise": "шум, σ² = 50", "jpeg": "JPEG, Q = 90"}
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for (att, mk) in zip(["none", "noise", "jpeg"], ["o-", "s-", "^-"]):
        t = s.loc[att].reset_index()
        ax.plot(t.nblocks, t.ber, mk, label=names[att])
        if att == "none":
            for _, r in t.iterrows():
                ax.annotate(f"B/{int(r.k)}" if r.k > 1 else "B", (r.nblocks, r.ber), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=9)
    ax.set_xscale("log"); ax.set_xticks([16, 49, 81, 169, 225, 361])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}".replace(".", ",")))
    ax.set_xlabel("Число блоков T (шаг блоков)"); ax.set_ylabel("Средняя BER, %"); ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig_stride.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    main()
