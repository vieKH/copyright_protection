"""Every experiment of chapters 2-4 in one pass.

    python experiments/run_all.py --workers 4      (from Source/; or python -m experiments.run_all)
    python experiments/run_all.py --trials 2       quick check
One CSV row = one image x one trial x one condition; column `exp` names the experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Source/

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import wm
from wm import attacks, metrics
from experiments import config as C
from experiments.common import load_image, seed


@dataclass
class Trial:
    """One image x one trial: the image and every random quantity derived from (name, trial)."""
    name: str
    trial: int
    img: np.ndarray
    base: dict = field(init=False)

    def __post_init__(self):
        self.base = dict(image=self.name, trial=self.trial)

    def rng(self, *parts) -> np.random.Generator:
        return np.random.default_rng(seed(self.name, self.trial, *parts))

    def bits(self, L: int) -> np.ndarray:
        return wm.binary_object(L, seed(self.name, self.trial, "bits", L))

    @staticmethod
    def theta(L: int) -> np.ndarray:
        return wm.phase_key(L, C.PHASE_SEED)

    def row(self, exp: str, **values) -> dict:
        return dict(self.base, exp=exp, **values)


def decoder_bers(res: dict) -> dict:
    return {f"ber_{d}": res[d] for d in wm.DECODERS}


def spread(lay: wm.Layout) -> dict:
    return dict(rho=lay.rho.mean(), rmin=lay.rho.min(), rmax=lay.rho.max())


# ---------------------------------------------------------------- experiments

def exp_stats(t: Trial):
    """Image statistics (once per image)."""
    if t.trial == 0:
        yield t.row("stats", **metrics.image_statistics(t.img))


def exp_variance(t: Trial):
    """2.4.1  watermark variance D: quality and BER of the original single-block method."""
    lay = wm.Layout(**C.default_layout_args(C.B0, C.L0))
    bits, theta = t.bits(C.L0), t.theta(C.L0)
    for D in C.D_GRID:
        y, _ = wm.embed(t.img, lay, bits, theta, D)
        yield t.row("D", D=D, psnr=metrics.psnr(t.img, y), mse=metrics.mse(t.img, y),
                    ssim=metrics.ssim(t.img, y) if t.trial < 5 else np.nan,
                    ber=wm.single_block_ber(y, lay, bits, t.rng("single", "D", D), C.SINGLE_TRIALS))


def exp_block_object(t: Trial):
    """2.4.2-2.4.3  object size L and block size B."""
    for B in C.B_GRID:
        for L in C.L_GRID:
            lay = wm.Layout(**C.default_layout_args(B, L))
            if not lay.feasible():
                continue
            bits = t.bits(L)
            y, _ = wm.embed(t.img, lay, bits, t.theta(L), C.D0)
            yield t.row("BL", B=B, L=L, **spread(lay),
                        ber=wm.single_block_ber(y, lay, bits, t.rng("single", B, L), C.SINGLE_TRIALS))


def exp_position(t: Trial):
    """2.4.4  position in the spectrum; chapter 4: blur / JPEG for every position."""
    bits, theta = t.bits(C.L0), t.theta(C.L0)
    for a in C.ANCHORS:
        lay = wm.Layout(C.B0, C.L0, a, C.STEP)
        y, _ = wm.embed(t.img, lay, bits, theta, C.D0)
        yield t.row("pos", anchor=a, **spread(lay),
                    ber=wm.single_block_ber(y, lay, bits, t.rng("single", "pos", a), C.SINGLE_TRIALS))
        for att, values in C.POSITION_ATTACKS.items():
            for v in values:
                z = attacks.apply(y, att, v)
                res, n = wm.multi_block_ber(z, lay, bits, theta, C.STRIDE0, C.GRID_START)
                yield t.row("pos_attack", anchor=a, rho=lay.rho.mean(), attack=att, value=v,
                            psnr_attack=metrics.psnr(y, z), **decoder_bers(res[n]))


def exp_multi_block(t: Trial):
    """3.2  number of blocks T (random subsets of a dense pool) and the recommended grid (all blocks, step B/2);
    chapter 4: the other attacks (recommended layout)."""
    lay = wm.Layout(C.B0, C.L0, C.A0, C.STEP)
    bits, theta = t.bits(C.L0), t.theta(C.L0)
    y, _ = wm.embed(t.img, lay, bits, theta, C.D0)
    res, n = wm.multi_block_ber(y, lay, bits, theta, C.DENSE_STRIDE, C.GRID_START, t.rng("pool"), C.COUNTS)
    for T, r in res.items():
        if T in C.COUNTS:
            yield t.row("multi", T=T, **decoder_bers(r))
    res, n = wm.multi_block_ber(y, lay, bits, theta, C.STRIDE0, C.GRID_START)
    yield t.row("multi_rec", T=n, **decoder_bers(res[n]))
    for att, values in [("none", [0])] + list(C.ATTACKS.items()):
        for v in values:
            z = attacks.apply(y, att, v, t.rng("noise", v))
            res, n = wm.multi_block_ber(z, lay, bits, theta, C.STRIDE0, C.GRID_START)
            yield t.row("attack", attack=att, value=v,
                        psnr_attack=metrics.psnr(y, z) if att != "none" else np.nan, **decoder_bers(res[n]))


EXPERIMENTS = [exp_stats, exp_variance, exp_block_object, exp_position, exp_multi_block]


def run_trial(job) -> list[dict]:
    path, trial = job
    t = Trial(path.stem, trial, load_image(path, check=True))
    return [row for exp in EXPERIMENTS for row in exp(t)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--trials", type=int, default=C.TRIALS)
    ap.add_argument("--out", type=Path, default=C.RESULTS / "raw.csv")
    args = ap.parse_args()
    jobs = [(p, t) for p in C.image_files() for t in range(args.trials)]
    t0, rows = time.time(), []
    with ProcessPoolExecutor(args.workers) as ex:
        for k, r in enumerate(ex.map(run_trial, jobs), 1):
            rows.extend(r)
            if k % 20 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} trials, {time.time() - t0:.0f} s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("saved", args.out, len(rows), "rows")


if __name__ == "__main__":
    main()
