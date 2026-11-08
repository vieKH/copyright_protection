# copyright_protection

Code for my graduation thesis (ВКР) on an image copyright protection method based on a periodic spectral (DFT) digital watermark. A balanced binary object is written into conjugate pairs of 2-D DFT coefficients with secret random phases. The inverse DFT gives a real noise-like period B×B, which is tiled over the whole image. The original method extracts the object from one arbitrary block using the DFT modulus and a median threshold. The modified method takes many blocks, compensates their phase, and votes on phase projections.

The scripts run every experiment in chapters 2–4 of the thesis and draw the illustrations used in chapter 1.

## Structure

```
Image/                 15 grayscale test images 512x512, 8 bit, TIFF (BOWS-2, USC-SIPI)
Source/
  wm/                  the method library
    embedding.py       Layout (carrier positions), binary_object, phase_key, spectrum, period,
                       embed (D -> alpha via Parseval)
    extraction.py      original single-block method + 4 multi-block decoders
                       (mag_vote, phase_vote, avg_mag, avg_phase), with phase compensation
    attacks.py         contrast, noise, blur, sharpen, jpeg (scaling and rotation are not studied)
    metrics.py         MSE, PSNR, SSIM (Gaussian 11x11, sigma 1.5), image statistics
  experiments/
    config.py          every parameter, the studied ranges and the Russian image names
    common.py          shared helpers: seeds, image loading, matplotlib setup and saving
    basic.py           basic DFT studies on a black image (1.3.2, figures 5-7) -> Results/basic/
    fig_example.py     4x4 object -> periodic watermark (1.3.2, figure 8) -> Results/figures/fig_example.png
    run_all.py         runs every experiment -> Results/raw.csv (one function exp_* per subsection)
    make_figures.py    Results/raw.csv -> Results/figures/*.png (Russian labels), Results/tables/*.csv
                       (one function fig_* per subsection)
    stride_study.py    window step B, B/2, ..., B/6 for the block grid (3.2) -> Results/stride/
  tests/test_core.py   4 quick correctness tests
Results/               precomputed results used in the thesis
```

## How to run (from the `Source` folder)

```bash
pip install -r ../requirements.txt
python tests/test_core.py                     # about 5 s
python experiments/basic.py                   # basic studies on a black image, a few seconds
python experiments/fig_example.py             # figure 8
python experiments/run_all.py --workers 4     # 15 images x 20 trials, about 4-6 min
python experiments/make_figures.py            # figures and tables
python experiments/stride_study.py --workers 4                         # choice of the window step
python experiments/run_all.py --trials 2 --out ../Results/quick.csv   # quick check, keeps raw.csv
python experiments/make_figures.py --raw ../Results/quick.csv          # figures from another CSV
```

You can also run each of these files directly in PyCharm (Run). Python 3.14 was used; the package versions are in `requirements.txt`.

Each row of `raw.csv` is one image × one trial × one condition. Results are averaged over the trials of each image first, then over the images, so every image has the same weight. The random seeds come from (image name, trial number), so the results reproduce exactly. The phase key is the same for all images (`PHASE_SEED` in config.py). Every `.tif`/`.tiff` file in `Image/` is used. To add an image, put it there and give it a Russian name in `NAMES_RU` (config.py), otherwise the scripts stop with an error.

## Outputs and thesis figures

| File | Thesis |
|---|---|
| `Results/basic/5_object.png`, `2_pairs_one_quadrant.png`, `3_pairs_two_quadrants.png` | figures 5, 6, 7 (1.3.2) |
| `Results/figures/fig_example.png` | figure 8 (1.3.2) |
| `Results/figures/dataset.png`, `Results/tables/dataset_stats.csv` | figure 11, table 4 (2.2) |
| `fig_D.png`, `tables/D.csv` | figure 12, table 5 (2.4.1) |
| `smooth.png` | figure 13 (2.4.1) |
| `fig_BL.png`, `fig_B.png`, `fig_BL_ratio.png`, `tables/BL.csv` | figures 14–16, table 6 (2.4.2–2.4.3) |
| `fig_pos.png`, `tables/pos.csv` | figure 17, table 7 (2.4.4) |
| `fig_T.png`, `tables/multi.csv`, `tables/multi_rec.csv` | figure 18, table 9 (3.2) |
| `fig_noise.png`, `fig_sharpen.png`, `tables/attacks.csv` | figures 19, 21, tables 11, 13 (4.3, 4.5) |
| `fig_blur.png`, `fig_jpeg.png`, `tables/pos_attacks.csv` | figures 20, 22, tables 12, 14 (4.4, 4.6) |
| `Results/stride/*` | choice of the window step B/2 (3.2) |

## Basic studies (`basic.py`)

These studies take a black 64×64 image, set DFT coefficients directly, and look at the inverse transform:

- `1_pair.png`: one conjugate pair gives a single 2-D cosine wave.
- `2_pairs_one_quadrant.png`: four pairs in one quadrant give waves of one orientation (figure 6).
- `3_pairs_two_quadrants.png`: the same pairs split between two quadrants give a more isotropic pattern (figure 7).
- `4_phase.png`: changing the phase of the carriers (φ = π/3) shifts the waves.
- `5_object.png`: the balanced binary object 14×14 (figure 5).
- `6_object_one_quadrant.png` / `7_object_halves.png`: the whole object written into one quadrant, compared with the object split into two halves in different quadrants (the layout used by the method).
- `8_spectrum.png`: a test image and its log-magnitude spectrum.

## Recommended parameters

| Parameter | Value | Subsection |
|---|---|---|
| Watermark variance D | 0.72 (PSNR ≈ 49.1 dB, SSIM ≈ 0.994, worst SSIM 0.988) | 2.4.1 |
| Binary object L×L | 21×21 = 441 bits (220 ones) | 2.4.2 |
| DFT block / period B | 128 | 2.4.3 |
| Position in the spectrum | a = 28 (ρ̄ ≈ 0.45): no attacks, weak blur σ ≤ 0.5, JPEG Q ≥ 95 | 2.4.4 |
| | a = 8 (ρ̄ ≈ 0.26): when blur σ ≥ 1 or JPEG Q ≤ 90 is expected | 4.4, 4.6 |
| Extraction | voting of phase projections over 49 blocks 128×128 on a grid with step 64 (B/2) from the image corner | 3.2 |

## Main results (15 images, 20 trials each)

- Without attacks the mean BER falls from 24.1 % (original single-block method) to 1.28 % (phase voting, 49 blocks), about 6 wrong bits out of 441. Extraction takes about 11 ms per image (NumPy, one CPU core).
- Linear contrast change without saturation does not change the decisions at all (checked: BER 1.12–1.32 % for c = 0.5–0.9).
- Gaussian noise with σ² ≤ 10 (attack PSNR ≥ 38 dB): BER ≤ 2 %; σ² = 100: 13.0 %.
- Sharpening (unsharp mask, λ = 5, window 3–15): BER ≤ 1.9 %.
- Blur and JPEG depend on the position in the spectrum. With a = 8, JPEG Q = 70 gives 25.7 % instead of 36.9 % for a = 28. After strong blur or compression the watermark survives only partly; exact recovery of the identifier would need error-correcting coding, which is not studied.
- Scaling and rotation are not studied: they need separate geometric synchronization.
