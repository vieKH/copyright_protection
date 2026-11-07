# copyright_protection

This is my research on an image copyright protection method based on a periodic spectral (DFT) digital watermark. The code runs every experiment in chapters 2–4 of the report (ВКР BKP_87).

## Structure

```
Image/                 15 grayscale test images 512x512 (BOWS-2, USC-SIPI)
Source/
  wm/                  the method library
    embedding.py       Layout (carrier positions), binary_object, phase_key, spectrum, period,
                       embed (D -> alpha via Parseval)
    extraction.py      original single-block method + 4 multi-block decoders
                       (mag_vote, phase_vote, avg_mag, avg_phase), with phase compensation
    attacks.py         contrast, noise, blur, sharpen, jpeg (scale and rotation are not studied)
    metrics.py         MSE, PSNR, SSIM (Gaussian 11x11, sigma 1.5), image statistics
  experiments/
    config.py          every parameter, studied range and the Russian image names
    common.py          shared helpers: seeds, image loading, matplotlib setup and saving
    basic.py           basic DFT studies on a black image (chapter 1, figures 5-7) -> Results/basic/
    run_all.py         runs every experiment -> Results/raw.csv (one function exp_* per section)
    make_figures.py    Results/raw.csv -> Results/figures/*.png (Russian labels), Results/tables/*.csv
                       (one function fig_* per section)
  tests/test_core.py   4 quick correctness tests
Results/               precomputed results used in the report
```

## How to run (from the `Source` folder)

```bash
pip install -r ../requirements.txt
python tests/test_core.py                     # about 5 s
python experiments/basic.py                   # basic studies on a black image, a few seconds
python experiments/run_all.py --workers 4     # 15 images x 20 trials, about 4-6 min
python experiments/make_figures.py            # figures and tables
python experiments/run_all.py --trials 2 --out ../Results/quick.csv   # quick check, keeps raw.csv
python experiments/make_figures.py --raw ../Results/quick.csv          # figures from another CSV
```

In PyCharm you can also run each of these files directly (Run).

Each row of `raw.csv` is one image × one trial × one condition. Results are averaged over the trials of each image first, then over the images, so every image has the same weight. The random seeds are derived from (image name, trial number), so results reproduce exactly. Every `.tif`/`.tiff` file in `Image/` is used; to add an image, put it there and give it a Russian name in `NAMES_RU` (config.py), otherwise the scripts stop with an error.

## Basic studies (`basic.py`)

These studies take a black 64×64 image, change the DFT coefficients directly, and look at the inverse transform:

- `1_pair.png`: one conjugate pair gives a single 2-D cosine wave.
- `2_pairs_one_quadrant.png`: four pairs in one quadrant give waves of one orientation (figure 6 of the report).
- `3_pairs_two_quadrants.png`: the same pairs split between two quadrants give a more isotropic pattern (figure 7).
- `4_phase.png`: changing the phase of the carriers (φ = π/3) shifts the waves.
- `5_object.png`: the balanced binary object 14×14 (figure 5).
- `6_object_one_quadrant.png` / `7_object_halves.png`: the whole object written into one quadrant, compared with the object split into two halves in different quadrants (the layout used by the method).
- `8_spectrum.png`: a test image and its log-magnitude spectrum.

## Recommended parameters

| Parameter | Value | Section |
|---|---|---|
| Watermark variance D | 0.72 (PSNR ≈ 49.1 dB, SSIM ≈ 0.994) | 2.4.1 |
| Binary object L×L | 21×21 = 441 bits | 2.4.2 |
| DFT block / period B | 128 | 2.4.3 |
| Position in the spectrum | a = 28 (ρ̄ ≈ 0.45), no attacks / weak blur / JPEG Q ≥ 95 | 2.4.4 |
| | a = 8 (ρ̄ ≈ 0.26) when blur σ ≥ 1 or JPEG Q ≤ 90 is expected | 4.4, 4.6 |
| Extraction | phase-projection voting, all 128×128 blocks with step 64 (50 % overlap) | 3.2 |

Main result (15 images): the mean BER without attack falls from 24.1 % (original single-block method) to 2.55 % (phase voting, about 31 blocks). Extraction takes about 10 ms per image.
