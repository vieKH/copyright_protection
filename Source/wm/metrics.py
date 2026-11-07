"""Quality metrics and image statistics."""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))


def psnr(a, b) -> float:
    e = mse(a, b)
    return float("inf") if e == 0 else 10 * np.log10(255 ** 2 / e)


def ssim(a, b) -> float:
    """SSIM with 11x11 Gaussian window, sigma = 1.5 (Wang et al., 2004)."""
    return float(structural_similarity(np.asarray(a, np.float64), np.asarray(b, np.float64), data_range=255,
                                       gaussian_weights=True, sigma=1.5, use_sample_covariance=False))


def image_statistics(img) -> dict:
    f = np.asarray(img, dtype=np.float64)
    hist = np.bincount(np.asarray(img).ravel(), minlength=256) / f.size
    nz = hist[hist > 0]
    h8, w8 = f.shape[0] // 8 * 8, f.shape[1] // 8 * 8
    bstd = f[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8).std(axis=(1, 3))
    gy, gx = np.gradient(f)
    # slope beta of the radial power spectrum S(rho) ~ rho^(-beta), 0.02 < rho < 0.45
    P = np.abs(np.fft.fft2(f - f.mean())) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(f.shape[0]), np.fft.fftfreq(f.shape[1]), indexing="ij")
    r = np.hypot(fy, fx).ravel()
    edges = np.logspace(np.log10(0.02), np.log10(0.45), 25)
    idx = np.digitize(r, edges)
    xs, ys = [], []
    for k in range(1, len(edges)):
        m = idx == k
        if m.any():
            xs.append(np.log10(r[m].mean()))
            ys.append(np.log10(P.ravel()[m].mean()))
    return dict(mean=f.mean(), std=f.std(), entropy=float(-(nz * np.log2(nz)).sum()),
                smooth_frac=float(np.mean(bstd < 2.0)), grad=float(np.hypot(gx, gy).mean()),
                psd_slope=float(-np.polyfit(xs, ys, 1)[0]))
