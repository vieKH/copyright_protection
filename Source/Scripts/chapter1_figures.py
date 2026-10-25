from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Source.Utils.utils import (
    build_wm_spectrum, calculate_q, count_psnr, embed_watermark_into_image,
    generate_watermark, qr_to_spectrum_positions,
)
from Source.Utils.phases import generate_phase_map
from Source.Utils.extraction_research import extract_watermark_limited_blocks
from Source.Utils.chapter1_validation import (
    count_ssim, extract_single_block_magnitude, predicted_embed_mse,
    verify_dft_properties,
)

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=Path('Image/512/512 (1).tif'))
    parser.add_argument('--output', type=Path, default=Path('Results/Chapter1'))
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--qr-size', type=int, default=21)
    parser.add_argument('--s-param', type=float, default=300.0)
    parser.add_argument('--gap', type=int, default=1)
    parser.add_argument('--payload-seed', type=int, default=42)
    parser.add_argument('--phase-seed', type=int, default=20261011)
    parser.add_argument('--start-row', type=int, default=5)
    parser.add_argument('--start-col', type=int, default=8)
    args = parser.parse_args(argv)
    file_path = args.image if args.image.is_absolute() else ROOT / args.image
    output = args.output if args.output.is_absolute() else ROOT / args.output
    with Image.open(file_path) as source:
        image = np.asarray(source.convert('L'))
    b, l = args.block_size, args.qr_size
    if image.shape[0] % b or image.shape[1] % b:
        parser.error('Image dimensions must be divisible by the square embedding period')
    qr = generate_watermark(l, seed=args.payload_seed)
    phi = generate_phase_map(l, seed=args.phase_seed)
    q = calculate_q(args.s_param, l, b)
    spectrum = build_wm_spectrum(qr, b, phi, gap=args.gap)
    spatial = q * np.fft.ifft2(spectrum).real
    watermarked = embed_watermark_into_image(image, qr, b, q, phi, gap=args.gap)

    mag = extract_single_block_magnitude(watermarked, l, b,
                                         args.start_row, args.start_col, gap=args.gap)
    # Uses exactly ONE window via the existing phase-aware extractor.
    aware, scores, _, blocks_used, aware_threshold = extract_watermark_limited_blocks(
        watermarked, l, b, phi, args.start_row, args.start_col,
        max_blocks=1, gap=args.gap, stride=b, phase_sign=-1,
    )
    if blocks_used != 1:
        raise RuntimeError('Expected a single-block phase-aware comparison')

    psnr = count_psnr(image, watermarked)
    ssim = count_ssim(image, watermarked)
    actual_mse = float(np.mean((image.astype(float)-watermarked.astype(float))**2))
    theory_mse = predicted_embed_mse(q, b, int(np.count_nonzero(qr)))
    data = {
        'image': str(file_path), 'image_shape': list(image.shape),
        'period': [b, b], 'payload_shape': list(qr.shape),
        'number_of_one_bits': int(np.count_nonzero(qr)),
        'S': args.s_param, 'q': q, 'gap': args.gap,
        'payload_seed': args.payload_seed, 'phase_seed': args.phase_seed,
        'extraction_start': [args.start_row, args.start_col],
        'PSNR_dB': psnr, 'SSIM': ssim,
        'MSE_actual_after_quantization': actual_mse,
        'MSE_theoretical_before_quantization': theory_mse,
        'baseline_one_block_accuracy': float(np.mean(qr == mag.bits)),
        'baseline_one_block_BER': float(np.mean(qr != mag.bits)),
        'baseline_threshold': mag.threshold,
        'phase_aware_one_block_accuracy': float(np.mean(qr == aware)),
        'phase_aware_one_block_BER': float(np.mean(qr != aware)),
        'phase_aware_threshold': aware_threshold,
        'dft_checks': verify_dft_properties(np.arange(17*23).reshape(17, 23)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for key, value in data.items():
            writer.writerow((key, json.dumps(value) if isinstance(value, (dict, list)) else value))

    carrier_map = np.zeros((b,b), dtype=np.float64)
    for p in qr_to_spectrum_positions(l, b, gap=args.gap):
        if qr[p['qr_i'], p['qr_j']]:
            u,v = p['row'],p['col']
            carrier_map[u,v] = 1
            carrier_map[(-u)%b,(-v)%b] = 1
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    panels = [
        (image, 'Original image', 'gray'),
        (watermarked, f'Watermarked, PSNR={psnr:.2f} dB', 'gray'),
        (watermarked.astype(float)-image.astype(float), 'Embedding error (rounded)', 'coolwarm'),
        (np.fft.fftshift(carrier_map), 'Carrier layout (centered FFT)', 'gray'),
        (qr, f'Ground truth ({l*l} bits)', 'gray'),
        (mag.bits, f'One block: magnitude acc={np.mean(qr==mag.bits):.3f}', 'gray'),
        (aware, f'One block: phase acc={np.mean(qr==aware):.3f}', 'gray'),
        (spatial, 'Periodic spatial watermark (float)', 'coolwarm'),
    ]
    for ax, (arr, title, cmap) in zip(axes.flat, panels):
        ax.imshow(arr, cmap=cmap, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'chapter1_overview.png', dpi=180)
    plt.close(fig)

    # Illustrate why independently randomized phases can reduce coherent peaks.
    fixed_phi = np.full((l,l), np.pi/3)
    fixed = q*np.fft.ifft2(build_wm_spectrum(qr,b,fixed_phi,gap=args.gap)).real
    fig, axes = plt.subplots(1,2,figsize=(9,4))
    for ax, pattern, title in zip(axes,(fixed,spatial),('Fixed carrier phases', 'Random per-bit phases')):
        ax.imshow(pattern, cmap='coolwarm')
        rms = float(np.sqrt(np.mean(pattern**2)))
        ax.set_title(f'{title}\npeak / RMS={np.max(np.abs(pattern))/max(rms,1e-12):.2f}')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'chapter1_phase_comparison.png', dpi=180)
    plt.close(fig)
    print('Chapter 1 outputs:', output)
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
