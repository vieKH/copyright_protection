"""Minimal reproducible embedding and overlapping-extraction example.

Run from the repository root: python -m Source.Scripts.execute
"""
from pathlib import Path
import numpy as np
from Source import config
from PIL import Image
from Source.Utils import (
    bit_accuracy, calculate_q, count_psnr, embed_watermark_into_image,
    extract_watermark, generate_watermark,
)


def main():
    image = np.array(Image.open(Path(__file__).resolve().parents[2] / 'Image' / 'lena.tif').convert('L'))
    qr = generate_watermark(14, seed=42)
    phi = np.pi / 3
    n = config.REGION_SIZE
    wm = embed_watermark_into_image(image, qr, n, calculate_q(300, 14, n), phi,
                                    x=config.EMBED_X, y=config.EMBED_Y, offset=config.EMBED_OFFSET)
    recovered, _, _, count, _ = extract_watermark(wm, 14, n, phi, 5, 8, stride=config.EXTRACT_STRIDE,
                                                         x=config.EMBED_X, y=config.EMBED_Y, offset=config.EMBED_OFFSET)
    print(f'Block size: {n}')
    print(f'Embed gap: {config.EMBED_GAP}')
    print(f'PSNR: {count_psnr(image, wm):.4f} dB')
    print(f'Extraction windows: {count}')
    print(f'Bit accuracy: {bit_accuracy(qr, recovered):.6f}')


if __name__ == '__main__':
    main()
