"""Minimal reproducible embedding and overlapping-extraction example.

Run from the repository root: python -m Source.Scripts.execute
"""
from pathlib import Path
import numpy as np
from Source import config
from PIL import Image
from Source.Utils import (
    bit_accuracy, calculate_q, count_psnr, embed_watermark_into_image,
    extract_watermark, generate_watermark, embed_synchronized, extract_synchronized,
)


def main():
    image = np.array(Image.open(Path(__file__).resolve().parents[2] / 'Image' / 'lena.tif').convert('L'))
    qr = generate_watermark(14, seed=42)
    phi = np.pi / 3
    n = config.REGION_SIZE
    layout = dict(x=config.EMBED_X, y=config.EMBED_Y, offset=config.EMBED_OFFSET,
                  gap=config.EMBED_GAP)
    embed = embed_synchronized if config.SYNC_ENABLED else embed_watermark_into_image
    wm = embed(image, qr, n, calculate_q(300,14,n), phi, **layout)
    if config.SYNC_ENABLED:
        result = extract_synchronized(wm,14,n,phi,**layout)
        print(f'Synchronization: {result.estimate}')
        recovered = result.recovered_qr
        count = 'not applicable (global Fourier projections)'
        if recovered is not None:
            print(f'Erased bits: {int((recovered < 0).sum())}')
    else:
        recovered, _, _, count, _ = extract_watermark(
            wm,14,n,phi,5,8,stride=config.EXTRACT_STRIDE,**layout)
    print(f'Block size: {n}')
    print(f'Embed gap: {config.EMBED_GAP}')
    print(f'PSNR: {count_psnr(image, wm):.4f} dB')
    print(f'Extraction windows: {count}')
    print('Bit accuracy (erasures count as errors):',
          'no payload: synchronization rejected' if recovered is None else float(np.mean(qr == recovered)))


if __name__ == '__main__':
    main()
