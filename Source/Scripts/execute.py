
from pathlib import Path
import numpy as np
from PIL import Image
from Source.Utils import (
    bit_accuracy, calculate_q, count_psnr, embed_watermark_into_image,
    extract_watermark, generate_watermark,
)


def main():
    image = np.array(Image.open(Path(__file__).resolve().parents[2] / 'Image' / 'lena.tif').convert('L'))
    qr = generate_watermark(14, seed=42)
    phi = np.pi / 3
    wm = embed_watermark_into_image(image, qr, 64, calculate_q(300, 14, 64), phi)
    recovered, _, _, count, _ = extract_watermark(wm, 14, 64, phi, 5, 8, stride=32)
    print(f'PSNR: {count_psnr(image, wm):.4f} dB')
    print(f'Extraction windows: {count}')
    print(f'Bit accuracy: {bit_accuracy(qr, recovered):.6f}')


if __name__ == '__main__':
    main()
