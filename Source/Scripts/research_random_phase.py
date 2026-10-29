import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from Source.Utils import (
    ImageDistorter, bit_accuracy, calculate_q, count_psnr,
    embed_watermark_into_image, extract_watermark, generate_phase_map, generate_watermark,
)
from Source.Utils.utils import build_wm_spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2021,2022,2023,2024,2025])
    parser.add_argument('--images', nargs='+', default=None, help='Image stems; default: every TIFF recursively in Image/')
    parser.add_argument('--output', type=Path, default=Path('Results/Random_Phase/smoke.csv'))
    parser.add_argument('--limit',type=int,help='Maximum number of input images')
    args = parser.parse_args()
    if args.limit is not None and args.limit<1: parser.error('--limit must be positive')
    root = Path(__file__).resolve().parents[2]
    images = sorted(p for p in (root/'Image').rglob('*') if p.is_file() and p.suffix.lower() in {'.tif','.tiff'})
    if not images: parser.error('No TIFF images found under Image/')
    if args.images is not None:
        available = {p.stem:p for p in images}
        missing = set(args.images)-available.keys()
        if missing:
            parser.error(f'Unknown images: {sorted(missing)}')
        images = [available[name] for name in args.images]
    if args.limit is not None: images=images[:args.limit]
    n, qr_size, gap = 64,14,1
    qr = generate_watermark(qr_size,seed=42)
    q = calculate_q(300,qr_size,n)
    phases = [('fixed','',np.pi/3)]+[('random',seed,generate_phase_map(qr_size,seed)) for seed in args.seeds]
    rows = []
    for path in images:
        image = np.asarray(Image.open(path).convert('L'))
        for mode,seed,phi in phases:
            wm = embed_watermark_into_image(image,qr,n,q,phi,gap=gap)
            spatial = q*np.fft.ifft2(build_wm_spectrum(qr,n,phi,gap=gap)).real
            peak = float(np.max(np.abs(spatial)))
            rms = float(np.sqrt(np.mean(spatial**2)))
            attacks = {
                'none':wm,
                'jpeg_70':ImageDistorter(wm).jpeg(70).image,
                'noise_variance_4':ImageDistorter(wm).white_noise(4,seed=123).image,
                'contrast_0.9':ImageDistorter(wm).contrast(0.9).image,
            }
            for attack,attacked in attacks.items():
                recovered,_,_,used,_ = extract_watermark(attacked,qr_size,n,phi,5,8,stride=32,gap=gap)
                rows.append(dict(image=path.stem,mode=mode,phase_seed=seed,attack=attack,
                                 accuracy=bit_accuracy(qr,recovered),psnr=count_psnr(image,wm),
                                 prequantization_peak=peak,prequantization_rms=rms,
                                 crest_factor=peak/rms,blocks_used=used,qr_seed=42,
                                 qr_size=qr_size,block_size=n,gap=gap,q=q,start_x=5,start_y=8,stride=32))
        print(f'Completed: {path.stem}',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='',encoding='utf-8') as out:
        writer = csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for attack in attacks:
        for mode in ('fixed','random'):
            acc = [r['accuracy'] for r in rows if r['mode']==mode and r['attack']==attack]
            print(f'{attack:18s} {mode:6s}: mean={np.mean(acc):.4f}, min={min(acc):.4f}, max={max(acc):.4f}')
    print(f'Saved: {args.output}')

if __name__ == '__main__':
    main()
