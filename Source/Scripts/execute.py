"""Minimal example using the repository's current block watermark."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from Source import config
from Source.Utils import (calculate_q,count_psnr,embed_watermark_into_image,
                          extract_watermark,generate_watermark,configured_phase)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,default=Path(__file__).resolve().parents[2]/'Image'/'1024'/'1024 (50).tif')
    args=parser.parse_args()
    if config.SYNC_ENABLED:
        parser.error('SYNC_ENABLED=True is unsupported: synchronization code is absent from this repository')
    with Image.open(args.image) as source: image=np.array(source.convert('L'))
    bits=generate_watermark(14,42); phi=configured_phase(14); n=config.REGION_SIZE
    layout=dict(x=config.EMBED_X,y=config.EMBED_Y,offset=config.EMBED_OFFSET,gap=config.EMBED_GAP)
    wm=embed_watermark_into_image(image,bits,n,calculate_q(300,14,n),phi,**layout)
    recovered,_,_,used,_=extract_watermark(wm,14,n,phi,5,8,stride=config.EXTRACT_STRIDE,**layout)
    print(f'Phase mode: {config.PHASE_MODE}; block size: {n}; gap: {config.EMBED_GAP}')
    print(f'PSNR: {count_psnr(image,wm):.4f} dB; extraction windows: {used}')
    print('Bit accuracy:',float(np.mean(bits==recovered)))

if __name__=='__main__': main()
