"""Reproduce fair comparisons on the two 100-image TIFF sets.

Example:
 python -m Source.Scripts.research_voting_200
 python -m Source.Scripts.research_voting_200 --count 2 --blocks 1 4 16 64

Outputs per-image observations, groupwise summaries and a complete run manifest.
Clean watermarked images only: attacks are a separate controlled experiment.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import subprocess

import numpy as np
from PIL import Image, ImageOps

from Source.Utils.batch_research import DATASETS, derived_seed, random_layout
from Source.Utils.extraction_research import random_extract_start
from Source.Utils.phases import generate_phase_map
from Source.Utils.utils import (calculate_q, count_psnr,
                                embed_watermark_into_image, generate_watermark)
from Source.Utils.voting_research import compare_methods

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('magnitude_vote', 'phase_vote', 'complex_average')


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_out(path, rows, headers):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def parse():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root',type=Path,default=ROOT/'Image')
    parser.add_argument('--output',type=Path,default=ROOT/'Results'/'Voting_Research'/'comparison_200')
    parser.add_argument('--count',type=int,default=100,help='per resolution, default full 200 images')
    parser.add_argument('--groups',type=int,nargs='+',choices=(512,1024),default=[512,1024])
    parser.add_argument('--blocks',type=int,nargs='+',default=[1,4,16,64])
    parser.add_argument('--seed',type=int,default=20261012)
    parser.add_argument('--payload-seed',type=int,default=42)
    parser.add_argument('--phase-seed',type=int,default=20261011)
    parser.add_argument('--gap',type=int,default=1)
    parser.add_argument('--s-param',type=float,default=300)
    parser.add_argument('--ordered-blocks',action='store_true',help='disable shared seeded random block order (for diagnostics)')
    return parser.parse_args()


def main():
    args=parse()
    if args.count < 1 or args.count > 100:
        raise ValueError('count must lie in 1..100')
    if any(k < 1 for k in args.blocks):
        raise ValueError('block counts must be positive')
    entries = [(size, i, args.image_root/str(size)/f'{size} ({i}).tif')
               for size in args.groups for i in range(1,args.count+1)]
    missing = [str(path) for _,_,path in entries if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} image files, first: {missing[:3]}')
    out = args.output
    if out.exists() and any(out.iterdir()): raise ValueError('Output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    data=[]
    manifest=dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        groups=args.groups, images_per_group=args.count, n_images=len(entries),
        blocks_requested=args.blocks, seed=args.seed,
        payload_seed=args.payload_seed,phase_seed=args.phase_seed,
        gap=args.gap,s_param=args.s_param,shuffle_blocks=not args.ordered_blocks,
        phase_vote='per-window known-phase projection, median threshold, majority (tie=0)',
        magnitude_vote='per-window FFT magnitudes, median threshold, majority (tie=0)',
        complex_average='phase-compensated complex coefficient mean, projection, single median threshold',
        data_scope='clean watermarked images; no attacks',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)):_hash(p) for p in sorted((ROOT/'Source').rglob('*.py'))},
        python=platform.python_version(),numpy=np.__version__,
        inputs=[])
    for idx, (size, image_id, path) in enumerate(entries,1):
        spec=DATASETS[size]
        n, l=spec.block_size,spec.qr_size
        image_hash=_hash(path)
        with Image.open(path) as source:
            image=np.asarray(ImageOps.exif_transpose(source).convert('L'))
        if image.shape!=(size,size):
            raise ValueError(f'Unexpected shape {path}: {image.shape}')
        qr=generate_watermark(l, args.payload_seed)
        phi=generate_phase_map(l,args.phase_seed)
        layout_seed=derived_seed(args.seed,size,image_id,1,'layout')
        extraction_seed=derived_seed(args.seed,size,image_id,1,'extraction')
        x,y=random_layout(n,l,args.gap,layout_seed)
        start_row,start_col=random_extract_start(n,seed=extraction_seed,avoid_zero_zero=True)
        q=calculate_q(args.s_param,l,n)
        watermarked=embed_watermark_into_image(image,qr,n,q,phi,x,y,n//16,gap=args.gap)
        psnr=count_psnr(image,watermarked)
        shuffled=(None if args.ordered_blocks else derived_seed(args.seed,size,image_id,1,'windows'))
        comparisons=compare_methods(
            watermarked,l,n,phi,x,y,n//16,args.gap,start_row,start_col,
            n//2,counts=args.blocks,shuffle_seed=shuffled)
        manifest['inputs'].append(dict(dataset=size,id=image_id,filename=path.name,sha256=image_hash))
        for result in comparisons:
            for method in METHODS:
                decoded=getattr(result,method)
                errors=int(np.count_nonzero(qr!=decoded))
                nbits=int(qr.size)
                data.append(dict(dataset=size,image_id=image_id,filename=path.name,
                                 block_size=n,payload_size=l,total_bits=nbits,
                                 blocks_used=result.blocks,method=method,
                                 wrong_bits=errors,BER=errors/nbits,
                                 bit_accuracy=1-errors/nbits,
                                 PSNR_dB=psnr,embed_x=x,embed_y=y,
                                 extract_row=start_row,extract_col=start_col,
                                 tie_bits=(getattr(result,method+'_ties') if method!='complex_average' else ''),
                                 image_sha256=image_hash))
        print(f'[{idx}/{len(entries)}] {size} ({image_id}) PSNR={psnr:.3f} dB',flush=True)
    row_keys=list(data[0])
    csv_out(out/'per_image.csv',data,row_keys)
    summary=[]
    for size in args.groups:
        for k in sorted(set(r['blocks_used'] for r in data if r['dataset']==size)):
            for method in METHODS:
                rr=[row for row in data if row['dataset']==size and row['blocks_used']==k and row['method']==method]
                if not rr:continue
                vals=np.asarray([row['BER'] for row in rr],dtype=float)
                summary.append(dict(dataset=size,blocks_used=k,method=method,images=len(rr),
                                    mean_BER=float(vals.mean()),std_BER=float(vals.std(ddof=1)) if len(vals)>1 else 0,
                                    median_BER=float(np.median(vals)),max_BER=float(vals.max()),
                                    min_BER=float(vals.min()),mean_accuracy=float(1-vals.mean()),
                                    mean_PSNR_dB=float(np.mean([row['PSNR_dB'] for row in rr]))))
    csv_out(out/'summary.csv',summary,list(summary[0]))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('SAVED',out, 'rows:',len(data),'groups:',len(summary),flush=True)
    for s in summary:
        print(f"{s['dataset']} K={s['blocks_used']:>3} {s['method']:>18} BER={s['mean_BER']:.6f}")

if __name__=='__main__':
    main()
