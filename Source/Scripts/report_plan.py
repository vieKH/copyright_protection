"""Experiments keyed to report's BKP plan, only on Image/{512,1024} TIFF data.

Execute from repository root:
 python -m Source.Scripts.research_report_plan --experiment capacity --count 100
 python -m Source.Scripts.research_report_plan --experiment layouts --count 100
 python -m Source.Scripts.research_report_plan --experiment blocks --count 100
 python -m Source.Scripts.research_report_plan --experiment attacks --count 100
 python -m Source.Scripts.research_report_plan --experiment all --count 100

Never reports a failed decoding or missing image as a valid BER score.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageOps

from Source.Utils.batch_research import DATASETS, apply_attack, derived_seed, random_layout
from Source.Utils.extraction_research import random_extract_start
from Source.Utils.phases import generate_phase_map
from Source.Utils.utils import (calculate_q, count_psnr, embed_watermark_into_image,
                                generate_watermark, qr_to_spectrum_positions)
from Source.Utils.voting_research import compare_methods

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ('capacity','layouts','blocks','attacks','frequency_attacks')
METHODS = ('magnitude_vote','phase_vote','complex_average')
ATTACKS = {
    'no_attack': [0.0],
    'contrast': [0.7,0.85,1.0,1.15,1.3],
    'wh_noise': [400,600,800,1000],
    'gauss_blur': [0.5,1.0,1.5,2.0,3.0],
    'jpeg': [30,50,70,90],
    'scale': [0.5,0.7,0.85,1.0,1.15,1.4,2.0],
    'rotation': [0,5,15,30,45,90,135,180],
}

def layout_candidates(n,l,gap):
    """Vary spectral placement over admissible geometries, including base location."""
    offset = n//16
    valid=[]
    for x in range(2, n//3+1, 3):
        for y in range(2, n//3+1, 3):
            try:
                positions=qr_to_spectrum_positions(l,n,x,y,offset,gap=gap)
                if len(positions)==l*l:
                    valid.append((x,y))
            except (ValueError, IndexError):
                continue
    return valid

def encode(img, n, l, x, y, gap, target_psnr, payload_seed, phase_seed):
    bits=generate_watermark(l,payload_seed)
    phi=generate_phase_map(l,phase_seed)
    # Calculate initial embedding strength via existing Parseval-based convention,
    # then iteratively calibrate to the actual uint8 image (quantization/clipping).
    s=255.0*10**(-target_psnr/20)
    # Repository's calculate_q normally accepts s_param, L, blocksize; calibration
    # uses actual MSE and does not assume all L^2 bits are 1.
    coeff=calculate_q(300.0,l,n)
    best=None
    for _ in range(14):
        wm=embed_watermark_into_image(img,bits,n,coeff,phi,x,y,n//16,gap=gap)
        psnr=float(count_psnr(img,wm))
        if not math.isfinite(psnr):
            coeff *= 1.5
            continue
        best=(wm,bits,phi,psnr,coeff)
        if abs(psnr-target_psnr)<0.08:
            break
        factor=10**((psnr-target_psnr)/20)
        coeff *= max(0.6,min(1.6,factor))
    if best is None:
        raise RuntimeError('PSNR calibration never produced a finite estimate')
    return best

def read_inputs(count):
    if count<1 or count>100:
        raise ValueError('count must be in [1,100] per size')
    entries=[(n,i,ROOT/'Image'/str(n)/f'{n} ({i}).tif')
             for n in (512,1024) for i in range(1,count+1)]
    missing=[str(path) for _,_,path in entries if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} TIFF files absent; first: {missing[:3]}')
    return entries

def hashfile(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def write_csv(path,rows,keys):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        w.writeheader();w.writerows(rows)

def evaluate(wm,bits,phi,n,l,x,y,gap,start_r,start_c,stride,ks,seed):
    outcomes=compare_methods(wm,l,n,phi,x,y,n//16,gap,start_r,start_c,stride,
                             counts=ks,shuffle_seed=seed)
    return [(a.blocks,method,int(np.count_nonzero(getattr(a,method)!=bits)),
             float(np.count_nonzero(getattr(a,method)!=bits)/bits.size))
            for a in outcomes for method in METHODS]

def variants(experiment,n,base_l):
    if experiment=='capacity':
        # Joint L and block-size sweep; constraints are tested rather than assumed.
        for b in sorted({n//16,n//8,n//4}):
            for l in ([5,9,13,17,21,25] if b<=64 else [9,17,21,29,37,41]):
                if l*l < b*b//2: yield b,l,'joint_size'
    elif experiment in ('layouts','frequency_attacks'):
        yield (n//8,base_l,'spectral_layout')
    elif experiment in ('blocks','attacks','frequency_attacks'):
        yield (n//8,base_l,'standard')

def run_one(experiment, size, image_id, path, args):
    img=np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('L'),dtype=np.uint8)
    if img.shape!=(size,size):raise ValueError(f'{path}: unexpected shape {img.shape}')
    base_l=DATASETS[size].qr_size
    rows=[]
    for b,l,kind in variants(experiment,size,base_l):
        gap=args.gap
        layouts=layout_candidates(b,l,gap)
        if not layouts:
            rows.append(dict(dataset=size,image_id=image_id,experiment=experiment,block_size=b,
                             payload_size=l,status='invalid_geometry',reason='no valid locations'))
            continue
        if experiment in ('layouts','frequency_attacks'):
            # Spatial frequency radius stratified, 5 locations spanning observed valid range.
            layouts=sorted(layouts,key=lambda xy: math.hypot(xy[0]/b,xy[1]/b))
            picks=np.linspace(0,len(layouts)-1,num=min(7,len(layouts)),dtype=int)
            layouts=[layouts[int(j)] for j in sorted(set(picks))]
        else:
            idx=derived_seed(args.seed,size,image_id,b,l,'layout')%len(layouts)
            layouts=[layouts[idx]]
        for x,y in layouts:
            condition=f'{experiment}|{size}|{image_id}|{b}|{l}|{x}|{y}'
            base=dict(experiment=experiment,dataset=size,image_id=image_id,
                      block_size=b,payload_size=l,frequency_x=x,frequency_y=y,
                      frequency_radius=round(math.hypot(x/b,y/b),6),
                      target_PSNR_dB=args.psnr_target,original_sha256=hashfile(path),
                      condition=condition,status='ok',reason='')
            try:
                wm,bits,phi,psnr,coeff=encode(img,b,l,x,y,gap,args.psnr_target,
                     args.payload_seed,args.phase_seed)
                start_r,start_c=random_extract_start(b,seed=derived_seed(args.seed,
                      size,image_id,b,l,'extract'),avoid_zero_zero=True)
                stride=b//2 if experiment!='blocks' else b
                overlap_levels=[0.0,0.5,0.75] if experiment=='blocks' else [0.5]
                levels=[('no_attack',0.0)]
                if experiment=='attacks':
                    levels=[(atk,a) for atk,values in ATTACKS.items() for a in values]
                if experiment=='frequency_attacks':
                    levels=[('gauss_blur',s) for s in (0.5,1,1.5,2)] + [('jpeg',q) for q in (50,70,90)]
                for overlap in overlap_levels:
                    stride=max(1,int(round(b*(1-overlap))))
                    for attack,attack_value in levels:
                        try:
                            applied=wm if attack=='no_attack' else apply_attack(wm,attack,attack_value,
                                    derived_seed(args.seed,size,image_id,attack,attack_value,'noise'))
                            applied=np.clip(applied,0,255).astype(np.uint8)
                            ks=args.blocks if experiment in ('blocks','attacks') else [1]
                            values=evaluate(applied,bits,phi,b,l,x,y,gap,start_r,start_c,
                                 stride,ks,derived_seed(args.seed,size,image_id,b,l,'windows'))
                            for k,method,errors,ber in values:
                                rows.append(dict(base,attack=attack,attack_value=attack_value,
                                   overlap=overlap,blocks_used=k,method=method,bit_errors=errors,
                                   total_bits=l*l,BER=ber,accuracy=1-ber,
                                   actual_PSNR_dB=psnr,embedding_coefficient=coeff))
                        except (ValueError,RuntimeError,IndexError) as exc:
                            rows.append(dict(base,attack=attack,attack_value=attack_value,
                                 overlap=overlap,status='error',reason=f'{type(exc).__name__}: {exc}'))
            except (ValueError,RuntimeError,IndexError) as exc:
                rows.append(dict(base,status='error',reason=f'{type(exc).__name__}: {exc}'))
    return rows

def summarize(rows):
    grouped=defaultdict(list)
    for r in rows:
        if r.get('status')!='ok' or 'BER' not in r:continue
        key=(r['experiment'],r['dataset'],r['block_size'],r['payload_size'],
             r['frequency_x'],r['frequency_y'],r['attack'],r['attack_value'],
             r['overlap'],r['blocks_used'],r['method'])
        grouped[key].append(r)
    keys=('experiment','dataset','block_size','payload_size','frequency_x','frequency_y',
          'attack','attack_value','overlap','blocks_used','method')
    result=[]
    for group,records in sorted(grouped.items()):
        vals=np.array([r['BER'] for r in records],dtype=float)
        result.append(dict(zip(keys,group),n_images=len(records),mean_BER=float(vals.mean()),
             std_BER=float(vals.std(ddof=1)) if len(vals)>1 else 0,
             median_BER=float(np.median(vals)),max_BER=float(vals.max()),
             min_BER=float(vals.min()),mean_PSNR_dB=float(np.mean([r['actual_PSNR_dB'] for r in records]))))
    return result

def draw_plots(summary,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for experiment in EXPERIMENTS:
      rr=[r for r in summary if r['experiment']==experiment and r['attack']=='no_attack']
      if experiment in ('attacks','frequency_attacks'):rr=[r for r in summary if r['experiment']==experiment]
      if not rr:continue
      for size in (512,1024):
        for method in METHODS:
          subset=[r for r in rr if r['dataset']==size and r['method']==method]
          if not subset: continue
          if experiment=='capacity': xlabel='Binary object side L'; xx=lambda r:r['payload_size']; groups=[str(x) for x in sorted({r['block_size'] for r in subset})]; grp=lambda r:str(r['block_size'])
          elif experiment in ('layouts','frequency_attacks'):xlabel='Spectral frequency radius';xx=lambda r:r['frequency_x']/r['block_size'];groups=['placement'];grp=lambda r:'placement'
          elif experiment=='blocks':xlabel='Number of blocks';xx=lambda r:r['blocks_used'];groups=[str(x) for x in sorted({r['overlap'] for r in subset})];grp=lambda r:str(r['overlap'])
          else:xlabel='Attack intensity';xx=lambda r:r['attack_value'];groups=sorted({r['attack'] for r in subset});grp=lambda r:r['attack']
          plt.figure(figsize=(8,4.8))
          for group in groups:
            ss=sorted([r for r in subset if grp(r)==group],key=xx)
            if ss:plt.plot([xx(r) for r in ss],[100*r['mean_BER'] for r in ss],marker='.',label=group)
          plt.title(f'{experiment}: {size}x{size}, {method} (TIFF dataset)')
          plt.xlabel(xlabel);plt.ylabel('BER (%)');plt.ylim(bottom=0);plt.grid(alpha=0.3)
          if len(groups)>1:plt.legend(fontsize='small')
          plt.tight_layout();plt.savefig(folder/f'{experiment}_{size}_{method}.png',dpi=180);plt.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=(*EXPERIMENTS,'all'),default='all')
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--output',type=Path,default=ROOT/'Results'/'Report_plan')
    p.add_argument('--seed',type=int,default=20261012)
    p.add_argument('--payload-seed',type=int,default=42)
    p.add_argument('--phase-seed',type=int,default=2026)
    p.add_argument('--gap',type=int,default=1)
    p.add_argument('--psnr-target',type=float,default=49.0)
    p.add_argument('--blocks',type=int,nargs='+',default=[1,4,16,64])
    args=p.parse_args()
    entries=read_inputs(args.count)
    experiments=EXPERIMENTS if args.experiment=='all' else (args.experiment,)
    args.output.mkdir(parents=True,exist_ok=True)
    try:sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):sha=None
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'repo_commit':sha,
              'data_source':'Image/512 and Image/1024 TIFF files only','n_input_images':len(entries),
              'experiments':experiments,'target_PSNR_dB':args.psnr_target,
              'seeds':dict(master=args.seed,payload=args.payload_seed,phase=args.phase_seed),
              'blocks':args.blocks,'input_files':[dict(path=str(path.relative_to(ROOT)),sha256=hashfile(path)) for _,_,path in entries]}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    allrows=[]
    columns=['experiment','dataset','image_id','block_size','payload_size','frequency_x','frequency_y',
           'frequency_radius','target_PSNR_dB','actual_PSNR_dB','original_sha256','condition',
           'attack','attack_value','overlap','blocks_used','method','bit_errors','total_bits',
           'BER','accuracy','embedding_coefficient','status','reason']
    for experiment in experiments:
        runrows=[]
        for i,(size,image_id,path) in enumerate(entries,1):
            runrows.extend(run_one(experiment,size,image_id,path,args))
            if i%5==0:print(f'{experiment}: {i}/{len(entries)} images',flush=True)
            if i%10==0:write_csv(args.output/f'{experiment}_per_image.csv',runrows,columns)
        write_csv(args.output/f'{experiment}_per_image.csv',runrows,columns)
        allrows.extend(runrows)
    summary=summarize(allrows)
    if summary:
        write_csv(args.output/'summary.csv',summary,list(summary[0]))
        draw_plots(summary,args.output)
    errors=[r for r in allrows if r.get('status')!='ok']
    if errors:
        write_csv(args.output/'errors.csv',errors,columns)
    print(f'Completed {len(allrows)} records; errors {len(errors)}; summary groups {len(summary)}')
    if errors:
        raise RuntimeError('Some conditions failed. Check errors.csv before drawing conclusions')

if __name__=='__main__':
    main()
