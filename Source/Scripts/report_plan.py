"""Experiments keyed to report's BKP plan, only on Image/{512,1024} TIFF data.

Execute from repository root:
 python -m Source.Scripts.report_plan --experiment capacity --count 100
 python -m Source.Scripts.report_plan --experiment layouts --count 100
 python -m Source.Scripts.report_plan --experiment blocks --count 100
 python -m Source.Scripts.report_plan --experiment attacks --count 100
 python -m Source.Scripts.report_plan --experiment all --count 100

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
from functools import lru_cache
import platform
import PIL
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageOps

from Source.Utils.batch_research import DATASETS, apply_attack, derived_seed
from Source.Utils.chapter1_validation import count_ssim
from Source.Utils.extraction_research import random_extract_start
from Source.Utils.phases import generate_phase_map
from Source.Utils.utils import (count_psnr, embed_watermark_into_image,
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

@lru_cache(maxsize=256)
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
    return tuple(valid)

def frequency_radius(n,l,x,y,gap):
    """Mean radial frequency of ALL payload carriers, in cycles/pixel."""
    positions=qr_to_spectrum_positions(l,n,x,y,n//16,gap=gap)
    f=np.fft.fftfreq(n)
    return float(np.mean([math.hypot(f[p['row']],f[p['col']]) for p in positions]))

def fixed_layout(n,l,gap):
    """Same admissible geometry for every image in a block/payload condition."""
    x=y=n//8
    try:
        qr_to_spectrum_positions(l,n,x,y,n//16,gap=gap)
        return x,y
    except ValueError:
        candidates=layout_candidates(n,l,gap)
        if not candidates: raise ValueError('No admissible spectral layout')
        return min(candidates,key=lambda xy: ((xy[0]-x)**2+(xy[1]-y)**2,xy))

def encode(img, n, l, x, y, gap, target_psnr, payload_seed, phase_seed):
    bits=generate_watermark(l,payload_seed)
    phi=generate_phase_map(l,phase_seed)
    if not np.isfinite(target_psnr) or target_psnr<=0:
        raise ValueError('target PSNR must be finite and positive')
    ones=int(bits.sum())
    if not ones: raise ValueError('Payload must contain at least one one-bit')
    # Parseval starting value; calibrate AFTER rounding and clipping.
    coeff=n*n*255*10**(-target_psnr/20)/math.sqrt(2*ones)
    best=None
    for _ in range(32):
        wm=embed_watermark_into_image(img,bits,n,coeff,phi,x,y,n//16,gap=gap)
        psnr=float(count_psnr(img,wm))
        if not math.isfinite(psnr):
            coeff*=1.5
            continue
        if best is None or abs(psnr-target_psnr)<abs(best[3]-target_psnr):
            best=(wm,bits,phi,psnr,coeff)
        if abs(psnr-target_psnr)<0.08: return best
        coeff*=max(0.6,min(1.6,10**((psnr-target_psnr)/20)))
    raise RuntimeError('PSNR calibration did not reach tolerance 0.08 dB; best='+str(None if best is None else best[3]))

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

def variants(experiment,n,base_l,gap=1):
    if experiment=='capacity':
        # Joint L and block-size sweep; constraints are tested rather than assumed.
        for b in sorted({n//16,n//8,n//4}):
            for l in ([5,9,13,17,21,25] if b<=64 else [9,17,21,29,37,41]):
                if l*l < b*b//2 and layout_candidates(b,l,gap): yield b,l,'joint_size'
    elif experiment in ('layouts','frequency_attacks'):
        yield (n//8,base_l,'spectral_layout')
    elif experiment in ('blocks','attacks'):
        yield (n//8,base_l,'standard')

def run_one(experiment, size, image_id, path, args):
    img=np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('L'),dtype=np.uint8)
    if img.shape!=(size,size):raise ValueError(f'{path}: unexpected shape {img.shape}')
    base_l=DATASETS[size].qr_size
    rows=[]
    for b,l,kind in variants(experiment,size,base_l,args.gap):
        gap=args.gap
        layouts=layout_candidates(b,l,gap)
        if not layouts:
            rows.append(dict(dataset=size,image_id=image_id,experiment=experiment,block_size=b,
                             payload_size=l,status='invalid_geometry',reason='no valid locations'))
            continue
        if experiment in ('layouts','frequency_attacks'):
            # Spatial frequency radius stratified, 5 locations spanning observed valid range.
            layouts=sorted(layouts,key=lambda xy: frequency_radius(b,l,*xy,gap))
            picks=np.linspace(0,len(layouts)-1,num=min(7,len(layouts)),dtype=int)
            layouts=[layouts[int(j)] for j in sorted(set(picks))]
        else:
            layouts=[fixed_layout(b,l,gap)]
        for x,y in layouts:
            condition=f'{experiment}|{size}|{image_id}|{b}|{l}|{x}|{y}'
            base=dict(experiment=experiment,dataset=size,image_id=image_id,
                      block_size=b,payload_size=l,frequency_x=x,frequency_y=y,
                      frequency_radius=round(frequency_radius(b,l,x,y,gap),9),
                      target_PSNR_dB=args.psnr_target,original_sha256=hashfile(path),
                      condition=condition,status='ok',reason='')
            try:
                wm,bits,phi,psnr,coeff=encode(img,b,l,x,y,gap,args.psnr_target,
                     args.payload_seed,args.phase_seed)
                mse=float(np.mean((wm.astype(float)-img.astype(float))**2))
                ssim=count_ssim(img,wm)
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
                                   actual_PSNR_dB=psnr,embedding_coefficient=coeff,
                                   embedding_MSE=mse,embedding_SSIM=ssim))
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
             min_BER=float(vals.min()),mean_PSNR_dB=float(np.mean([r['actual_PSNR_dB'] for r in records])),
             frequency_radius=float(records[0]['frequency_radius']),
             mean_SSIM=float(np.mean([r['embedding_SSIM'] for r in records]))))
    return result

def draw_plots(summary,folder):
    """One curve per comparable condition; separate unrelated attack units."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for experiment in EXPERIMENTS:
      for size in (512,1024):
        for method in METHODS:
          subset=[r for r in summary if r['experiment']==experiment and r['dataset']==size and r['method']==method]
          if not subset: continue
          panels=sorted({r['attack'] for r in subset}) if experiment=='attacks' else ['all']
          for panel in panels:
            ss=[r for r in subset if panel=='all' or r['attack']==panel]
            curves=defaultdict(list)
            for r in ss:
              if experiment=='capacity':
                xlabel='Binary object side L'; xx=r['payload_size']; group=f"DFT B={r['block_size']}"
              elif experiment in ('layouts','frequency_attacks'):
                xlabel='Mean carrier radius (cycles/pixel)'; xx=r['frequency_radius']; group=f"{r['attack']}={r['attack_value']}, K={r['blocks_used']}"
              elif experiment=='blocks':
                xlabel='Number of extraction windows K'; xx=r['blocks_used']; group=f"overlap={r['overlap']}"
              else:
                xlabel=f'{panel}: attack parameter'; xx=r['attack_value']; group=f"K={r['blocks_used']}, overlap={r['overlap']}"
              curves[group].append((xx,r))
            fig,ax=plt.subplots(figsize=(8,4.8))
            for label,points in sorted(curves.items()):
              points.sort(key=lambda t:t[0])
              ax.plot([t[0] for t in points],[100*t[1]['mean_BER'] for t in points],'.-',label=label)
            ax.set(title=f'{experiment}: {size}x{size}, {method}',xlabel=xlabel,ylabel='Mean BER (%)',ylim=(0,100))
            ax.grid(alpha=.3)
            if len(curves)>1: ax.legend(fontsize='small')
            fig.tight_layout(); suffix='' if panel=='all' else '_'+panel
            fig.savefig(folder/f'{experiment}_{size}_{method}{suffix}.png',dpi=160); plt.close(fig)

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
    if not math.isfinite(args.psnr_target) or args.psnr_target<=0: p.error('--psnr-target must be finite and positive')
    if args.gap<0 or min(args.seed,args.payload_seed,args.phase_seed)<0: p.error('Gap and seeds must be nonnegative')
    if min(args.blocks)<1: p.error('--blocks must be positive')
    if args.output.exists() and any(args.output.iterdir()): p.error('Output directory must be empty')
    entries=read_inputs(args.count)
    experiments=EXPERIMENTS if args.experiment=='all' else (args.experiment,)
    args.output.mkdir(parents=True,exist_ok=True)
    try:sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):sha=None
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'repo_commit':sha,
              'data_source':'Image/512 and Image/1024 TIFF files only','n_input_images':len(entries),
              'experiments':experiments,'target_PSNR_dB':args.psnr_target,
              'seeds':dict(master=args.seed,payload=args.payload_seed,phase=args.phase_seed),
              'status':'running','blocks':args.blocks,'gap':args.gap,'calibration_tolerance_dB':0.08,
              'layout_protocol':'fixed geometry per block/payload; explicit placement sweep',
              'frequency_radius':'mean signed FFT carrier radius, cycles/pixel',
              'versions':dict(python=platform.python_version(),numpy=np.__version__,pillow=PIL.__version__),
              'code_sha256':{str(p.relative_to(ROOT)):hashfile(p) for p in sorted((ROOT/'Source').rglob('*.py'))},
              'input_files':[dict(path=str(path.relative_to(ROOT)),sha256=hashfile(path)) for _,_,path in entries]}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    allrows=[]
    columns=['experiment','dataset','image_id','block_size','payload_size','frequency_x','frequency_y',
           'frequency_radius','target_PSNR_dB','actual_PSNR_dB','original_sha256','condition',
           'attack','attack_value','overlap','blocks_used','method','bit_errors','total_bits',
           'BER','accuracy','embedding_coefficient','embedding_MSE','embedding_SSIM','status','reason']
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
    skipped=[r for r in allrows if r.get('status')=='invalid_geometry']
    if skipped: write_csv(args.output/'skipped.csv',skipped,columns)
    errors=[r for r in allrows if r.get('status')=='error']
    manifest.update(status='failed' if errors else 'complete',n_records=len(allrows),n_errors=len(errors),n_skipped=len(skipped))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if errors:
        write_csv(args.output/'errors.csv',errors,columns)
    print(f'Completed {len(allrows)} records; errors {len(errors)}; summary groups {len(summary)}')
    if errors:
        raise RuntimeError('Some conditions failed. Check errors.csv before drawing conclusions')

if __name__=='__main__':
    main()
