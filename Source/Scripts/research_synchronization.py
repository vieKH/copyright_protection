"""Compare legacy, uncorrected sync, blind sync, and known-geometry reference.

Run: python -m Source.Scripts.research_synchronization
The decoder receives neither the original image, payload bits nor attack truth.
Truth is used ONLY for scoring and the explicitly named oracle reference.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from Source import config
from Source.Utils.utils import generate_watermark, calculate_q, count_psnr, embed_watermark_into_image
from Source.Utils.extraction_research import extract_watermark
from Source.Utils.synchronization import (
    default_sync_settings, embed_synchronized, extract_synchronized,
    SyncEstimate, _decode_registered,
)
from Source.Utils.run_output import create_run_directory

ROOT=Path(__file__).resolve().parents[2]


def geometry_cases():
    cases=[dict(name='clean')]
    for a in [-120,-25,-12,7,23,90,170]:
        cases.append(dict(name=f'rotate_{a}',angle=a))
    for scale in [.8,.85,.95,1.05,1.15,1.2]:
        cases.append(dict(name=f'scale_{scale}',scale=scale))
    for t in [(17,-23),(-39,51),(133,-145)]:
        cases.append(dict(name=f'translate_{t[0]}_{t[1]}',translation=t))
    for h,w,top,left in [(384,384,31,67),(320,352,43,71),(256,256,31,67)]:
        cases.append(dict(name=f'crop_{h}x{w}',shape=(h,w),top_left=(top,left)))
    cases.append(dict(name='rotate_scale_crop',angle=13,scale=1.1,
                      shape=(384,416),translation=(17,-23)))
    return cases


def apply_geometry(image, case):
    # Independent affine attack implementation; no synchronization helpers.
    shape=tuple(case.get('shape',image.shape))
    angle=float(case.get('angle',0));scale=float(case.get('scale',1))
    a=np.deg2rad(angle)
    inverse=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])/scale
    centre=(np.array(image.shape)-1)/2
    output_centre=(np.array(shape)-1)/2
    origin=centre-inverse@np.array(case.get('translation',(0,0)))
    if 'top_left' in case:
        origin=np.array(case['top_left'])+output_centre
    shifted=origin-inverse@output_centre
    output=ndimage.affine_transform(image.astype(float),inverse,offset=shifted,
                                    output_shape=shape,order=1,mode='constant',cval=0,
                                    prefilter=False)
    return np.rint(np.clip(output,0,255)).astype(np.uint8),angle,scale,origin


def accuracy(bits, recovered):
    return '' if recovered is None else float(np.mean(bits==recovered))


def legacy_extract(image, qr_size,n,phi,layout):
    try:
        return extract_watermark(image,qr_size,n,phi,5,8,stride=config.EXTRACT_STRIDE,**layout)[0]
    except ValueError as exc:
        if 'No blocks' not in str(exc):raise
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',nargs='+',default=['lena','goldhilf'])
    parser.add_argument('--full',action='store_true',help='Also run all legacy configured attacks; no oracle for these')
    args=parser.parse_args()
    output=Path(create_run_directory(str(ROOT/'Results'/'Synchronization_Research'),config.EMBED_GAP))
    settings=default_sync_settings()
    n=config.REGION_SIZE;qr_size=14;phi=np.pi/3
    bits=generate_watermark(qr_size,seed=42)
    q=calculate_q(300,qr_size,n)
    layout=dict(x=config.EMBED_X,y=config.EMBED_Y,offset=config.EMBED_OFFSET,gap=config.EMBED_GAP)
    metadata=dict(base_commit='273cc3568c42c284af66d57f8c757f260a3083ad',
                  sync=asdict(settings),region_size=n,qr_size=qr_size,qr_seed=42,
                  q=q,phi=phi,layout=layout,images=args.images,
                  note='accuracy counts erasures as errors; rejected payload accuracy is blank')
    (output/'config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    rows=[];negative=[]
    for name in args.images:
        image=np.array(Image.open(ROOT/'Image'/f'{name}.tif').convert('L'))
        old=embed_watermark_into_image(image,bits,n,q,phi,**layout)
        marked=embed_synchronized(image,bits,n,q,phi,settings=settings,**layout)
        Image.fromarray(marked).save(output/f'{name}_synchronized.png')
        cases=geometry_cases()
        for case in cases:
            attacked,a,scale,origin=apply_geometry(marked,case)
            old_attacked,*_=apply_geometry(old,case)
            start=time.perf_counter()
            result=extract_synchronized(attacked,qr_size,n,phi,settings=settings,**layout)
            elapsed=time.perf_counter()-start
            # Oracle is evaluated AFTER blind estimation and never fed into it.
            truth=SyncEstimate(a,scale,float(origin[0]%n),float(origin[1]%n),1,1,True)
            oracle=_decode_registered(attacked,qr_size,n,phi,truth,**layout)
            e=result.estimate
            row=dict(image=name,attack=case['name'],height=attacked.shape[0],width=attacked.shape[1],
                     legacy_accuracy=accuracy(bits,legacy_extract(old_attacked,qr_size,n,phi,layout)),
                     pilot_without_sync_accuracy=accuracy(bits,legacy_extract(attacked,qr_size,n,phi,layout)),
                     sync_accepted=e.accepted,sync_accuracy=accuracy(bits,result.recovered_qr),
                     oracle_accuracy=accuracy(bits,oracle.recovered_qr),
                     erased_bits='' if result.usable_bits is None else int(np.sum(~result.usable_bits)),
                     angle_true=a,angle_est=e.angle_deg,scale_true=scale,scale_est=e.scale,
                     angle_error=abs((e.angle_deg-a+180)%360-180),scale_error=abs(e.scale-scale),
                     origin_error=float(np.linalg.norm((np.array([e.origin_row,e.origin_col])-origin+n/2)%n-n/2)),
                     coherence=e.coherence,spectral_score=e.spectral_score,seconds=elapsed,
                     legacy_psnr=count_psnr(image,old),sync_psnr=count_psnr(image,marked))
            rows.append(row)
            print(name,case['name'],'accepted',e.accepted,'accuracy',row['sync_accuracy'],flush=True)
        # Negative controls are descriptive, not a calibrated false-positive bound.
        rng=np.random.default_rng(992)
        noise=np.clip(rng.normal(128,25,image.shape),0,255)
        for label,negative_image,opts in [('unmarked',image,settings),('legacy_no_pilot',old,settings),
                                         ('wrong_seed',marked,replace(settings,seed=31337)),
                                         ('noise',noise,settings),('constant',np.full(image.shape,128),settings)]:
            e=extract_synchronized(negative_image,qr_size,n,phi,settings=opts,**layout).estimate
            negative.append(dict(image=name,case=label,**asdict(e)))
        if args.full:
            from Source.Scripts.research_distorter import build_attack_specs
            for spec in build_attack_specs():
                for value in spec.values:
                    attacked=spec.apply(marked,value);old_attacked=spec.apply(old,value)
                    t=time.perf_counter()
                    result=extract_synchronized(attacked,qr_size,n,phi,settings=settings,**layout)
                    e=result.estimate
                    row={key:'' for key in rows[0]}
                    row.update(image=name,attack=f'legacy_suite/{spec.name}/{value}',height=attacked.shape[0],width=attacked.shape[1],
                               legacy_accuracy=accuracy(bits,legacy_extract(old_attacked,qr_size,n,phi,layout)),
                               pilot_without_sync_accuracy=accuracy(bits,legacy_extract(attacked,qr_size,n,phi,layout)),
                               sync_accepted=e.accepted,sync_accuracy=accuracy(bits,result.recovered_qr),
                               erased_bits='' if result.usable_bits is None else int(np.sum(~result.usable_bits)),
                               angle_est=e.angle_deg,scale_est=e.scale,coherence=e.coherence,spectral_score=e.spectral_score,
                               seconds=time.perf_counter()-t,legacy_psnr=count_psnr(image,old),sync_psnr=count_psnr(image,marked))
                    rows.append(row)
            print(name,'full legacy suite finished',flush=True)
    for filename,data in [('results.csv',rows),('negative_controls.csv',negative)]:
        with (output/filename).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    import matplotlib.pyplot as plt
    geometric=[r for r in rows if not r['attack'].startswith('legacy_suite/')]
    fig,axes=plt.subplots(len(args.images),1,figsize=(13,5*len(args.images)),squeeze=False)
    for ax,name in zip(axes[:,0],args.images):
        subset=[r for r in geometric if r['image']==name]
        for field,label in [('legacy_accuracy','Legacy'),('sync_accuracy','Blind sync'),('oracle_accuracy','Known geometry reference')]:
            ax.plot([r['attack'] for r in subset], [np.nan if r[field]=='' else 100*r[field] for r in subset],'.-',label=label)
        ax.set_title(name+' | rejected estimates are missing, not zero accuracy')
        ax.set_ylabel('Correct bits / all bits (%)');ax.set_ylim(0,101);ax.tick_params(axis='x',rotation=65);ax.legend()
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    print('Saved:',output,flush=True)


if __name__=='__main__':main()
