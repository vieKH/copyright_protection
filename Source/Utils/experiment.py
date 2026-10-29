"""Paired, fixed-grid experiments. Attack parameters never enter decode()."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import PIL
from PIL import Image, ImageOps
from .phases import generate_phase_map
from .utils import (generate_watermark, calculate_q, embed_watermark_into_image,
    qr_to_spectrum_positions, iter_offset_blocks, count_psnr)


def decode(image, phases, n, layout, method='phase', radius=0, start=(0, 0), stride=None):
    """Blind payload estimation; no true bits, attack angle, scale or reference image."""
    m = phases.shape[0]
    pos = qr_to_spectrum_positions(m, n, **layout)
    score = np.zeros((m, m), float)
    count = 0
    neighborhoods = [(du, dv) for du in range(-radius, radius+1)
                     for dv in range(-radius, radius+1) if du*du+dv*dv <= radius*radius]
    # Disjoint fixed neighborhoods: exclude bins closer to another bit's
    # carrier OR conjugate carrier; ties between different bits are discarded.
    centers = np.array([(p['row'],p['col']) for p in pos])
    pools = {}
    for k,p in enumerate(pos):
        pool=[]
        for du,dv in neighborhoods:
            z=np.array([(p['row']+du)%n,(p['col']+dv)%n])
            if not np.any(z): continue
            distances=[]
            for sign in (1,-1):
                delta=(z-sign*centers+n//2)%n-n//2
                distances.append(np.sum(delta**2,axis=1))
            d=np.minimum(*distances)
            winners=np.flatnonzero(d==d.min())
            if len(winners)==1 and winners[0]==k: pool.append(tuple(z))
        if not pool: raise ValueError('Empty spectral neighborhood')
        pools[(p['qr_i'],p['qr_j'])]=pool
    for block, r, c in iter_offset_blocks(image, n, *start, stride=n if stride is None else stride):
        f = np.fft.fft2(block)
        for p in pos:
            i,j,u,v = p['qr_i'],p['qr_j'],p['row'],p['col']
            if method == 'phase':
                # Known window coordinates, not an estimated geometric transform.
                carrier = np.exp(-1j*(phases[i,j]+2*np.pi*(u*r+v*c)/n))
                value = np.real(f[u,v]*carrier)
            elif method == 'energy':
                value = np.mean([abs(f[uu,vv])**2 for uu,vv in pools[(i,j)]])
            else:
                raise ValueError('Unknown decoder')
            score[i,j] += value
        count += 1
    if not count:
        raise ValueError('Image contains no complete extraction window')
    score /= count
    threshold = float(np.median(score))
    return (score > threshold).astype(np.uint8), score, count, threshold


def attack(image, kind, value, resample, canvas):
    im = Image.fromarray(image)
    if kind == 'rotation':
        return np.asarray(im.rotate(value, resample=resample, expand=False, fillcolor=0)).copy()
    size = tuple(max(1, int(round(d*value))) for d in im.size)
    scaled = im.resize(size, resample=resample)
    if canvas == 'native':
        return np.asarray(scaled).copy()
    result = Image.new('L', im.size, 0)
    result.paste(scaled, ((im.width-scaled.width)//2, (im.height-scaled.height)//2))
    return np.asarray(result).copy()


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main(kind):
    p = argparse.ArgumentParser(description=f'{kind}: random-phase embedding; fixed-grid baselines and optional spectral geometric synchronization')
    p.add_argument('--input', type=Path, required=True, help='Image or directory; scanned recursively')
    p.add_argument('--output', type=Path, required=True, help='New directory (refuses overwrite)')
    p.add_argument('--values', type=float, nargs='+', default=([.5,.75,.9,.98,1,1.02,1.1,1.5,2] if kind=='scale' else [-120,-25,-12,-3,-1,0,1,3,7,23,90,170]))
    p.add_argument('--seeds', type=int, nargs='+', default=[2021,2022,2023,2024,2025])
    p.add_argument('--payload-seed', type=int, default=42)
    p.add_argument('--block-size', type=int, default=64)
    p.add_argument('--payload-size', type=int, default=14)
    p.add_argument('--s-param', type=float, default=300.)
    p.add_argument('--start', type=int, nargs=2, default=[0,0], metavar=('ROW','COL'))
    p.add_argument('--stride', type=int, default=None)
    p.add_argument('--accuracy-target', type=float, default=.75, help='Descriptive target; NOT a presence detector')
    p.add_argument('--interpolation', choices=['bilinear','bicubic'], default='bilinear')
    p.add_argument('--canvas', choices=['native','fixed'], default='fixed', help='Scale only; rotation always fixed canvas')
    p.add_argument('--limit', type=int)
    p.add_argument('--save-images', action='store_true')
    p.add_argument('--spectral-search', action='store_true',
                   help='OPT IN: blind Fourier-domain scale/angle estimation; this IS geometric synchronization')
    p.add_argument('--search-scale-min',type=float,default=.5)
    p.add_argument('--search-scale-max',type=float,default=2.)
    p.add_argument('--search-scale-step',type=float,default=.002)
    p.add_argument('--search-angle-step',type=float,default=.25)
    a = p.parse_args()
    n,m=a.block_size,a.payload_size
    if n<8 or m<2 or not np.isfinite(a.s_param) or a.s_param<=0: p.error('Invalid block/payload/strength')
    if any(s<0 for s in a.seeds) or a.payload_seed<0: p.error('Seeds must be nonnegative')
    if any(not np.isfinite(v) or (kind=='scale' and v<=0) for v in a.values): p.error('Invalid attack values')
    if not 0<=a.accuracy_target<=1: p.error('accuracy-target must be in [0,1]')
    if any(x<0 or x>=n for x in a.start) or (a.stride is not None and a.stride<=0): p.error('Invalid extraction grid')
    if a.limit is not None and a.limit<1: p.error('limit must be positive')
    if a.output.exists(): p.error('Output exists; choose a new directory')
    if not a.input.exists(): p.error('Input does not exist')
    paths=([a.input] if a.input.is_file() else sorted(x for x in a.input.rglob('*') if x.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}))
    paths=paths[:a.limit] if a.limit else paths
    if not paths: p.error('No input images found')
    baseline=dict(x=n//8,y=n//8,offset=n//16,gap=1)
    low=dict(x=1,y=1,offset=0,gap=1)
    variants=([('baseline',baseline,'phase',0),('lower_band',low,'phase',0)] if kind=='scale' else [('phase',baseline,'phase',0),('energy_point',baseline,'energy',0),('energy_local',baseline,'energy',1)])
    try:
        for _,layout,_,_ in variants: qr_to_spectrum_positions(m,n,**layout)
    except ValueError as e: p.error(str(e))
    if not (np.isfinite(a.search_scale_min) and np.isfinite(a.search_scale_max) and 0<a.search_scale_min<a.search_scale_max): p.error('Invalid scale search range')
    if any(not np.isfinite(v) or v<=0 for v in (a.search_scale_step,a.search_angle_step)): p.error('Search steps must be finite and positive')
    if a.spectral_search:
        if kind=='rotation':
            variants=[('phase',baseline,'phase',0)]
        variants.append(('spectral_search',baseline,'spectral_search',0))
        if kind=='scale':
            variants.append(('spectral_search_lower',low,'spectral_search',0))
    a.output.mkdir(parents=True)
    bits=generate_watermark(m,a.payload_seed); q=calculate_q(a.s_param,m,n)
    np.save(a.output/'payload.npy',bits)
    rows=[]; failures=[]; sources=[]
    values=sorted(set(a.values+[1. if kind=='scale' else 0.]))
    resample=getattr(Image.Resampling,a.interpolation.upper())
    for index,path in enumerate(paths):
        with Image.open(path) as im: original=np.asarray(ImageOps.exif_transpose(im).convert('L')).copy()
        h,w=original.shape; hh,ww=h//n*n,w//n*n
        sources.append(dict(image=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original_shape=[h,w],used_shape=[hh,ww]))
        if min(hh,ww)<n:
            failures.append(dict(image=str(path),reason='smaller than one block')); continue
        host=original[:hh,:ww]
        for seed in sorted(set(a.seeds)):
            phase=generate_phase_map(m,seed); np.save(a.output/f'phase_{seed}.npy',phase)
            for variant,layout,method,radius in variants:
                wm=embed_watermark_into_image(host,bits,n,q,phase,**layout)
                embedding_psnr=count_psnr(host,wm)
                for value in values:
                    transformed=attack(wm,kind,value,resample,a.canvas)
                    estimate=None; search_quality=None; supported_bits=m*m; hypotheses=0
                    try:
                        if method=='spectral_search':
                            from .spectral_geometry import extract_spectral_geometry
                            result=extract_spectral_geometry(transformed,m,n,kind,**layout,
                                scale_min=a.search_scale_min,scale_max=a.search_scale_max,
                                scale_step=a.search_scale_step,angle_step=a.search_angle_step)
                            recovered,score,threshold=result.bits,result.score,result.threshold
                            blocks=1  # One GLOBAL observed image, not an embedding block.
                            estimate,search_quality=result.estimate,result.quality
                            supported_bits,hypotheses=result.supported_bits,result.hypotheses
                        else:
                            recovered,score,blocks,threshold=decode(transformed,phase,n,layout,method,radius,tuple(a.start),a.stride)
                    except ValueError as e:
                        failures.append(dict(image=str(path),seed=seed,variant=variant,value=value,reason=str(e))); continue
                    acc=float(np.mean(recovered==bits))
                    row=dict(image=str(path.resolve()),phase_seed=seed,variant=variant,value=value,accuracy=acc,ber=1-acc,meets_accuracy_target=int(acc>=a.accuracy_target),psnr_embedding=embedding_psnr,blocks_used=blocks,threshold=threshold,predicted_ones=int(recovered.sum()),height=transformed.shape[0],width=transformed.shape[1],
                             estimated_parameter=estimate,search_quality=search_quality,
                             supported_bits=supported_bits,unsupported_bits=m*m-supported_bits,
                             search_hypotheses=hypotheses,geometric_synchronization=(method=='spectral_search'),
                             watermarked_sha256=hashlib.sha256(wm.tobytes()).hexdigest())
                    rows.append(row)
                    if a.save_images:
                        stem=f'{index:04d}_{seed}_{variant}_{value:g}'
                        Image.fromarray(transformed).save(a.output/(stem+'_attacked.png'))
                        Image.fromarray(recovered*255).save(a.output/(stem+'_bits.png'))
                print(f'{index+1}/{len(paths)} {path.name} seed={seed} {variant}',flush=True)
    manifest=dict(source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('utils.py'),Path(__file__).with_name('phases.py'),Path(__file__).with_name('spectral_geometry.py')] if p.is_file()},arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},kind=kind,values=values,q=q,variants=variants,sources=sources,failures=failures,versions=dict(python=platform.python_version(),numpy=np.__version__,pillow=PIL.__version__),synchronization=bool(a.spectral_search),
                  protocol='fixed-window baselines; optional blind Fourier-domain geometric search; no inverse image transform; upper-left crop to complete blocks',
                  search_protocol='Fixed public range for all observations; attack values and payload bits never passed to spectral extractor; rotation estimates modulo 180 degrees')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    if not rows: raise SystemExit('No valid results. See manifest.json')
    save_csv(a.output/'results.csv',rows)
    summary=[]
    for variant,_,_,_ in variants:
        for value in values:
            rr=[r for r in rows if r['variant']==variant and r['value']==value]
            if not rr: continue
            # Image is the unit; average phase seeds before summarizing images.
            means=[np.mean([r['accuracy'] for r in rr if r['image']==img]) for img in sorted({r['image'] for r in rr})]
            summary.append(dict(variant=variant,value=value,images=len(means),trials=len(rr),mean_accuracy=float(np.mean(means)),std_between_images=float(np.std(means,ddof=1)) if len(means)>1 else '',min_image_mean=float(min(means)),target_fraction_trials=float(np.mean([r['meets_accuracy_target'] for r in rr])) ))
    save_csv(a.output/'summary.csv',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    for variant,_,_,_ in variants:
        ss=[r for r in summary if r['variant']==variant]
        ax.plot([r['value'] for r in ss],[r['mean_accuracy'] for r in ss],'.-',label=variant)
    ax.axhline(a.accuracy_target,color='gray',linestyle='--',label='descriptive target')
    ax.set(xlabel='Scale factor' if kind=='scale' else 'Angle (degrees)',ylabel='Mean bit accuracy',ylim=(0,1.02))
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(a.output/'accuracy.png',dpi=180); plt.close(fig)
    if a.spectral_search:
        from collections import defaultdict
        paired=defaultdict(dict)
        for r in rows: paired[(r['image'],r['phase_seed'],r['value'])][r['variant']]=r
        effects=[]
        comparisons=[('baseline' if kind=='scale' else 'phase','spectral_search')]
        if kind=='scale': comparisons.append(('lower_band','spectral_search_lower'))
        for base_name,new_name in comparisons:
            for value in values:
                observations=[]
                for (image,seed,v),group in paired.items():
                    if v==value and base_name in group and new_name in group:
                        first,second=group[base_name],group[new_name]
                        if first['watermarked_sha256']!=second['watermarked_sha256']:
                            raise RuntimeError('Unpaired embeddings: cannot claim an extraction improvement')
                        observations.append((image,second['accuracy']-first['accuracy']))
                images=sorted({im for im,d in observations})
                differences=np.array([np.mean([d for im,d in observations if im==image]) for image in images])
                if not len(differences): continue
                rng=np.random.default_rng(27029)
                boot=differences[rng.integers(len(differences),size=(2000,len(differences)))].mean(axis=1)
                effects.append(dict(baseline=base_name,candidate=new_name,value=value,images=len(images),trials=len(observations),
                                    gain_accuracy=float(differences.mean()),
                                    ci95_low=float(np.quantile(boot,.025)) if len(images)>1 else '',
                                    ci95_high=float(np.quantile(boot,.975)) if len(images)>1 else '',
                                    images_improved=int(np.sum(differences>0)),images_worsened=int(np.sum(differences<0))))
        if effects: save_csv(a.output/'paired_gains.csv',effects)
    print(f'Saved {len(rows)} trials; {len(failures)} failures to {a.output}')
    if failures: raise SystemExit('Partial results: inspect failures in manifest.json')
