"""Run the two numbered image datasets and export one Excel sheet per attack.

python -m Source.Scripts.research_dataset
python -m Source.Scripts.research_dataset --count 2 --attacks jpeg wh_noise
python -m Source.Scripts.research_dataset --dry-run
python -m Source.Scripts.research_dataset --resume Results/Dataset_Research/RUN_DIRECTORY
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from Source.Utils.batch_research import (
    DATASETS, BatchDecoder, apply_attack, attack_specs, derived_seed, random_layout, valid_layouts,
)
from Source.Utils.dataset_report import export_reports
from Source.Utils.extraction_research import random_extract_start
from Source.Utils.phases import generate_phase_map
from Source.Utils.utils import calculate_q, count_psnr, embed_watermark_into_image, generate_watermark

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(image_root, groups, count, start):
    records=[]
    errors=[]
    for size in groups:
        for image_id in range(start,start+count):
            path=Path(image_root)/str(size)/f'{size} ({image_id}).tif'
            try:
                with Image.open(path) as image:
                    if image.size!=(size,size):
                        raise ValueError(f'expected {size}x{size}, got {image.size[0]}x{image.size[1]}')
                    image.verify()
                records.append(dict(dataset=size,image_id=image_id,filename=path.name,
                                    path=str(path.resolve()),sha256=_sha256(path)))
            except (OSError,ValueError) as exc:
                errors.append(f'{path}: {exc}')
    if errors:
        preview='\n'.join(errors[:12])
        if len(errors)>12: preview+=f'\n... and {len(errors)-12} more input errors'
        raise ValueError(f'Input validation failed ({len(errors)} files). No experiment was started.\n{preview}')
    return records


def settings_from_args(args):
    selected=set(args.attacks) if args.attacks else {s.name for s in attack_specs()}
    selected.add('no_attack')
    attacks=[s for s in attack_specs() if s.name in selected]
    return dict(groups=args.groups,count=args.count,start=args.start,repeats=args.repeats,
                seed=args.seed,gap=args.gap,s_param=args.s_param,
                payload_seed=args.payload_seed,phase_seed=args.phase_seed,
                attacks=[a.name for a in attacks],sweeps={a.name:list(a.values) for a in attacks},
                parameter_names={a.name:a.parameter for a in attacks},stride_mode=args.stride,
                layout_range=[4,12],extract_range='0..block_size-1, excluding (0,0)',
                phase_mode='random',phase_sign=-1,detrend=False,decision_rule='median',
                noise_stream='independent per image/trial/attack/parameter',
                payload_phase_policy='fixed per dataset; random coordinates change per image/trial')


def code_hashes():
    names=('Source/Scripts/research_dataset.py','Source/Utils/batch_research.py',
           'Source/Utils/dataset_report.py','Source/Utils/utils.py','Source/Utils/extraction_research.py',
           'Source/Utils/phases.py','Source/Utils/distorter.py','Source/Utils/my_function.py')
    return {name:_sha256(ROOT/name) for name in names}


def experiment_signature(settings,records,hashes):
    # Input paths are recorded, but content identity allows moving the input tree.
    identities=[{k:r[k] for k in ('dataset','image_id','filename','sha256')} for r in records]
    data=dict(settings=settings,inputs=identities,code=hashes)
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def _psnr(a,b):
    value=count_psnr(a,b)
    return (None,True) if np.isinf(value) else (value,False)


def run_image(task):
    """One image per worker; no figure generation or storage of recovered arrays."""
    record,settings,trial_ids=task
    spec=DATASETS[record['dataset']]
    n,qr_size=spec.block_size,spec.qr_size
    qr=generate_watermark(qr_size,settings['payload_seed'])
    phases=generate_phase_map(qr_size,settings['phase_seed'])
    q=calculate_q(settings['s_param'],qr_size,n)
    stride=n//2 if settings['stride_mode']=='half' else n
    offset=n//16
    attacks=[a for a in attack_specs() if a.name in settings['attacks']]
    rows=[]
    try:
        # Detect changes after preflight, including resumed inputs.
        if _sha256(record['path'])!=record['sha256']:
            raise ValueError('Image changed after input validation')
        with Image.open(record['path']) as source:
            image=np.asarray(ImageOps.exif_transpose(source).convert('L'))
        image_error=None
    except (OSError,ValueError) as exc:
        image=None
        image_error=str(exc)
    for trial in trial_ids:
        layout_seed=derived_seed(settings['seed'],spec.size,record['image_id'],trial,'layout')
        extract_seed=derived_seed(settings['seed'],spec.size,record['image_id'],trial,'extraction')
        x,y=random_layout(n,qr_size,settings['gap'],layout_seed)
        start_x,start_y=random_extract_start(n,seed=extract_seed,avoid_zero_zero=True)
        base=dict(dataset=spec.size,image_id=record['image_id'],filename=record['filename'],trial=trial,
                  embed_x=x,embed_y=y,embed_offset=offset,extract_x=start_x,extract_y=start_y,
                  block_size=n,qr_size=qr_size,gap=settings['gap'],stride=stride,q=q,
                  payload_seed=settings['payload_seed'],phase_seed=settings['phase_seed'],
                  layout_seed=layout_seed,extract_seed=extract_seed,total_bits=qr.size,
                  psnr_embed_db=None,psnr_embed_infinite=None)
        embed_error=image_error
        wm=None
        decoder=None
        if embed_error is None:
            try:
                wm=embed_watermark_into_image(image,qr,n,q,phases,x,y,offset,gap=settings['gap'])
                base['psnr_embed_db'],base['psnr_embed_infinite']=_psnr(image,wm)
                decoder=BatchDecoder(n,qr_size,phases,x,y,offset,settings['gap'],start_x,start_y,stride)
            except (ValueError,RuntimeError) as exc:
                embed_error=str(exc)
        for attack in attacks:
            for value in attack.values:
                noise_seed=derived_seed(settings['seed'],spec.size,record['image_id'],trial,attack.name,value,'noise')
                row=dict(base,attack=attack.name,parameter_name=attack.parameter,parameter_value=value,
                         noise_seed=noise_seed if attack.name in ('wh_noise','salt_pepper') else None,
                         status='ok',accuracy=None,ber=None,correct_bits=None,blocks_used=None,
                         psnr_attack_db=None,psnr_attack_infinite=None,error='')
                started=time.perf_counter()
                try:
                    if embed_error is not None:
                        raise ValueError(embed_error)
                    attacked=apply_attack(wm,attack.name,value,noise_seed)
                    attacked=np.clip(attacked,0,255).astype(np.uint8)
                    recovered,used,_=decoder.decode(attacked)
                    correct=int(np.count_nonzero(qr==recovered))
                    row.update(correct_bits=correct,accuracy=correct/qr.size,ber=1-correct/qr.size,blocks_used=used)
                    row['psnr_attack_db'],row['psnr_attack_infinite']=_psnr(wm,attacked)
                except Exception as exc:
                    # Count failed measurements explicitly; never convert them to 0% accuracy.
                    row.update(status='error',error=f'{type(exc).__name__}: {exc}',
                               accuracy=None,ber=None,correct_bits=None,blocks_used=None)
                row['seconds']=time.perf_counter()-started
                rows.append(row)
    return rows


def load_checkpoint(path):
    path=Path(path)
    if not path.exists(): return []
    rows=[]
    raw=path.read_bytes()
    lines=raw.splitlines(keepends=True)
    offset=0
    for i,line in enumerate(lines):
        try:
            row=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if i!=len(lines)-1:
                raise ValueError(f'Invalid checkpoint JSON at line {i+1}; do not resume this file')
            print('Discarding an incomplete final checkpoint line.',file=sys.stderr)
            with path.open('r+b') as f: f.truncate(offset)
            break
        rows.append(row)
        offset+=len(line)
    keys=[row_key(r) for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate result keys in checkpoint; refusing to double-count')
    if path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        # A complete JSON record may survive a crash without its final newline.
        with path.open('ab') as f: f.write(b'\n')
    return rows


def row_key(row):
    return (row['dataset'],row['image_id'],row['trial'],row['attack'],row['parameter_value'])


def make_tasks(records,settings,rows):
    done={row_key(r) for r in rows}
    tasks=[]
    for record in records:
        trials=[]
        for trial in range(1,settings['repeats']+1):
            expected={(record['dataset'],record['image_id'],trial,name,value)
                      for name in settings['attacks'] for value in settings['sweeps'][name]}
            if not expected.issubset(done): trials.append(trial)
        if trials: tasks.append((record,settings,trials))
    return tasks


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image-root',type=Path,default=ROOT/'Image')
    parser.add_argument('--output-root',type=Path,default=ROOT/'Results'/'Dataset_Research')
    parser.add_argument('--groups',type=int,nargs='+',choices=sorted(DATASETS),default=[512,1024])
    parser.add_argument('--count',type=int,default=100,help='Images per selected dataset')
    parser.add_argument('--start',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=1,help='Random placement repetitions per image')
    parser.add_argument('--seed',type=int,default=20261012,help='Placement/extraction/noise master seed')
    parser.add_argument('--payload-seed',type=int,default=42)
    parser.add_argument('--phase-seed',type=int,default=20261011)
    parser.add_argument('--gap',type=int,default=1)
    parser.add_argument('--s-param',type=float,default=300)
    parser.add_argument('--stride',choices=['half','block'],default='half')
    parser.add_argument('--attacks',nargs='+',choices=[s.name for s in attack_specs()],default=None)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--dry-run',action='store_true',help='Validate files/layouts; run no embedding or attacks')
    parser.add_argument('--resume',type=Path,help='Resume with identical protocol/settings; repeat original CLI settings')
    parser.add_argument('--export-only',type=Path,help='Rebuild Excel/CSVs from a saved run; no input images needed')
    parser.add_argument('--json-only',action='store_true',help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser=build_parser()
    args=parser.parse_args(argv)
    if args.export_only:
        run_dir=args.export_only.resolve()
        manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
        rows=load_checkpoint(run_dir/'results.jsonl')
        if not rows: parser.error('Checkpoint contains no completed measurements')
        expected=manifest['measurements_expected']
        # Exporting an interrupted run must visibly expose incompleteness.
        manifest['label']+=f' | completed measurements {len(rows)}/{expected}'
        export_reports(run_dir,rows,manifest,json_only=args.json_only)
        print(f'Rebuilt reports in: {run_dir}')
        return 0
    for field in ('count','start','repeats','workers'):
        if getattr(args,field)<1: parser.error(f'--{field} must be >= 1')
    if any(s<0 for s in (args.seed,args.phase_seed,args.payload_seed,args.gap)):
        parser.error('Seeds and gap must be non-negative')
    if not np.isfinite(args.s_param) or args.s_param<=0:
        parser.error('--s-param must be finite and positive')
    if len(args.groups)!=len(set(args.groups)):
        parser.error('Duplicate dataset groups are not allowed')
    settings=settings_from_args(args)
    try:
        for size in settings['groups']:
            group=DATASETS[size]
            pairs=valid_layouts(group.block_size,group.qr_size,args.gap)
            print(f'{size}: block={group.block_size}, QR={group.qr_size}, valid (x,y)={len(pairs)}/81')
        records=preflight(args.image_root,args.groups,args.count,args.start)
    except ValueError as exc:
        parser.error(str(exc))
    measurements=len(records)*args.repeats*sum(len(v) for v in settings['sweeps'].values())
    print(f'Validated {len(records)} images. Planned measurements: {measurements}; attack sheets: {len(settings["attacks"])}')
    if args.dry_run: return 0
    hashes=code_hashes()
    signature=experiment_signature(settings,records,hashes)
    if args.resume:
        run_dir=args.resume.resolve()
        manifest=json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
        if manifest['signature']!=signature:
            parser.error('Resume protocol, image content, or implementation differs. Repeat the original CLI options or start a new run.')
        rows=load_checkpoint(run_dir/'results.jsonl')
    else:
        args.output_root.mkdir(parents=True,exist_ok=True)
        run_dir=args.output_root/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%f}_seed_{args.seed}'
        run_dir.mkdir()
        manifest=dict(version=1,label='User image dataset experiment',settings=settings,inputs=records,
                      code_hashes=hashes,signature=signature,measurements_expected=measurements,
                      created_utc=datetime.now(timezone.utc).isoformat())
        (run_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        rows=[]
    tasks=make_tasks(records,settings,rows)
    existing={row_key(r) for r in rows}
    print(f'Run directory: {run_dir}',flush=True)
    print(f'Pending images: {len(tasks)}; checkpoint measurements: {len(rows)}',flush=True)
    pool=ProcessPoolExecutor(max_workers=args.workers) if args.workers>1 else None
    iterator=pool.map(run_image,tasks,chunksize=1) if pool else map(run_image,tasks)
    try:
        with (run_dir/'results.jsonl').open('ab') as checkpoint:
            for index,batch in enumerate(iterator,start=1):
                new=[r for r in batch if row_key(r) not in existing]
                for row in new:
                    checkpoint.write((json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n').encode())
                    existing.add(row_key(row))
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
                rows.extend(new)
                record=tasks[index-1][0]
                errors=sum(r['status']!='ok' for r in new)
                print(f'[{index}/{len(tasks)}] {record["filename"]}: {len(new)} measurements, {errors} errors; saved {len(rows)}/{measurements}',flush=True)
    except KeyboardInterrupt:
        print('\nInterrupted. Completed images are in results.jsonl; use --resume with the same options.',file=sys.stderr)
        return 130
    finally:
        if pool: pool.shutdown(wait=True,cancel_futures=True)
    export_reports(run_dir,rows,manifest,json_only=args.json_only)
    errors=sum(r['status']!='ok' for r in rows)
    print(f'Completed {len(rows)}/{measurements} measurements; errors: {errors}')
    print(f'Reports: {run_dir}/' + ('report_spec.json, ' if args.json_only else 'dataset_results.xlsx, ') + 'summary.csv, details.csv')
    return 1 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
