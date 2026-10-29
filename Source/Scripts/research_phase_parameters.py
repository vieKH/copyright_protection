"""Parameter sweeps for the current random-phase block decoder.

Run from repository root:
 python -m Source.Scripts.research_phase_parameters --count 2 --output Results/phase_smoke
 python -m Source.Scripts.research_phase_parameters --count 100 --workers 4 --output Results/phase_200
Resume with the SAME arguments plus --resume. Independent files per image are
atomically checkpointed. A failure is an error row, never an invented BER.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import PIL
from PIL import Image, ImageOps

from Source.Utils.phase_parameters import (ATTACKS, METHODS, Condition, GROUP,
    attacks_for, available_windows, evaluate_image, frequency_metrics, make_plan,
    paired_effects, summarize)

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    first = list(rows[0])
    fields = first + sorted(set().union(*(r.keys() for r in rows))-set(first))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def preflight(image_root, settings, conditions):
    entries = []
    for size in settings['groups']:
        for image_id in range(1, settings['count']+1):
            path = image_root/str(size)/f'{size} ({image_id}).tif'
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source)
                if image.size != (size, size) or image.mode != 'L':
                    raise ValueError(f'{path}: expected {size}x{size} 8-bit grayscale, got {image.size}/{image.mode}')
            for c in conditions:
                n = c.block_size
                if size % n:
                    raise ValueError(f'{size}: dimensions not divisible by DFT size {n}')
                available = available_windows((size,size), n, settings['start'], n//2)
                if available < max(settings['blocks']):
                    raise ValueError(f'{size}, B={n}: only {available} windows; requested K={settings["blocks"]}')
            entries.append(dict(dataset=size, image_id=image_id, filename=path.name,
                                path=str(path.resolve()), sha256=sha256(path)))
    return entries


def checkpoint_path(output, record):
    return output/'checkpoints'/f'{record["dataset"]}_{record["image_id"]:03d}.json'


def expected_keys(record, settings, conditions):
    return {(c.condition_id, repeat, attack, method, k)
            for c in conditions for repeat in range(1, settings['repeats']+1)
            for attack in attacks_for(c, settings)
            for method in settings['methods'] for k in settings['blocks']}


def validate_checkpoint(data, signature, record, settings, conditions):
    if data.get('signature') != signature or data.get('input_sha256') != record['sha256']:
        raise ValueError('Checkpoint belongs to different code, settings or input')
    rows = data['rows']
    keys = [(r['condition_id'], r['repeat'], r['attack'], r['method'], r['blocks_used']) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected_keys(record, settings, conditions):
        raise ValueError('Checkpoint has missing, duplicate or unexpected measurements')
    if any(r['dataset'] != record['dataset'] or r['image_id'] != record['image_id'] for r in rows):
        raise ValueError('Checkpoint has observations from another image')
    return rows


def run_image(task):
    record, settings, condition_dicts, output_string, signature = task
    conditions = [Condition(**c) for c in condition_dicts]
    if sha256(record['path']) != record['sha256']:
        raise ValueError(f'Input changed after validation: {record["path"]}')
    with Image.open(record['path']) as source:
        image = np.array(ImageOps.exif_transpose(source).convert('L'))
    rows = evaluate_image(image, record, settings, conditions)
    data = dict(signature=signature, input_sha256=record['sha256'], rows=rows)
    validate_checkpoint(data, signature, record, settings, conditions)
    write_json(checkpoint_path(Path(output_string), record), data)
    return len(rows), sum(r['status'] != 'ok' for r in rows)


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    grouped = {}
    for r in summary:
        if r['images_ok'] == 0:
            continue
        key = (r['experiment'], r['dataset'], r['attack'], r['method'])
        grouped.setdefault(key, []).append(r)
    for (experiment, size, attack, method), records in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for k in sorted({r['blocks_used'] for r in records}):
            selected = [r for r in records if r['blocks_used'] == k]
            field, label = {'payload':('payload_size','Binary payload side L'),
                            'dft':('block_size','DFT side B'),
                            'frequency':('mean_frequency_radius_mean','Mean carrier radius (cycles/pixel)')}[experiment]
            selected.sort(key=lambda r: (r[field], r['condition_id']))
            xx = [r[field] for r in selected]
            yy = [100*r['mean_BER'] for r in selected]
            ax.plot(xx, yy, '.-', label=f'K={k}')
            for x, r in zip(xx, selected):
                if r.get('ci95_low') is not None:
                    ax.vlines(x, 100*r['ci95_low'], 100*r['ci95_high'], linewidth=1)
        ax.set(xlabel=label, ylabel='Mean BER (%)', ylim=(0,100),
               title=f'{experiment}; {size}x{size}; {method}; {attack}')
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(output/f'{experiment}_{size}_{attack}_{method}.png', dpi=160)
        plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--image-root', type=Path, default=ROOT/'Image')
    p.add_argument('--output', type=Path, default=ROOT/'Results'/'phase_smoke')
    p.add_argument('--experiments', nargs='+', choices=['payload','dft','frequency'], default=['payload','dft','frequency'])
    p.add_argument('--groups', type=int, nargs='+', choices=[512,1024], default=[512,1024])
    p.add_argument('--count', type=int, default=100, help='Images per resolution, 1..100')
    p.add_argument('--base-block', type=int, default=128)
    p.add_argument('--base-payload', type=int, default=14)
    p.add_argument('--payload-sizes', type=int, nargs='+', default=[7,14,21,28,41])
    p.add_argument('--dft-sizes', type=int, nargs='+', default=[64,128,256])
    p.add_argument('--frequency-locations', type=int, default=7)
    p.add_argument('--frequency-attacks', nargs='+', choices=ATTACKS, default=list(ATTACKS))
    p.add_argument('--variance', type=float, default=.72, help='Prequantization spatial watermark variance in pixel^2')
    p.add_argument('--gap', type=int, default=1)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=['complex_average'])
    p.add_argument('--blocks', type=int, nargs='+', default=[1,4], help='Exactly matched K for every DFT size')
    p.add_argument('--start', type=int, nargs=2, default=[5,8], metavar=('ROW','COL'))
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261029)
    p.add_argument('--accuracy-target', type=float, default=.8, help='Descriptive bit-accuracy target, not presence detection')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--no-plots', action='store_true')
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if not 1 <= args.count <= 100: p.error('count must be in 1..100')
    if min(args.repeats,args.bootstrap,args.workers,args.frequency_locations) < 1: p.error('repeat/bootstrap/worker/location counts must be positive')
    if min(args.payload_sizes+[args.base_payload]) < 2 or min(args.blocks) < 1: p.error('payload side >=2 and K >=1 required')
    if args.seed < 0 or args.gap < 0: p.error('seed and gap must be nonnegative')
    if not np.isfinite(args.variance) or args.variance <= 0: p.error('variance must be finite and positive')
    if not 0 <= args.accuracy_target <= 1: p.error('accuracy-target must be in [0,1]')
    if any(n < 16 or n % 16 for n in args.dft_sizes+[args.base_block]): p.error('DFT sides must be positive multiples of 16')
    settings = {k:v for k,v in vars(args).items() if k not in ('output','image_root','workers','resume','dry_run','no_plots')}
    for key in ('groups','experiments','payload_sizes','dft_sizes','blocks','frequency_attacks','methods'):
        settings[key] = sorted(set(settings[key]))
    try:
        conditions = make_plan(settings)
        entries = preflight(args.image_root, settings, conditions)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    expected = len(entries)*sum(len(expected_keys(r,settings,conditions)) for r in entries[:1])
    print(f'Validated {len(entries)} images; {len(conditions)} geometries; {expected} measurements', flush=True)
    if args.dry_run:
        return 0
    output = args.output.resolve()
    # Include all source content in resume identity, independent of Git commit state.
    hashes = {str(path.relative_to(ROOT)):sha256(path) for path in sorted((ROOT/'Source').rglob('*.py'))}
    identity = dict(settings=settings, inputs=entries, code_sha256=hashes,
                    conditions=[asdict(c) for c in conditions])
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest_path = output/'manifest.json'
    if args.resume:
        if not manifest_path.is_file(): p.error('Cannot resume: manifest.json is absent')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('signature') != signature: p.error('Resume refused: settings, source code or inputs changed')
    else:
        if output.exists() and any(output.iterdir()): p.error('Output must be empty; use --resume for the same protocol')
        git = subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, capture_output=True, text=True)
        manifest = dict(identity, signature=signature, measurements_expected=expected,
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        source_commit=git.stdout.strip() or None,
                        versions=dict(python=platform.python_version(),numpy=np.__version__,pillow=PIL.__version__),
                        energy_protocol='Fixed prequantization spatial variance; no PSNR recalibration',
                        decoder='Existing compare_methods; complex_average uses known per-bit random phases and coordinate compensation',
                        window_protocol='Same image/B windows across payload and placement; exact requested K; half-block stride',
                        statistical_unit='Image; repeat seeds averaged within image before bootstrap',
                        phase_mode='random', synchronization=False, status='running')
    manifest['status'] = 'running'; write_json(manifest_path, manifest)
    tasks = []
    for record in entries:
        path = checkpoint_path(output, record)
        if path.is_file():
            rows = validate_checkpoint(json.loads(path.read_text()),signature,record,settings,conditions)
            if all(r['status'] == 'ok' for r in rows):
                continue
        tasks.append((record,settings,[asdict(c) for c in conditions],str(output),signature))
    print(f'Pending images: {len(tasks)}; atomic per-image checkpoints enabled', flush=True)
    if args.workers == 1:
        for i, task in enumerate(tasks,1):
            total, errors = run_image(task)
            print(f'[{i}/{len(tasks)}] {task[0]["filename"]}: {total} rows, {errors} errors', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,(total,errors) in enumerate(pool.map(run_image,tasks),1):
                print(f'[{i}/{len(tasks)}] {total} rows, {errors} errors', flush=True)
    rows = []
    for record in entries:
        data = json.loads(checkpoint_path(output,record).read_text())
        rows.extend(validate_checkpoint(data,signature,record,settings,conditions))
    per_image, summary = summarize(rows, settings)
    effects = paired_effects(per_image, settings)
    write_csv(output/'raw.csv',rows); write_csv(output/'per_image.csv',per_image)
    write_csv(output/'summary.csv',summary); write_csv(output/'paired_effects.csv',effects)
    write_csv(output/'conditions.csv',[dict(asdict(c),condition_id=c.condition_id,**frequency_metrics(c)) for c in conditions])
    errors = [r for r in rows if r['status'] != 'ok']
    errors_path = output/'errors.csv'
    if errors:
        write_csv(errors_path,errors)
    elif errors_path.exists():
        errors_path.unlink()
    if not args.no_plots:
        plot_summary(summary,output)
    manifest.update(status='failed' if errors else 'complete',measurements_completed=len(rows),
                    measurements_error=len(errors), completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(manifest_path,manifest)
    print(f'Completed {len(rows)}/{expected} measurements; {len(errors)} errors; {output}', flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
