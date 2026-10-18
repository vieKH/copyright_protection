import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
from PIL import Image, ImageOps
import PIL

from Source.Utils.batch_research import DATASETS, derived_seed, random_layout
from Source.Utils.detection import WatermarkDetector, calibrate_threshold, classify_score
from Source.Utils.extraction_research import random_extract_start
from Source.Utils.phases import generate_phase_map
from Source.Utils.utils import calculate_q, count_psnr, embed_watermark_into_image, generate_watermark

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_image(path):
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert('L'))


def prepare_records(image_root, seed):
    records = []
    for size in sorted(DATASETS):
        for image_id in range(1, 101):
            path = image_root / str(size) / f'{size} ({image_id}).tif'
            image = read_image(path)
            if image.shape != (size, size):
                raise ValueError(f'Wrong dimensions: {path}: {image.shape}')
            records.append(dict(dataset=size, image_id=image_id, filename=path.name,
                                path=str(path), sha256=sha256(path),
                                pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    assign_folds(records, seed)
    for size in sorted(DATASETS):
        rng = np.random.default_rng(derived_seed(seed, size, 'half-embedded'))
        chosen = set(map(int, rng.choice(np.arange(1, 101), 50, replace=False)))
        for record in records:
            if record['dataset'] == size:
                record['half_embedded'] = record['image_id'] in chosen
    return records


def assign_folds(records, seed, folds=5):
    """Keep pixel-identical source images together; balance within resolution."""
    for size in sorted({r['dataset'] for r in records}):
        grouped = {}
        for record in records:
            if record['dataset'] == size:
                grouped.setdefault(record['pixel_sha256'], []).append(record)
        groups = list(grouped.values())
        rng = np.random.default_rng(derived_seed(seed, size, 'folds'))
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        counts = [0] * folds
        for group in groups:
            fold = counts.index(min(counts))
            for record in group:
                record['fold'] = fold
            counts[fold] += len(group)
    if {r['fold'] for r in records} != set(range(folds)):
        raise ValueError('Not enough distinct sources for all folds')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows):
    tp = sum(r['embedded'] and r['detected'] for r in rows)
    tn = sum(not r['embedded'] and not r['detected'] for r in rows)
    fp = sum(not r['embedded'] and r['detected'] for r in rows)
    fn = sum(r['embedded'] and not r['detected'] for r in rows)
    return dict(n=len(rows), TP=tp, TN=tn, FP=fp, FN=fn,
                accuracy=(tp + tn) / len(rows),
                TPR=tp / (tp + fn) if tp + fn else None,
                FPR=fp / (fp + tn) if fp + tn else None,
                FNR=fn / (tp + fn) if tp + fn else None)


def crossfit(records, alpha):
    calibrations = []
    for fold in range(5):
        calibration = [r for r in records if r['fold'] != fold]
        held_out = [r for r in records if r['fold'] == fold]
        if {r['pixel_sha256'] for r in calibration} & {r['pixel_sha256'] for r in held_out}:
            raise ValueError('Pixel-identical source leaked across folds')
        # Duplicated files are not extra independent calibration observations.
        unique = {}
        for r in calibration:
            unique.setdefault(r['pixel_sha256'], r)
        negatives = [r['score_clean'] for r in unique.values()]
        threshold = calibrate_threshold(negatives, alpha)
        calibrations.append(dict(fold=fold, calibration_n=len(negatives), test_n=len(held_out),
                                 threshold=threshold, alpha=alpha,
                                 calibration_sources=[f"{r['dataset']}/{r['filename']}" for r in unique.values()],
                                 test_sources=[f"{r['dataset']}/{r['filename']}" for r in held_out]))
        for r in held_out:
            r['threshold'] = threshold
    decisions = []
    for experiment in ('none', 'half', 'all'):
        for r in records:
            embedded = experiment == 'all' or (experiment == 'half' and r['half_embedded'])
            score = r['score_embedded'] if embedded else r['score_clean']
            result = classify_score(score, r['threshold'], r['blocks_used'])
            decisions.append(dict(experiment=experiment, dataset=r['dataset'], image_id=r['image_id'],
                                  filename=r['filename'], fold=r['fold'], embedded=embedded,
                                  score=score, threshold=result.threshold, detected=result.detected,
                                  correct=result.detected == embedded, blocks_used=result.blocks_used))
    summaries = []
    for experiment in ('none', 'half', 'all'):
        for group in ('512', '1024', 'all'):
            rows = [r for r in decisions if r['experiment'] == experiment
                    and (group == 'all' or str(r['dataset']) == group)]
            summaries.append(dict(experiment=experiment, dataset=group, **metrics(rows)))
    return calibrations, decisions, summaries


def report_text(summaries, calibrations, records, settings):
    def rate(value):
        return 'N/A' if value is None else f'{100 * value:.2f}%'
    names = {'none': '1: no watermark', 'half': '2: half watermarked', 'all': '3: all watermarked'}
    lines = ['# Watermark Detection Results on 200 Images', '',
             'Detection of a specific reference watermark with known phases and spectral positions; the original image is not required.',
             'Only images without attacks are evaluated. A negative result means that the watermark being tested was not detected; it does not prove that the image has never been watermarked.', '',
             '| Experiment | Dataset | N | TP | TN | FP | FN | Accuracy | TPR | FPR |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {names[s['experiment']]} | {s['dataset']} | {s['n']} | {s['TP']} | {s['TN']} | {s['FP']} | {s['FN']} | {rate(s['accuracy'])} | {rate(s['TPR'])} | {rate(s['FPR'])} |")
    lines += ['', 'TP: watermarked and detected; TN: unwatermarked and not detected; FP: false positive; FN: false negative.',
              'N/A: the corresponding class is absent from the experiment; do not replace this with 0%.', '',
              '## Protocol Fixed Before Execution', '',
              '- 100 images of size 512×512: block size 64, binary watermark size 21×21; 100 images of size 1024×1024: block size 128, watermark size 41×41.',
              '- The watermark is an approximately balanced binary 0/1 matrix, not a QR code whose content is decoded.',
              f"- S={settings['s_param']}; gap={settings['gap']}; payload seed={settings['payload_seed']}; phase seed={settings['phase_seed']}; master seed={settings['seed']}.",
              '- Valid spectral positions are selected randomly within [4,12] and are known at both embedding and detection; the extraction start is random and differs from (0,0), with a stride of half the block size.',
              '- Detection score: Pearson correlation between phase-projection scores and the reference watermark. A positive decision is made when score > threshold.',
              '- Five folds are formed by source image and balanced across the two resolutions; images with identical pixel content remain in the same fold. Each image is evaluated using a threshold calibrated only on images outside its fold.',
              f"- Target alpha={settings['alpha']}; the threshold is the order statistic at rank ceil((n+1)(1-alpha)) of the scores of UNWATERMARKED images. Scores of watermarked images are not used to select the threshold.",
              '- Alpha is a target under the assumption that negative samples are exchangeable, not the probability that an individual decision is correct. Calibration pools both datasets and provides no separate guarantee for each resolution.',
              '- Experiment 2 uses a reproducible random selection of exactly 50 watermarked images in each group; the assignments are recorded in decisions.csv.',
              '- Scores are computed for 400 inputs (200 original images and 200 watermarked versions) and reused for the 600 decisions across the three scenarios. The three scenarios are not independent samples.', '',
              '## Thresholds Calibrated Outside Each Test Fold', '', '| Fold | Unique Calibration Images | Test Images | Threshold |', '|---:|---:|---:|---:|']
    for c in calibrations:
        lines.append(f"| {c['fold']} | {c['calibration_n']} | {c['test_n']} | {c['threshold']:.8f} |")
    lines += ['', '## Images with False Positive or False Negative Decisions', '']
    errors = 0
    for r in records:
        if r['score_clean'] > r['threshold']:
            scenarios = '1 and 2' if not r['half_embedded'] else '1'
            lines.append(f"- False positive: `{r['filename']}`, experiment {scenarios}; score={r['score_clean']:.8f}, threshold={r['threshold']:.8f}, fold={r['fold']}.")
            errors += 1
        if r['score_embedded'] <= r['threshold']:
            scenarios = '2 and 3' if r['half_embedded'] else '3'
            lines.append(f"- False negative: `{r['filename']}`, experiment {scenarios}; score={r['score_embedded']:.8f}, threshold={r['threshold']:.8f}, fold={r['fold']}.")
            errors += 1
    if not errors:
        lines.append('No errors were observed in this test sample.')
    psnr = [r['psnr_embed_db'] for r in records if r['psnr_embed_db'] is not None]
    lines += ['', f'PSNR of watermarked images: mean {np.mean(psnr):.3f} dB; minimum {min(psnr):.3f} dB.' if psnr else 'PSNR: all values are infinite.', '',
              '## Scope of the Conclusions', '',
              'This is a cross-validation evaluation of the protocol, not an independent test of a final deployment threshold. Thresholds are not adjusted based on test results. Related images with nonidentical pixel content (the same scene, cropped or resized versions) may still introduce dependencies between folds.',
              'Other watermarks, incorrect phase keys, geometric attacks, JPEG compression, noise, and searches over multiple keys or positions have not been evaluated. Each protocol change requires recalibration and reevaluation.',
              'Do not infer that the true FPR is zero merely because no false positives are observed in a finite sample. These results must not be interpreted as proof of authorship.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-root', type=Path, default=ROOT / 'Image')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=20261018)
    parser.add_argument('--payload-seed', type=int, default=42)
    parser.add_argument('--phase-seed', type=int, default=20261011)
    parser.add_argument('--s-param', type=float, default=300)
    parser.add_argument('--gap', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.01)
    args = parser.parse_args(argv)
    if min(args.seed, args.payload_seed, args.phase_seed, args.gap) < 0:
        parser.error('seeds and gap must be nonnegative')
    if not np.isfinite(args.s_param) or args.s_param <= 0:
        parser.error('s-param must be positive and finite')
    calibrate_threshold(np.zeros(160), args.alpha)
    records = prepare_records(args.image_root, args.seed)
    output = args.output or ROOT / 'Results' / 'Detection_Research' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Output directory must be empty; refusing to overwrite results')
    settings = {k: v for k, v in vars(args).items() if k not in ('image_root', 'output')}
    settings.update(folds=5, score='pearson_soft_reference_v1', decision='strict_greater',
                    protocol='known_reference_known_layout_no_attack', calibration='clean_negatives_only')
    code_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'Source').rglob('*.py'))}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    manifest = dict(settings=settings, code_sha256=code_hashes, commit=commit.stdout.strip() or None,
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__),
                    inputs=records, status='running')
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for index, r in enumerate(records, 1):
        spec = DATASETS[r['dataset']]
        n, m = spec.block_size, spec.qr_size
        path = Path(r['path'])
        if sha256(path) != r['sha256']:
            raise ValueError(f'Input changed after preflight: {path}')
        image = read_image(path)
        reference = generate_watermark(m, args.payload_seed)
        phases = generate_phase_map(m, args.phase_seed)
        x, y = random_layout(n, m, args.gap, derived_seed(args.seed, r['dataset'], r['image_id'], 'layout'))
        start_x, start_y = random_extract_start(n, derived_seed(args.seed, r['dataset'], r['image_id'], 'extract'))
        layout = dict(x=x, y=y, offset=n // 16, gap=args.gap)
        detector = WatermarkDetector(reference, n, phases, **layout,
                                     start_x=start_x, start_y=start_y, stride=n // 2)
        q = calculate_q(args.s_param, m, n)
        wm = embed_watermark_into_image(image, reference, n, q, phases, **layout)
        clean_score, used = detector.measure(image)
        embedded_score, used_wm = detector.measure(wm)
        if used != used_wm:
            raise AssertionError('Extraction window mismatch')
        psnr = count_psnr(image, wm)
        r.update(score_clean=clean_score, score_embedded=embedded_score, blocks_used=used,
                 embed_x=x, embed_y=y, embed_offset=n // 16, extract_x=start_x, extract_y=start_y,
                 block_size=n, watermark_size=m, q=q,
                 psnr_embed_db=float(psnr) if np.isfinite(psnr) else None)
        with (output / 'scores.jsonl').open('a', encoding='utf-8') as checkpoint:
            checkpoint.write(json.dumps(r, allow_nan=False) + '\n')
        print(f'[{index}/200] {r["filename"]}: clean={clean_score:.6f}, embedded={embedded_score:.6f}', flush=True)
    calibrations, decisions, summaries = crossfit(records, args.alpha)
    write_csv(output / 'decisions.csv', decisions)
    write_csv(output / 'summary.csv', summaries)
    (output / 'calibration.json').write_text(json.dumps(calibrations, indent=2), encoding='utf-8')
    report = report_text(summaries, calibrations, records, settings)
    (output / 'REPORT_EN.md').write_text(report, encoding='utf-8')
    manifest.update(status='complete', scored_inputs=400, decisions=600, inputs=records)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    print(report, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
