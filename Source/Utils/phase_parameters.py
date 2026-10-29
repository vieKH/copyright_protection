"""Controlled parameter experiments using the existing phase-aware decoder.

The controlled energy is the spatial watermark variance BEFORE uint8 rounding.
Actual MSE/PSNR/SSIM and clipping are measured separately for every embedding.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np

from .batch_research import apply_attack, derived_seed
from .chapter1_validation import count_ssim
from .phases import generate_phase_map
from .utils import (build_wm_spectrum, count_psnr, embed_watermark_into_image,
                    generate_watermark, qr_to_spectrum_positions)
from .voting_research import compare_methods

METHODS = ('complex_average', 'phase_vote', 'magnitude_vote')
ATTACKS = ('none', 'jpeg90', 'jpeg70', 'blur1')


@dataclass(frozen=True)
class Condition:
    experiment: str
    block_size: int
    payload_size: int
    x: int
    y: int
    offset: int
    gap: int

    @property
    def condition_id(self):
        return (f'{self.experiment}_B{self.block_size}_L{self.payload_size}'
                f'_x{self.x}_y{self.y}_o{self.offset}_g{self.gap}')

    def layout(self):
        return dict(x=self.x, y=self.y, offset=self.offset, gap=self.gap)


def frequency_metrics(condition):
    """Use signed, unshifted FFT bins; include ALL payload positions."""
    c = condition
    pp = qr_to_spectrum_positions(c.payload_size, c.block_size, **c.layout())
    ff = np.fft.fftfreq(c.block_size)
    radii = np.array([np.hypot(ff[p['row']], ff[p['col']]) for p in pp])
    return dict(frequency_radius_mean=float(radii.mean()),
                frequency_radius_min=float(radii.min()),
                frequency_radius_max=float(radii.max()))


def make_plan(settings):
    """Only one nominated axis changes within each experiment.

    Payload sweep: same B/x/y/offset/gap. DFT sweep: same L and normalized
    x/y/offset anchors, with fixed gap in FFT bins. Frequency sweep: same B/L/gap.
    Invalid user-requested geometries raise; none are silently replaced.
    """
    b, l, gap = settings['base_block'], settings['base_payload'], settings['gap']
    conditions = []
    if 'payload' in settings['experiments']:
        for side in sorted(set(settings['payload_sizes'] + [l])):
            conditions.append(Condition('payload', b, side, b//8, b//8, b//16, gap))
    if 'dft' in settings['experiments']:
        for n in sorted(set(settings['dft_sizes'] + [b])):
            conditions.append(Condition('dft', n, l, n//8, n//8, n//16, gap))
    if 'frequency' in settings['experiments']:
        candidates = []
        for x in range(2, b//3 + 1, 3):
            for y in range(2, b//3 + 1, 3):
                c = Condition('frequency', b, l, x, y, b//16, gap)
                try:
                    radius = frequency_metrics(c)['frequency_radius_mean']
                except ValueError:
                    continue
                candidates.append((radius, x, y, c))
        if not candidates:
            raise ValueError('No admissible frequency locations')
        candidates.sort(key=lambda t: t[:3])
        picks = np.linspace(0, len(candidates)-1,
                            min(settings['frequency_locations'], len(candidates)), dtype=int)
        chosen = {candidates[int(i)][3] for i in picks}
        chosen.add(Condition('frequency', b, l, b//8, b//8, b//16, gap))
        conditions.extend(sorted(chosen, key=lambda c: (frequency_metrics(c)['frequency_radius_mean'], c.x, c.y)))
    for c in conditions:
        qr_to_spectrum_positions(c.payload_size, c.block_size, **c.layout())
    if not conditions:
        raise ValueError('Empty experiment plan')
    return conditions


def available_windows(shape, block_size, start, stride):
    if stride < 1 or any(v < 0 or v >= block_size for v in start):
        raise ValueError('Invalid extraction origin or stride')
    return math.prod(max(0, (dim-block_size-origin)//stride + 1)
                     for dim, origin in zip(shape, start))


def embed_at_variance(image, condition, variance, payload_seed, phase_seed):
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('variance must be finite and positive')
    c = condition
    bits = generate_watermark(c.payload_size, payload_seed)
    phase = generate_phase_map(c.payload_size, phase_seed)
    ones = int(bits.sum())
    if not ones:
        raise ValueError('Payload needs at least one one-bit')
    q = c.block_size**2 * math.sqrt(variance / (2*ones))
    spectrum = build_wm_spectrum(bits, c.block_size, phase, **c.layout())
    spatial = q*np.fft.ifft2(spectrum)
    if np.max(np.abs(spatial.imag)) > 1e-9:
        raise RuntimeError('Watermark spectrum is not Hermitian')
    spatial = spatial.real
    measured = float(spatial.var())
    if not np.isclose(measured, variance, rtol=1e-10, atol=1e-12):
        raise RuntimeError('Spatial variance does not match the fixed budget')
    wm = embed_watermark_into_image(image, bits, c.block_size, q, phase, **c.layout())
    tiled = np.tile(spatial, (image.shape[0]//c.block_size, image.shape[1]//c.block_size))
    ideal = image.astype(float)+tiled
    psnr = count_psnr(image, wm)
    metrics = dict(embedding_coefficient=q, variance_target=variance,
                   variance_before_quantization=measured,
                   watermark_mean_before_quantization=float(spatial.mean()),
                   embedding_MSE=float(np.mean((wm.astype(float)-image.astype(float))**2)),
                   embedding_PSNR_dB=psnr if math.isfinite(psnr) else None,
                   embedding_PSNR_infinite=int(math.isinf(psnr)),
                   embedding_SSIM=count_ssim(image, wm),
                   clipping_fraction=float(np.mean((ideal < 0) | (ideal > 255))),
                   watermark_peak=float(np.max(np.abs(spatial))),
                   payload_ones=ones,
                   payload_sha256=hashlib.sha256(bits.tobytes()).hexdigest(),
                   phase_sha256=hashlib.sha256(phase.tobytes()).hexdigest())
    return wm, bits, phase, metrics


def attacks_for(condition, settings):
    return settings['frequency_attacks'] if condition.experiment == 'frequency' else ['none']


def trial_seeds(settings, repeat):
    # Matched trials across images, B and placement. Payload/phase streams differ.
    return (derived_seed(settings['seed'], 'payload', repeat),
            derived_seed(settings['seed'], 'phase', repeat))


def evaluate_image(image, record, settings, conditions):
    """Ground truth enters metric calculation only, never compare_methods()."""
    rows = []
    for c in conditions:
        stride = c.block_size//2
        available = available_windows(image.shape, c.block_size, settings['start'], stride)
        if image.shape[0] % c.block_size or image.shape[1] % c.block_size:
            raise ValueError('Image dimensions must be divisible by every block size')
        if available < max(settings['blocks']):
            raise ValueError(f'{c.condition_id}: requested K exceeds {available} windows')
        for repeat in range(1, settings['repeats']+1):
            payload_seed, phase_seed = trial_seeds(settings, repeat)
            # Same selected spatial windows for a given image/B across L and placement.
            window_seed = derived_seed(settings['seed'], record['dataset'], record['image_id'],
                                       c.block_size, 'windows')
            base = dict(asdict(c), condition_id=c.condition_id,
                        dataset=record['dataset'], image_id=record['image_id'],
                        filename=record['filename'], original_sha256=record['sha256'],
                        repeat=repeat, payload_seed=payload_seed, phase_seed=phase_seed,
                        window_seed=window_seed, start_row=settings['start'][0],
                        start_col=settings['start'][1], stride=stride,
                        available_windows=available, total_bits=c.payload_size**2,
                        status='ok', error='', **frequency_metrics(c))
            try:
                wm, bits, phase, metrics = embed_at_variance(
                    image, c, settings['variance'], payload_seed, phase_seed)
                embed_error = None
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                metrics = {}
                embed_error = exc
            for attack in attacks_for(c, settings):
                try:
                    if embed_error is not None:
                        raise embed_error
                    if attack == 'none':
                        attacked = wm
                    elif attack.startswith('jpeg'):
                        attacked = apply_attack(wm, 'jpeg', int(attack[4:]), 0)
                    elif attack == 'blur1':
                        attacked = apply_attack(wm, 'gauss_blur', 1.0, 0)
                    else:
                        raise ValueError(f'Unknown attack: {attack}')
                    outcomes = compare_methods(
                        attacked, c.payload_size, c.block_size, phase,
                        c.x, c.y, c.offset, c.gap, *settings['start'], stride,
                        counts=settings['blocks'], shuffle_seed=window_seed)
                    if [r.blocks for r in outcomes] != settings['blocks']:
                        raise RuntimeError('Decoder silently changed the requested K')
                    for outcome in outcomes:
                        for method in settings['methods']:
                            errors = int(np.count_nonzero(getattr(outcome, method) != bits))
                            accuracy = 1-errors/bits.size
                            rows.append(dict(base, **metrics, attack=attack, method=method,
                                             blocks_used=outcome.blocks, bit_errors=errors,
                                             BER=errors/bits.size, accuracy=accuracy,
                                             exact_recovery=int(errors == 0),
                                             meets_target=int(accuracy >= settings['accuracy_target'])))
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    for k in settings['blocks']:
                        for method in settings['methods']:
                            rows.append(dict(base, **metrics, attack=attack, method=method,
                                             blocks_used=k, status='error',
                                             error=f'{type(exc).__name__}: {exc}'))
    return rows


GROUP = ('experiment', 'condition_id', 'dataset', 'block_size', 'payload_size',
         'x', 'y', 'offset', 'gap', 'attack', 'method', 'blocks_used')
METRICS = ('BER', 'accuracy', 'exact_recovery', 'meets_target', 'embedding_PSNR_dB',
           'embedding_MSE', 'embedding_SSIM', 'clipping_fraction', 'embedding_PSNR_infinite',
           'variance_before_quantization', 'frequency_radius_mean')


def summarize(rows, settings):
    """Average repeat seeds WITHIN each image before any between-image statistics."""
    images = defaultdict(list)
    for r in rows:
        images[tuple(r[k] for k in GROUP)+(r['image_id'],)].append(r)
    per_image = []
    for key, trials in sorted(images.items()):
        record = dict(zip(GROUP+('image_id',), key))
        complete = (len(trials) == settings['repeats'] and
                    {r['repeat'] for r in trials} == set(range(1, settings['repeats']+1)) and
                    all(r['status'] == 'ok' for r in trials))
        record.update(status='ok' if complete else 'error', repeats_expected=settings['repeats'],
                      repeats_ok=sum(r['status'] == 'ok' for r in trials))
        if complete:
            for k in METRICS:
                values = [r[k] for r in trials if r[k] is not None]
                record[k] = float(np.mean(values)) if values else None
        per_image.append(record)
    grouped = defaultdict(list)
    for r in per_image:
        grouped[tuple(r[k] for k in GROUP)].append(r)
    summary = []
    for key, records in sorted(grouped.items()):
        good = [r for r in records if r['status'] == 'ok']
        result = dict(zip(GROUP, key), images_attempted=len(records), images_ok=len(good),
                      images_error=len(records)-len(good),
                      coverage_complete=len(good) == settings['count'])
        if good:
            values = np.array([r['BER'] for r in good])
            rng = np.random.default_rng(derived_seed(settings['seed'], *key, 'bootstrap'))
            boot = values[rng.integers(len(values), size=(settings['bootstrap'], len(values)))].mean(axis=1)
            result.update(mean_BER=float(values.mean()), median_BER=float(np.median(values)),
                          std_BER=float(values.std(ddof=1)) if len(values)>1 else None,
                          ci95_low=float(np.quantile(boot, .025)) if len(values)>1 else None,
                          ci95_high=float(np.quantile(boot, .975)) if len(values)>1 else None,
                          min_BER=float(values.min()), max_BER=float(values.max()),
                          images_meeting_mean_target=sum(r['accuracy'] >= settings['accuracy_target'] for r in good))
            for k in METRICS:
                if k == 'BER': continue
                values = [r[k] for r in good if r[k] is not None]
                result['mean_'+k] = float(np.mean(values)) if values else None
            result['images_with_finite_PSNR'] = sum(r['embedding_PSNR_dB'] is not None for r in good)
        summary.append(result)
    return per_image, summary


def paired_effects(per_image, settings):
    """Compare each condition with the SAME experiment's B/L/default-layout base."""
    groups = defaultdict(dict)
    for r in per_image:
        key = (r['experiment'], r['dataset'], r['attack'], r['method'], r['blocks_used'])
        groups[key].setdefault(r['condition_id'], []).append(r)
    effects = []
    for key, conditions in sorted(groups.items()):
        experiment = key[0]
        b, l = settings['base_block'], settings['base_payload']
        baseline = Condition(experiment, b, l, b//8, b//8, b//16, settings['gap']).condition_id
        reference = {r['image_id']: r for r in conditions.get(baseline, []) if r['status'] == 'ok'}
        for cid, rr in sorted(conditions.items()):
            if cid == baseline:
                continue
            current = {r['image_id']: r for r in rr if r['status'] == 'ok'}
            ids = sorted(reference.keys() & current.keys())
            effect = dict(zip(('experiment','dataset','attack','method','blocks_used'), key),
                          condition_id=cid, baseline_condition_id=baseline,
                          n_paired_images=len(ids), coverage_complete=len(ids) == settings['count'])
            if ids:
                delta = np.array([current[i]['BER']-reference[i]['BER'] for i in ids])
                rng = np.random.default_rng(derived_seed(settings['seed'], cid, *key, 'paired'))
                boot = delta[rng.integers(len(delta), size=(settings['bootstrap'], len(delta)))].mean(axis=1)
                effect.update(delta_BER=float(delta.mean()), ci95_low=float(np.quantile(boot,.025)) if len(delta)>1 else None,
                              ci95_high=float(np.quantile(boot,.975)) if len(delta)>1 else None)
            effects.append(effect)
    return effects
