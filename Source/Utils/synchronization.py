"""Blind similarity synchronization for the periodic Fourier watermark.

Coordinates are (row, column). Observed q = scale * R(angle) * p + t.
The estimated origin is p at the observed image centre, modulo block size.
No original image, payload values, or attack parameters enter the estimator.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage, optimize, fft
from .utils import build_wm_spectrum, qr_to_spectrum_positions
from .extraction_research import recover_qr_from_score


@dataclass(frozen=True)
class SyncSettings:
    seed: int = 20261002
    pilot_pairs: int = 24
    energy_fraction: float = 0.35
    angle_min: float = -180.0
    angle_max: float = 180.0
    angle_step: float = 1.0
    scale_min: float = 0.75
    scale_max: float = 1.30
    scale_step: float = 0.01
    min_coherence: float = 0.72
    min_spectral_score: float = 1.5

    def __post_init__(self):
        if not isinstance(self.pilot_pairs, int) or self.pilot_pairs < 8:
            raise ValueError('pilot_pairs must be an integer >= 8')
        values = [self.energy_fraction, self.angle_min, self.angle_max, self.angle_step,
                  self.scale_min, self.scale_max, self.scale_step, self.min_coherence,
                  self.min_spectral_score]
        if not np.all(np.isfinite(values)):
            raise ValueError('Synchronization settings must be finite')
        if not 0 < self.energy_fraction < 1:
            raise ValueError('energy_fraction must be in (0, 1)')
        if not (-180 <= self.angle_min <= self.angle_max <= 180) or self.angle_step <= 0:
            raise ValueError('Invalid angle search range/step')
        if not 0 < self.scale_min <= self.scale_max or self.scale_step <= 0:
            raise ValueError('Invalid scale search range/step')
        if not 0 < self.min_coherence <= 1 or self.min_spectral_score <= 0:
            raise ValueError('Invalid synchronization acceptance thresholds')


def default_sync_settings():
    """Read shared globals at call time; explicit SyncSettings override them."""
    from .. import config
    return SyncSettings(**{name: getattr(config, 'SYNC_'+name.upper())
                           for name in SyncSettings.__dataclass_fields__})


@dataclass(frozen=True)
class SyncEstimate:
    angle_deg: float
    scale: float
    origin_row: float
    origin_col: float
    coherence: float
    spectral_score: float
    accepted: bool


@dataclass(frozen=True)
class SynchronizedExtraction:
    estimate: SyncEstimate
    recovered_qr: np.ndarray | None
    score_map: np.ndarray | None
    threshold: float | None
    usable_bits: np.ndarray | None


def rotation(angle):
    a = np.deg2rad(angle)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def pilot_layout(qr_size, n, x=None, y=None, offset=None, gap=None, settings=None):
    """Deterministic pilot pairs outside payload AND its conjugates, with guards."""
    settings = settings or default_sync_settings()
    payload = qr_to_spectrum_positions(qr_size, n, x, y, offset, gap=gap)
    occupied = np.array([(p['row'], p['col']) for p in payload])
    occupied = (occupied + n // 2) % n - n // 2
    occupied = np.concatenate([occupied, -occupied])
    candidates = []
    for u in range(3, n // 2):
        for v in range(-n // 2 + 1, n // 2):
            radius = np.hypot(u, v) / n
            if .16 <= radius <= .32 and abs(v) >= 3:
                p = np.array([u, v])
                if np.min(np.linalg.norm(occupied - p, axis=1)) >= 2:
                    candidates.append(p)
    rng = np.random.default_rng(settings.seed)
    selected = []
    for index in rng.permutation(len(candidates)):
        p = candidates[index]
        if not selected or min(np.linalg.norm(p-q) for q in selected + [-q for q in selected]) >= 4:
            selected.append(p)
            if len(selected) == settings.pilot_pairs:
                break
    if len(selected) != settings.pilot_pairs:
        raise ValueError('Not enough guarded pilot positions: increase block size or reduce pilot_pairs')
    points = np.asarray(selected, dtype=int)
    phases = rng.uniform(-np.pi, np.pi, settings.pilot_pairs)
    return points, phases


def synchronized_spectrum(qr, n, q, phi, x=None, y=None, offset=None, gap=None, settings=None):
    """Same total pre-quantization energy as the legacy payload at Q=q."""
    settings = settings or default_sync_settings()
    if not np.isfinite(q) or q < 0:
        raise ValueError('q must be finite and nonnegative')
    spectrum = q * build_wm_spectrum(qr, n, phi, x, y, offset, gap)
    points, phases = pilot_layout(len(qr), n, x, y, offset, gap, settings)
    energy = float(np.sum(np.abs(spectrum)**2))
    if energy == 0:
        raise ValueError('Synchronized embedding needs nonzero payload energy')
    spectrum *= np.sqrt(1 - settings.energy_fraction)
    amplitude = np.sqrt(energy * settings.energy_fraction / (2 * len(points)))
    for (u, v), phase in zip(points, phases):
        spectrum[u % n, v % n] = amplitude * np.exp(1j * phase)
        spectrum[-u % n, -v % n] = amplitude * np.exp(-1j * phase)
    return spectrum


def embed_synchronized(image, qr, size_region, q, phi, x=None, y=None, offset=None,
                       gap=None, settings=None):
    """Embed payload and pilot in one rounding operation (not two uint8 additions)."""
    image = np.asarray(image)
    if image.ndim != 2 or not np.all(np.isfinite(image)) or np.any(image < 0) or np.any(image > 255):
        raise ValueError('image must be finite grayscale pixels in [0,255]')
    if any(d % size_region for d in image.shape):
        raise ValueError('Embedding image dimensions must be divisible by block size')
    spec = synchronized_spectrum(qr, size_region, q, phi, x, y, offset, gap, settings)
    pattern = np.fft.ifft2(spec).real
    tiled = np.tile(pattern, (image.shape[0]//size_region, image.shape[1]//size_region))
    return np.rint(np.clip(image.astype(float) + tiled, 0, 255)).astype(np.uint8)


def _windowed_image(image):
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or min(image.shape) < 32 or not np.all(np.isfinite(image)):
        raise ValueError('Synchronization needs a finite 2D image at least 32x32')
    window = np.hanning(image.shape[0])[:, None] * np.hanning(image.shape[1])[None, :]
    mean = np.sum(image * window) / window.sum()
    return (image-mean) * window, float(window.sum())


def _coefficients(data, frequencies, normalization):
    """Direct centred Fourier samples: no second image interpolation after attack."""
    h, w = data.shape
    r = np.arange(h) - (h-1)/2
    c = np.arange(w) - (w-1)/2
    out = []
    for first in range(0, len(frequencies), 32):
        f = frequencies[first:first+32]
        er = np.exp(-2j*np.pi*f[:, 0, None]*r)
        ec = np.exp(-2j*np.pi*f[:, 1, None]*c)
        rows = np.einsum('kh,hw->kw', er, data, optimize=False)
        out.extend(np.sum(rows*ec, axis=1)/normalization)
    return np.asarray(out)


def _observed_frequencies(points, n, angle, scale):
    return (points/n) @ rotation(angle).T / scale


def _origin_from_phases(z, points, phases, n):
    z = z * np.exp(-1j*phases)
    # Equal phase votes prevent a few strong host-image coefficients dominating.
    unit = z / np.maximum(np.abs(z), 1e-12)
    # A half-pixel true origin can sit below a false integer-grid sidelobe.
    # Oversample using SIGNED Fourier indices, then refine continuously.
    oversampling = 4
    grid_n = oversampling*n
    grid = np.zeros((grid_n, grid_n), complex)
    for (u, v), value in zip(points, unit):
        grid[u % grid_n, v % grid_n] = value
        grid[-u % grid_n, -v % grid_n] = value.conjugate()
    corr = np.fft.fft2(grid).real
    origin = np.array(np.unravel_index(np.argmax(corr), corr.shape), dtype=float)/oversampling
    def objective(o):
        return -float(np.mean(np.real(unit*np.exp(-2j*np.pi*(points@o)/n))))
    fit = optimize.minimize(objective, origin, method='Nelder-Mead',
                            options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 150})
    # Robust complex fitting uses amplitude information but limits host outliers.
    amplitude = max(float(np.median(np.abs(z))), 1e-12)
    def residual(p):
        difference = z/amplitude - p[2]*np.exp(2j*np.pi*(points@p[:2])/n)
        return np.concatenate([difference.real,difference.imag])
    robust = optimize.least_squares(residual, [*fit.x,1.0], loss='soft_l1', f_scale=.35,
                                    max_nfev=100)
    origin = robust.x[:2] % n
    coherence = -objective(origin)
    return origin, coherence


def estimate_synchronization(image, qr_size, size_region, x=None, y=None, offset=None,
                             gap=None, settings=None):
    settings = settings or default_sync_settings()
    n = size_region
    points, phases = pilot_layout(qr_size, n, x, y, offset, gap, settings)
    data, norm = _windowed_image(image)
    shape = tuple(fft.next_fast_len(2*d) for d in data.shape)
    magnitude = np.abs(fft.fftshift(fft.fft2(data, s=shape)))
    background = ndimage.gaussian_filter(magnitude, 6)
    whitened = magnitude / np.maximum(background, 1e-9)
    # Bound individual votes: isolated host peaks must not win the whole search.
    whitened = np.minimum(whitened, 15)
    def scores(params):
        params = np.atleast_2d(params)
        a = np.deg2rad(params[:, 0])
        f = points/n
        fr = (np.cos(a)[:, None]*f[:,0] - np.sin(a)[:,None]*f[:,1])/params[:,1,None]
        fc = (np.sin(a)[:, None]*f[:,0] + np.cos(a)[:,None]*f[:,1])/params[:,1,None]
        coords = np.array([fr*shape[0]+shape[0]//2, fc*shape[1]+shape[1]//2])
        samples = ndimage.map_coordinates(whitened, coords, order=1, mode='constant', cval=0)
        return np.mean(samples, axis=1)
    angles = np.arange(settings.angle_min, settings.angle_max+settings.angle_step*.1, settings.angle_step)
    scales = np.arange(settings.scale_min, settings.scale_max+settings.scale_step*.1, settings.scale_step)
    if len(angles)*len(scales) > 250000:
        raise ValueError('Search grid too large; increase angle_step or scale_step')
    params = np.array(np.meshgrid(angles, scales, indexing='ij')).reshape(2,-1).T
    # Include identity exactly even when it is not on the requested grid.
    if settings.angle_min <= 0 <= settings.angle_max and settings.scale_min <= 1 <= settings.scale_max:
        params = np.vstack([params, [0,1]])
    params[:, 0] = np.clip(params[:, 0], settings.angle_min, settings.angle_max)
    params[:, 1] = np.clip(params[:, 1], settings.scale_min, settings.scale_max)
    coarse = scores(params)
    order = np.argsort(coarse)[::-1]
    starts = []
    for i in order:
        p = params[i]
        if all(abs(p[0]-q[0]) > 2 or abs(p[1]-q[1]) > .025 for q in starts):
            starts.append(p)
        if len(starts) == 6:
            break
    candidates = []
    for start in starts:
        bounds = [(max(settings.angle_min,start[0]-2), min(settings.angle_max,start[0]+2)),
                  (max(settings.scale_min,start[1]-.025), min(settings.scale_max,start[1]+.025))]
        fit = optimize.minimize(lambda p: -scores(p)[0], start, method='Nelder-Mead', bounds=bounds,
                                options={'xatol':1e-5, 'fatol':1e-6, 'maxiter':180})
        angle, scale = fit.x
        z = _coefficients(data, _observed_frequencies(points,n,angle,scale), norm)
        origin, coherence = _origin_from_phases(z,points,phases,n)
        candidates.append((coherence, -float(fit.fun), angle, scale, origin))
    coherence, score, angle, scale, origin = max(candidates, key=lambda c:c[0])
    accepted = coherence >= settings.min_coherence and score >= settings.min_spectral_score
    return SyncEstimate(float(angle),float(scale),float(origin[0]),float(origin[1]),
                        float(coherence),float(score),bool(accepted))


def extract_synchronized(image, qr_size, size_region, phi, x=None, y=None, offset=None,
                         gap=None, settings=None, expected_ones=None):
    """Blind pilot-only registration, then payload projection at transformed bins.

    Rejected registration returns no payload. Acceptance is heuristic, not a
    calibrated watermark-presence/authentication test.
    """
    settings = settings or default_sync_settings()
    estimate = estimate_synchronization(image,qr_size,size_region,x,y,offset,gap,settings)
    if not estimate.accepted:
        return SynchronizedExtraction(estimate,None,None,None,None)
    return _decode_registered(image,qr_size,size_region,phi,estimate,x,y,offset,gap,expected_ones)


def _decode_registered(image, qr_size, size_region, phi, estimate, x=None, y=None,
                       offset=None, gap=None, expected_ones=None):
    """Decode fixed geometry. Private hook for the explicitly labelled oracle benchmark."""
    if not np.isfinite(phi):
        raise ValueError('phi must be finite')
    positions = qr_to_spectrum_positions(qr_size,size_region,x,y,offset,gap)
    points = np.array([(p['row'],p['col']) for p in positions])
    points = (points + size_region//2) % size_region - size_region//2
    freqs = _observed_frequencies(points,size_region,estimate.angle_deg,estimate.scale)
    usable = np.max(np.abs(freqs),axis=1) < .49
    data,norm = _windowed_image(image)
    z = _coefficients(data,freqs,norm)
    origin = np.array([estimate.origin_row,estimate.origin_col])
    raw = np.real(z*np.exp(-1j*phi-2j*np.pi*(points@origin)/size_region))
    # Erasures are explicit (-1), not silently guessed from aliased frequencies.
    score = raw.reshape(qr_size,qr_size)
    mask = usable.reshape(qr_size,qr_size)
    if not np.any(usable):
        return SynchronizedExtraction(estimate,None,None,None,mask)
    if expected_ones is not None and not np.all(usable):
        raise ValueError('expected_ones decoding requires all payload bins to be usable')
    known, threshold = recover_qr_from_score(raw[usable], expected_ones)
    recovered = np.full(qr_size*qr_size,-1,dtype=np.int8)
    recovered[usable] = known
    score = score.copy()
    score[~mask] = np.nan
    return SynchronizedExtraction(estimate,recovered.reshape(qr_size,qr_size),score,threshold,mask)
