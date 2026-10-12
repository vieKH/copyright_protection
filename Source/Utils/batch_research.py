"""Reproducible dataset experiments without changing the single-image APIs."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib

import numpy as np

from .distorter import ImageDistorter
from .extraction_research import _robust_zscore, recover_qr_from_score
from .phases import resolve_phase_map
from .utils import qr_to_spectrum_positions


@dataclass(frozen=True)
class DatasetSpec:
    size: int
    block_size: int
    qr_size: int


DATASETS = {512: DatasetSpec(512, 64, 21), 1024: DatasetSpec(1024, 128, 41)}


@dataclass(frozen=True)
class AttackSpec:
    name: str
    parameter: str
    values: tuple


def _frange(start, stop, step):
    return tuple(round(start + i * step, 6) for i in range(int(round((stop-start)/step)) + 1))


def attack_specs():
    """Same sweeps and distortion semantics as research_distorter, plus a baseline."""
    return (
        AttackSpec('no_attack', 'none', (0,)),
        AttackSpec('contrast', 'alpha', _frange(0.7,1.3,0.1)),
        AttackSpec('rot_rest', 'angle_deg', _frange(0,42,7)),
        AttackSpec('rotation', 'angle_deg', _frange(1,90,8.9)),
        AttackSpec('scale_rest', 'scale_factor', _frange(0.55,1.45,0.15)),
        AttackSpec('scale', 'scale_factor', _frange(0.55,1.45,0.15)),
        AttackSpec('cut', 'area_fraction', _frange(0.2,0.9,0.1)),
        AttackSpec('shift', 'shift_fraction', _frange(0.02,0.20,0.02)),
        AttackSpec('smooth', 'window_size', tuple(range(3,16,2))),
        AttackSpec('gauss_blur', 'sigma', _frange(1,4,0.5)),
        AttackSpec('sharpen', 'window_size', tuple(range(3,16,2))),
        AttackSpec('median', 'window_size', tuple(range(3,16,2))),
        AttackSpec('wh_noise', 'variance', tuple(range(400,1001,100))),
        AttackSpec('salt_pepper', 'noise_fraction', _frange(0.05,0.5,0.05)),
        AttackSpec('jpeg', 'quality_factor', tuple(range(30,91,10))),
    )


def derived_seed(master, *labels):
    """Stable RNG streams independent of scheduling or selection order."""
    key = '|'.join([str(master), *map(str, labels)])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') & ((1 << 63)-1)


@lru_cache(maxsize=16)
def valid_layouts(n, qr_size, gap, lower=4, upper=12):
    offset = n // 16
    pairs = []
    for x in range(lower, upper+1):
        for y in range(lower, upper+1):
            try:
                qr_to_spectrum_positions(qr_size,n,x,y,offset,gap=gap)
            except ValueError:
                continue
            pairs.append((x,y))
    if not pairs:
        raise ValueError(f'No valid layout: N={n}, QR={qr_size}, gap={gap}, range={lower}..{upper}')
    return tuple(pairs)


def random_layout(n, qr_size, gap, seed):
    pairs = valid_layouts(n, qr_size, gap)
    return pairs[int(np.random.default_rng(seed).integers(len(pairs)))]


def tiled_median(image, size, rows_per_tile=16):
    """Exact edge-padded odd median, bounded temporary memory even at 1024x1024."""
    size = ImageDistorter._make_odd(int(size))
    pad = size // 2
    padded = np.pad(image, pad, mode='edge')
    result = np.empty_like(image)
    for r in range(0, image.shape[0], rows_per_tile):
        end = min(r+rows_per_tile,image.shape[0])
        windows = np.lib.stride_tricks.sliding_window_view(padded[r:end+2*pad],(size,size))
        result[r:end] = np.median(windows,axis=(-2,-1))
    return result


def apply_attack(image, name, value, seed):
    if name == 'no_attack':
        return image
    d = ImageDistorter(image)
    if name == 'contrast': return d.contrast(float(value)).image
    if name == 'rot_rest': return d.rotation_rest(float(value),interpolation='bilinear',fill_value=0).image
    if name == 'rotation': return d.rotation(float(value),interpolation='bilinear',fill_value=0).image
    if name == 'scale_rest': return d.scale_rest(float(value),interpolation='bilinear',fill_value=0).image
    if name == 'scale': return d.scale(float(value),interpolation='bilinear',fill_value=0).image
    if name == 'cut': return d.cutout(float(value),position='center',fill_value=0).image
    if name == 'shift':
        return d.translate(int(round(float(value)*image.shape[0])),int(round(float(value)*image.shape[1])),fill_value=0).image
    if name == 'smooth': return d.smooth(int(value)).image
    if name == 'gauss_blur': return d.gauss_blur(float(value)).image
    if name == 'sharpen': return d.sharpen(int(value)).image
    if name == 'median': return tiled_median(image,int(value))
    if name == 'wh_noise': return d.white_noise(float(value),seed=seed).image
    if name == 'salt_pepper': return d.salt_pepper(float(value),seed=seed).image
    if name == 'jpeg': return d.jpeg(int(value)).image
    raise ValueError(f'Unknown attack: {name}')


class BatchDecoder:
    """Vectorized equivalent of phase_sign=-1, detrend=False block extraction.

    Keep only the selected bins when accumulating FFTs. Offset correction uses
    actual window coordinates, including non-multiple block strides.
    """
    def __init__(self, n, qr_size, phi, x, y, offset, gap, start_x, start_y, stride):
        self.n, self.qr_size = n, qr_size
        self.start_x, self.start_y, self.stride = start_x,start_y,stride
        positions = qr_to_spectrum_positions(qr_size,n,x,y,offset,gap=gap)
        self.u = np.array([p['row'] for p in positions])
        self.v = np.array([p['col'] for p in positions])
        self.uc, self.vc = (-self.u)%n, (-self.v)%n
        self.carrier = np.exp(1j * resolve_phase_map(phi,qr_size).ravel())
        self.corrections = {}

    def decode(self, image):
        image = np.asarray(image)
        if image.ndim != 2 or not np.all(np.isfinite(image)):
            raise ValueError('Input must be finite grayscale pixels')
        n = self.n
        total = np.zeros(self.qr_size**2,dtype=np.complex128)
        total_conj = np.zeros_like(total)
        used = 0
        for r in range(self.start_x,image.shape[0]-n+1,self.stride):
            for c in range(self.start_y,image.shape[1]-n+1,self.stride):
                key = (r%n,c%n)
                if key not in self.corrections:
                    self.corrections[key] = (
                        np.exp(-2j*np.pi*(self.u*key[0]+self.v*key[1])/n),
                        np.exp(-2j*np.pi*(self.uc*key[0]+self.vc*key[1])/n),
                    )
                correction,correction_conj = self.corrections[key]
                spectrum = np.fft.fft2(image[r:r+n,c:c+n].astype(np.float64))
                total += spectrum[self.u,self.v] * correction
                total_conj += spectrum[self.uc,self.vc] * correction_conj
                used += 1
        if not used:
            raise ValueError('No complete extraction window is available')
        raw = 0.5*(np.real(total*np.conj(self.carrier))+np.real(total_conj*self.carrier))/used
        score = _robust_zscore(raw.reshape(self.qr_size,self.qr_size))
        recovered,threshold = recover_qr_from_score(score)
        return recovered,used,threshold
