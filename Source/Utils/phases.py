"""Shared per-bit phases, repeated in every block; seeds are not cryptographic keys."""
from typing import Union
import numpy as np

Phase = Union[float, np.ndarray]


def generate_phase_map(qr_size: int, seed: int) -> np.ndarray:
    """Independent uniform phases for ALL bit positions, including zero bits."""
    if isinstance(qr_size, (bool, np.bool_)) or not isinstance(qr_size, (int, np.integer)) or qr_size <= 0:
        raise ValueError("qr_size must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("phase seed must be a non-negative integer")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return rng.uniform(-np.pi, np.pi, size=(int(qr_size), int(qr_size)))


def resolve_phase_map(phi: Phase, qr_size: int) -> np.ndarray:
    """Expand a legacy scalar phase or validate a per-bit map."""
    phases = np.asarray(phi, dtype=np.float64)
    if phases.ndim == 0:
        phases = np.full((qr_size, qr_size), float(phases))
    elif phases.shape != (qr_size, qr_size):
        raise ValueError("phi must be a finite scalar or a (qr_size, qr_size) phase map")
    if not np.all(np.isfinite(phases)):
        raise ValueError("phi must contain only finite phases")
    return phases


def configured_phase(qr_size: int) -> Phase:
    """Resolve the explicit phase mode used by non-synchronized scripts."""
    from .. import config
    if config.PHASE_MODE == "random":
        return generate_phase_map(qr_size, config.PHASE_SEED)
    if config.PHASE_MODE == "fixed":
        return float(config.FIXED_PHASE)
    raise ValueError("PHASE_MODE must be 'random' or 'fixed'")
