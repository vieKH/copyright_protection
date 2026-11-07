"""Basic properties of the method:  python -m pytest tests  (or python tests/test_core.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wm


def test_period_is_real_with_prescribed_variance():
    lay = wm.Layout(128, 21, 28, 2)
    bits, th = wm.binary_object(21, 1), wm.phase_key(21, 2)
    w = wm.alpha_from_variance(128, bits, 0.72) * wm.period(lay, bits, th)
    assert abs(w.mean()) < 1e-12 and np.isclose(np.mean(w ** 2), 0.72)       # Parseval


def test_carriers_have_no_conjugate_collisions():
    for B, L in [(64, 7), (128, 21), (256, 42)]:
        lay = wm.Layout(B, L, B // 8, max(1, B // 64))
        u, v = lay.positions
        assert len(set(zip(u, v))) == L * L


def test_flat_image_is_decoded_without_errors():
    img = np.full((512, 512), 128, np.uint8)
    lay = wm.Layout(128, 21, 28, 2)
    bits, th = wm.binary_object(21, 3), wm.phase_key(21, 4)
    y, _ = wm.embed(img, lay, bits, th, 0.72)
    res, n = wm.multi_block_ber(y, lay, bits, th, 64, (5, 9), np.random.default_rng(0))
    assert all(res[n][d] == 0 for d in ("phase_vote", "avg_phase", "avg_mag"))


def test_contrast_invariance():
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(120, 20, (512, 512)), 0, 255).astype(np.uint8)
    lay = wm.Layout(128, 21, 28, 2)
    bits, th = wm.binary_object(21, 5), wm.phase_key(21, 6)
    y, _ = wm.embed(img, lay, bits, th, 0.72)
    co = wm.grid(y.shape, 128, 64, (0, 0))
    d0 = wm.decode(wm.carriers(y, lay, co), th, [len(co)])[len(co)]["phase_vote"]
    d1 = wm.decode(wm.carriers(0.5 * y.astype(float), lay, co), th, [len(co)])[len(co)]["phase_vote"]
    assert np.array_equal(d0, d1)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
