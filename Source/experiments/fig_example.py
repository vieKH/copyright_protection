"""Illustration for subsection 1.3.2: how a 4x4 binary object becomes a periodic watermark."""
import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "Source"))
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wm import Layout, binary_object, phase_key, period
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 13})
B, L = 32, 4
lay = Layout(B, L, 7)
bits = binary_object(L, 7); theta = phase_key(L, 11)
u, v = lay.positions; b = bits.ravel().astype(bool)
w = period(lay, bits, theta)
fig, axs = plt.subplots(2, 2, figsize=(9.5, 9.6)); ax = axs.ravel()
a0 = ax[0]; a0.imshow(1 - bits, cmap="gray", vmin=0, vmax=1)
for r in range(L):
    for s in range(L): a0.text(s, r, str(bits[r, s]), ha="center", va="center", color="white" if bits[r, s] else "black", fontsize=16)
a0.set_xticks([]); a0.set_yticks([]); a0.set_title("а) бинарный объект 4×4")
def c(x): return ((x + B // 2) % B) - B // 2
a1 = ax[1]
a1.scatter(c(v[~b]), c(u[~b]), s=55, facecolors="none", edgecolors="0.6", label="носитель бита 0")
a1.scatter(c(v[b]), c(u[b]), s=55, color="#1f5fa8", label="бит 1: $e^{j\\theta}$")
a1.scatter(c(-v[b]), c(-u[b]), s=55, color="#d9792b", marker="s", label="сопряжённая пара")
a1.set_xlim(-B // 2, B // 2); a1.set_ylim(B // 2, -B // 2); a1.set_aspect("equal")
a1.axhline(0, color="0.85", lw=0.8); a1.axvline(0, color="0.85", lw=0.8)
a1.set_xlabel("v"); a1.set_ylabel("u"); a1.set_title("б) спектр одного периода")
a1.legend(fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
lim = np.abs(w).max()
ax[2].imshow(w, cmap="gray", vmin=-lim, vmax=lim); ax[2].set_title("в) один период ЦВЗ, 32×32")
ax[3].imshow(np.tile(w, (4, 4)), cmap="gray", vmin=-lim, vmax=lim); ax[3].set_title("г) ЦВЗ на фрагменте 128×128")
for k in (32, 64, 96):
    ax[3].axhline(k - 0.5, color="#d9792b", ls="--", lw=1); ax[3].axvline(k - 0.5, color="#d9792b", ls="--", lw=1)
for a in ax[2:]: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); out = ROOT / "Results" / "figures" / "fig_example.png"; out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=200)
print(out)
