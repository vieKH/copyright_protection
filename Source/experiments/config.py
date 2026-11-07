"""All parameters of the study in one place (values used in the report BKP_95)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # diplom/
IMAGE_DIR = ROOT / "Image"
RESULTS = ROOT / "Results"

TRIALS = 20                 # random binary objects (and random block subsets) per image
PHASE_SEED = 20261011       # secret phase key, shared by all images
DEMO_BITS_SEED = 7          # binary object of the illustration "smooth.png"

# Russian names of the test images (figure titles and tables)
NAMES_RU = {"boat": "Лодка", "bridge": "Мост", "cloudandsun": "Облака и солнце", "drink": "Напиток",
            "flower": "Цветок", "forest": "Лес", "house": "Дом", "livingroom": "Гостиная", "people": "Люди",
            "plane": "Самолёт", "port": "Порт", "tank": "Танк", "text": "Текст", "tower": "Башня",
            "tree": "Дерево"}


def image_files():
    """Every .tif/.tiff test image in Image/; each one needs a Russian name in NAMES_RU."""
    files = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    missing = [p.name for p in files if p.stem not in NAMES_RU]
    if missing:
        raise ValueError(f"add Russian names to NAMES_RU (config.py) for: {', '.join(missing)}")
    return files


# recommended (fixed) parameters
D0 = 0.72                   # watermark variance                     (2.4.1)
L0 = 21                     # object side, 441 bits                  (2.4.2)
B0 = 128                    # DFT block = period                     (2.4.3)
A0 = 28                     # object offset in the spectrum, rho~0.45 (2.4.4)
STEP = 2                    # spacing between carriers (bins) at B = 128
STRIDE0 = 64                # window step B/2: 49 blocks on 512x512   (3.2)
GRID_START = (0, 0)         # the block grid starts at the image corner
DECODER = "phase_vote"      # voting of phase projections, all blocks (3.2)

# studied ranges
D_GRID = [0.1, 0.25, 0.5, 0.72, 1.0, 2.0, 4.0, 8.0, 16.0]
B_GRID = [64, 128, 256]
L_GRID = [7, 14, 21, 28, 42]
ANCHORS = [2, 8, 14, 20, 24, 28]          # rho ~ 0.21 ... 0.45 for L = 21
DENSE_STRIDE = 16                          # pool of 625 blocks for the study of T (3.2)
COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256] # numbers of blocks T, random subsets of the pool
SINGLE_TRIALS = 16                         # random positions of the single block

ATTACKS = {"contrast": [0.5, 0.7, 0.9],
           "noise": [1, 2, 5, 10, 20, 50, 100],
           "sharpen": [3, 5, 7, 9, 11, 13, 15]}
POSITION_ATTACKS = {"blur": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
                    "jpeg": [30, 40, 50, 60, 70, 80, 90, 95]}


def default_layout_args(B: int, L: int):
    """Same normalised band for every B: offset B/8, step B/64 (used in 2.4.1-2.4.3)."""
    return dict(B=B, L=L, a=B // 8, step=max(1, B // 64))
