"""Shared research configuration. Restart scripts after editing this file."""
# Empty Fourier bins between payload bits on BOTH axes; step = gap + 1.
# Research range for this release: 0, 1, 2.
EMBED_GAP = 1
REGION_SIZE = 64
EXTRACT_STRIDE = 32
EMBED_X = REGION_SIZE // 8
EMBED_Y = REGION_SIZE // 8
EMBED_OFFSET = REGION_SIZE // 16

# execute.py uses the synchronized path by default. The two old research scripts
# deliberately remain legacy baselines; use research_synchronization.py for sync.
SYNC_ENABLED = True
SYNC_SEED = 20261002
SYNC_PILOT_PAIRS = 24
# Fraction of the ORIGINAL payload energy allocated to synchronization.
SYNC_ENERGY_FRACTION = 0.35
SYNC_ANGLE_MIN = -180.0
SYNC_ANGLE_MAX = 180.0
SYNC_ANGLE_STEP = 1.0
SYNC_SCALE_MIN = 0.75
SYNC_SCALE_MAX = 1.30
SYNC_SCALE_STEP = 0.01
# Heuristic quality gates; not statistically calibrated authentication thresholds.
SYNC_MIN_COHERENCE = 0.72
SYNC_MIN_SPECTRAL_SCORE = 1.5
