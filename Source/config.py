"""Shared research configuration. Restart scripts after editing this file."""
# Empty Fourier bins between payload bits on BOTH axes; step = gap + 1.
# Research range for this release: 0, 1, 2.
EMBED_GAP = 2
REGION_SIZE = 128
EXTRACT_STRIDE = 32
EMBED_X = REGION_SIZE // 8
EMBED_Y = REGION_SIZE // 8
EMBED_OFFSET = REGION_SIZE // 16
