import numpy as np
import matplotlib.pyplot as plt
import os
from Source.Utils import research_qr

SIZE_QR = 7
SIZE_REGION = 64
X = SIZE_REGION // 8
Y = SIZE_REGION // 8
OFFSET = SIZE_REGION // 16
PHASE = np.pi / 3
S = 140


def calculate_q(snr: float, l: int, n: int) -> float:
    return (255 * n * n) / (snr * l)

IMAGE_PATH = os.path.join('Image', 'lena.tif')
SAVE_PATH = os.path.join('Results','QR_Research')

if __name__ == '__main__':
    image = plt.imread(IMAGE_PATH)

    path_save = os.path.join(SAVE_PATH, 'QR_Research.png')

    # plot_qr_s_grid(image, SIZE_QR, SIZE_REGION, X, Y, OFFSET, PHASE, S)

    q = calculate_q(S, SIZE_QR, SIZE_REGION)
    research_qr(image, SIZE_QR, SIZE_REGION, X, Y, OFFSET, PHASE, q, path_save)


