# import numpy as np
import matplotlib.pyplot as plt
import os
# from Source.Utils import research_qr
#
# SIZE_QR = 14
# SIZE_REGION = 64
# X = SIZE_REGION // 8
# Y = SIZE_REGION // 8
# OFFSET = SIZE_REGION // 16
# PHASE = np.pi / 3
# S = 300
#
#
# def calculate_q(snr: float, l: int, n: int) -> float:
#     """ Calculate the QR decomposition of a signal """
#     return (255 * n * n) / (snr * l)
#
IMAGE_PATH = os.path.join('Image', 'lena.tif')
SAVE_PATH = os.path.join('Results','QR_Research')
#
# if __name__ == '__main__':
#     os.makedirs(SAVE_PATH, exist_ok=True)
#     image = plt.imread(IMAGE_PATH)
#
#     path_save = os.path.join(SAVE_PATH, 'QR_Research.png')
#
#     q = calculate_q(S, SIZE_QR, SIZE_REGION)
#
#     research_qr(image, SIZE_QR, SIZE_REGION, X, Y, OFFSET, PHASE, q, path_save)
#
#
import numpy as np

image = plt.imread(IMAGE_PATH)
test = image[:64][:64]
F = np.fft.fft2(test)
N= 64

for i in range(14):
    F[i+1][i+1] += 300 * np.exp(np.pi/3)
    F[N-i-1][N-i-1] += 300 * np.exp(-np.pi/3)

print(np.sum(np.imag(F)))
print(np.sum(np.real(F)))