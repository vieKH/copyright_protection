import numpy as np
import matplotlib.pyplot as plt
import os
from Source.Utils import research_qr

SIZE_QR = 5
SIZE_REGION = 64
X = 1
Y = 1
OFFSET = 5
PHASE = np.pi / 3
Q = 1.0


IMAGE_PATH = os.path.join('Image', 'lena.tif')
SAVE_PATH = os.path.join('Results','QR_Research')

if __name__ == '__main__':
    image = plt.imread(IMAGE_PATH)

    path_save = os.path.join(SAVE_PATH, 'QR_Research.png')
    research_qr = research_qr(image, SIZE_QR, SIZE_REGION,X, Y, OFFSET, Q, PHASE, path_save)

