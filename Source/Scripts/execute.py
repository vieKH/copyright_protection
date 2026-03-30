import numpy as np
import matplotlib.pyplot as plt
import os
from Source.Utils import research_qr


IMAGE_PATH = os.path.join('Image', 'lena.tif')
SAVE_PATH = os.path.join('Results','QR_Research')

if __name__ == '__main__':
    image = plt.imread(IMAGE_PATH)

    path_save = os.path.join(SAVE_PATH, 'QR_Research.png')
    research_qr = research_qr(image, 5, 4,4, path_save, phase=np.pi / 2)

