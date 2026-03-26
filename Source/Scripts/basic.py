import os
import numpy as np
from Source.Utils import  couple_of_points, phase_research, show_frequency, many_couple_of_points

IMAGE_PATH = os.path.join('Image', 'lena.tif')
SAVE_PATH = os.path.join('Results','Basic')
N = 64

if __name__ == "__main__":
    blackImage = np.zeros((N, N))

    path_couple_of_points = os.path.join(SAVE_PATH, "1_couple_of_points.png")
    couple_of_points(blackImage, 3, 4, path_couple_of_points)

    x = np.array([2, 4, 14])
    y = np.array([2, 0, 2])

    path_many_couple_of_points = os.path.join(SAVE_PATH, "2_many_couple_of_points.png")
    many_couple_of_points(blackImage, x, y, path_many_couple_of_points )

    path_phase_research = os.path.join(SAVE_PATH, "3_phase_research.png")
    phase_research(blackImage, x, y, path_phase_research, np.pi / 3)

    path_show_frequency = os.path.join(SAVE_PATH, "4_show_frequency.png")
    show_frequency(IMAGE_PATH, path_show_frequency)

    # research_qr(blackImage, 8, 1, 1)

