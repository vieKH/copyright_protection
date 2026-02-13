from Source.Research.research_plot import *
image_path = "C://Users//Hoang//OneDrive//Desktop//Study//copyright_protection//Image//goldhilf.tif"

if __name__ == "__main__":
    N = 32
    blackImage = np.zeros((N, N))
    image = plt.imread(image_path)
    couple_of_points(blackImage, 3, 4)

    x = np.array([2, 4, 14])
    y = np.array([2, 0, 2])
    #many_couple_of_points(blackImage, x, y)
    #phase_research(blackImage, x, y)

    #high_frequency()
    #research_qr(blackImage, 9, 3, 4)
    #research_qr(image, 41, 40, 50)