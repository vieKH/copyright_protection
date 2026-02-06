import numpy as np
import matplotlib.pyplot as plt

blackImage = np.zeros((16,16))
N = 16

def compressionSpectrum(spectrum):
    spectrum_shift = np.fft.fftshift(spectrum)
    return np.log(1 + np.abs(spectrum_shift))


def coupleOfPoints():
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(blackImage, cmap="gray")
    plt.axis('off')
    plt.title("Original image")

    plt.subplot(1, 3, 2)
    spectrum = np.fft.fft2(blackImage)
    spectrum[2][14] = 1
    spectrum[14][2] = 1
    spectrumCompression = compressionSpectrum(spectrum)
    plt.imshow(spectrumCompression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum after changing couple of points")

    plt.subplot(1, 3, 3)
    image = np.real(np.fft.ifft2(spectrum))
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title("Image after changing couple of points")

    plt.show()


def manyCoupleOfPoints():
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(blackImage, cmap="gray")
    plt.axis('off')
    plt.title("Original image")

    plt.subplot(1, 3, 2)
    spectrum = np.fft.fft2(blackImage)
    m = [2, 4, 14]
    n = [2, 0, 2]

    for i in range(3):
        spectrum[m[i], n[i]] = 1
        spectrum[(N-m[i]) % N, (N-n[i]) % N] = 1

    spectrumCompression = compressionSpectrum(spectrum)
    plt.imshow(spectrumCompression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum after changing couple of points")

    plt.subplot(1, 3, 3)
    image = np.real(np.fft.ifft2(spectrum))
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title("Image after changing couple of points")

    plt.show()


def phase():
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(blackImage, cmap="gray")
    plt.axis('off')
    plt.title("Original image")

    plt.subplot(1, 3, 2)
    spectrum = np.fft.fft2(blackImage)
    spectrum[2, 2] = np.exp(1j * np.pi)
    spectrum[14, 14] = np.exp(1j * np.pi)
    spectrumCompression = compressionSpectrum(spectrum)
    plt.imshow(spectrumCompression, cmap="gray")
    plt.axis('off')
    plt.title("Spectrum after changing phase")

    plt.subplot(1, 3, 3)
    image = np.real(np.fft.ifft2(spectrum))
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title("Image after changing phase")

    plt.show()

if __name__ == "__main__":
    coupleOfPoints()
    manyCoupleOfPoints()
    phase()