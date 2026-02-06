import numpy as np
import matplotlib.pyplot as plt
blackImage = np.zeros((8,8))


def countSpectrum(N, img, a, b):
    sum = 0
    for i in range(N):
        for j in range(N):
            sum += img[i][j] * np.exp(-1j * 2*np.pi * (i*a + j*b) / N)
    return sum


def countImage(N, spectrum, a, b):
    sum = 0
    for i in range(N):
        for j in range(N):
            sum += spectrum[i][j] * np.exp(1j * 2*np.pi * (i*a + j*b) / N)
    return sum / N**2


def myDPF(img):
    N, _ = img.shape
    spectrum = np.zeros_like(img, dtype=np.complex128)
    for i in range(N):
        for j in range(N):
            spectrum[i][j] = countSpectrum(N,img,i,j)

    return spectrum


def myODFP(spectrum):
    image = np.zeros_like(spectrum)
    N, _ = spectrum.shape
    for i in range(N):
        for j in range(N):
            image[i][j] = countImage(N,spectrum,i,j)
    return image


def conpressionSpectrum(spectrum):
    res = np.fft.fftshift(spectrum)
    res = np.log(1 + np.abs(res))
    return res


if __name__ == "__main__":
    test_matrix = np.random.randint(0, 256, (32, 32))
    print("-----Original Image-------")
    print(test_matrix)

    print("------------ My Function DPF --------------")
    spectrum = myDPF(test_matrix)
    print(spectrum)

    print("------My function ODPF------")
    image = myODFP(spectrum)
    print(image)

    print("------------ Function DPF from Numpy ------------")
    spectrum2 = np.fft.fft2(test_matrix)
    print(spectrum2)

    print("------Function ODPF from Numpy------")
    image2 = np.fft.ifft2(spectrum)
    print(image2)

    print("------Function ODPF from Numpy (take spectrum after using myDPF)------")
    image2 = np.fft.ifft2(spectrum2)
    print(image2)

    print("------My function ODPF (take spectrum after using function in numpy ------")
    image = myODFP(spectrum)
    print(image)

