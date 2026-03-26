import numpy as np
from Source.Utils import my_fft2, my_ifft2


if __name__ == "__main__":
    test_matrix = np.random.randint(0, 256, (32, 32))
    print("-----Original Image-------")
    print(test_matrix)

    print('-' * 30)
    print("------------ My Function FFT2 --------------")
    spectrum = my_fft2(test_matrix)
    print(spectrum)

    print("------------ Function FFT2 from Numpy ------------")
    spectrum2 = np.fft.fft2(test_matrix)
    print(spectrum2)

    print('-' * 30)
    print("------ My function IFFT2------")
    image = my_ifft2(spectrum)
    print(image)

    print("------Function IFFT2 from Numpy------")
    image2 = np.fft.ifft2(spectrum)
    print(image2)

    print('-' * 30)
    print("------Function IFFT2 from Numpy (take spectrum after using myDPF)------")
    image2 = np.fft.ifft2(spectrum2)
    print(image2)

    print("------My function IFFT2 (take spectrum after using function in numpy ------")
    image = my_ifft2(spectrum)
    print(image)
