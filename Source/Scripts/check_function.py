import numpy as np
from Source.Utils import my_fft2, my_ifft2

EPSILON = 1e-20

def calculate_mse(image1: np.ndarray, image2: np.ndarray):
    """ Mean Square Error (MSE)"""
    if image1.shape != image2.shape:
        raise ValueError("Функция calculate_mse: 2 данных имеют разный размер")

    # Избегать переполнения числовых значений в формате uint8.
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    return np.mean((image1 - image2) ** 2)


if __name__ == "__main__":
    test_matrix = np.random.randint(0, 256, (32, 32))

    print('-' * 30)
    print("Check function my_fft2 and np.fft.fft2")
    print('-' * 30)
    spectrum = my_fft2(test_matrix)
    spectrum2 = np.fft.fft2(test_matrix)
    mse = calculate_mse(np.real(spectrum), np.real(spectrum2))
    if mse < EPSILON:
        print(f'Result can be accepted. MSE = {mse}')
    else:
        print(f'Result can be rejected. MSE = {mse}')

    print()
    print('-' * 30)
    print("Check my function my_ifft2 and np.fft.ifft2 (take spectrum after using myDPF)")
    print('-' * 30)
    image = my_ifft2(spectrum)
    image2 = np.fft.ifft2(spectrum)
    mse = calculate_mse(np.real(image2), np.real(image))
    if mse < EPSILON:
        print(f'Result can be accepted. MSE = {mse}')
    else:
        print(f'Result can be rejected. MSE = {mse}')

    print()
    print('-' * 30)
    print("Check my function my_ifft2 and np.fft.ifft2 (take spectrum after using function in numpy)")
    print('-' * 30)
    image2 = np.fft.ifft2(spectrum2)
    image = my_ifft2(spectrum2)
    mse = calculate_mse(np.real(image), np.real(image2))
    if mse < EPSILON:
        print(f'Result can be accepted. MSE = {mse}')
    else:
        print(f'Result can be rejected. MSE =  {mse}')
