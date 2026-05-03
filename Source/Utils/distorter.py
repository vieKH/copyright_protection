import io
import numpy as np

from PIL import Image
from scipy.signal import convolve2d


class ImageDistorter:
    def __init__(self, image: np.ndarray):
        """Конструктор класса для применения искажений к изображениям"""
        self._image = image.copy()
        self._height = self._image.shape[0]
        self._width = self._image.shape[1]

    @property
    def image(self):
        """Геттер для получения изображения"""
        return self._image

    def contrast(self, contrast_factor: float) -> "ImageDistorter":
        """Искажение с помощью линейного контрастирования"""
        contrasted = np.clip(contrast_factor * self._image, 0, 255).astype(self._image.dtype)

        return ImageDistorter(contrasted)

    def rotation_rest(self, angle: float) -> "ImageDistorter":
        """Искажение с помощью поворота с последующим восстановлением"""

        rotated = self._rotate(self._image, angle)
        restored = self._rotate(rotated, -angle)
        cropped = self._crop(restored, self._height, self._width)

        return ImageDistorter(cropped)

    def rotation(self, angle: float) -> "ImageDistorter":
        """Искажение с помощью поворота с обрезкой"""
        rotated = self._rotate(self._image, angle)
        cropped = self._crop(rotated, self._height, self._width)

        return ImageDistorter(cropped)

    def scale_rest(self, scale_factor: float) -> "ImageDistorter":
        """Искажение с помощью масштабирования с последующим восстановлением"""
        rotated = self._scale(self._image, scale_factor)
        restored = self._scale(rotated, 1 / scale_factor)
        cropped = self._crop(restored, self._height, self._width)

        return ImageDistorter(cropped)

    def scale(self, scale_factor: float) -> "ImageDistorter":
        """Искажение с помощью масштабирования с обрезкой/дополнением нулями"""
        rotated = self._scale(self._image, scale_factor)
        cropped = self._crop(rotated, self._height, self._width)

        return ImageDistorter(cropped)

    def cut(self, replacement_image: np.ndarray, area_fraction: float) -> "ImageDistorter":
        """Искажение с помощью обрезки с заменой данных из другого изображения"""
        area_fraction_sqrt = np.sqrt(area_fraction)

        rows, cols = np.indices((self._height, self._width))

        cutted = replacement_image.copy()
        mask = (rows < np.floor(self._height * area_fraction_sqrt)) & (
                    cols < np.floor(self._width * area_fraction_sqrt))

        if len(self._image.shape) == 2:
            cutted[mask] = self._image[mask]
        else:
            cutted[mask] = self._image[mask, :]

        return ImageDistorter(cutted)

    def cyclic_shift(self, shift_fraction: float) -> "ImageDistorter":
        """Искажение с помощью циклического сдвига изображения по диагонали"""
        shift_height = int(np.floor(shift_fraction * self._height))
        shift_width = int(np.floor(shift_fraction * self._width))

        shifted = np.roll(self._image, shift=(shift_height, shift_width), axis=(0, 1))

        return ImageDistorter(shifted)

    def smooth(self, window_size: int) -> "ImageDistorter":
        """Искажение с помощью сглаживания"""
        if window_size % 2 == 0:
            window_size += 1
            print(f'Warning: window_size has been changed to {window_size} (must be odd)')

        window = np.ones((window_size, window_size))
        factor = 1 / window_size ** 2

        smoothed = (factor * self._convolve(self._image, window)).astype(self._image.dtype)

        return ImageDistorter(smoothed)

    def gauss_blur(self, sigma: float) -> "ImageDistorter":
        """Искажение с помощью сглаживания гауссовой маской"""
        window_size = 2 * int(3 * sigma) + 1
        window = self._gaussian_kernel(window_size, sigma)

        smoothed = self._convolve(self._image, window).astype(self._image.dtype)

        return ImageDistorter(smoothed)

    def sharpen(self, window_size: int, gain_factor: float = 5) -> "ImageDistorter":
        """Искажение с помощью повешения резкости"""
        if window_size % 2 == 0:
            window_size += 1
            print(f'Warning: window_size has been changed to {window_size} (must be odd)')

        window = np.ones((window_size, window_size))
        factor = 1 / window_size ** 2

        smoothed = (factor * self._convolve(self._image, window)).astype(self._image.dtype)
        sharpened = self._image + gain_factor * (self._image - smoothed)
        sharpened = np.clip(sharpened, 0, 255).astype(self._image.dtype)

        return ImageDistorter(sharpened)

    def median(self, window_size: int) -> "ImageDistorter":
        """Искажение с помощью медианной фильтрации"""
        if window_size % 2 == 0:
            window_size += 1
            print(f'Warning: window_size has been changed to {window_size} (must be odd)')

        pad = window_size // 2

        if len(self._image.shape) == 2:
            image_padded = np.pad(self._image, pad, mode='edge')

            shape = (self._height, self._width, window_size, window_size)
            strides = (image_padded.strides[0], image_padded.strides[1],
                       image_padded.strides[0], image_padded.strides[1])
            windows = np.lib.stride_tricks.as_strided(image_padded, shape=shape, strides=strides)

            filtered = np.median(windows, axis=(2, 3)).astype(self._image.dtype)

        else:
            filtered = np.zeros_like(self._image)

            for c in range(self._image.shape[2]):
                image_padded = np.pad(self._image[:, :, c], pad, mode='edge')

                shape = (self._height, self._width, window_size, window_size)
                strides = (image_padded.strides[0], image_padded.strides[1],
                           image_padded.strides[0], image_padded.strides[1])
                windows = np.lib.stride_tricks.as_strided(image_padded, shape=shape, strides=strides)

                filtered[:, :, c] = np.median(windows, axis=(2, 3))

            filtered = filtered.astype(self._image.dtype)

        return ImageDistorter(filtered)

    def white_noise(self, variance: float) -> "ImageDistorter":
        """Искажение при помощи сложения с белым гауссовым шумом"""
        noise = np.random.normal(loc=0, scale=np.sqrt(variance), size=self._image.shape)

        noised = np.clip(self._image.astype(np.float32) + noise, 0, 255).astype(self._image.dtype)

        return ImageDistorter(noised)

    def salt_pepper(self, noise_fraction: float) -> "ImageDistorter":
        """Искажение при помощи сложения с импульсным шумом"""
        if noise_fraction < 0 or noise_fraction > 1:
            raise ValueError("noise_fraction should be between 0 and 1")

        noise = np.zeros(self._image.size)
        noise_length = int(noise_fraction * noise.size / 2)
        noise[:noise_length] = 255
        noise[noise_length: 2 * noise_length] = -255
        np.random.shuffle(noise)
        noise = np.reshape(noise, shape=self._image.shape)

        noised = np.clip(self._image + noise, 0, 255).astype(self._image.dtype)

        return ImageDistorter(noised)

    def jpeg(self, quality_factor: int) -> "ImageDistorter":
        """Искажение при помощи сжатия с потерями в формате JPEG"""
        if quality_factor < 1 or quality_factor > 100:
            raise ValueError("quality_factor should be between 1 and 100")

        if len(self._image.shape) == 3:
            pil_image = Image.fromarray(self._image)
        else:
            pil_image = Image.fromarray(self._image, mode='L')

        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality_factor)
        buffer.seek(0)

        compressed = np.array(Image.open(buffer))

        return ImageDistorter(compressed)

    @staticmethod
    def _crop(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """Обрезка/дополнение нулями изображения до целевого размера (центрирование)"""
        height, width = image.shape[:2]

        # Вычисляем сдвиги для центрирования
        y_offset = (target_height - height) // 2
        x_offset = (target_width - width) // 2

        # Создаём новый холст целевого размера
        if len(image.shape) == 2:
            result = np.zeros((target_height, target_width), dtype=image.dtype)
        else:
            result = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)

        # Определяем область вставки исходного изображения
        y_start = max(0, y_offset)
        y_end = min(target_height, y_offset + height)
        x_start = max(0, x_offset)
        x_end = min(target_width, x_offset + width)

        # Определяем область выреза из исходного изображения
        src_y_start = max(0, -y_offset)
        src_y_end = src_y_start + (y_end - y_start)
        src_x_start = max(0, -x_offset)
        src_x_end = src_x_start + (x_end - x_start)

        # Вставляем изображение в новый холст
        if len(image.shape) == 2:
            result[y_start:y_end, x_start:x_end] = image[src_y_start:src_y_end, src_x_start:src_x_end]
        else:
            result[y_start:y_end, x_start:x_end, :] = image[src_y_start:src_y_end, src_x_start:src_x_end, :]

        return result

    @staticmethod
    def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """Поворот изображения на заданный угол с увеличением холста"""
        height, width = image.shape[:2]
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Вычисляем размер нового холста
        corners = np.array([
            [-width / 2, -height / 2], [width / 2, -height / 2],
            [-width / 2, height / 2], [width / 2, height / 2]
        ])

        rotated_corners = corners @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        new_width = int(np.ceil(rotated_corners[:, 0].max() - rotated_corners[:, 0].min()))
        new_height = int(np.ceil(rotated_corners[:, 1].max() - rotated_corners[:, 1].min()))

        # Создаём новый холст
        if len(image.shape) == 2:
            result = np.zeros((new_height, new_width), dtype=image.dtype)
        else:
            result = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)

        # Смещения для центрирования
        offset_x = new_width // 2
        offset_y = new_height // 2
        center_x, center_y = width // 2, height // 2

        # Обратное отображение: для каждого пикселя результата находим пиксель в исходнике
        y_coords, x_coords = np.ogrid[:new_height, :new_width]

        x_orig = (x_coords - offset_x) * cos_a + (y_coords - offset_y) * sin_a + center_x
        y_orig = (y_coords - offset_y) * cos_a - (x_coords - offset_x) * sin_a + center_y

        x_orig = np.clip(np.round(x_orig), 0, width - 1).astype(int)
        y_orig = np.clip(np.round(y_orig), 0, height - 1).astype(int)

        # Заполняем результат
        if len(image.shape) == 2:
            result[y_coords, x_coords] = image[y_orig, x_orig]
        else:
            result[y_coords, x_coords] = image[y_orig, x_orig, :]

        return result

    @staticmethod
    def _scale(image: np.ndarray, factor: float) -> np.ndarray:
        """Масштабирование изображения (увеличение/уменьшение)"""
        height, width = image.shape[:2]
        new_height = int(height * factor)
        new_width = int(width * factor)

        # Создаём координатную сетку нового размера
        y_coords, x_coords = np.ogrid[:new_height, :new_width]

        # Обратное отображение: из нового размера в исходный
        x_orig = x_coords / factor
        y_orig = y_coords / factor

        x_orig = np.clip(np.round(x_orig), 0, width - 1).astype(int)
        y_orig = np.clip(np.round(y_orig), 0, height - 1).astype(int)

        # Заполняем результат
        if len(image.shape) == 2:
            result = image[y_orig, x_orig]
        else:
            result = image[y_orig, x_orig, :]

        return result

    @staticmethod
    def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Двумерная свёртка с ядром"""
        if len(image.shape) == 3:
            # Несколько каналов
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = convolve2d(image[:, :, i], kernel, mode='same')
            return result
        else:
            # Один канал
            return convolve2d(image, kernel, mode='same')

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """Создание гауссова ядра размера size x size"""
        # Центр ядра
        center = (size - 1) / 2

        # Создаём координатные сетки
        m1, m2 = np.meshgrid(np.arange(size), np.arange(size))

        # Вычисляем экспоненту без нормировки
        exponent = -((m1 - center) ** 2 + (m2 - center) ** 2) / (2 * sigma ** 2)
        kernel = np.exp(exponent)

        # Нормируем, чтобы сумма элементов = 1
        kernel = kernel / kernel.sum()

        return kernel
