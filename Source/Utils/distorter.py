"""Image distortion utilities for watermark robustness experiments.

The methods return a new ImageDistorter instance and keep the output image size
unchanged whenever possible. Keeping the size unchanged is convenient for the
current block-based extractor, which expects a fixed block grid.
"""

from __future__ import annotations

import io
from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.signal import convolve2d


Interpolation = Literal["nearest", "bilinear", "bicubic", "lanczos"]
Position = Literal["center", "top_left", "top_right", "bottom_left", "bottom_right", "random"]


class ImageDistorter:
    def __init__(self, image: np.ndarray):
        """Create a distortion wrapper for a grayscale or RGB image."""
        self._image = np.asarray(image).copy()
        self._height = self._image.shape[0]
        self._width = self._image.shape[1]

    @property
    def image(self) -> np.ndarray:
        """Return the distorted image."""
        return self._image

    # ------------------------------------------------------------------
    # Non-geometric attacks: they mainly modify pixel values.
    # ------------------------------------------------------------------
    def contrast(self, contrast_factor: float) -> "ImageDistorter":
        """Linear contrast change: I' = contrast_factor * I."""
        contrasted = np.clip(
            contrast_factor * self._image.astype(np.float64),
            0,
            255,
        ).astype(self._image.dtype)
        return ImageDistorter(contrasted)

    def white_noise(
        self,
        variance: float,
        seed: Optional[int] = None,
    ) -> "ImageDistorter":
        """Additive white Gaussian noise with the given variance."""
        if variance < 0:
            raise ValueError("variance must be non-negative")

        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0, scale=np.sqrt(variance), size=self._image.shape)
        noised = np.clip(self._image.astype(np.float64) + noise, 0, 255).astype(self._image.dtype)
        return ImageDistorter(noised)

    def salt_pepper(
        self,
        noise_fraction: float,
        seed: Optional[int] = None,
    ) -> "ImageDistorter":
        """Impulse noise: randomly set pixels to 0 or 255."""
        if not 0 <= noise_fraction <= 1:
            raise ValueError("noise_fraction must be between 0 and 1")

        rng = np.random.default_rng(seed)
        result = self._image.copy()
        h, w = self._height, self._width
        n_pixels = h * w
        n_noisy = int(round(noise_fraction * n_pixels))

        if n_noisy <= 0:
            return ImageDistorter(result)

        flat_indices = rng.choice(n_pixels, size=n_noisy, replace=False)
        salt_mask = rng.random(n_noisy) < 0.5

        if result.ndim == 2:
            flat = result.reshape(-1)
            flat[flat_indices[salt_mask]] = 255
            flat[flat_indices[~salt_mask]] = 0
        else:
            flat = result.reshape(n_pixels, result.shape[2])
            flat[flat_indices[salt_mask], :] = 255
            flat[flat_indices[~salt_mask], :] = 0

        return ImageDistorter(result)

    def smooth(self, window_size: int) -> "ImageDistorter":
        """Mean filtering with a square window."""
        window_size = self._make_odd(window_size)
        kernel = np.ones((window_size, window_size), dtype=np.float64) / (window_size ** 2)
        smoothed = np.clip(self._convolve(self._image, kernel), 0, 255).astype(self._image.dtype)
        return ImageDistorter(smoothed)

    def gauss_blur(self, sigma: float) -> "ImageDistorter":
        """Gaussian blur."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        window_size = 2 * int(3 * sigma) + 1
        kernel = self._gaussian_kernel(window_size, sigma)
        blurred = np.clip(self._convolve(self._image, kernel), 0, 255).astype(self._image.dtype)
        return ImageDistorter(blurred)

    def sharpen(self, window_size: int, gain_factor: float = 5.0) -> "ImageDistorter":
        """Unsharp-mask style sharpening."""
        window_size = self._make_odd(window_size)
        kernel = np.ones((window_size, window_size), dtype=np.float64) / (window_size ** 2)
        smoothed = self._convolve(self._image, kernel)
        sharpened = self._image.astype(np.float64) + gain_factor * (self._image.astype(np.float64) - smoothed)
        sharpened = np.clip(sharpened, 0, 255).astype(self._image.dtype)
        return ImageDistorter(sharpened)

    def median(self, window_size: int) -> "ImageDistorter":
        """Median filtering."""
        window_size = self._make_odd(window_size)
        pad = window_size // 2

        if self._image.ndim == 2:
            filtered = self._median_2d(self._image, window_size, pad)
        else:
            filtered = np.zeros_like(self._image)
            for c in range(self._image.shape[2]):
                filtered[:, :, c] = self._median_2d(self._image[:, :, c], window_size, pad)

        return ImageDistorter(filtered.astype(self._image.dtype))

    def jpeg(self, quality_factor: int) -> "ImageDistorter":
        """JPEG lossy compression and decompression."""
        if not 1 <= quality_factor <= 100:
            raise ValueError("quality_factor must be between 1 and 100")

        pil_image = self._to_pil(self._image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=int(quality_factor))
        buffer.seek(0)
        compressed = np.asarray(Image.open(buffer)).astype(self._image.dtype)
        return ImageDistorter(compressed)

    # ------------------------------------------------------------------
    # Geometric attacks: they modify the spatial coordinate system.
    # ------------------------------------------------------------------
    def cyclic_shift(self, shift_fraction: float) -> "ImageDistorter":
        """Cyclic diagonal shift. This preserves size but changes block alignment."""
        shift_height = int(np.floor(shift_fraction * self._height))
        shift_width = int(np.floor(shift_fraction * self._width))
        shifted = np.roll(self._image, shift=(shift_height, shift_width), axis=(0, 1))
        return ImageDistorter(shifted)

    def translate(
        self,
        shift_y: int,
        shift_x: int,
        fill_value: int = 0,
    ) -> "ImageDistorter":
        """Non-cyclic translation with constant padding."""
        result = self._constant_canvas(self._height, self._width, fill_value)

        src_y1 = max(0, -shift_y)
        src_y2 = min(self._height, self._height - shift_y)
        src_x1 = max(0, -shift_x)
        src_x2 = min(self._width, self._width - shift_x)

        dst_y1 = max(0, shift_y)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        dst_x1 = max(0, shift_x)
        dst_x2 = dst_x1 + (src_x2 - src_x1)

        if src_y2 > src_y1 and src_x2 > src_x1:
            result[dst_y1:dst_y2, dst_x1:dst_x2, ...] = self._image[src_y1:src_y2, src_x1:src_x2, ...]

        return ImageDistorter(result.astype(self._image.dtype))

    def crop_restore(
        self,
        retained_fraction: float,
        position: Position = "center",
        fill_value: int = 0,
        seed: Optional[int] = None,
    ) -> "ImageDistorter":
        """Crop a retained image region and pad it back to the original size.

        This is the direct implementation of обрезка изображения for the current
        extractor. The visible image still has the same size, but part of the
        original content is removed.
        """
        if not 0 < retained_fraction <= 1:
            raise ValueError("retained_fraction must be in (0, 1]")

        scale = np.sqrt(retained_fraction)
        crop_h = max(1, int(round(self._height * scale)))
        crop_w = max(1, int(round(self._width * scale)))
        y1, x1 = self._position_to_top_left(crop_h, crop_w, position, seed)
        cropped = self._image[y1:y1 + crop_h, x1:x1 + crop_w, ...]
        restored = self._center_to_canvas(cropped, self._height, self._width, fill_value)
        return ImageDistorter(restored.astype(self._image.dtype))

    def cutout(
        self,
        area_fraction: float,
        position: Position = "center",
        fill_value: int = 0,
        seed: Optional[int] = None,
    ) -> "ImageDistorter":
        """Remove one rectangular area by filling it with a constant value."""
        if not 0 <= area_fraction <= 1:
            raise ValueError("area_fraction must be between 0 and 1")

        result = self._image.copy()
        if area_fraction == 0:
            return ImageDistorter(result)

        scale = np.sqrt(area_fraction)
        cut_h = max(1, int(round(self._height * scale)))
        cut_w = max(1, int(round(self._width * scale)))
        y1, x1 = self._position_to_top_left(cut_h, cut_w, position, seed)
        result[y1:y1 + cut_h, x1:x1 + cut_w, ...] = fill_value
        return ImageDistorter(result.astype(self._image.dtype))

    def cut(self, replacement_image: np.ndarray, area_fraction: float) -> "ImageDistorter":
        """Backward-compatible cut method from the earlier version.

        It keeps the top-left sqrt(area_fraction) area from the current image and
        fills the rest from replacement_image.
        """
        if not 0 <= area_fraction <= 1:
            raise ValueError("area_fraction must be between 0 and 1")

        replacement = np.asarray(replacement_image).copy()
        if replacement.shape != self._image.shape:
            raise ValueError("replacement_image must have the same shape")

        area_fraction_sqrt = np.sqrt(area_fraction)
        rows, cols = np.indices((self._height, self._width))
        mask = (
            rows < np.floor(self._height * area_fraction_sqrt)
        ) & (
            cols < np.floor(self._width * area_fraction_sqrt)
        )

        if self._image.ndim == 2:
            replacement[mask] = self._image[mask]
        else:
            replacement[mask, :] = self._image[mask, :]

        return ImageDistorter(replacement.astype(self._image.dtype))

    def rotation(
        self,
        angle: float,
        interpolation: Interpolation = "bilinear",
        fill_value: int = 0,
    ) -> "ImageDistorter":
        """Rotate and return the central crop with the original size."""
        rotated = self._rotate(self._image, angle, interpolation, expand=True, fill_value=fill_value)
        cropped = self._crop_or_pad_center(rotated, self._height, self._width, fill_value)
        return ImageDistorter(cropped.astype(self._image.dtype))

    def rotation_rest(
        self,
        angle: float,
        interpolation: Interpolation = "bilinear",
        fill_value: int = 0,
    ) -> "ImageDistorter":
        """Rotate by angle, then rotate back by -angle.

        This tests interpolation damage even when the final orientation is
        restored.
        """
        rotated = self._rotate(self._image, angle, interpolation, expand=True, fill_value=fill_value)
        restored = self._rotate(rotated, -angle, interpolation, expand=True, fill_value=fill_value)
        cropped = self._crop_or_pad_center(restored, self._height, self._width, fill_value)
        return ImageDistorter(cropped.astype(self._image.dtype))

    def scale(
        self,
        scale_factor: float,
        interpolation: Interpolation = "bilinear",
        fill_value: int = 0,
    ) -> "ImageDistorter":
        """Scale image, then crop/pad to the original size."""
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")

        scaled = self._resize_by_factor(self._image, scale_factor, interpolation)
        restored_size = self._crop_or_pad_center(scaled, self._height, self._width, fill_value)
        return ImageDistorter(restored_size.astype(self._image.dtype))

    def scale_rest(
        self,
        scale_factor: float,
        interpolation: Interpolation = "bilinear",
        fill_value: int = 0,
    ) -> "ImageDistorter":
        """Scale by factor and then restore by 1/factor."""
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")

        scaled = self._resize_by_factor(self._image, scale_factor, interpolation)
        restored = self._resize_by_factor(scaled, 1.0 / scale_factor, interpolation)
        cropped = self._crop_or_pad_center(restored, self._height, self._width, fill_value)
        return ImageDistorter(cropped.astype(self._image.dtype))

    def resampling(
        self,
        sampling_factor: float,
        interpolation: Interpolation = "bilinear",
    ) -> "ImageDistorter":
        """Change sampling step and restore the original image size.

        sampling_factor < 1 means downsample then upsample. This is the clearest
        implementation of изменение шага дискретизации.
        """
        if sampling_factor <= 0:
            raise ValueError("sampling_factor must be positive")

        tmp = self._resize_by_factor(self._image, sampling_factor, interpolation)
        restored = self._resize_to_shape(tmp, self._height, self._width, interpolation)
        return ImageDistorter(restored.astype(self._image.dtype))

    # ------------------------------------------------------------------
    # Static helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _make_odd(window_size: int) -> int:
        window_size = int(window_size)
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        return window_size if window_size % 2 == 1 else window_size + 1

    @staticmethod
    def _pil_resample(interpolation: Interpolation):
        if interpolation == "nearest":
            return Image.Resampling.NEAREST
        if interpolation == "bilinear":
            return Image.Resampling.BILINEAR
        if interpolation == "bicubic":
            return Image.Resampling.BICUBIC
        if interpolation == "lanczos":
            return Image.Resampling.LANCZOS
        raise ValueError("Unsupported interpolation")

    @staticmethod
    def _to_pil(image: np.ndarray) -> Image.Image:
        image_u8 = np.clip(image, 0, 255).astype(np.uint8)
        if image_u8.ndim == 2:
            return Image.fromarray(image_u8, mode="L")
        return Image.fromarray(image_u8)

    @classmethod
    def _resize_by_factor(
        cls,
        image: np.ndarray,
        factor: float,
        interpolation: Interpolation,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        new_height = max(1, int(round(height * factor)))
        new_width = max(1, int(round(width * factor)))
        return cls._resize_to_shape(image, new_height, new_width, interpolation)

    @classmethod
    def _resize_to_shape(
        cls,
        image: np.ndarray,
        target_height: int,
        target_width: int,
        interpolation: Interpolation,
    ) -> np.ndarray:
        pil_image = cls._to_pil(image)
        resized = pil_image.resize(
            (int(target_width), int(target_height)),
            resample=cls._pil_resample(interpolation),
        )
        return np.asarray(resized).astype(image.dtype)

    @classmethod
    def _rotate(
        cls,
        image: np.ndarray,
        angle: float,
        interpolation: Interpolation,
        expand: bool,
        fill_value: int,
    ) -> np.ndarray:
        pil_image = cls._to_pil(image)
        rotated = pil_image.rotate(
            angle,
            resample=cls._pil_resample(interpolation),
            expand=expand,
            fillcolor=int(fill_value),
        )
        return np.asarray(rotated).astype(image.dtype)

    def _constant_canvas(self, height: int, width: int, fill_value: int) -> np.ndarray:
        if self._image.ndim == 2:
            return np.full((height, width), fill_value, dtype=self._image.dtype)
        return np.full((height, width, self._image.shape[2]), fill_value, dtype=self._image.dtype)

    def _center_to_canvas(
        self,
        image: np.ndarray,
        target_height: int,
        target_width: int,
        fill_value: int,
    ) -> np.ndarray:
        result = self._constant_canvas(target_height, target_width, fill_value)
        h, w = image.shape[:2]

        y0 = (target_height - h) // 2
        x0 = (target_width - w) // 2
        y_start = max(0, y0)
        x_start = max(0, x0)
        y_end = min(target_height, y0 + h)
        x_end = min(target_width, x0 + w)

        src_y_start = max(0, -y0)
        src_x_start = max(0, -x0)
        src_y_end = src_y_start + (y_end - y_start)
        src_x_end = src_x_start + (x_end - x_start)

        result[y_start:y_end, x_start:x_end, ...] = image[src_y_start:src_y_end, src_x_start:src_x_end, ...]
        return result

    def _crop_or_pad_center(
        self,
        image: np.ndarray,
        target_height: int,
        target_width: int,
        fill_value: int,
    ) -> np.ndarray:
        return self._center_to_canvas(image, target_height, target_width, fill_value)

    def _position_to_top_left(
        self,
        patch_h: int,
        patch_w: int,
        position: Position,
        seed: Optional[int],
    ) -> Tuple[int, int]:
        max_y = self._height - patch_h
        max_x = self._width - patch_w

        if position == "center":
            return max_y // 2, max_x // 2
        if position == "top_left":
            return 0, 0
        if position == "top_right":
            return 0, max_x
        if position == "bottom_left":
            return max_y, 0
        if position == "bottom_right":
            return max_y, max_x
        if position == "random":
            rng = np.random.default_rng(seed)
            return int(rng.integers(0, max_y + 1)), int(rng.integers(0, max_x + 1))

        raise ValueError("Unsupported position")

    @staticmethod
    def _median_2d(image: np.ndarray, window_size: int, pad: int) -> np.ndarray:
        image_padded = np.pad(image, pad, mode="edge")
        shape = (image.shape[0], image.shape[1], window_size, window_size)
        strides = (
            image_padded.strides[0],
            image_padded.strides[1],
            image_padded.strides[0],
            image_padded.strides[1],
        )
        windows = np.lib.stride_tricks.as_strided(image_padded, shape=shape, strides=strides)
        return np.median(windows, axis=(2, 3))

    @staticmethod
    def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            result = np.zeros_like(image, dtype=np.float64)
            for i in range(image.shape[2]):
                result[:, :, i] = convolve2d(image[:, :, i], kernel, mode="same", boundary="symm")
            return result
        return convolve2d(image, kernel, mode="same", boundary="symm")

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        center = (size - 1) / 2
        y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        exponent = -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2)
        kernel = np.exp(exponent)
        return kernel / np.sum(kernel)
