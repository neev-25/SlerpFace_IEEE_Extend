"""Bounded image decoding shared by CLI, dashboard, and research tools."""

import os
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MIN_IMAGE_SIDE = 3


def _check_size(size):
    width, height = size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"Images must be at least {MIN_IMAGE_SIDE} pixels on each side.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit.")


def _from_array(value):
    if value.ndim not in (2, 3) or (value.ndim == 3 and value.shape[2] not in (1, 3, 4)):
        raise ValueError("Expected a grayscale, RGB, or RGBA image array.")
    _check_size((value.shape[1], value.shape[0]))
    if value.dtype.kind not in "uif" or not np.isfinite(value).all():
        raise ValueError("Image pixels must be finite numeric values.")
    if value.min() < 0 or value.max() > 255:
        raise ValueError("Image pixels must be in [0, 255] or normalized [0, 1].")
    if value.dtype.kind == "f" and value.max() <= 1:
        value = np.rint(value * 255)
    value = value.astype(np.uint8, copy=False)
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[:, :, 0]
    return Image.fromarray(value)


def load_rgb_image(value, size=None):
    """Return an independent RGB PIL image without retaining an open file.

    Dimensions are checked before pixel decoding. Caller-owned streams/images
    remain open. Resizing is explicit and always occurs after validation.
    """
    def decode(image):
        _check_size(image.size)
        result = ImageOps.exif_transpose(image).convert("RGB")
        if size is not None:
            result = result.resize(size, Image.Resampling.BILINEAR)
        return result

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            if isinstance(value, np.ndarray):
                return decode(_from_array(value))
            if isinstance(value, Image.Image):
                return decode(value)
            if isinstance(value, (str, os.PathLike)):
                # Check and decode the same open file, even if its path is replaced.
                with open(value, "rb") as source:
                    if os.fstat(source.fileno()).st_size > MAX_IMAGE_BYTES:
                        raise ValueError("Image file exceeds the 20 MiB limit.")
                    with Image.open(source) as image:
                        return decode(image)
            if hasattr(value, "read") and hasattr(value, "seek"):
                start = value.tell()
                try:
                    value.seek(0, 2)
                    if value.tell() > MAX_IMAGE_BYTES:
                        raise ValueError("Image file exceeds the 20 MiB limit.")
                    value.seek(0)
                    with Image.open(value) as image:
                        return decode(image)
                finally:
                    if not value.closed:
                        value.seek(start)
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Image is invalid or exceeds safe decoding limits.") from exc
    raise ValueError("Expected an image path, PIL image, numeric array, or image stream.")
