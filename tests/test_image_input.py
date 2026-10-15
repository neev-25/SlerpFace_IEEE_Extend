import io

import numpy as np
import pytest
from PIL import Image

from modules.image_input import load_rgb_image


@pytest.mark.parametrize("shape", [(0, 2, 3), (1, 1, 3), (2, 2, 3), (5, 5, 2), (5,)])
def test_rejects_bad_dimensions(shape):
    with pytest.raises(ValueError):
        load_rgb_image(np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize("pixel", [np.nan, np.inf, -1, 256])
def test_rejects_invalid_pixel_values(pixel):
    with pytest.raises(ValueError):
        load_rgb_image(np.full((4, 4, 3), pixel))


def test_normalized_and_single_channel_arrays():
    array = np.ones((4, 4, 1), dtype=np.float32)
    assert np.array(load_rgb_image(array)).min() == 255
    assert array.shape == (4, 4, 1)


def test_file_stream_position_preserved():
    stream = io.BytesIO()
    Image.new("RGB", (5, 6)).save(stream, "PNG")
    stream.seek(2)
    result = load_rgb_image(stream)
    assert result.size == (5, 6)
    assert stream.tell() == 2
    assert not stream.closed


def test_rejects_pixel_limit_before_conversion(monkeypatch):
    image = Image.new("RGB", (10, 10))
    monkeypatch.setattr("modules.image_input.MAX_IMAGE_PIXELS", 50)
    with pytest.raises(ValueError, match="pixel limit"):
        load_rgb_image(image)


def test_corrupt_stream_is_a_validation_error():
    with pytest.raises(ValueError, match="invalid"):
        load_rgb_image(io.BytesIO(b"not an image"))


def test_path_byte_limit_checked_on_open_file(tmp_path, monkeypatch):
    path = tmp_path / "input.png"
    Image.new("RGB", (8, 8)).save(path)
    monkeypatch.setattr("modules.image_input.MAX_IMAGE_BYTES", path.stat().st_size - 1)
    with pytest.raises(ValueError, match="20 MiB"):
        load_rgb_image(path)
