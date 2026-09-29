import numpy as np
import pytest

from suture_scoring.preprocess import DEFAULT_SIZE, preprocess_image


def test_output_shape_dtype_and_range(fixture_images_dir):
    for path in sorted(fixture_images_dir.glob("*.png")):
        arr = preprocess_image(path)
        assert arr.shape == (DEFAULT_SIZE, DEFAULT_SIZE, 3)
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0


def test_wildly_different_resolutions_land_same_shape(fixture_images_dir):
    shapes = {preprocess_image(p).shape for p in fixture_images_dir.glob("*.png")}
    assert len(shapes) == 1
    assert shapes.pop() == (DEFAULT_SIZE, DEFAULT_SIZE, 3)


def test_deterministic(fixture_images_dir):
    path = sorted(fixture_images_dir.glob("*.png"))[0]
    a = preprocess_image(path)
    b = preprocess_image(path)
    assert np.array_equal(a, b)