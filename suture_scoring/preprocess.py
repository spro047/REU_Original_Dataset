"""Image preprocessing: normalize wildly inconsistent resolutions to one input size.

Images in the dataset range from 383x549 to 1694x1406. Every image is
aspect-preserving resized so its longer side equals `size`, then center-padded
to a `size x size` square on a white background, and scaled to float32 in [0, 1].
Per-channel mean/std normalization (e.g. ImageNet stats) is intentionally NOT
applied here: it belongs to the model input specification decided in ticket 03.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_SIZE = 512


def preprocess_image(path: str | Path, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Load and normalize an image to ``(size, size, 3)`` float32 in [0, 1]."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        width, height = im.size
        scale = size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        im = im.resize((new_width, new_height), Image.LANCZOS)

        canvas = Image.new("RGB", (size, size), (255, 255, 255))
        canvas.paste(im, ((size - new_width) // 2, (size - new_height) // 2))

        arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr