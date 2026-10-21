"""Camera frames as the Dream API takes them: one base64 JPEG per camera, no ``data:`` prefix."""

from __future__ import annotations

import base64
import io

import numpy as np
from inspect_robots.types import ImageArray
from PIL import Image

# Dream-1 reads JPEG quality 85; frames already at its size upload fastest.
JPEG_QUALITY = 85


def encode_jpeg(image: ImageArray, size: tuple[int, int]) -> str:
    """Stretch an RGB ``(H, W, 3)`` uint8 frame to ``size`` (width, height) and base64 its JPEG.

    Stretching, never cropping or padding, is how Dream resized the recordings it trained on.
    """
    arr = np.asarray(image)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an RGB uint8 (H, W, 3) frame, got {arr.dtype} {arr.shape}")
    picture = Image.fromarray(arr, "RGB")
    if picture.size != size:
        picture = picture.resize(size, Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
