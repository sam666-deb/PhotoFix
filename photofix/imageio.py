"""Image decoding/encoding. All images inside the pipeline are float32 RGB in [0, 1], shape (H, W, 3)."""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # iPhone photos (HEIC/HEIF)

BROWSER_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "GIF", "BMP"}


def image_format(data: bytes) -> str | None:
    return Image.open(io.BytesIO(data)).format


def load_image(data: bytes) -> tuple[np.ndarray, bytes | None]:
    """Decode image bytes, apply EXIF orientation, return (rgb float32, icc_profile)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    icc = img.info.get("icc_profile")
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return rgb, icc


def load_path(path) -> np.ndarray:
    with open(path, "rb") as f:
        return load_image(f.read())[0]


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def encode_jpeg(rgb: np.ndarray, quality: int = 95, icc: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(to_uint8(rgb)).save(buf, "JPEG", quality=quality, subsampling=0, icc_profile=icc)
    return buf.getvalue()


def resize_max_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is at most max_side (never upscales)."""
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return rgb
    return cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def center_crop(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    ch, cw = min(h, size), min(w, size)
    top, left = (h - ch) // 2, (w - cw) // 2
    return rgb[top : top + ch, left : left + cw]
