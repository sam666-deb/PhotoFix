"""Synthetic degradation engine.

Takes a clean photo and applies random, *labeled* defects. This produces unlimited
(degraded, clean, labels) training triples for the analyzer (Phase 1) and restorers (Phase 3),
and is what the evaluation harness uses to score the pipeline.
"""

import cv2
import numpy as np

from photofix.analyzer import DEFECTS
from photofix.imageio import to_uint8


def exposure(rgb: np.ndarray, ev: float) -> np.ndarray:
    """Exposure change in stops, applied in (approximately) linear light."""
    linear = np.clip(rgb, 0.0, 1.0) ** 2.2 * (2.0**ev)
    return np.clip(linear, 0.0, 1.0) ** (1 / 2.2)


def low_contrast(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Compress toward mid-gray, like a hazy or washed-out shot. amount in [0, 1)."""
    return rgb * (1.0 - amount) + 0.5 * amount


def color_cast(rgb: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    return np.clip(rgb * np.asarray(gains, dtype=np.float32), 0.0, 1.0)


def harsh_shadows(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Crush the shadows (lower tones darkened, highlights kept)."""
    return np.clip(rgb, 0.0, 1.0) ** (1.0 + 1.5 * amount)


def gaussian_noise(rgb: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Signal-dependent sensor-like noise: shot noise + read noise."""
    std = np.sqrt(sigma**2 * 0.5 + rgb * sigma**2)
    return np.clip(rgb + rng.standard_normal(rgb.shape, dtype=np.float32) * std, 0.0, 1.0)


def defocus_blur(rgb: np.ndarray, radius: float) -> np.ndarray:
    r = max(1, int(round(radius)))
    kernel = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(kernel, (r, r), r, 1.0, -1)
    # filter2D uses an FFT for large kernels, which can leave tiny negative values; clip them.
    return np.clip(cv2.filter2D(rgb, -1, kernel / kernel.sum()), 0.0, 1.0)


def motion_blur(rgb: np.ndarray, length: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    return np.clip(cv2.filter2D(rgb, -1, kernel / max(kernel.sum(), 1e-6)), 0.0, 1.0)


def jpeg(rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", to_uint8(rgb)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0


def random_degrade(
    rgb: np.ndarray, rng: np.random.Generator, max_defects: int = 3, p_clean: float = 0.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply 1..max_defects random defects. Returns (degraded, labels) with labels as severity in [0, 1].

    With probability p_clean the image is returned untouched (all labels 0), so a model trained on this
    data also learns what a photo that needs no fixing looks like.
    """
    out = rgb.astype(np.float32, copy=True)
    labels = dict.fromkeys(DEFECTS, 0.0)
    if rng.random() < p_clean:
        return out, labels
    exposure_kind = rng.choice(["underexposure", "overexposure"])
    candidates = [exposure_kind, "low_contrast", "harsh_shadows", "color_cast", "noise", "blur"]
    chosen = rng.choice(candidates, size=rng.integers(1, max_defects + 1), replace=False)
    scale = max(rgb.shape[:2]) / 1024.0

    # Order follows the physical capture pipeline: optics -> exposure/tone -> color -> sensor noise.
    if "blur" in chosen:
        sev = rng.uniform(0.3, 1.0)
        if rng.random() < 0.5:
            out = defocus_blur(out, radius=sev * 6 * scale)
        else:
            out = motion_blur(out, length=max(3, int(sev * 20 * scale)), angle=rng.uniform(0, 180))
        labels["blur"] = sev
    if "underexposure" in chosen:
        sev = rng.uniform(0.3, 1.0)
        out = exposure(out, -0.75 - 1.75 * sev)
        labels["underexposure"] = sev
    if "overexposure" in chosen:
        sev = rng.uniform(0.3, 1.0)
        out = exposure(out, 0.5 + 1.5 * sev)
        labels["overexposure"] = sev
    if "harsh_shadows" in chosen:
        sev = rng.uniform(0.3, 1.0)
        out = harsh_shadows(out, sev)
        labels["harsh_shadows"] = sev
    if "low_contrast" in chosen:
        sev = rng.uniform(0.3, 1.0)
        out = low_contrast(out, 0.25 + 0.4 * sev)
        labels["low_contrast"] = sev
    if "color_cast" in chosen:
        sev = rng.uniform(0.3, 1.0)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        gains = 1.0 + 0.25 * sev * direction
        out = color_cast(out, tuple(gains / gains.max()))
        labels["color_cast"] = sev
    if "noise" in chosen:
        sev = rng.uniform(0.3, 1.0)
        out = gaussian_noise(out, sigma=0.01 + 0.07 * sev, rng=rng)
        labels["noise"] = sev

    if rng.random() < 0.3:
        out = jpeg(out, int(rng.integers(60, 95)))
    return np.clip(out, 0.0, 1.0).astype(np.float32), {k: round(float(v), 3) for k, v in labels.items()}


def restoration_degrade(
    patch: np.ndarray, rng: np.random.Generator, margin: int, p_clean: float = 0.15
) -> tuple[np.ndarray, dict[str, float]]:
    """Realistic noise/blur/compression for training the restorer (Phase 3).

    `patch` includes `margin` extra pixels per side so blur kernels see real context; the returned
    image is cropped back. Compared with `random_degrade`, the noise here mimics what reaches a phone
    or camera JPEG: signal-dependent, often luminance-dominant, sometimes spatially correlated
    (demosaicing / in-camera denoising smear it into blotches), then JPEG-compressed.
    """
    out = patch.astype(np.float32, copy=True)
    labels = {"noise": 0.0, "blur": 0.0, "jpeg": 0.0}
    if rng.random() >= p_clean:
        kinds = [k for k, p in (("blur", 0.55), ("noise", 0.65), ("jpeg", 0.5)) if rng.random() < p]
        if not kinds:
            kinds = [rng.choice(["blur", "noise"])]
        if "blur" in kinds:
            kind = rng.choice(["gaussian", "defocus", "motion"])
            if kind == "gaussian":
                sigma = rng.uniform(0.6, 2.5)
                out = cv2.GaussianBlur(out, (0, 0), sigma)
                labels["blur"] = sigma / 2.5
            elif kind == "defocus":
                radius = rng.uniform(1.0, 4.0)
                out = defocus_blur(out, radius)
                labels["blur"] = radius / 4.0
            else:
                length = int(rng.integers(3, 14))
                out = motion_blur(out, length, rng.uniform(0, 180))
                labels["blur"] = length / 13
        if "noise" in kinds:
            sigma = rng.uniform(0.01, 0.08)  # std at mid-gray
            shape = out.shape
            per_channel = rng.standard_normal(shape, dtype=np.float32)
            luma = rng.standard_normal(shape[:2], dtype=np.float32)[..., None]
            mix = rng.uniform(0.0, 0.8)  # how luminance-dominant the noise is
            field = np.sqrt(1 - mix**2) * per_channel + mix * luma
            if rng.random() < 0.4:  # spatially correlated (blotchy) noise
                field = cv2.GaussianBlur(field, (0, 0), rng.uniform(0.5, 1.2))
                field /= field.std() + 1e-6
            std = sigma * np.sqrt(0.3 + 1.4 * np.clip(out, 0, 1))  # shot noise grows with brightness
            out = np.clip(out + field * std, 0.0, 1.0)
            labels["noise"] = sigma / 0.08
        if "jpeg" in kinds:
            q = int(rng.integers(40, 96))
            out = jpeg(out, q)
            labels["jpeg"] = (95 - q) / 55
    if margin:
        out = out[margin:-margin, margin:-margin]
    return np.clip(out, 0.0, 1.0).astype(np.float32), {k: round(float(v), 3) for k, v in labels.items()}
