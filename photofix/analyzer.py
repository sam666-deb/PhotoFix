"""Stage A (baseline): classical defect analyzer.

Scores each defect in [0, 1] (0 = no problem, 1 = severe). These hand-tuned heuristics are the
baseline that the deep-learning analyzer (Phase 1) must beat on the evaluation harness.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from photofix.imageio import center_crop, resize_max_side

DEFECTS = ("underexposure", "overexposure", "low_contrast", "harsh_shadows", "color_cast", "noise", "blur")

PREVIEW_SIDE = 1024
NOISE_CROP = 1024


@dataclass
class Analysis:
    scores: dict[str, float]
    stats: dict[str, float] = field(default_factory=dict)

    def defects(self, threshold: float = 0.5) -> list[str]:
        return [name for name, score in self.scores.items() if score >= threshold]

    def to_dict(self) -> dict:
        return {"scores": self.scores, "stats": self.stats, "defects": self.defects()}


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0 -> 1 as x goes from lo to hi, clamped."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def estimate_noise_sigma(gray: np.ndarray, block: int = 32) -> float:
    """Immerkaer noise estimate, taken over the flatter blocks so texture isn't mistaken for noise."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    resp = np.abs(cv2.filter2D(gray, -1, kernel))[1:-1, 1:-1]
    h, w = (resp.shape[0] // block) * block, (resp.shape[1] // block) * block
    if h == 0 or w == 0:
        return float(np.sqrt(np.pi / 2) * resp.mean() / 6)
    blocks = resp[:h, :w].reshape(h // block, block, w // block, block).mean(axis=(1, 3))
    return float(np.sqrt(np.pi / 2) * np.percentile(blocks, 40) / 6)


def estimate_illuminant(rgb: np.ndarray, p: int = 6, sigma: float = 1.0) -> np.ndarray:
    """Gray-edge color constancy (van de Weijer et al. 2007): the average edge color is assumed neutral.

    Returns per-channel illuminant strength in linear light, normalized to mean 1 ([1, 1, 1] = neutral).
    """
    linear = cv2.GaussianBlur(rgb**2.2, (0, 0), sigma)
    grad = np.sqrt(cv2.Sobel(linear, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(linear, cv2.CV_32F, 0, 1) ** 2)
    unclipped = rgb.max(axis=-1) < 0.98
    grad = grad[unclipped] if unclipped.sum() > 100 else grad.reshape(-1, 3)
    e = (grad.astype(np.float64) ** p).mean(axis=0) ** (1.0 / p) + 1e-8
    return (e / e.mean()).astype(np.float32)


def analyze(rgb: np.ndarray) -> Analysis:
    rgb = np.clip(rgb, 0.0, 1.0)
    small = resize_max_side(rgb, PREVIEW_SIDE)
    y = luminance(small)

    p1, p50, p99 = (float(v) for v in np.percentile(y, [1, 50, 99]))
    dark_frac = float((y < 0.08).mean())
    clipped_frac = float((y > 0.97).mean())

    illuminant = estimate_illuminant(small)
    cast_dev = float(np.abs(illuminant - 1.0).max())

    # Noise is measured at full resolution (downscaling hides it); blur on the normalized preview.
    noise_sigma = estimate_noise_sigma(luminance(center_crop(rgb, NOISE_CROP)))
    lap_var = float(cv2.Laplacian(y * 255.0, cv2.CV_32F).var())

    underexposure = _ramp(0.34 - p50, 0.0, 0.18)
    overexposure = max(_ramp(p50 - 0.68, 0.0, 0.2), _ramp(clipped_frac, 0.03, 0.2))
    scores = {
        "underexposure": underexposure,
        "overexposure": overexposure,
        "low_contrast": _ramp(0.62 - (p99 - p1), 0.0, 0.35),
        "harsh_shadows": _ramp(dark_frac, 0.2, 0.5) * (1.0 - underexposure),
        "color_cast": _ramp(cast_dev, 0.12, 0.35),
        "noise": _ramp(noise_sigma, 0.008, 0.028),
        "blur": _ramp(2.2 - np.log10(lap_var + 1e-6), 0.0, 1.0),
    }
    stats = {
        "p1": p1,
        "median": p50,
        "p99": p99,
        "dark_frac": dark_frac,
        "clipped_frac": clipped_frac,
        "illuminant_r": float(illuminant[0]),
        "illuminant_g": float(illuminant[1]),
        "illuminant_b": float(illuminant[2]),
        "noise_sigma": noise_sigma,
        "laplacian_var": lap_var,
    }
    return Analysis(scores={k: round(v, 3) for k, v in scores.items()}, stats=stats)
