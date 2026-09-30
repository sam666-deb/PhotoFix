"""Stage B/C (baseline): turn an Analysis into edit parameters, then apply them at full resolution.

The split matters: `plan_edits` decides *what* to do from preview statistics only, and `apply_edits`
executes those parameters on the full-resolution image. In Phase 2 a neural network replaces
`plan_edits` (predicting EditParams directly), while `apply_edits` stays deterministic, so output
quality never depends on the network's input resolution.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from photofix.analyzer import Analysis, analyze, luminance
from photofix.imageio import to_uint8

TARGET_MEDIAN = 0.45


@dataclass
class EditParams:
    denoise_h: float = 0.0  # NL-means filter strength (uint8 units); 0 = off
    wb_r: float = 1.0  # white-balance gains, applied in linear light
    wb_g: float = 1.0
    wb_b: float = 1.0
    saturation: float = 1.0
    black_point: float = 0.0  # levels, in luma units
    white_point: float = 1.0
    gamma: float = 1.0  # <1 brightens, >1 darkens
    shadow_lift: float = 0.0
    highlight_recover: float = 0.0
    sharpen_amount: float = 0.0

    def is_identity(self) -> bool:
        return self == EditParams()

    def describe(self) -> list[str]:
        steps = []
        if self.denoise_h:
            steps.append(f"Denoise (strength {self.denoise_h:.1f})")
        if (self.wb_r, self.wb_g, self.wb_b) != (1.0, 1.0, 1.0):
            steps.append(f"White balance (R ×{self.wb_r:.2f}, G ×{self.wb_g:.2f}, B ×{self.wb_b:.2f})")
        if self.black_point or self.white_point != 1.0:
            steps.append(f"Levels ({self.black_point:.2f} – {self.white_point:.2f})")
        if self.gamma != 1.0:
            steps.append(f"Exposure ({'brighten' if self.gamma < 1 else 'darken'}, gamma {self.gamma:.2f})")
        if self.shadow_lift:
            steps.append(f"Lift shadows ({self.shadow_lift:.2f})")
        if self.highlight_recover:
            steps.append(f"Recover highlights ({self.highlight_recover:.2f})")
        if self.saturation != 1.0:
            steps.append(f"Saturation ×{self.saturation:.2f}")
        if self.sharpen_amount:
            steps.append(f"Sharpen ({self.sharpen_amount:.2f})")
        return steps

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def plan_edits(analysis: Analysis) -> EditParams:
    s, st = analysis.scores, analysis.stats
    p = EditParams()

    if s["noise"] > 0.25:
        p.denoise_h = round(1.2 * st["noise_sigma"] * 255.0, 2)  # the estimate runs low in dark regions

    if s["color_cast"] > 0.0:
        # Von Kries correction, scaled by confidence so borderline casts are only partly removed.
        illuminant = np.array([st["illuminant_r"], st["illuminant_g"], st["illuminant_b"]])
        gains = illuminant ** -s["color_cast"]
        gains /= gains @ np.array([0.2126, 0.7152, 0.0722])  # keep overall brightness
        p.wb_r, p.wb_g, p.wb_b = (float(g) for g in gains)

    lc = s["low_contrast"]
    if lc > 0.0:
        p.black_point = st["p1"] * lc
        p.white_point = 1.0 - (1.0 - st["p99"]) * lc
        p.saturation = 1.0 + 0.15 * lc

    exposure_fix = max(s["underexposure"], s["overexposure"])
    if exposure_fix > 0.0:
        median = np.clip((st["median"] - p.black_point) / (p.white_point - p.black_point), 0.02, 0.98)
        target = median + (TARGET_MEDIAN - median) * exposure_fix
        p.gamma = float(np.clip(np.log(target) / np.log(median), 0.35, 2.5))

    p.shadow_lift = 0.6 * s["harsh_shadows"]
    p.highlight_recover = 0.5 * s["overexposure"]

    if s["blur"] > 0.3 and s["noise"] < 0.5:
        p.sharpen_amount = 0.8 * s["blur"]

    return p


def tone_curve(x: np.ndarray, p: EditParams) -> np.ndarray:
    """Monotonic curve on luma: levels -> gamma -> shadow lift -> highlight compression."""
    x = np.clip((x - p.black_point) / max(p.white_point - p.black_point, 1e-3), 0.0, 1.0)
    if p.gamma != 1.0:
        x = x**p.gamma
    if p.shadow_lift:
        x = x + 3.0 * p.shadow_lift * x * (1.0 - x) ** 3
    if p.highlight_recover:
        x = x - 3.0 * p.highlight_recover * (1.0 - x) * x**3
    return np.clip(x, 0.0, 1.0)


def apply_edits(rgb: np.ndarray, p: EditParams) -> np.ndarray:
    out = rgb.astype(np.float32, copy=True)

    if p.denoise_h:
        den = cv2.fastNlMeansDenoisingColored(to_uint8(out), None, p.denoise_h, p.denoise_h, 7, 21)
        out = den.astype(np.float32) / 255.0

    if (p.wb_r, p.wb_g, p.wb_b) != (1.0, 1.0, 1.0):
        gains = np.array([p.wb_r, p.wb_g, p.wb_b], dtype=np.float32)
        out = np.clip(out**2.2 * gains, 0.0, 1.0) ** (1 / 2.2)

    if p.saturation != 1.0:
        y = luminance(out)[..., None]
        out = np.clip(y + (out - y) * p.saturation, 0.0, 1.0)

    if (p.black_point, p.white_point, p.gamma, p.shadow_lift, p.highlight_recover) != (0.0, 1.0, 1.0, 0.0, 0.0):
        # Scale RGB by the luma ratio so hue is preserved while tone changes.
        y = luminance(out)
        ratio = tone_curve(y, p) / np.maximum(y, 1e-4)
        out = np.clip(out * ratio[..., None], 0.0, 1.0)

    if p.sharpen_amount:
        sigma = max(1.0, max(out.shape[:2]) / 1500.0)
        blurred = cv2.GaussianBlur(out, (0, 0), sigma)
        out = np.clip(out + p.sharpen_amount * (out - blurred), 0.0, 1.0)

    return out


def enhance(rgb: np.ndarray, strength: float = 1.0) -> tuple[np.ndarray, Analysis, EditParams]:
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
    analysis = analyze(rgb)
    params = plan_edits(analysis)
    out = apply_edits(rgb, params)
    if strength != 1.0:
        out = np.clip(rgb + (out - rgb) * strength, 0.0, 1.0)
    return out, analysis, params
