"""Stage B/C (baseline): turn an Analysis into edit parameters, then apply them at full resolution.

The split matters: `plan_edits` decides *what* to do from preview statistics only, and `apply_edits`
executes those parameters on the full-resolution image. In Phase 2 a neural network replaces
`plan_edits` (predicting EditParams directly), while `apply_edits` stays deterministic, so output
quality never depends on the network's input resolution.

Editing happens in two passes, like a photographer's workflow:
  1. corrections (`plan_edits`): fix the detected defects (exposure, shadows, color, noise, ...)
  2. finishing (`plan_finish`): measure the corrected preview and restore what corrections take away.
     Lifting shadows flattens contrast, dulls highlights and washes out color, so this pass adds
     contrast, whites, vibrance and output sharpening, each only as much as that photo needs.
"""

from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np

from photofix.analyzer import Analysis, ScoreModel, analyze, luminance
from photofix.imageio import resize_max_side, to_uint8

TARGET_MEDIAN = 0.45
LOW_KEY_TARGET_MEDIAN = 0.32  # storms, dusk: brighten enough to see detail, not into daylight
MIN_CONFIDENCE = 0.3  # scores below this are treated as "no defect": never partially edit a fine photo
MAX_DARKEN_GAMMA = 1.6  # beyond this, bright high-key photos turn muddy
MAX_TONE_RATIO = 3.0  # cap on per-pixel RGB scaling; the rest of the lift is added neutrally
WB_STRENGTH = 0.75  # remove most of a cast but keep some warmth/coolness; full neutralization looks sterile

TARGET_CONTRAST = 0.22  # luma std of a well-exposed photo; finishing adds contrast below this
TARGET_WHITE = 0.97  # where the brightest highlights (99.5th percentile) should land
TARGET_SATURATION = 0.32  # mean saturation below which vibrance kicks in
MAX_WHITES_BOOST = 1.15  # brighten highlights by at most 15%
NIGHT_MEDIAN = 0.2  # original median luma below this = night / low-light scene
LOW_KEY_P99 = 0.5  # original 99th percentile below this = no real highlights (storms, dusk)
MAX_VIBRANCE = 0.25
OUTPUT_SHARPEN = 0.35  # standard capture/output sharpening for any edited photo
MAX_SHARPEN = 0.6  # beyond this, edges get crunchy halos
WARM_CAST_STRENGTH = 0.5  # warm casts (sunsets, golden hour) are usually intended; only half-correct them
LOW_LIGHT_MEDIAN = 0.35  # below this, colored light is usually the subject (blue hour, aurora, city lights)
LOW_LIGHT_CAST_STRENGTH = 0.3
PREVIEW_SIDE = 1024
LOG_EPS = 0.01


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
    # finishing
    clarity: float = 0.0  # local-contrast boost (gain on the detail layer)
    contrast: float = 0.0  # global S-curve strength, 0..0.5
    whites: float = 1.0  # final luma white point; <1 brightens the highlights
    vibrance: float = 0.0  # saturation boost weighted toward muted colors
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
        if self.clarity:
            steps.append(f"Clarity (+{self.clarity:.2f})")
        if self.contrast:
            steps.append(f"Contrast (+{self.contrast:.2f})")
        if self.whites != 1.0:
            steps.append(f"Brighten whites (+{1 / self.whites - 1:.0%})")
        if self.vibrance:
            steps.append(f"Vibrance (+{self.vibrance:.2f})")
        if self.sharpen_amount:
            steps.append(f"Sharpen ({self.sharpen_amount:.2f})")
        return steps

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def plan_edits(analysis: Analysis) -> EditParams:
    s = {k: (v if v >= MIN_CONFIDENCE else 0.0) for k, v in analysis.scores.items()}
    st = analysis.stats
    p = EditParams()

    if s["noise"] > 0.25:
        p.denoise_h = round(1.2 * st["noise_sigma"] * 255.0, 2)  # the estimate runs low in dark regions

    if s["color_cast"] > 0.0:
        # Von Kries correction, scaled by confidence so borderline casts are only partly removed.
        illuminant = np.array([st["illuminant_r"], st["illuminant_g"], st["illuminant_b"]])
        strength = s["color_cast"] * WB_STRENGTH
        if illuminant[0] > illuminant[2]:  # warm scene: correcting would cool it down
            strength *= WARM_CAST_STRENGTH
        if st["median"] < LOW_LIGHT_MEDIAN:
            strength *= LOW_LIGHT_CAST_STRENGTH
        gains = illuminant ** -strength
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
        goal = LOW_KEY_TARGET_MEDIAN if is_low_key(analysis) else TARGET_MEDIAN
        target = median + (goal - median) * exposure_fix
        p.gamma = float(np.clip(np.log(target) / np.log(median), 0.35, MAX_DARKEN_GAMMA))

    p.shadow_lift = 0.6 * s["harsh_shadows"]
    p.highlight_recover = 0.5 * s["overexposure"]

    if s["blur"] > 0.3 and s["noise"] < 0.5:
        p.sharpen_amount = 0.8 * s["blur"]

    return _drop_negligible(p)


def is_low_key(analysis: Analysis) -> bool:
    """No real highlights anywhere: a storm, dusk, a dim interior. Usually a mood, not a mistake."""
    return analysis.stats["p99"] < LOW_KEY_P99


def plan_finish(corrected_preview: np.ndarray, p: EditParams, analysis: Analysis) -> EditParams:
    """Measure the corrected preview and add the finishing a retoucher would: each step is sized to the
    gap between this photo and a well-finished one, so a photo that's already punchy gets almost none."""
    s = analysis.scores
    y = luminance(corrected_preview)
    p = replace(p)

    # Clarity: put back the local contrast that brightening compresses.
    lifting = p.shadow_lift + max(0.0, 1.0 - p.gamma)
    if lifting > 0 or s["low_contrast"] >= MIN_CONFIDENCE:
        p.clarity = float(np.clip(0.1 + 0.4 * lifting, 0.0, 0.4))

    # Contrast: S-curve when the overall tonal range came out flat.
    p.contrast = float(np.clip((TARGET_CONTRAST - y.std()) / TARGET_CONTRAST * 1.5, 0.0, 0.45))

    # Whites: dull highlights get stretched up to TARGET_WHITE (0.5% of pixels may clip). Skipped for
    # night and low-key scenes: their brightest tones are glows, clouds or light sources (an aurora,
    # a storm sky), and stretching those to white destroys the color and the mood.
    has_real_highlights = analysis.stats["median"] >= NIGHT_MEDIAN and not is_low_key(analysis)
    p995 = float(np.percentile(y, 99.5))
    if has_real_highlights and p995 < TARGET_WHITE - 0.03:
        p.whites = float(np.clip(p995 / TARGET_WHITE, 1 / MAX_WHITES_BOOST, 1.0))

    # Vibrance: lifted shadows look washed out; boost muted colors. Not in low-key scenes, where muted
    # color is part of the mood (vibrance turns a gray storm sepia).
    rgb = corrected_preview[y > 0.08]
    if len(rgb) and not is_low_key(analysis):
        sat = float(((rgb.max(axis=1) - rgb.min(axis=1)) / (rgb.max(axis=1) + 1e-4)).mean())
        p.vibrance = float(np.clip((TARGET_SATURATION - sat) * 1.5, 0.0, MAX_VIBRANCE))

    # Output sharpening on every edited photo, backed off when there's noise to amplify.
    if s["noise"] < 0.5:
        noise_backoff = 1.0 - s["noise"] if s["noise"] >= MIN_CONFIDENCE else 1.0
        p.sharpen_amount = max(p.sharpen_amount, OUTPUT_SHARPEN * noise_backoff)
    p.sharpen_amount = min(p.sharpen_amount, MAX_SHARPEN)

    return _drop_negligible(p)


def _drop_negligible(p: EditParams) -> EditParams:
    """Reset edits too small to see, so they're neither applied nor reported."""
    if max(abs(p.wb_r - 1), abs(p.wb_g - 1), abs(p.wb_b - 1)) < 0.01:
        p.wb_r = p.wb_g = p.wb_b = 1.0
    if p.black_point < 0.005 and p.white_point > 0.995:
        p.black_point, p.white_point = 0.0, 1.0
    if abs(p.saturation - 1) < 0.01:
        p.saturation = 1.0
    if abs(p.gamma - 1) < 0.01:
        p.gamma = 1.0
    if p.whites > 0.99:
        p.whites = 1.0
    for name in ("shadow_lift", "highlight_recover", "clarity", "contrast", "vibrance", "sharpen_amount"):
        if getattr(p, name) < 0.02:
            setattr(p, name, 0.0)
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


def guided_filter(img: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Self-guided edge-preserving smoothing (He et al. 2010): smooths texture, keeps strong edges."""
    size = (2 * radius + 1, 2 * radius + 1)
    mean = cv2.boxFilter(img, -1, size)
    var = cv2.boxFilter(img * img, -1, size) - mean * mean
    a = var / (var + eps)
    b = mean - a * mean
    return cv2.boxFilter(a, -1, size) * img + cv2.boxFilter(b, -1, size)


def local_tone(y: np.ndarray, p: EditParams) -> np.ndarray:
    """Apply the tone curve to an edge-aware base layer and add the detail layer back on top.

    A global curve that lifts shadows also squashes the texture inside them. Working on the base keeps
    texture intact (in log space, so it scales with the new brightness), and `clarity` amplifies it,
    mostly in the darker tones that were lifted; bright areas like skies get little, or they turn gritty.
    """
    radius = max(2, round(0.015 * max(y.shape)))
    base = np.clip(guided_filter(y, radius, eps=0.01), 0.0, 1.0)
    detail = np.log(y + LOG_EPS) - np.log(base + LOG_EPS)
    gain = 1.0 + p.clarity * (1.0 - base) ** 2
    return np.clip((tone_curve(base, p) + LOG_EPS) * np.exp(detail * gain) - LOG_EPS, 0.0, 1.0)


def s_curve(x: np.ndarray, k: float) -> np.ndarray:
    """Contrast S-curve pivoting at mid-gray; monotonic for k < 1."""
    return np.clip(x - k * np.sin(2 * np.pi * x) / (2 * np.pi), 0.0, 1.0)


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

    tonal = (p.black_point, p.white_point, p.gamma, p.shadow_lift, p.highlight_recover, p.clarity)
    if tonal != (0.0, 1.0, 1.0, 0.0, 0.0, 0.0) or p.contrast or p.whites != 1.0:
        y = luminance(out)
        target = local_tone(y, p) if tonal != (0.0, 1.0, 1.0, 0.0, 0.0, 0.0) else y
        if p.contrast:
            target = s_curve(target, p.contrast)
        if p.whites != 1.0:
            target = np.clip(target / p.whites, 0.0, 1.0)
        # Scale RGB by the luma ratio so hue is preserved while tone changes. Near black the ratio
        # explodes and would amplify faint color into a saturated tint, so it's capped and the
        # remaining lift is added equally to all channels (neutral gray).
        ratio = np.minimum(target / np.maximum(y, 1e-4), MAX_TONE_RATIO)
        out = np.clip(out * ratio[..., None] + (target - y * ratio)[..., None], 0.0, 1.0)

    if p.vibrance:
        y = luminance(out)[..., None]
        sat = (out.max(axis=-1, keepdims=True) - out.min(axis=-1, keepdims=True)) / (
            out.max(axis=-1, keepdims=True) + 1e-4
        )
        out = np.clip(y + (out - y) * (1.0 + p.vibrance * (1.0 - sat)), 0.0, 1.0)

    if p.sharpen_amount:
        # Luma-only unsharp mask (no color fringes) with a small threshold so flat areas/noise stay clean.
        sigma = max(1.0, max(out.shape[:2]) / 1500.0)
        y = luminance(out)
        detail = y - cv2.GaussianBlur(y, (0, 0), sigma)
        t = 0.004
        detail *= np.clip((np.abs(detail) - t) / t, 0.0, 1.0)
        out = np.clip(out + (p.sharpen_amount * detail)[..., None], 0.0, 1.0)

    return out


def enhance(
    rgb: np.ndarray, strength: float = 1.0, net: ScoreModel | None = None
) -> tuple[np.ndarray, Analysis, EditParams]:
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
    analysis = analyze(rgb, net)
    params = plan_edits(analysis)
    if not params.is_identity():  # a photo that needs no fixing gets no finishing either
        preview = resize_max_side(rgb, PREVIEW_SIDE)
        corrected = apply_edits(preview, replace(params, denoise_h=0.0, sharpen_amount=0.0))
        params = plan_finish(corrected, params, analysis)
    out = apply_edits(rgb, params)
    if strength != 1.0:
        out = np.clip(rgb + (out - rgb) * strength, 0.0, 1.0)
    return out, analysis, params
