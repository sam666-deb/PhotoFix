"""Phase 4: output guardrail, the last check before a photo is returned.

Every earlier stage decides from the *input* (is it dark? noisy?). This one inspects the *result*:
it compares the edited photo against the original for the classic signs of over-editing, and if any
limit is exceeded it blends the edit back toward the original just far enough to pass, and says why.
Each check is measured *relative to the original*, so a photo that was already clipped or saturated
isn't blamed on the editor.
"""

import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from photofix.analyzer import luminance
from photofix.imageio import resize_max_side

PREVIEW_SIDE = 768
STRENGTHS = (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.0)  # tried in order; 0.0 = edit fully reverted

# Limits, and the plain-language reason shown to the user when one is hit.
LIMITS = {
    "highlights_clipped": (0.03, "protect highlights from blowing out"),  # +fraction of pixels newly at white
    "shadows_crushed": (0.02, "keep shadow detail"),  # +fraction of pixels newly at black
    "skin_shift": (7.0, "keep skin tones natural"),  # mean color shift (ΔE a*b*, lightness excluded) on skin
    "oversaturated": (0.01, "avoid oversaturated colors"),  # +fraction of pixels with garish chroma
    "texture_amplified": (1.8, "avoid amplifying noise and halos"),  # fine-detail energy in flat areas, ×
}
MIN_SKIN_PIXELS = 400  # skin check only when faces show enough skin to measure
FACE_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
GARISH_CHROMA = 95.0  # LAB chroma beyond which colors look neon


@dataclass
class GuardReport:
    strength: float = 1.0  # how much of the edit was kept
    full_strength: dict[str, float] = field(default_factory=dict)  # measurements of the unmodified edit
    violations: list[str] = field(default_factory=list)  # checks that failed at full strength

    @property
    def adjusted(self) -> bool:
        return self.strength < 1.0

    def describe(self) -> str | None:
        if not self.adjusted:
            return None
        reasons = " and ".join(LIMITS[v][1] for v in self.violations)
        if self.strength == 0.0:
            return f"Edit reverted to {reasons}"
        return f"Toned down to {self.strength:.0%} to {reasons}"

    def to_dict(self) -> dict:
        return {"strength": self.strength, "violations": self.violations,
                "measurements": {k: round(v, 4) for k, v in self.full_strength.items()}, "note": self.describe()}


_face_lock = threading.Lock()
_face_detector = None


def detect_faces(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Face boxes (x, y, w, h) via YuNet. Returns [] if the model file is missing."""
    global _face_detector
    if not FACE_MODEL.exists():
        return []
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    with _face_lock:  # the detector object isn't thread-safe; the web server uses a thread pool
        if _face_detector is None:
            _face_detector = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (w, h), 0.7, 0.3, 5000)
        _face_detector.setInputSize((w, h))
        _, faces = _face_detector.detect(bgr)
    return [] if faces is None else [tuple(int(v) for v in f[:4]) for f in faces]


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Skin-colored pixels (YCrCb box) inside detected faces. Color alone confuses skin with sunset
    clouds, sand and wood; requiring a face makes "keep skin natural" mean faces, which is what matters."""
    mask = np.zeros(rgb.shape[:2], bool)
    for x, y, w, h in detect_faces(rgb):
        mask[max(y, 0) : y + h, max(x, 0) : x + w] = True
    if not mask.any():
        return mask
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)  # float input: Y in [0,1], Cr/Cb centered on 0.5
    yy, cr, cb = ycrcb[..., 0], ycrcb[..., 1] * 255, ycrcb[..., 2] * 255
    return mask & (yy > 0.15) & (cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)


def _fine_detail(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Contrast-normalized fine detail (so plain brightening doesn't count) and local mean luma."""
    y = luminance(rgb)
    local = cv2.GaussianBlur(y, (0, 0), 1.5)
    return (y - local) / (local + 0.03), local


def measure(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    """Over-editing measurements of `after` relative to `before` (same-size float RGB previews)."""
    m = {}
    m["highlights_clipped"] = float((after.max(-1) >= 0.995).mean() - (before.max(-1) >= 0.995).mean())
    m["shadows_crushed"] = float((luminance(after) <= 0.01).mean() - (luminance(before) <= 0.01).mean())

    lab_b, lab_a = cv2.cvtColor(before, cv2.COLOR_RGB2LAB), cv2.cvtColor(after, cv2.COLOR_RGB2LAB)
    skin = skin_mask(before)
    m["skin_shift"] = (
        float(np.linalg.norm(lab_a[..., 1:][skin] - lab_b[..., 1:][skin], axis=-1).mean())
        if skin.sum() >= MIN_SKIN_PIXELS else 0.0
    )
    chroma = lambda lab: np.hypot(lab[..., 1], lab[..., 2])
    m["oversaturated"] = float((chroma(lab_a) > GARISH_CHROMA).mean() - (chroma(lab_b) > GARISH_CHROMA).mean())

    detail_b, local_b = _fine_detail(before)
    detail_a, _ = _fine_detail(after)
    texture_std = np.sqrt(cv2.GaussianBlur(detail_b**2, (0, 0), 4))
    flat = (texture_std < np.percentile(texture_std, 30)) & (local_b > 0.03)  # smooth areas: sky, skin, walls
    if flat.sum() > 100:
        rms = lambda d: float(np.sqrt((d[flat] ** 2).mean()))
        m["texture_amplified"] = rms(detail_a) / max(rms(detail_b), 2e-3)
    else:
        m["texture_amplified"] = 1.0
    return m


def violations(m: dict[str, float]) -> list[str]:
    return [name for name, (limit, _) in LIMITS.items() if m[name] > limit]


def guard(original: np.ndarray, edited: np.ndarray) -> tuple[np.ndarray, GuardReport]:
    """Return the edit at the highest strength that passes every check, plus a report."""
    before = resize_max_side(original, PREVIEW_SIDE)
    after = resize_max_side(edited, PREVIEW_SIDE)
    full = measure(before, after)
    report = GuardReport(full_strength=full, violations=violations(full))
    if not report.violations:
        return edited, report
    for s in STRENGTHS[1:]:
        if not violations(measure(before, np.clip(before + (after - before) * s, 0.0, 1.0))):
            report.strength = s
            break
    else:
        report.strength = 0.0
    out = np.clip(original + (edited - original) * report.strength, 0.0, 1.0)
    return out, report
