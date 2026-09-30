"""Full-reference image quality metrics used by the evaluation harness."""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    if np.array_equal(pred, target):
        return 100.0  # identical images; avoid log(0)
    return float(peak_signal_noise_ratio(target, pred, data_range=1.0))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(structural_similarity(target, pred, channel_axis=-1, data_range=1.0))


def delta_e(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean CIE76 color difference (lower is better). ~2.3 is the 'just noticeable' threshold."""
    a = cv2.cvtColor(pred.astype(np.float32), cv2.COLOR_RGB2LAB)
    b = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2LAB)
    return float(np.linalg.norm(a - b, axis=-1).mean())


def all_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {"psnr": psnr(pred, target), "ssim": ssim(pred, target), "delta_e": delta_e(pred, target)}
