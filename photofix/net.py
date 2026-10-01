"""Stage A (Phase 1): deep-learning defect analyzer.

Dual-view network. Global defects (exposure, contrast, color) need to see the whole image, while
pixel-level defects (noise, blur) vanish when an image is downscaled. So the model looks at two views:
  * global: the whole photo squashed to VIEW_SIZE²
  * local:  a VIEW_SIZE² crop at native pixel scale (after capping the photo at WORK_SIDE)
Each view has its own EfficientNet-B0 trunk; pooled features are concatenated into a small head that
outputs one logit per defect = probability that the defect is present.
"""

import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from photofix.analyzer import DEFECTS
from photofix.imageio import resize_max_side

VIEW_SIZE = 224
WORK_SIDE = 2048

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_views(rgb: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (global, local) float32 HWC views. rng=None takes the center crop (inference)."""
    work = resize_max_side(rgb, WORK_SIDE)
    global_view = cv2.resize(work, (VIEW_SIZE, VIEW_SIZE), interpolation=cv2.INTER_AREA)

    h, w = work.shape[:2]
    if h < VIEW_SIZE or w < VIEW_SIZE:
        pad = ((0, max(0, VIEW_SIZE - h)), (0, max(0, VIEW_SIZE - w)), (0, 0))
        work = np.pad(work, pad, mode="reflect")
        h, w = work.shape[:2]
    if rng is None:
        top, left = (h - VIEW_SIZE) // 2, (w - VIEW_SIZE) // 2
    else:
        top, left = int(rng.integers(0, h - VIEW_SIZE + 1)), int(rng.integers(0, w - VIEW_SIZE + 1))
    local_view = work[top : top + VIEW_SIZE, left : left + VIEW_SIZE]
    return global_view.astype(np.float32), np.ascontiguousarray(local_view, dtype=np.float32)


def to_tensor(view: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(view).permute(2, 0, 1).contiguous()


class DefectNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.global_trunk = efficientnet_b0(weights=weights).features
        self.local_trunk = efficientnet_b0(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(2 * 1280, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, len(DEFECTS))
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor) -> torch.Tensor:
        """Inputs: (B, 3, H, W) RGB in [0, 1]. Returns (B, len(DEFECTS)) logits."""
        g = self.pool(self.global_trunk((global_view - self.mean) / self.std)).flatten(1)
        l = self.pool(self.local_trunk((local_view - self.mean) / self.std)).flatten(1)
        return self.head(torch.cat([g, l], dim=1))


class DLAnalyzer:
    """Loads a trained checkpoint and scores photos. Used by `analyze(rgb, net=...)`."""

    name = "dl"

    def __init__(self, checkpoint: str | Path, device: torch.device | None = None):
        from photofix.device import get_device

        self.device = device or get_device()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if tuple(ckpt["defects"]) != DEFECTS:
            raise ValueError(f"Checkpoint defects {ckpt['defects']} don't match {DEFECTS}")
        self.model = DefectNet(pretrained=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self._lock = threading.Lock()  # the web server calls predict from a thread pool

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> dict[str, float]:
        g, l = make_views(np.clip(rgb, 0.0, 1.0))
        with self._lock:
            logits = self.model(to_tensor(g)[None].to(self.device), to_tensor(l)[None].to(self.device))
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        return {d: round(float(p), 3) for d, p in zip(DEFECTS, probs)}
