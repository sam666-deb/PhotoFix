"""Phase 2: learned global color/tone editing with an Image-Adaptive 3D LUT (Zeng et al., TPAMI 2020).

A 3D LUT maps every input RGB color to an output color, which can express white balance, exposure,
contrast, curves, saturation and selective color shifts in one operation. A small CNN looks at a 256²
thumbnail and predicts weights that blend N learned basis LUTs into one LUT for *this* photo; the LUT is
then applied to every pixel by trilinear interpolation, at any resolution. Trained on MIT-Adobe FiveK
(input -> Expert C), so the edit style comes from a professional retoucher rather than hand-written rules.
"""

import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LUT_DIM = 33
N_BASIS = 3
THUMB = 256


def identity_lut(dim: int = LUT_DIM) -> torch.Tensor:
    """(3, D, D, D) LUT indexed [channel, b, g, r] that maps every color to itself."""
    ramp = torch.linspace(0, 1, dim)
    b, g, r = torch.meshgrid(ramp, ramp, ramp, indexing="ij")
    return torch.stack([r, g, b])


def apply_lut(img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """img (B, 3, H, W) in [0, 1]; lut (B, 3, D, D, D). Trilinear lookup -> (B, 3, H, W)."""
    grid = (img.permute(0, 2, 3, 1) * 2 - 1).unsqueeze(1)  # (B, 1, H, W, 3) as (x=r, y=g, z=b)
    out = F.grid_sample(lut, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(2)


class LUTNet(nn.Module):
    def __init__(self, n_basis: int = N_BASIS, dim: int = LUT_DIM):
        super().__init__()

        def block(cin, cout, norm=True):
            layers = [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
            return layers + ([nn.InstanceNorm2d(cout, affine=True)] if norm else [])

        self.backbone = nn.Sequential(
            *block(3, 16, norm=False), *block(16, 32), *block(32, 64), *block(64, 128), *block(128, 128),
            nn.Dropout(0.5), nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(128 * 4, n_basis),
        )
        # Basis LUTs: the first starts as identity, the rest as zero offsets. The predictor starts at
        # weights (1, 0, 0, ...), so an untrained model is a no-op rather than a random color mess.
        basis = torch.zeros(n_basis, 3, dim, dim, dim)
        basis[0] = identity_lut(dim)
        self.basis = nn.Parameter(basis)
        head = self.backbone[-1]
        nn.init.zeros_(head.weight)
        with torch.no_grad():
            head.bias.zero_()
            head.bias[0] = 1.0

    def forward(self, thumb: torch.Tensor) -> torch.Tensor:
        """thumb (B, 3, THUMB, THUMB) in [0, 1] -> per-image LUT (B, 3, D, D, D)."""
        weights = self.backbone(thumb)
        return torch.einsum("bn,ncxyz->bcxyz", weights, self.basis)



def lut_regularization(lut: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(smoothness, monotonicity) penalties on predicted LUTs (B, 3, D, D, D).

    Smoothness prevents banding; monotonicity keeps e.g. a brighter red input from ever mapping to a
    darker red output. Applied to the final per-image LUT (not each basis), since that's what touches pixels.
    """
    tv, mono = 0.0, 0.0
    for axis, channel in ((4, 0), (3, 1), (2, 2)):  # r varies along W, g along H, b along D
        diff = torch.diff(lut, dim=axis)
        tv = tv + diff.pow(2).mean()
        mono = mono + F.relu(-diff.select(1, channel)).mean()
    return tv, mono


def make_thumb(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (THUMB, THUMB), interpolation=cv2.INTER_AREA)


class LUTEnhancer:
    """Inference wrapper: predicts a LUT from a thumbnail, applies it to the full-resolution photo."""

    name = "lut"

    def __init__(self, checkpoint: str | Path, device: torch.device | None = None):
        from photofix.device import get_device

        self.device = device or get_device()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = LUTNet(n_basis=ckpt["n_basis"], dim=ckpt["dim"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self._lock = threading.Lock()  # the web server calls this from a thread pool

    @torch.inference_mode()
    def predict_lut(self, rgb: np.ndarray) -> torch.Tensor:
        thumb = torch.from_numpy(make_thumb(rgb)).permute(2, 0, 1)[None].to(self.device)
        with self._lock:
            return self.model(thumb).cpu()

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray, rows_per_chunk: int = 1024) -> np.ndarray:
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
        lut = self.predict_lut(rgb)
        out = np.empty_like(rgb)
        for top in range(0, rgb.shape[0], rows_per_chunk):  # bounded memory on 50 MP photos
            chunk = torch.from_numpy(np.ascontiguousarray(rgb[top : top + rows_per_chunk])).permute(2, 0, 1)[None]
            out[top : top + rows_per_chunk] = apply_lut(chunk, lut)[0].permute(1, 2, 0).numpy()
        return np.clip(out, 0.0, 1.0)
