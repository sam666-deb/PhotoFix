"""Phase 3: neural denoising + deblurring with NAFNet (Chen et al., "Simple Baselines for Image
Restoration", ECCV 2022; architecture re-implemented here from the paper, MIT-licensed original).

One blind model handles noise, blur and JPEG artifacts together, since real photos rarely have only
one. It predicts a residual on top of the input, so a clean photo passes through almost unchanged.
Large photos are processed in overlapping tiles with feathered blending, so memory stays bounded
and there are no seams.
"""

import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an NCHW tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    """NAFNet's replacement for nonlinear activations: split channels in two and multiply."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, 2 * c, 1)
        self.dwconv = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))  # simplified channel attention
        self.conv2 = nn.Conv2d(c, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.ffn1 = nn.Conv2d(c, 2 * c, 1)
        self.ffn2 = nn.Conv2d(c, c, 1)
        # Residual scales start at zero, so each block begins as an identity (stable training).
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.gate(self.dwconv(self.conv1(self.norm1(x))))
        y = self.conv2(y * self.sca(y))
        x = x + y * self.beta
        y = self.ffn2(self.gate(self.ffn1(self.norm2(x))))
        return x + y * self.gamma


class NAFNet(nn.Module):
    def __init__(self, width: int = 24, enc_blocks=(1, 1, 2, 2), middle_blocks: int = 2, dec_blocks=(1, 1, 1, 1)):
        super().__init__()
        self.config = {"width": width, "enc_blocks": list(enc_blocks), "middle_blocks": middle_blocks,
                       "dec_blocks": list(dec_blocks)}
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 3, 3, padding=1)
        self.encoders, self.downs, self.ups, self.decoders = (nn.ModuleList() for _ in range(4))
        c = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(middle_blocks)])
        for n in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.multiple = 2 ** len(enc_blocks)
        # Zero the output layer so an untrained model returns its input unchanged (residual = 0).
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

    def forward(self, inp):
        """inp (B, 3, H, W) in [0, 1] -> restored image, same size."""
        h, w = inp.shape[-2:]
        pad_h, pad_w = (-h) % self.multiple, (-w) % self.multiple
        x = F.pad(inp, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else inp
        feat = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)
        feat = self.middle(feat)
        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = dec(up(feat) + skip)
        out = self.ending(feat) + x  # predict the residual
        return out[..., :h, :w]


def _feather(size: int, overlap: int) -> np.ndarray:
    """1D blend weights: ramp up over `overlap` px at both ends, flat in the middle."""
    ramp = np.ones(size, np.float32)
    if overlap > 0:
        edge = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
        ramp[:overlap], ramp[-overlap:] = edge, edge[::-1]
    return ramp


class Restorer:
    """Inference wrapper: tiled, feather-blended NAFNet over a full-resolution photo."""

    name = "nafnet"

    def __init__(self, checkpoint: str | Path, device: torch.device | None = None):
        from photofix.device import get_device

        self.device = device or get_device()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = NAFNet(**ckpt["config"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self._lock = threading.Lock()  # the web server calls this from a thread pool

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray, tile: int = 512, overlap: int = 32) -> np.ndarray:
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)
        h, w = rgb.shape[:2]
        if h <= tile and w <= tile:
            return self._run(rgb)
        acc = np.zeros_like(rgb)
        weight = np.zeros((h, w, 1), np.float32)
        step = tile - overlap
        tops = sorted({min(t, max(h - tile, 0)) for t in range(0, h, step)})
        lefts = sorted({min(l, max(w - tile, 0)) for l in range(0, w, step)})
        for top in tops:
            for left in lefts:
                patch = rgb[top : top + tile, left : left + tile]
                ph, pw = patch.shape[:2]
                wy = _feather(ph, overlap if ph == tile else 0)
                wx = _feather(pw, overlap if pw == tile else 0)
                # Don't fade the outer border of the image: only shared edges blend.
                if top == 0:
                    wy[: overlap] = 1.0
                if top + ph >= h:
                    wy[-overlap:] = 1.0
                if left == 0:
                    wx[: overlap] = 1.0
                if left + pw >= w:
                    wx[-overlap:] = 1.0
                mask = (wy[:, None] * wx[None, :])[..., None]
                acc[top : top + ph, left : left + pw] += self._run(patch) * mask
                weight[top : top + ph, left : left + pw] += mask
        return np.clip(acc / np.maximum(weight, 1e-6), 0.0, 1.0)

    def _run(self, patch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(patch)).permute(2, 0, 1)[None].to(self.device)
        with self._lock:
            y = self.model(x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return y
