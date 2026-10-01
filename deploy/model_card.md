---
license: other
license_name: research-use-only
license_link: https://data.csail.mit.edu/graphics/fivek/
library_name: pytorch
pipeline_tag: image-to-image
tags:
  - image-enhancement
  - photo-editing
  - denoising
  - 3d-lut
  - nafnet
---

# PhotoFix models

Trained weights for [PhotoFix](https://github.com/sam666-deb/PhotoFix), an automatic photo enhancement pipeline.
See the [project README](https://github.com/sam666-deb/PhotoFix) for a demo video, benchmarks and the write-up.

| File | Model | Params | Trained on | Held-out result |
|---|---|---|---|---|
| `analyzer.pt` | Dual-view EfficientNet-B0 defect classifier (7 defects) | 8.7M | DIV2K + synthetic defects | mean AUC 0.978; noise/blur F1 ≥ 0.99 |
| `lut.pt` | Image-Adaptive 3D LUT (3 basis LUTs, 33³) | 0.57M | MIT-Adobe FiveK, Expert C (1,897 pairs) | 22.41 dB vs expert (no edit: 21.01); ΔE on finished photos 2.45 |
| `restorer.pt` | NAFNet, width 24 (blind denoise/deblur/JPEG cleanup) | 3.9M | DIV2K + realistic synthetic damage | 28.80 dB (input 26.51); noise +2.9 dB over NL-means |

All three were trained on a single Apple M1 Pro (PyTorch MPS) in about 35–70 minutes each.

## Usage

```bash
git clone https://github.com/sam666-deb/PhotoFix && cd PhotoFix
hf download {MODEL_REPO} analyzer.pt lut.pt restorer.pt --local-dir checkpoints
pip install -r requirements.txt && uvicorn server.main:app
```

```python
from photofix.pipeline import Pipeline
from photofix.imageio import load_path

result = Pipeline.load("checkpoints").run(load_path("photo.jpg"), style="natural")  # or "pro"
print(result.steps)  # e.g. ['Neural denoise & deblur', 'Exposure (brighten, gamma 0.54)', ...]
```

## Intended use and limitations

- These weights are for photo enhancement research and demos. They're trained on synthetic defects and
  one retoucher's style, so they don't represent every taste or camera.
- Deliberately dark or bright photos (silhouettes, night skies) can be over-corrected. PhotoFix's guardrail
  mitigates this but doesn't solve it.
- Restoration was validated on synthetic noise and blur; real sensor noise may behave differently.

## Licenses

- `lut.pt` is trained on **MIT-Adobe FiveK**, which is licensed for **research use only**
  ([terms](https://data.csail.mit.edu/graphics/fivek/)). Don't use it commercially.
- `analyzer.pt` and `restorer.pt` are trained on **DIV2K** (academic research use). The analyzer starts from
  torchvision's ImageNet EfficientNet-B0 weights.
- Architectures re-implemented from Zeng et al. (TPAMI 2020, 3D LUT) and Chen et al. (ECCV 2022, NAFNet).
