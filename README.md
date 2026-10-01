# PhotoFix

**Automatic photo enhancement that finds what's wrong with a photo and fixes it at full resolution, with deep-learning models, measured improvements, and a safety net against over-editing.**

![Python](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-web%20app-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-48%20passing-brightgreen)

Drop in a photo and PhotoFix detects seven kinds of defects: under/over-exposure, low contrast, harsh shadows,
color casts, noise and blur. It corrects only what's actually wrong, and shows you what it did and why. It
ships with two editing styles: a vivid **Natural** look and a **Pro** look learned from a professional retoucher.

Every component started as a classical baseline and was replaced by a learned model **only when it measurably
beat it**. Every claim below comes from a reproducible benchmark in this repo.

---

## Highlights

- **Neural defect analyzer:** a dual-view EfficientNet-B0 that sees the whole frame (exposure, color) and a
  native-resolution crop (noise, blur). It raised the share of edited photos that got better while cutting visible
  changes to *already-good* photos by 64% (ΔE 7.35 → 2.67).
- **Learned "Pro" style:** an Image-Adaptive 3D LUT trained on MIT-Adobe FiveK. It scores +1.8 dB closer to a
  professional retoucher than the rule-based pipeline, and is trained not to "fix" photos that are already finished.
- **Neural denoise and deblur:** a compact NAFNet (3.9M params), +2.9 dB over classical denoising. It runs in
  seamless tiles at any resolution, and only on photos flagged as noisy or blurry.
- **Output guardrail:** checks the *result* for blown highlights, crushed shadows, unnatural skin tones (in
  detected faces only), garish color and amplified noise. It automatically tones the edit down and tells you why.
- **Human evaluation built in:** a blind A/B rating page with a Bradley–Terry leaderboard, because metrics and
  human eyes disagreed more than once during this project.
- **Runs locally on a laptop:** all models were trained on an Apple M1 Pro (PyTorch MPS) in about 35–70 minutes each.

## How it works

```
                 ┌──────────────┐   noisy/blurry?   ┌──────────────┐  Pro style?  ┌──────────────┐
  photo ───────► │   Analyzer   │ ────────────────► │   Restorer   │ ───────────► │ Learned LUT  │
                 │ 7 defect     │                   │ NAFNet,      │              │ expert color │
                 │ probabilities│                   │ tiled        │              │ & tone       │
                 └──────┬───────┘                   └──────────────┘              └──────┬───────┘
                        │ preview statistics                                             │
                        ▼                                                                ▼
                 ┌──────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────┐
                 │  Plan edits  │──►│ Render at full resolution:   │──►│ Guardrail: check result, │──► result
                 │  (+ scene    │   │ local tone mapping, finishing│   │ tone down if overdone    │
                 │   rules)     │   │ (contrast, clarity, vibrance)│   └──────────────────────────┘
                 └──────────────┘   └──────────────────────────────┘
```

**Decide on a preview, render at full resolution.** Every model looks at a small image, while every edit is a
deterministic operation applied to the original pixels: a parametric tone curve, a 3D LUT, or a tiled
restoration network. Output quality therefore never depends on a model's input size, and a 12 MP photo
finishes in about 0.5–2 s (about 8 s when neural restoration is needed).

| Stage | Classical baseline | Learned replacement | Gain |
|---|---|---|---|
| Detect defects | Hand-tuned image statistics | Dual-view EfficientNet-B0 | Change to good photos ΔE 7.35 → 2.67; harsh-shadow detection 0 → 0.72 precision |
| Global color/tone | Gray-edge white balance, curves | Image-Adaptive 3D LUT (FiveK Expert C) | +1.8 dB PSNR vs a professional retoucher |
| Denoise/deblur | NL-means + unsharp mask | NAFNet, blind restoration | +2.9 dB (noise), +1.4 dB (noise + blur) |
| Safety | none | Output guardrail (5 checks) | Change to good photos 2.85 → 2.65, PSNR +0.13 dB |

## Results

All numbers come from held-out data never seen in training and can be reproduced with the scripts in
[Training and evaluation](#training-and-evaluation).

<details open>
<summary><b>End to end: repairing synthetic damage</b> (DIV2K validation, 300 samples)</summary>

| Pipeline | PSNR ↑ | SSIM ↑ | ΔE ↓ | Photos improved | ΔE on already-good photos ↓ |
|---|---|---|---|---|---|
| Damaged input | 17.15 | 0.600 | 19.62 | n/a | n/a |
| Classical baseline | 18.84 | 0.706 | 16.18 | 70% | 7.35 |
| + neural analyzer | 19.62 | 0.718 | 14.50 | 75% | 2.67 |
| + finishing pass | 19.45 | 0.708 | 14.96 | 76% | 2.85 |
| + neural restorer | 19.68 | 0.736 | 14.69 | 77% | 2.85 |
| **+ guardrail (shipped)** | **19.81** | **0.730** | **14.78** | **78%** | **2.65** |

ΔE below about 2.3 is invisible to the eye. The finishing pass *lowers* PSNR on purpose: this benchmark rewards
matching the original, while finishing (contrast, clarity, vibrance) goes beyond it to look better, which the
human ratings and the expert benchmark below capture instead.
</details>

<details>
<summary><b>Compared with a professional retoucher</b> (MIT-Adobe FiveK Expert C, 492 photos)</summary>

| Variant | PSNR vs expert ↑ | ΔE vs expert ↓ | Change to finished photos ↓ |
|---|---|---|---|
| No edit | 21.01 | 12.87 | 0 |
| Rule-based (Natural) | 20.65 | 13.55 | n/a |
| Learned LUT v1 | 22.70 | 10.22 | 4.75 |
| **Learned LUT v3 (shipped as Pro)** | **22.41** | **10.70** | **2.45** |

The rules fix defects but move photos *away* from what a professional would do, which is why the learned style
exists. v1 matched the expert best but also "fixed" phone photos that were already finished, turning skin grey. v3
adds 40% *finished photo → itself* training samples and only keeps checkpoints that leave finished photos visually
unchanged.

The absolute numbers aren't comparable to published FiveK results, because this dataset mirror renders the RAW
inputs differently.
</details>

<details>
<summary><b>Neural restoration vs classical</b> (DIV2K validation crops, 400 samples, PSNR dB)</summary>

| Damage | Unedited | NL-means + unsharp | **NAFNet** |
|---|---|---|---|
| Noise | 28.40 | 29.14 | **32.02** |
| Blur | 26.23 | 26.63 | **27.02** |
| Noise + blur | 23.49 | 24.08 | **25.50** |
| JPEG artifacts | 32.57 | 32.25 | **33.21** |
</details>

<details>
<summary><b>Defect detection</b> (precision / recall)</summary>

| Defect | Classical | Neural |
|---|---|---|
| Underexposure | 0.32 / 0.65 | 0.82 / 1.00 |
| Overexposure | 0.82 / 0.50 | 0.74 / 0.97 |
| Low contrast | 0.68 / 0.84 | 0.97 / 1.00 |
| Harsh shadows | 0.00 / 0.00 | 0.72 / 0.88 |
| Color cast | 0.70 / 0.63 | 0.77 / 0.80 |
| Noise | 0.89 / 0.88 | 1.00 / 1.00 |
| Blur | 0.77 / 0.73 | 1.00 / 1.00 |
</details>

**Real photos.** Synthetic benchmarks only go so far, so each stage was also reviewed by eye on 31 real phone
photos (portraits, sunsets, night skies, storms). That review produced the scene-aware rules, the guardrail limits
and the Natural/Pro split. Neither style won everywhere, so the user chooses.

## Getting started

**Requirements:** Python 3.14 (the tested version), about 2 GB of disk. Apple Silicon (MPS), NVIDIA (CUDA) and CPU are all supported.
Trained on an M1 Pro with 32 GB of RAM.

```bash
git clone https://github.com/sam666-deb/PhotoFix.git
cd PhotoFix
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn server.main:app --reload
```

Open http://127.0.0.1:8000, drop in a photo (JPEG, PNG, WebP or iPhone HEIC), and compare before and after with
the slider.

**Model weights** are not stored in git. Without them the app still works fully on the classical pipeline, and each
trained model is picked up automatically once it's in `checkpoints/`:

| Checkpoint | Enables | Train with |
|---|---|---|
| `analyzer.pt` | Neural defect detection | `scripts.train_analyzer` (about 35 min) |
| `lut.pt` | Pro style | `scripts.train_lut` (about 65 min) |
| `restorer.pt` | Neural denoise/deblur | `scripts.train_restorer` (about 70 min) |

To compare against the baselines, switch any component off with environment variables:
`PHOTOFIX_ANALYZER=classical`, `PHOTOFIX_LUT=off`, `PHOTOFIX_RESTORER=off`, `PHOTOFIX_GUARDRAIL=off`.

## Training and evaluation

```bash
# Data
curl -L -o data/raw/DIV2K_train_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip   # ~3.5 GB
curl -L -o data/raw/DIV2K_valid_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip   # ~0.4 GB
.venv/bin/python -m scripts.prepare_data          # DIV2K -> data/div2k
.venv/bin/python -m scripts.prepare_fivek         # FiveK Expert C, streamed and aligned to 480p (~60 GB download, ~2 GB kept)

# Train
.venv/bin/python -m scripts.train_analyzer
.venv/bin/python -m scripts.train_lut
.venv/bin/python -m scripts.train_restorer

# Evaluate
.venv/bin/python -m scripts.evaluate --images data/div2k/val --per-image 3 --analyzer dl \
    --restorer checkpoints/restorer.pt --guardrail   # end-to-end benchmark
.venv/bin/python -m scripts.eval_fivek             # vs a professional retoucher
.venv/bin/python -m scripts.eval_restore           # neural vs classical restoration
.venv/bin/python -m scripts.gallery --images DIR   # before/after sheets for your own photos
.venv/bin/python -m pytest                         # 48 tests
```

### Human ratings

```bash
.venv/bin/python -m scripts.build_rating_set --images DIR   # pre-render every variant
# rate at http://127.0.0.1:8000/rate.html  (← → better, ↓ same, X both bad)
.venv/bin/python -m scripts.analyze_ratings                 # Bradley–Terry leaderboard + feedback report
```

Ratings are blind: two versions of the same photo appear in random order, and the unedited original is one of the
contenders, so the leaderboard also answers whether editing helps at all. The editor additionally logs which style
and strength people keep when downloading. All of this data stays local in `data/` (gitignored).

## Project structure

```
photofix/
  pipeline.py      orchestration: loads available models, runs every stage, per-stage timings
  analyzer.py      defect statistics + classical scores          net.py       neural analyzer
  enhancer.py      edit planning, finishing, full-res rendering   lut.py       learned 3D LUT (Pro)
  restore.py       NAFNet + tiled inference                       guardrail.py output checks
  degradations.py  synthetic damage for training/evaluation       dataset.py   training datasets
  ratings.py       Bradley–Terry ranking of human ratings         metrics.py   PSNR / SSIM / ΔE
server/            FastAPI app: /api/enhance, rating + feedback endpoints
web/               editor (before/after, strength, Natural/Pro) and rating page; no build step
scripts/           data prep, training, benchmarks, gallery, rating tools
models/            bundled YuNet face detector (used by the guardrail's skin check)
tests/             48 tests: pipeline, models, guardrail, API, ratings
```

## Design decisions and lessons learned

- **The model decides *whether*; image statistics decide *how much*.** The analyzer outputs defect
  probabilities, not edit amounts. That kept the edits interpretable and let each stage be swapped out on its own.
- **"Do no harm" is a first-class metric.** Every model was trained with untouched samples (25% clean photos for
  the analyzer, 15% clean crops for the restorer, 40% finished photos for the LUT) and is evaluated on how much it
  changes photos that need nothing.
- **Domain gap is real.** The FiveK-trained style "fixed" already-processed phone JPEGs and turned skin grey,
  because it had never seen a finished photo. This was found by looking at real photos, not from the metrics.
- **Metrics and eyes disagree.** The finishing pass lowered PSNR but looked clearly better; the best-scoring LUT
  made faces pale. That's why human ratings are the final scorecard.
- **Check outputs, not just inputs.** Every stage decides from the input; the guardrail is the one place that
  checks the result. Its first skin check used color alone and flagged sunset clouds, so it now only looks inside
  detected faces.

## Limitations

- Deliberately dark or bright photos (silhouettes, night skies, white backdrops) can still be over-corrected.
  Telling intent from a mistake needs scene understanding the current models don't have.
- Deblurring gains are modest (+0.4 dB). Real defocus is hard for a 3.9M-parameter model with a short training run.
- Restoration was validated on synthetic damage; the real test set contained no genuinely noisy photos, because
  modern phones denoise heavily.
- Edits are global or tone-based. Region-aware editing (sky, faces, subject) is the most promising next step.

## Roadmap

- [x] Classical baseline, synthetic damage engine, evaluation harness, web app
- [x] Neural defect analyzer
- [x] Learned Pro style (3D LUT, MIT-Adobe FiveK)
- [x] Neural denoise and deblur (NAFNet)
- [x] Orchestrated pipeline and output guardrail
- [x] Blind A/B rating page and feedback loop
- [ ] Public demo and release of pretrained weights
- [ ] Region-aware editing (sky, skin, subject segmentation)

## Acknowledgements and licenses

- **DIV2K:** Agustsson & Timofte, NTIRE 2017. Academic research use.
- **MIT-Adobe FiveK:** Bychkovsky et al., CVPR 2011, used through a
  [Hugging Face mirror](https://huggingface.co/datasets/KlyaT/mit-adobe-fivek). Research use only, under the
  [dataset license](https://data.csail.mit.edu/graphics/fivek/).
- **Image-Adaptive 3D LUT:** Zeng et al., TPAMI 2020 (architecture re-implemented).
- **NAFNet:** Chen et al., ECCV 2022 (architecture re-implemented; original under MIT).
- **EfficientNet-B0:** ImageNet weights from torchvision.
- **YuNet face detector:** Shiqi Yu et al., OpenCV model zoo, MIT. See [models/README.md](models/README.md).

Models trained on these datasets inherit their research-use terms.

---

Built by [@sam666-deb](https://github.com/sam666-deb) as a portfolio project in computer vision and deep learning.
