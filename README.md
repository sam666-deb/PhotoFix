# PhotoFix

A web app that finds defects in a photo (bad exposure, harsh shadows, low contrast, color cast, noise, blur)
and fixes them automatically at full resolution. The pipeline starts as a classical baseline and is then
replaced stage by stage with deep-learning models, and every step is measured on the same evaluation harness.

## Quick start (macOS, Apple Silicon)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn server.main:app --reload
# open http://127.0.0.1:8000
```

```bash
.venv/bin/python -m pytest                          # tests
.venv/bin/python -m scripts.evaluate                # scorecard on built-in sample photos
.venv/bin/python -m scripts.evaluate --images DIR   # scorecard on your own clean photos
.venv/bin/python -m scripts.gallery --images DIR    # before/after sheets for real photos -> results/gallery/
```

The server uses the neural analyzer automatically when `checkpoints/analyzer.pt` exists
(`PHOTOFIX_ANALYZER=classical` forces the baseline). The checkpoint isn't in git, so train it:

```bash
mkdir -p data/raw   # DIV2K, ~4 GB
curl -L -o data/raw/DIV2K_train_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
curl -L -o data/raw/DIV2K_valid_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
.venv/bin/python -m scripts.prepare_data
.venv/bin/python -m scripts.train_analyzer          # ~35 min on an M1 Pro (MPS)
.venv/bin/python -m scripts.evaluate --images data/div2k/val --per-image 3 --analyzer dl
```

## How it works

```
photo ─► [A] Analyzer ─► defect scores ─► [B] plan_edits ─► EditParams ─► [C] apply_edits ─► result
          (preview)                        (preview)                       (full resolution)
```

| Module | Role |
|---|---|
| `photofix/analyzer.py` | Stage A: scores 7 defects in [0, 1] (classical heuristics, or a trained model via `net=`) |
| `photofix/net.py` | Stage A (DL): dual-view EfficientNet-B0 defect classifier + `DLAnalyzer` inference wrapper |
| `photofix/dataset.py` | Training data: clean DIV2K photos + on-the-fly synthetic defects (25% left clean) |
| `photofix/enhancer.py` | Stages B/C: `plan_edits` corrects defects; `plan_finish` measures the corrected preview and adds contrast/clarity/whites/vibrance/sharpening as needed; `apply_edits` renders at full resolution with edge-aware local tone mapping |
| `photofix/degradations.py` | Synthetic defect engine: clean photo → degraded photo + labels (training/eval data) |
| `photofix/metrics.py` | PSNR, SSIM, ΔE |
| `scripts/evaluate.py` | Scorecard: restoration quality, "do no harm", per-defect precision/recall |
| `scripts/train_analyzer.py` | Trains the DL analyzer, saves the best checkpoint by validation loss |
| `photofix/lut.py` | Phase 2: Image-Adaptive 3D LUT (CNN predicts a per-photo LUT from a thumbnail, applied at full res) |
| `scripts/prepare_fivek.py`, `train_lut.py`, `eval_fivek.py` | FiveK data prep (streamed, aligned, 480p), LUT training, expert-referenced scorecard |
| `scripts/gallery.py` | Runs a folder of real photos (JPEG/PNG/HEIC) through the pipeline, writes before/after sheets + report |
| `server/main.py` | FastAPI: `POST /api/enhance`, serves the UI |
| `web/` | Upload, before/after slider, strength control, defect report, download |

Decisions are made on a small preview, and the pixels are rendered at full resolution. Because of this,
the DL models in later phases can predict `EditParams` from a small image without ever reducing output quality.

## Two editing styles

The app offers a **Natural | Pro** switch:

- **Natural** (default): the neural analyzer + rule-based corrections + finishing pass. Vivid, and suits
  phone photos that are already processed.
- **Pro**: global color and tone come from a learned 3D LUT that imitates a professional retoucher
  (`photofix/lut.py`). It's more refined and understated. The analyzer still drives denoising, shadow
  recovery and sharpening.

Neither style wins everywhere (see the Phase 2 scorecard), so the user chooses. Pro needs `checkpoints/lut.pt`:

```bash
.venv/bin/python -m scripts.prepare_fivek   # ~60 GB streamed from Hugging Face, ~2 GB kept; research-use license
.venv/bin/python -m scripts.train_lut       # ~65 min on an M1 Pro
.venv/bin/python -m scripts.eval_fivek      # score all variants against the expert
```

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Classical baseline, degradation engine, eval harness, web app | ✅ done |
| 1 | DL analyzer (dual-view CNN on synthetic defects) replaces `analyze` | ✅ done |
| 2 | Learned "Pro" style: Image-Adaptive 3D LUT trained on MIT-Adobe FiveK (Expert C) | ✅ done |
| 3 | DL local restorers (NAFNet denoise/deblur, tiled inference on MPS) | next |
| 4 | Orchestration + safety checks against over-editing | |
| 5 | Polish, deploy (e.g. Hugging Face Spaces), portfolio write-up | |

## Scorecard

DIV2K validation set: 100 photos never seen in training, 3 seeded random defect mixes each (300 samples).
Both rows use the same enhancer; only the analyzer differs.

| Analyzer | PSNR ↑ | SSIM ↑ | ΔE ↓ | Improved | Do-no-harm ΔE ↓ |
|---|---|---|---|---|---|
| none (degraded input) | 17.15 | 0.600 | 19.62 | | |
| classical (Phase 0) | 18.84 | 0.706 | 16.18 | 70% | 7.35 |
| neural (Phase 1) | 19.62 | 0.718 | 14.50 | 75% | 2.67 |
| **neural + finishing pass** | **19.45** | **0.708** | **14.96** | **76%** | **2.85** |

The finishing pass (local tone mapping, clarity, contrast, whites, vibrance, output sharpening)
scores slightly *lower* because this metric rewards matching the original photo exactly, and finishing
deliberately goes past the original to look better. Phase 2 trains against expert retouches (FiveK), and
from then on the scorecard measures "looks like a pro edited it" instead of "matches the original."

Detection, precision / recall:

| Defect | classical | neural |
|---|---|---|
| underexposure | 0.32 / 0.65 | 0.82 / 1.00 |
| overexposure | 0.82 / 0.50 | 0.74 / 0.97 |
| low contrast | 0.68 / 0.84 | 0.97 / 1.00 |
| harsh shadows | 0.00 / 0.00 | 0.72 / 0.88 |
| color cast | 0.70 / 0.63 | 0.77 / 0.80 |
| noise | 0.89 / 0.88 | 1.00 / 1.00 |
| blur | 0.77 / 0.73 | 1.00 / 1.00 |

**How the neural analyzer works:** a global view (the whole photo at 224²) sees exposure, contrast and color.
A native-resolution 224² crop sees noise and blur, which vanish when downscaled. Each view has its own
EfficientNet-B0 trunk. The model outputs the *probability* each defect is present, and `plan_edits` gates on it
(scores < 0.3 mean "leave it alone"). The image statistics still decide *how much* to correct.

**Scene-aware rules** (found by reviewing 31 real phone photos, see `scripts/gallery.py`):
- *Low-key scenes* (no real highlights: storms, dusk) are brightened less, and get no vibrance or whites boost.
- *Dim scenes* get only light white-balance correction: colored light there is usually the subject.
- *Warm casts* (sunsets) are only half-corrected.

Result on those 31 photos: 21 left untouched (already-good iPhone shots), 8 clearly improved, 1 (an aurora
with a bright horizon glow) still slightly washed out.

## Phase 2 scorecard: compared with a professional retoucher

MIT-Adobe FiveK, Expert C, 492 held-out photos at 480p. *No-harm ΔE* is how much the model changes a photo
that is already finished (the expert's own output); below ~2.3 the change is invisible.

| Variant | PSNR vs expert ↑ | ΔE vs expert ↓ | No-harm ΔE ↓ |
|---|---|---|---|
| no edit | 21.01 | 12.87 | 0 |
| rules (Phase 1) | 20.65 | 13.55 | n/a |
| learned LUT v1 | 22.70 | 10.22 | 4.75 |
| **learned LUT v3 (shipped)** | **22.41** | **10.70** | **2.45** |

- **The rules edit *away* from the expert.** They fix defects, but don't edit the way a retoucher would. That's why Phase 2 exists.
- **v1 learned the expert's style, but also "fixed" photos that were already finished.** On phone JPEGs it
  cooled and greyed skin tones. FiveK inputs are all unprocessed camera renders, so v1 had never seen a finished photo.
- **v3 adds 40% "finished photo → itself" training samples,** and only saves checkpoints with no-harm ΔE ≤ 2.5.
  It gives up 0.3 dB of accuracy to leave finished photos (and skin tones) alone.
- **The numbers aren't comparable to published FiveK results (~25 dB).** This mirror renders the raw inputs
  differently from the papers' preprocessed set (no-edit already scores 21 dB here vs ~18 dB there).

On 31 real phone photos: Pro wins the stormy seascape, Natural wins a vivid portrait and an outdoor
cat photo, and the rest are close. That's why it's a user choice, not a replacement.

**Known weaknesses** (what Phase 3+ targets):
- **Intent:** silhouettes, night skies and white-backdrop (high-key) photos still get "fixed." The model
  sees a dark or bright photo, and the rule-based corrections can't tell that it's deliberate.
- **White balance direction** comes from classical gray-edge, which is fooled by strongly colored content
  (e.g. yellow star trails get pushed blue). The model decides *whether* there is a cast, not *which way*.
- Metrics are on synthetic defects; real-world defects are harder.
- Sharpening can't recover true defocus. That needs the Phase 3 deblur model.
