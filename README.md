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
```

## How it works

```
photo ─► [A] Analyzer ─► defect scores ─► [B] plan_edits ─► EditParams ─► [C] apply_edits ─► result
          (preview)                        (preview)                       (full resolution)
```

| Module | Role |
|---|---|
| `photofix/analyzer.py` | Stage A: scores 7 defects in [0, 1] |
| `photofix/enhancer.py` | Stages B/C: `plan_edits` chooses parameters; `apply_edits` renders them at full resolution |
| `photofix/degradations.py` | Synthetic defect engine: clean photo → degraded photo + labels (training/eval data) |
| `photofix/metrics.py` | PSNR, SSIM, ΔE |
| `scripts/evaluate.py` | Scorecard: restoration quality, "do no harm", per-defect precision/recall |
| `server/main.py` | FastAPI: `POST /api/enhance`, serves the UI |
| `web/` | Upload, before/after slider, strength control, defect report, download |

Decisions are made on a small preview, and the pixels are rendered at full resolution. Because of this,
the DL models in later phases can predict `EditParams` from a small image without ever reducing output quality.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 0 | Classical baseline, degradation engine, eval harness, web app | ✅ done |
| 1 | DL analyzer (multi-label CNN on synthetic + real data) replaces `analyze` | next |
| 2 | DL global enhancer (predict `EditParams` / 3D LUT, trained on MIT-Adobe FiveK) replaces `plan_edits` | |
| 3 | DL local restorers (NAFNet denoise/deblur, tiled inference on MPS) | |
| 4 | Orchestration + safety checks against over-editing | |
| 5 | Polish, deploy (e.g. Hugging Face Spaces), portfolio write-up | |

## Baseline scorecard (Phase 0, built-in samples, seed 0)

| | degraded | enhanced |
|---|---|---|
| PSNR | 18.80 | 19.08 |
| SSIM | 0.649 | 0.759 |

Noise detection: 1.00 precision / 1.00 recall. Blur: 1.00 / 0.82. Low contrast: 0.69 / 0.92.

**Known weaknesses** (these are what the DL phases target):
- Color cast is confused with naturally colored scenes (an orange cat gets "corrected"). Do-no-harm ΔE is 8.45.
- Exposure has no notion of intent: moody or dusk shots get brightened.
- Harsh-shadow and overexposure detection are close to useless.
- Sharpening can't recover true defocus. That needs the Phase 3 deblur model.

Four sample images are only a smoke test. For real numbers, run the eval on a larger clean set,
such as the Kodak 24 or DIV2K validation images.
