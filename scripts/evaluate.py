"""Evaluation harness: every pipeline version (classical baseline, then each DL model) is scored here.

For each clean image it creates N randomly degraded copies (known labels), runs the pipeline, and reports:
  * restoration quality: PSNR / SSIM / ΔE of degraded vs. enhanced, against the clean original
  * "do no harm": how much the pipeline changes the already-clean image (lower is better)
  * analyzer detection: per-defect precision / recall

Usage:
  .venv/bin/python -m scripts.evaluate                      # built-in sample photos
  .venv/bin/python -m scripts.evaluate --images data/kodak  # your own folder of clean photos
  .venv/bin/python -m scripts.evaluate --images data/div2k/val --analyzer dl   # trained analyzer
  .venv/bin/python -m scripts.evaluate --images data/div2k/val --analyzer dl \
      --restorer checkpoints/restorer.pt --guardrail                            # full shipped pipeline
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from photofix.analyzer import DEFECTS
from photofix.degradations import random_degrade
from photofix.imageio import load_path, resize_max_side
from photofix.metrics import all_metrics
from photofix.pipeline import Pipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DETECT_THRESHOLD = 0.3


def load_images(folder: str | None, max_side: int) -> dict[str, np.ndarray]:
    if folder:
        paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            raise SystemExit(f"No images found in {folder}")
        return {p.name: resize_max_side(load_path(p), max_side) for p in paths}
    from skimage import data

    names = ["astronaut", "coffee", "chelsea", "rocket"]
    return {n: resize_max_side(getattr(data, n)()[..., :3].astype(np.float32) / 255.0, max_side) for n in names}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="folder of clean reference photos (default: built-in samples)")
    ap.add_argument("--per-image", type=int, default=8, help="degraded variants per image")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--analyzer", choices=["classical", "dl"], default="classical")
    ap.add_argument("--checkpoint", default="checkpoints/analyzer.pt")
    ap.add_argument("--restorer", default=None, help="neural restorer checkpoint, e.g. checkpoints/restorer.pt")
    ap.add_argument("--guardrail", action="store_true", help="apply the Phase 4 output guardrail")
    args = ap.parse_args()

    restorer = None
    if args.restorer:
        from photofix.restore import Restorer

        restorer = Restorer(args.restorer)

    net = None
    if args.analyzer == "dl":
        from photofix.net import DLAnalyzer

        net = DLAnalyzer(args.checkpoint)

    pipeline = Pipeline(analyzer=net, restorer=restorer, guardrail=args.guardrail)
    adjusted = 0
    rng = np.random.default_rng(args.seed)
    images = load_images(args.images, args.max_side)
    rows, no_harm = [], []
    detect = {d: {"tp": 0, "fp": 0, "fn": 0} for d in DEFECTS}
    start = time.perf_counter()

    for name, clean in images.items():
        out = pipeline.run(clean).image
        no_harm.append(all_metrics(out, clean)["delta_e"])

        for _ in range(args.per_image):
            degraded, labels = random_degrade(clean, rng)
            result = pipeline.run(degraded)
            out, analysis = result.image, result.analysis
            adjusted += result.guard.adjusted
            before, after = all_metrics(degraded, clean), all_metrics(out, clean)
            rows.append({"image": name, "labels": labels, "scores": analysis.scores, "before": before, "after": after})
            for d in DEFECTS:
                truth, pred = labels[d] > 0, analysis.scores[d] >= DETECT_THRESHOLD
                if pred and truth:
                    detect[d]["tp"] += 1
                elif pred:
                    detect[d]["fp"] += 1
                elif truth:
                    detect[d]["fn"] += 1

    def mean(key, metric):
        return float(np.mean([r[key][metric] for r in rows]))

    summary = {
        "analyzer": args.analyzer,
        "restorer": bool(restorer),
        "guardrail": args.guardrail,
        "guardrail_adjusted_pct": 100.0 * adjusted / max(len(rows), 1),
        "n_images": len(images),
        "n_samples": len(rows),
        "restoration": {
            m: {"degraded": mean("before", m), "enhanced": mean("after", m)} for m in ("psnr", "ssim", "delta_e")
        },
        "improved_pct": 100.0 * float(np.mean([r["after"]["psnr"] > r["before"]["psnr"] for r in rows])),
        "no_harm_delta_e": float(np.mean(no_harm)),
        "detection": {
            d: {
                "precision": c["tp"] / max(c["tp"] + c["fp"], 1),
                "recall": c["tp"] / max(c["tp"] + c["fn"], 1),
                "support": c["tp"] + c["fn"],
            }
            for d, c in detect.items()
        },
        "seconds": round(time.perf_counter() - start, 1),
    }

    print(f"\n[{args.analyzer} analyzer{' + neural restorer' if restorer else ''}] {summary['n_samples']} degraded samples from {summary['n_images']} images ({summary['seconds']} s)\n")
    print(f"{'metric':<10}{'degraded':>10}{'enhanced':>10}")
    for m, v in summary["restoration"].items():
        print(f"{m:<10}{v['degraded']:>10.3f}{v['enhanced']:>10.3f}")
    print(f"\nImproved (PSNR up): {summary['improved_pct']:.0f}% of samples")
    if args.guardrail:
        print(f"Guardrail toned down: {summary['guardrail_adjusted_pct']:.0f}% of samples")
    print(f"Do-no-harm ΔE on clean photos: {summary['no_harm_delta_e']:.2f}  (lower is better, <2.3 ≈ invisible)\n")
    print(f"{'defect':<15}{'precision':>10}{'recall':>8}{'n':>5}")
    for d, v in summary["detection"].items():
        print(f"{d:<15}{v['precision']:>10.2f}{v['recall']:>8.2f}{v['support']:>5}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"eval_{args.analyzer}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps({"summary": summary, "samples": rows}, indent=2))
    print(f"\nSaved {out_file}")


if __name__ == "__main__":
    main()
