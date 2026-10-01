"""Before/after gallery for real photos: the qualitative check that synthetic metrics can't replace.

  .venv/bin/python -m scripts.gallery --images "Original Photos"

Writes to results/gallery/:
  <name>.jpg         side-by-side before | after, full size
  sheet_NN.jpg       contact sheets (several photos per page) for quick review
  report.json        detected defects and applied edits per photo
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from photofix.pipeline import Pipeline
from photofix.imageio import load_path, to_uint8

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff"}
SHEET_ROW_HEIGHT = 300
SHEET_ROWS = 6
GAP = 6


def side_by_side(before: np.ndarray, after: np.ndarray, height: int | None = None) -> np.ndarray:
    if height:
        size = (round(before.shape[1] * height / before.shape[0]), height)
        before, after = (cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in (before, after))
    gap = np.ones((before.shape[0], GAP, 3), np.float32)
    return np.hstack([before, gap, after])


def write_jpeg(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), to_uint8(rgb)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="results/gallery")
    ap.add_argument("--style", choices=["natural", "pro"], default="natural")
    ap.add_argument("--no-guardrail", action="store_true", help="skip the Phase 4 output guardrail")
    ap.add_argument("--checkpoints", default="checkpoints")
    args = ap.parse_args()

    pipeline = Pipeline.load(args.checkpoints, guardrail=not args.no_guardrail)
    print(f"pipeline: {pipeline.describe()}  style: {args.style}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    report, rows = [], []
    for i, path in enumerate(paths, 1):
        rgb = load_path(path)
        start = time.perf_counter()
        result = pipeline.run(rgb, args.style)
        out, analysis = result.image, result.analysis
        elapsed = time.perf_counter() - start
        write_jpeg(out_dir / f"{path.stem}.jpg", side_by_side(rgb, out))
        rows.append(side_by_side(rgb, out, SHEET_ROW_HEIGHT))
        report.append({
            "index": i,
            "file": path.name,
            "size": [rgb.shape[1], rgb.shape[0]],
            "defects": {k: v for k, v in analysis.scores.items() if v >= 0.3},
            "edits": result.steps,
            "guardrail": result.guard.to_dict(),
            "seconds": round(elapsed, 2),
        })
        print(f"{i:3d}. {path.name[:40]:40s} {', '.join(report[-1]['defects']) or 'no defects':45s} {elapsed:.1f}s")

    for page, start in enumerate(range(0, len(rows), SHEET_ROWS), 1):
        chunk = rows[start : start + SHEET_ROWS]
        width = max(r.shape[1] for r in chunk)
        padded = [np.pad(r, ((0, GAP), (0, width - r.shape[1]), (0, 0)), constant_values=1.0) for r in chunk]
        write_jpeg(out_dir / f"sheet_{page:02d}.jpg", np.vstack(padded))

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{len(paths)} photos -> {out_dir}/")


if __name__ == "__main__":
    main()
