"""Score pipeline variants against a professional retoucher (FiveK Expert C) on held-out photos.

This is the "does it look like a pro edited it?" scorecard, complementing scripts/evaluate.py
(which measures defect repair on synthetic damage).

  .venv/bin/python -m scripts.eval_fivek
  .venv/bin/python -m scripts.eval_fivek --variants input rules --limit 100
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from photofix import enhance
from photofix.dataset import list_images, read_rgb
from photofix.metrics import delta_e, psnr

VARIANTS = {
    "input": "no edit",
    "rules": "neural analyzer + rule-based corrections + finishing (Phase 1)",
    "lut": "learned expert LUT only (Phase 2)",
    "lut+local": "learned LUT + analyzer-driven denoise/shadows/sharpening",
    "lut+local+finish": "learned LUT + analyzer-driven fixes + finishing pass",
}


def quantize(rgb: np.ndarray) -> np.ndarray:
    return np.round(np.clip(rgb, 0, 1) * 255) / 255


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/fivek/test")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--analyzer", default="checkpoints/analyzer.pt")
    ap.add_argument("--lut", default="checkpoints/lut.pt")
    args = ap.parse_args()

    from photofix.net import DLAnalyzer

    net = DLAnalyzer(args.analyzer)
    lut = None
    if any(v.startswith("lut") for v in args.variants):
        from photofix.lut import LUTEnhancer

        lut = LUTEnhancer(args.lut)

    run = {
        "input": lambda x: x,
        "rules": lambda x: enhance(x, net=net)[0],
        "lut": lambda x: lut(x),
        "lut+local": lambda x: enhance(x, net=net, lut=lut, finish=False)[0],
        "lut+local+finish": lambda x: enhance(x, net=net, lut=lut)[0],
    }
    inputs = list_images(Path(args.data) / "input")[: args.limit]
    scores = {v: {"psnr": [], "delta_e": [], "seconds": 0.0} for v in args.variants}
    for path in inputs:
        inp, tgt = read_rgb(path), read_rgb(Path(args.data) / "target" / path.name)
        for v in args.variants:
            start = time.perf_counter()
            out = quantize(run[v](inp))
            scores[v]["seconds"] += time.perf_counter() - start
            scores[v]["psnr"].append(psnr(out, tgt))
            scores[v]["delta_e"].append(delta_e(out, tgt))

    summary = {
        v: {
            "description": VARIANTS[v],
            "psnr": float(np.mean(s["psnr"])),
            "delta_e": float(np.mean(s["delta_e"])),
            "ms_per_image": 1000 * s["seconds"] / len(inputs),
        }
        for v, s in scores.items()
    }
    print(f"\nFiveK Expert C, {len(inputs)} held-out photos at 480p\n")
    print(f"{'variant':<18}{'PSNR ↑':>8}{'ΔE ↓':>8}{'ms':>7}   description")
    for v, m in summary.items():
        print(f"{v:<18}{m['psnr']:>8.2f}{m['delta_e']:>8.2f}{m['ms_per_image']:>7.0f}   {m['description']}")
    out = Path("results") / f"eval_fivek_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
