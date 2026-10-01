"""Head-to-head: neural restorer (Phase 3) vs the classical fixes it replaces (NL-means + unsharp mask),
broken down by damage type, on DIV2K validation crops the model never trained on.

  .venv/bin/python -m scripts.eval_restore
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from photofix.analyzer import analyze
from photofix.dataset import RestorationPatches
from photofix.enhancer import EditParams, apply_edits, plan_edits
from photofix.metrics import psnr, ssim


def damage_type(labels: dict) -> str:
    kinds = [k for k in ("noise", "blur") if labels[k] > 0]
    if not kinds:
        return "jpeg only" if labels["jpeg"] > 0 else "clean"
    return " + ".join(kinds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/div2k/val")
    ap.add_argument("--per-image", type=int, default=4, help="seeded damage variants per photo")
    ap.add_argument("--restorer", default="checkpoints/restorer.pt")
    ap.add_argument("--analyzer", default="checkpoints/analyzer.pt")
    args = ap.parse_args()

    from photofix.net import DLAnalyzer
    from photofix.restore import Restorer

    net, restorer = DLAnalyzer(args.analyzer), Restorer(args.restorer)
    ds = RestorationPatches(args.data, train=False, crop=256, per_image=args.per_image, seed=1, with_labels=True)
    groups = defaultdict(lambda: defaultdict(list))
    for i in range(len(ds)):
        bad_t, good_t, labels = ds[i]
        bad, good = (t.permute(1, 2, 0).numpy() for t in (bad_t, good_t))
        # Classical = exactly what the Natural style does for noise/blur today.
        plan = plan_edits(analyze(bad, net))
        classical = apply_edits(bad, EditParams(denoise_h=plan.denoise_h, sharpen_amount=plan.sharpen_amount))
        neural = restorer(bad)
        g = groups[damage_type(labels)]
        for name, img in (("input", bad), ("classical", classical), ("neural", neural)):
            g[f"{name}_psnr"].append(psnr(img, good))
            g[f"{name}_ssim"].append(ssim(img, good))

    order = ["noise", "blur", "noise + blur", "jpeg only", "clean"]
    summary = {k: {m: float(np.mean(v)) for m, v in groups[k].items()} | {"n": len(groups[k]["input_psnr"])}
               for k in order if k in groups}
    print(f"\nDIV2K validation crops (256²), {len(ds)} samples\n")
    print(f"{'damage':<14}{'n':>4}   {'PSNR: input  classical  neural':<34}{'SSIM: input  classical  neural'}")
    for k, m in summary.items():
        print(f"{k:<14}{m['n']:>4}   {m['input_psnr']:>11.2f}{m['classical_psnr']:>11.2f}{m['neural_psnr']:>8.2f}"
              f"     {m['input_ssim']:>11.3f}{m['classical_ssim']:>11.3f}{m['neural_ssim']:>8.3f}")
    out = Path("results") / f"eval_restore_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
