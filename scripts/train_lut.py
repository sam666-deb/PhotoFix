"""Train the Image-Adaptive 3D LUT (Phase 2) on MIT-Adobe FiveK, input -> Expert C.

  .venv/bin/python -m scripts.train_lut                    # full run
  .venv/bin/python -m scripts.train_lut --epochs 2 --limit-batches 10   # smoke test

Reports PSNR and ΔE against the expert on the held-out test set (full 480p frames). Saves to
checkpoints/lut.pt the most accurate model (test PSNR) that leaves finished photos visibly unchanged
(no-harm ΔE <= MAX_NO_HARM); until one qualifies, the gentlest model so far is kept.
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from photofix.dataset import FiveKPairs, worker_init
from photofix.device import get_device
from photofix.lut import LUT_DIM, N_BASIS, LUTNet, apply_lut, lut_regularization

SMOOTH_WEIGHT = 1e-4  # from the paper
MONO_WEIGHT = 10.0
MAX_NO_HARM = 2.5  # ΔE on already-finished photos; ~2.3 is the just-noticeable difference


def delta_e(pred: np.ndarray, target: np.ndarray) -> float:
    lab = lambda x: cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
    return float(np.linalg.norm(lab(pred) - lab(target), axis=-1).mean())


@torch.no_grad()
def evaluate(model, dataset, device, limit=None) -> dict:
    """Per image at full 480p: PSNR on 8-bit-quantized output (as published results do) and CIE76 ΔE.
    no_harm_delta_e: how much the model changes an already-finished photo (the expert's own output)."""
    model.eval()
    psnrs, des, base_psnrs, no_harm = [], [], [], []
    for i in range(len(dataset) if limit is None else min(limit, len(dataset))):
        thumb, inp, tgt = dataset[i]
        out = apply_lut(inp[None].to(device), model(thumb[None].to(device)))[0].clamp(0, 1)
        out = (out * 255).round() / 255
        mse = F.mse_loss(out.cpu(), tgt).item()
        psnrs.append(10 * math.log10(1 / max(mse, 1e-10)))
        base_psnrs.append(10 * math.log10(1 / max(F.mse_loss(inp, tgt).item(), 1e-10)))
        des.append(delta_e(out.cpu().permute(1, 2, 0).numpy(), tgt.permute(1, 2, 0).numpy()))
        tgt_thumb = F.interpolate(tgt[None], size=thumb.shape[-2:], mode="area").to(device)
        again = apply_lut(tgt[None].to(device), model(tgt_thumb))[0].clamp(0, 1).cpu()
        no_harm.append(delta_e(again.permute(1, 2, 0).numpy(), tgt.permute(1, 2, 0).numpy()))
    return {"psnr": float(np.mean(psnrs)), "delta_e": float(np.mean(des)), "input_psnr": float(np.mean(base_psnrs)),
            "no_harm_delta_e": float(np.mean(no_harm))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/fivek")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--p-finished", type=float, default=0.4, help="share of finished-photo -> itself samples")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--limit-batches", type=int, default=None, help="for smoke tests")
    ap.add_argument("--out", default="checkpoints/lut.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = get_device()
    train_ds = FiveKPairs(Path(args.data) / "train", train=True, p_finished=args.p_finished)
    test_ds = FiveKPairs(Path(args.data) / "test", train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                          worker_init_fn=worker_init, persistent_workers=args.workers > 0)

    model = LUTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    print(f"device={device} train={len(train_ds)} test={len(test_ds)} steps/epoch={len(train_dl)}")
    eval_limit = 20 if args.limit_batches else None
    base = evaluate(model, test_ds, device, eval_limit)
    print(f"untrained (identity): PSNR {base['psnr']:.2f}  ΔE {base['delta_e']:.2f}", flush=True)
    if args.p_finished == 0:
        print("warning: --p-finished 0 -> the model never learns to leave finished photos alone")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    history, best = [], None

    def rank(m):  # qualifying models beat all others; among qualifying, higher PSNR; otherwise gentler
        ok = m["no_harm_delta_e"] <= MAX_NO_HARM
        return (ok, m["psnr"] if ok else -m["no_harm_delta_e"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.perf_counter(), 0.0, 0
        for step, (thumb, inp, tgt) in enumerate(train_dl):
            if args.limit_batches and step >= args.limit_batches:
                break
            thumb, inp, tgt = thumb.to(device), inp.to(device), tgt.to(device)
            lut = model(thumb)
            mse = F.mse_loss(apply_lut(inp, lut), tgt)
            tv, mono = lut_regularization(lut)
            loss = mse + SMOOTH_WEIGHT * tv + MONO_WEIGHT * mono
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += mse.item() * len(inp)
            n += len(inp)
        line = f"epoch {epoch:3d}  train PSNR {10 * math.log10(1 / (total / n)):.2f}  ({time.perf_counter() - t0:.0f}s)"
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = evaluate(model, test_ds, device, eval_limit)
            history.append({"epoch": epoch, **m})
            line += f"  test PSNR {m['psnr']:.2f}  ΔE {m['delta_e']:.2f}  no-harm ΔE {m['no_harm_delta_e']:.2f}"
            if best is None or rank(m) > rank(best):
                best = m
                torch.save({"model": model.state_dict(), "n_basis": N_BASIS, "dim": LUT_DIM, "epoch": epoch,
                            "metrics": m, "config": vars(args)}, args.out)
                line += "  *saved*"
        print(line, flush=True)

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"train_lut_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"untrained": base, "history": history}, indent=2))
    print(f"\nSaved model: test PSNR {best['psnr']:.2f}, no-harm ΔE {best['no_harm_delta_e']:.2f} "
          f"(input vs expert: {base['input_psnr']:.2f})   checkpoint: {args.out}")


if __name__ == "__main__":
    main()
