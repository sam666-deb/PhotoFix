"""Train the NAFNet restorer (Phase 3): blind denoise + deblur + JPEG cleanup, on DIV2K patches.

  .venv/bin/python -m scripts.train_restorer                 # full run (~65 min on an M1 Pro)
  .venv/bin/python -m scripts.train_restorer --steps 40 --eval-every 20   # smoke test

Validation: one fixed crop per DIV2K validation photo with seeded damage (some left clean). Saves the
best model by validation PSNR to checkpoints/restorer.pt, and reports "no-harm" PSNR on the clean
crops (how untouched an already-sharp, noise-free patch stays).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from photofix.dataset import RestorationPatches, worker_init
from photofix.device import get_device
from photofix.restore import NAFNet


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return 10 * math.log10(1 / max(F.mse_loss(a, b).item(), 1e-10))


@torch.no_grad()
def evaluate(model, dataset, device) -> dict:
    model.eval()
    restored, degraded, clean_kept = [], [], []
    for i in range(len(dataset)):
        bad, good = dataset[i]
        out = model(bad[None].to(device)).clamp(0, 1).cpu()[0]
        out = (out * 255).round() / 255
        if torch.equal(bad, good):  # a clean sample: measures do-no-harm
            clean_kept.append(psnr(out, good))
        else:
            restored.append(psnr(out, good))
            degraded.append(psnr(bad, good))
    model.train()
    return {"psnr": float(np.mean(restored)), "input_psnr": float(np.mean(degraded)),
            "no_harm_psnr": float(np.mean(clean_kept)) if clean_kept else None, "n_clean": len(clean_kept)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/div2k")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", default="checkpoints/restorer.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = get_device()
    train_ds = RestorationPatches(Path(args.data) / "train", train=True, crop=args.crop, per_image=50)
    val_ds = RestorationPatches(Path(args.data) / "val", train=False, crop=256)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                        worker_init_fn=worker_init, persistent_workers=True)

    model = NAFNet(width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)
    n_params = sum(p.numel() for p in model.parameters())
    base = evaluate(model, val_ds, device)
    print(f"device={device} params={n_params / 1e6:.2f}M val={len(val_ds)} ({base['n_clean']} clean)")
    print(f"untrained (identity): restored PSNR {base['psnr']:.2f} (= input {base['input_psnr']:.2f})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    history, best, step, t0, running = [], -math.inf, 0, time.perf_counter(), []
    while step < args.steps:
        for bad, good in loader:
            bad, good = bad.to(device), good.to(device)
            loss = F.l1_loss(model(bad), good)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            running.append(loss.item())
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(model, val_ds, device)
                history.append({"step": step, **m})
                line = (f"step {step:5d}  train L1 {np.mean(running):.4f}  val PSNR {m['psnr']:.2f} "
                        f"(input {m['input_psnr']:.2f})  no-harm PSNR {m['no_harm_psnr']:.1f}  "
                        f"({(time.perf_counter() - t0) / 60:.0f} min)")
                running = []
                if m["psnr"] > best:
                    best = m["psnr"]
                    torch.save({"model": model.state_dict(), "config": model.config, "step": step, "metrics": m,
                                "args": vars(args)}, args.out)
                    line += "  *saved*"
                print(line, flush=True)
            if step >= args.steps:
                break

    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"train_restorer_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"untrained": base, "history": history}, indent=2))
    print(f"\nBest val PSNR {best:.2f} (input {base['input_psnr']:.2f})   checkpoint: {args.out}")


if __name__ == "__main__":
    main()
