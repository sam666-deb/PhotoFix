"""Train the dual-view defect analyzer (Phase 1) on Apple Silicon (MPS), CUDA, or CPU.

  .venv/bin/python -m scripts.train_analyzer                 # full run
  .venv/bin/python -m scripts.train_analyzer --epochs 1 --limit-batches 20   # smoke test

Saves the best (lowest validation loss) model to checkpoints/analyzer.pt and a history to results/.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from photofix.analyzer import DEFECTS
from photofix.dataset import DegradedPhotos, worker_init
from photofix.device import get_device
from photofix.net import DefectNet


def roc_auc(scores: np.ndarray, truth: np.ndarray) -> float:
    """Rank-based ROC AUC (probability a random positive outranks a random negative)."""
    pos, neg = scores[truth > 0.5], scores[truth <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def detection_metrics(probs: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> dict:
    out = {}
    for i, d in enumerate(DEFECTS):
        pred, t = probs[:, i] >= threshold, truth[:, i] > 0.5
        tp, fp, fn = int((pred & t).sum()), int((pred & ~t).sum()), int((~pred & t).sum())
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[d] = {
            "auc": roc_auc(probs[:, i], truth[:, i]),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        }
    # "Clean" = model says nothing is wrong. False alarms here are the over-editing we want to kill.
    clean = truth.sum(axis=1) == 0
    out["_clean_false_alarm_rate"] = float((probs[clean] >= threshold).any(axis=1).mean()) if clean.any() else None
    return out


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, limit=None):
    train = optimizer is not None
    model.train(train)
    total, n, all_probs, all_truth = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for step, (g, l, presence, _) in enumerate(loader):
            if limit and step >= limit:
                break
            g, l, presence = g.to(device), l.to(device), presence.to(device)
            logits = model(g, l)
            loss = criterion(logits, presence)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
            total += loss.item() * len(g)
            n += len(g)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_truth.append(presence.cpu().numpy())
    return total / max(n, 1), np.concatenate(all_probs), np.concatenate(all_truth)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/div2k")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--repeat", type=int, default=4, help="passes over the train photos per epoch")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit-batches", type=int, default=None, help="for smoke tests")
    ap.add_argument("--out", default="checkpoints/analyzer.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = get_device()
    train_ds = DegradedPhotos(Path(args.data) / "train", train=True, repeat=args.repeat)
    val_ds = DegradedPhotos(Path(args.data) / "val", train=False, repeat=4)
    loader_kw = dict(batch_size=args.batch, num_workers=args.workers, worker_init_fn=worker_init,
                     persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = DefectNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = min(len(train_dl), args.limit_batches or math.inf)
    total_steps, warmup = args.epochs * steps_per_epoch, steps_per_epoch // 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / max(warmup, 1) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total_steps - warmup, 1))),
    )
    criterion = nn.BCEWithLogitsLoss()

    print(f"device={device} train={len(train_ds)} val={len(val_ds)} steps/epoch={steps_per_epoch}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    history, best = [], math.inf
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, _, _ = run_epoch(model, train_dl, device, criterion, optimizer, scheduler, args.limit_batches)
        val_loss, probs, truth = run_epoch(model, val_dl, device, criterion, limit=args.limit_batches)
        metrics = detection_metrics(probs, truth)
        mean_auc = float(np.nanmean([metrics[d]["auc"] for d in DEFECTS]))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "mean_auc": mean_auc,
                        "metrics": metrics})
        flag = ""
        if val_loss < best:
            best, flag = val_loss, "  *saved*"
            torch.save({"model": model.state_dict(), "defects": list(DEFECTS), "epoch": epoch,
                        "val_loss": val_loss, "metrics": metrics, "config": vars(args)}, args.out)
        print(f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}  mean AUC {mean_auc:.3f}  "
              f"clean false-alarm {metrics['_clean_false_alarm_rate']:.2f}  "
              f"({time.perf_counter() - t0:.0f}s){flag}", flush=True)

    best_metrics = min(history, key=lambda h: h["val_loss"])["metrics"]
    print(f"\n{'defect':<15}{'AUC':>7}{'prec':>7}{'recall':>8}{'F1':>7}")
    for d in DEFECTS:
        m = best_metrics[d]
        print(f"{d:<15}{m['auc']:>7.3f}{m['precision']:>7.2f}{m['recall']:>8.2f}{m['f1']:>7.2f}")
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"train_analyzer_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(history, indent=2))
    print(f"\nBest checkpoint: {args.out}   history: {out}")


if __name__ == "__main__":
    main()
