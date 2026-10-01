"""Download MIT-Adobe FiveK (Expert C) and convert it to aligned 480p input/target pairs.

Source: the Hugging Face mirror KlyaT/mit-adobe-fivek (lossless WebP; "original" = the camera raw
rendered to sRGB, "augmented" = Expert C's Lightroom retouch). Full resolution is ~124 GB, so this
streams one parquet shard at a time, writes 480p JPEGs, and deletes the shard.

  .venv/bin/python -m scripts.prepare_fivek                  # 500 test + ~1,900 train pairs (~60 GB download)
  .venv/bin/python -m scripts.prepare_fivek --train-shards 44   # all 3,500 train pairs (~100 GB)

Output: data/fivek/{train,test}/{input,target}/<name>.jpg (short side 480).
Licensed for research use only (https://data.csail.mit.edu/graphics/fivek/).
"""

import argparse
import io
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

REPO = "https://huggingface.co/datasets/KlyaT/mit-adobe-fivek/resolve/main/c"
RAW = Path("data/raw/fivek")
OUT = Path("data/fivek")
SHORT_SIDE = 480
MIN_MATCH = 0.85  # correlation below this = the expert cropped/reframed; the pair is skipped
# The mirror's "validation" split (500 photos) is our held-out test set.
SPLITS = {"test": ("validation", 7), "train": ("train", 44)}


def _thumb(img: Image.Image, size=(64, 64)) -> np.ndarray:
    g = np.asarray(img.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def align(original: Image.Image, target: Image.Image) -> tuple[Image.Image, float]:
    """Rotate/crop the original to match the expert's framing. Returns (aligned original, match score).

    Raw files store pixels unrotated and include a few border pixels the expert's export crops off;
    some exports are also rotated to portrait.
    """
    best, best_score = original, -1.0
    for op in (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270, Image.Transpose.ROTATE_180):
        cand = original.transpose(op) if op is not None else original
        if abs(cand.width / cand.height - target.width / target.height) > 0.05:
            continue
        # Center-crop the raw border so both frames cover the same area.
        if cand.width >= target.width and cand.height >= target.height:
            left, top = (cand.width - target.width) // 2, (cand.height - target.height) // 2
            cand = cand.crop((left, top, left + target.width, top + target.height))
        score = float((_thumb(cand) * _thumb(target)).mean())
        if score > best_score:
            best, best_score = cand, score
    return best, best_score


def resize_short(img: Image.Image, short: int = SHORT_SIDE) -> Image.Image:
    scale = short / min(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)


def convert_row(args) -> str:
    orig_bytes, target_bytes, name, out_dir = args
    stem = Path(name).stem
    if (Path(out_dir) / "target" / f"{stem}.jpg").exists():
        return "exists"
    original = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
    target = Image.open(io.BytesIO(target_bytes)).convert("RGB")
    original, score = align(original, target)
    if score < MIN_MATCH:
        return f"skipped {stem} (match {score:.2f})"
    target = resize_short(target)
    original = original.resize(target.size, Image.Resampling.LANCZOS)
    for kind, img in (("input", original), ("target", target)):
        img.save(Path(out_dir) / kind / f"{stem}.jpg", "JPEG", quality=95, subsampling=0)
    return "ok"


def download(name: str) -> Path:
    dest = RAW / name
    if not (dest.exists() and dest.with_suffix(".done").exists()):
        subprocess.run(["curl", "-sfL", "--retry", "5", "-C", "-", "-o", str(dest), f"{REPO}/{name}"], check=True)
        dest.with_suffix(".done").touch()
    return dest


def process_shard(path: Path, out_dir: Path, pool: ProcessPoolExecutor) -> list[str]:
    table = pq.read_table(path, columns=["original", "augmented"])
    jobs = [
        (row["original"]["bytes"], row["augmented"]["bytes"], row["augmented"]["path"], str(out_dir))
        for row in table.to_pylist()
    ]
    return list(pool.map(convert_row, jobs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-shards", type=int, default=24, help="of 44 (~78 photos each)")
    ap.add_argument("--splits", nargs="+", default=["test", "train"], choices=list(SPLITS))
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    shards = []
    for split in args.splits:
        prefix, total = SPLITS[split]
        count = args.train_shards if split == "train" else total
        for kind in ("input", "target"):
            (OUT / split / kind).mkdir(parents=True, exist_ok=True)
        shards += [(f"{prefix}-{i:05d}-of-{total:05d}.parquet", OUT / split) for i in range(count)]

    # Download the next shard while the current one is being converted.
    with ThreadPoolExecutor(1) as downloader, ProcessPoolExecutor() as pool:
        pending = downloader.submit(download, shards[0][0])
        for i, (name, out_dir) in enumerate(shards):
            path = pending.result()
            if i + 1 < len(shards):
                pending = downloader.submit(download, shards[i + 1][0])
            results = process_shard(path, out_dir, pool)
            path.unlink()
            path.with_suffix(".done").unlink(missing_ok=True)
            for r in results:
                if r.startswith("skipped"):
                    print("  " + r)
            ok = sum(r in ("ok", "exists") for r in results)
            total = len(list((out_dir / "target").glob("*.jpg")))
            print(f"[{i + 1}/{len(shards)}] {name}: {ok}/{len(results)} pairs  ({out_dir.name}: {total} total)", flush=True)


if __name__ == "__main__":
    main()
