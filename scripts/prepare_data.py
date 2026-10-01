"""Extract DIV2K zips (from data/raw) into data/div2k/{train,val} as high-quality JPEGs (fast to decode).

Download first (~4 GB):
  curl -L -o data/raw/DIV2K_train_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
  curl -L -o data/raw/DIV2K_valid_HR.zip https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
Then:
  .venv/bin/python -m scripts.prepare_data
"""

import argparse
import io
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

RAW = Path("data/raw")
OUT = Path("data/div2k")
SPLITS = {"train": "DIV2K_train_HR.zip", "val": "DIV2K_valid_HR.zip"}


def convert(args: tuple[str, str, str]) -> None:
    zip_path, member, dest = args
    with zipfile.ZipFile(zip_path) as zf:
        img = Image.open(io.BytesIO(zf.read(member))).convert("RGB")
    img.save(dest, "JPEG", quality=95, subsampling=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    for split in ap.parse_args().splits:
        zip_name = SPLITS[split]
        zip_path = RAW / zip_name
        if not zip_path.exists():
            print(f"skip {split}: {zip_path} not found")
            continue
        out_dir = OUT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".png")]
        jobs = [
            (str(zip_path), m, str(out_dir / (Path(m).stem + ".jpg")))
            for m in members
            if not (out_dir / (Path(m).stem + ".jpg")).exists()
        ]
        with ProcessPoolExecutor() as pool:
            list(pool.map(convert, jobs, chunksize=8))
        print(f"{split}: {len(members)} images in {out_dir} ({len(jobs)} newly converted)")


if __name__ == "__main__":
    main()
