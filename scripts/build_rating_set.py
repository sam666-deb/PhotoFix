"""Pre-render every pipeline variant of a folder of photos, for blind A/B rating at /rate.html.

  .venv/bin/python -m scripts.build_rating_set --images "Original Photos"

Writes data/rating/<photo>/<variant>.jpg (long side 1600) plus data/rating/manifest.json. Variants
that come out pixel-identical (e.g. Natural leaves an already-good photo untouched) are recorded as
duplicates, so the rating page never asks you to compare two identical images.
"""

import argparse
import hashlib
import json
from pathlib import Path

from photofix.imageio import encode_jpeg, load_path, resize_max_side, to_uint8
from photofix.pipeline import Pipeline

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
OUT = Path("data/rating")
SIDE = 1600

# name -> (style, guardrail). "original" is the unedited photo: the baseline every edit must beat.
VARIANTS = {
    "original": (None, None),
    "natural": ("natural", True),
    "natural-noguard": ("natural", False),
    "pro": ("pro", True),
    "pro-noguard": ("pro", False),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--checkpoints", default="checkpoints")
    args = ap.parse_args()

    pipeline = Pipeline.load(args.checkpoints)
    variants = {k: v for k, v in VARIANTS.items() if v[0] in (None, *pipeline.styles)}
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    manifest = {"variants": list(variants), "photos": {}}
    for i, path in enumerate(paths, 1):
        rgb = resize_max_side(load_path(path), SIDE)
        photo_dir = OUT / path.stem
        photo_dir.mkdir(parents=True, exist_ok=True)
        by_hash, entry = {}, {}
        for name, (style, guardrail) in variants.items():
            if style is None:
                out, notes = rgb, []
            else:
                pipeline.guardrail = guardrail
                result = pipeline.run(rgb, style)
                out, notes = result.image, result.steps
            digest = hashlib.sha1(to_uint8(out).tobytes()).hexdigest()
            if digest in by_hash:  # identical to an earlier variant: don't store or compare it twice
                entry[name] = {"same_as": by_hash[digest], "steps": notes}
                continue
            by_hash[digest] = name
            (photo_dir / f"{name}.jpg").write_bytes(encode_jpeg(out, quality=92))
            entry[name] = {"file": f"{path.stem}/{name}.jpg", "steps": notes}
        manifest["photos"][path.stem] = entry
        distinct = sum("file" in v for v in entry.values())
        print(f"{i:3d}/{len(paths)} {path.name[:40]:40s} {distinct} distinct variants", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pairs = sum(n * (n - 1) // 2 for n in (sum("file" in v for v in e.values()) for e in manifest["photos"].values()))
    print(f"\n{len(paths)} photos, {pairs} distinct comparisons available -> open http://127.0.0.1:8000/rate.html")


if __name__ == "__main__":
    main()
