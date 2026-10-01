"""Report from human ratings (/rate.html) and editor feedback (strength people actually keep).

  .venv/bin/python -m scripts.analyze_ratings
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from photofix.ratings import RATINGS_DIR, leaderboard, load_jsonl


def main():
    manifest_path = Path("data/rating/manifest.json")
    records = load_jsonl(RATINGS_DIR / "ratings.jsonl")
    if manifest_path.exists() and records:
        board = leaderboard(records, json.loads(manifest_path.read_text()))
        raters = {r["rater"] for r in records}
        print(f"\nBlind A/B ratings: {board['ratings']} from {len(raters)} rater(s)\n")
        print(f"{'variant':<18}{'preferred over original':>26}{'90% CI':>16}{'W/L/T':>12}")
        for v in board["variants"]:
            lo, hi = v["ci90"] or (float("nan"), float("nan"))
            print(f"{v['variant']:<18}{v['preferred_over_original']:>25.0%} {lo:>8.0%} – {hi:<5.0%}"
                  f"{v['wins']:>5}/{v['losses']}/{v['ties']}")
        median_ms = np.median([r["ms"] for r in records if r.get("ms")]) if records else 0
        print(f"\nMedian decision time: {median_ms / 1000:.1f} s")
        if len(records) < 30:
            print("Fewer than 30 ratings: treat the ranking as provisional.")
    else:
        print("No ratings yet. Build a set (scripts/build_rating_set.py) and rate at /rate.html.")

    feedback = load_jsonl(RATINGS_DIR / "feedback.jsonl")
    if feedback:
        print(f"\nEditor downloads: {len(feedback)}")
        by_style = defaultdict(list)
        for f in feedback:
            by_style[f["style"]].append(f["strength"])
        for style, s in by_style.items():
            print(f"  {style:<8} n={len(s):<3} mean strength kept {np.mean(s):.0%}  (lowered in {np.mean(np.array(s) < 1):.0%})")
        by_defect = defaultdict(list)
        for f in feedback:
            for d in f["defects"] or ["(none)"]:
                by_defect[d].append(f["strength"])
        print("  strength kept, by detected defect:")
        for d, s in sorted(by_defect.items(), key=lambda kv: np.mean(kv[1])):
            print(f"    {d:<15} n={len(s):<3} {np.mean(s):.0%}")
        print("  A defect where people keep lowering strength is one the defaults overcorrect.")


if __name__ == "__main__":
    main()
