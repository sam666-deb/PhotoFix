"""Human preference data: blind A/B ratings -> a ranking of pipeline variants.

Metrics like PSNR repeatedly disagreed with what looked better (finishing scored lower but looked
better; LUT v1 scored best but turned skin grey), so human judgment is the final scorecard. Each rating
says which of two variants of the same photo looked better. Bradley–Terry turns many such comparisons
into one strength per variant, reported as "probability of being preferred over the unedited original",
with bootstrap confidence intervals so a lead is only trusted once it's real.
"""

import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

RATINGS_DIR = Path("data/ratings")
CHOICES = ("left", "right", "same", "both_bad")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def distinct_pairs(manifest: dict) -> list[tuple[str, str, str]]:
    """(photo, variant_a, variant_b) for every pair of *visually different* variants."""
    pairs = []
    for photo, variants in manifest["photos"].items():
        stored = [v for v in manifest["variants"] if "file" in variants.get(v, {})]
        pairs += [(photo, a, b) for a, b in combinations(stored, 2)]
    return pairs


def resolve(manifest: dict, photo: str, variant: str) -> str:
    """The stored variant a (possibly duplicate) variant is identical to."""
    entry = manifest["photos"][photo][variant]
    return entry.get("same_as", variant)


def outcomes(records: list[dict], manifest: dict) -> list[tuple[str, str, float]]:
    """(variant_a, variant_b, score_a) per rating; 'same' and 'both bad' count as a tie (0.5).

    A rating also says something about variants that were pixel-identical to the ones shown (e.g. when
    Natural changed nothing, it *is* the original), so duplicates inherit the outcome.
    """
    out = []
    for r in records:
        if r["photo"] not in manifest["photos"]:
            continue
        score = {"left": 1.0, "right": 0.0}.get(r["choice"], 0.5)
        groups = {}
        for v in manifest["variants"]:
            groups.setdefault(resolve(manifest, r["photo"], v), []).append(v)
        for a in groups.get(r["left"], [r["left"]]):
            for b in groups.get(r["right"], [r["right"]]):
                out.append((a, b, score))
    return out


def bradley_terry(games: list[tuple[str, str, float]], variants: list[str], iters: int = 200) -> dict[str, float]:
    """Maximum-likelihood strengths (MM algorithm, Hunter 2004), normalized to geometric mean 1.
    A tiny prior (one tie against every other variant) keeps strengths finite with little data."""
    idx = {v: i for i, v in enumerate(variants)}
    n = len(variants)
    wins = np.full(n, 0.5 * (n - 1))
    pair_counts = np.ones((n, n)) - np.eye(n)
    for a, b, s in games:
        i, j = idx[a], idx[b]
        wins[i] += s
        wins[j] += 1 - s
        pair_counts[i, j] += 1
        pair_counts[j, i] += 1
    p = np.ones(n)
    for _ in range(iters):
        denom = (pair_counts / (p[:, None] + p[None, :])).sum(axis=1)
        p = wins / denom
        p /= np.exp(np.log(p).mean())
    return {v: float(p[idx[v]]) for v in variants}


def leaderboard(records: list[dict], manifest: dict, n_boot: int = 300, seed: int = 0) -> dict:
    variants = manifest["variants"]
    games = outcomes(records, manifest)
    strengths = bradley_terry(games, variants)

    def vs_original(p: dict[str, float]) -> dict[str, float]:
        base = p.get("original", 1.0)
        return {v: p[v] / (p[v] + base) for v in variants}

    point = vs_original(strengths)
    rng = random.Random(seed)
    boots = {v: [] for v in variants}
    if records:
        for _ in range(n_boot):  # resample ratings (not games): duplicates share one human judgment
            sample = [rng.choice(records) for _ in records]
            for v, x in vs_original(bradley_terry(outcomes(sample, manifest), variants, iters=100)).items():
                boots[v].append(x)
    tally = {v: Counter() for v in variants}
    for a, b, s in games:
        tally[a]["win" if s == 1 else "loss" if s == 0 else "tie"] += 1
        tally[b]["win" if s == 0 else "loss" if s == 1 else "tie"] += 1
    rows = []
    for v in variants:
        ci = [float(x) for x in np.percentile(boots[v], [5, 95])] if boots[v] else None  # None until rated
        rows.append({"variant": v, "preferred_over_original": point[v], "ci90": ci,
                     "wins": tally[v]["win"], "losses": tally[v]["loss"], "ties": tally[v]["tie"]})
    rows.sort(key=lambda r: -r["preferred_over_original"])
    return {"ratings": len(records), "variants": rows}
