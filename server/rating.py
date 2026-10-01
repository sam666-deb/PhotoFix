"""Blind A/B rating (/rate.html) and implicit feedback from the main app.

Enabled when data/rating/manifest.json exists (build it with scripts/build_rating_set.py). Ratings
and feedback are appended to data/ratings/*.jsonl: local, gitignored, never sent anywhere.
"""

import json
import random
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from photofix.ratings import RATINGS_DIR, append_jsonl, distinct_pairs, leaderboard, load_jsonl

ROOT = Path(__file__).resolve().parent.parent
RATING_SET = ROOT / "data" / "rating"
RATINGS_FILE = ROOT / RATINGS_DIR / "ratings.jsonl"
FEEDBACK_FILE = ROOT / RATINGS_DIR / "feedback.jsonl"

router = APIRouter(prefix="/api")
_lock = threading.Lock()  # appends from the thread pool must not interleave


def _manifest() -> dict:
    path = RATING_SET / "manifest.json"
    if not path.exists():
        raise HTTPException(404, "No rating set yet. Run: .venv/bin/python -m scripts.build_rating_set --images <folder>")
    return json.loads(path.read_text())


def _key(photo: str, a: str, b: str) -> tuple[str, str, str]:
    return (photo, *sorted((a, b)))


@router.get("/rate/next")
def next_pair(rater: str = "anonymous"):
    """The least-rated pair (preferring ones this rater hasn't seen), in random left/right order."""
    manifest = _manifest()
    records = load_jsonl(RATINGS_FILE)
    counts = Counter(_key(r["photo"], r["left"], r["right"]) for r in records)
    seen = {_key(r["photo"], r["left"], r["right"]) for r in records if r.get("rater") == rater}
    pairs = distinct_pairs(manifest)
    if not pairs:
        raise HTTPException(404, "The rating set has no photos with differing variants.")
    fresh = [p for p in pairs if _key(*p) not in seen] or pairs
    fewest = min(counts[_key(*p)] for p in fresh)
    photo, a, b = random.choice([p for p in fresh if counts[_key(*p)] == fewest])
    left, right = random.sample([a, b], 2)
    url = lambda v: f"/rating/{manifest['photos'][photo][v]['file']}"
    return {"photo": photo, "left": left, "right": right, "left_url": url(left), "right_url": url(right),
            "progress": {"rated_by_you": len(seen), "total_pairs": len(pairs)}}


class Rating(BaseModel):
    photo: str
    left: str
    right: str
    choice: Literal["left", "right", "same", "both_bad"]
    rater: str = Field("anonymous", max_length=40)
    ms: int = Field(0, ge=0, description="time spent deciding")


@router.post("/rate")
def rate(rating: Rating):
    manifest = _manifest()
    variants = manifest["photos"].get(rating.photo, {})
    if rating.left not in variants or rating.right not in variants or rating.left == rating.right:
        raise HTTPException(400, "Unknown photo or variants.")
    with _lock:
        append_jsonl(RATINGS_FILE, {"ts": time.time(), **rating.model_dump()})
    return {"ok": True}


@router.get("/rate/stats")
def stats():
    return leaderboard(load_jsonl(RATINGS_FILE), _manifest())


class Feedback(BaseModel):
    """What a user chose in the main app before downloading: an implicit rating of the defaults."""

    style: str
    strength: float = Field(ge=0, le=1)
    guard_strength: float = Field(1.0, ge=0, le=1)
    defects: list[str] = []
    steps: list[str] = []


@router.post("/feedback")
def feedback(item: Feedback):
    with _lock:
        append_jsonl(FEEDBACK_FILE, {"ts": time.time(), **item.model_dump()})
    return {"ok": True}
