import json
import random

import pytest
from fastapi.testclient import TestClient

from photofix.ratings import bradley_terry, distinct_pairs, leaderboard, outcomes

MANIFEST = {
    "variants": ["original", "natural", "pro"],
    "photos": {
        "p1": {"original": {"file": "p1/original.jpg"}, "natural": {"file": "p1/natural.jpg"},
               "pro": {"file": "p1/pro.jpg"}},
        # Natural changed nothing on p2: it's pixel-identical to the original.
        "p2": {"original": {"file": "p2/original.jpg"}, "natural": {"same_as": "original"},
               "pro": {"file": "p2/pro.jpg"}},
    },
}


def test_distinct_pairs_skip_identical_variants():
    pairs = distinct_pairs(MANIFEST)
    assert ("p1", "original", "natural") in pairs
    assert len([p for p in pairs if p[0] == "p2"]) == 1  # only original vs pro


def test_rating_counts_for_identical_duplicates():
    games = outcomes([{"photo": "p2", "left": "pro", "right": "original", "choice": "left"}], MANIFEST)
    # Beating the original on p2 also beats Natural, which *is* the original there.
    assert ("pro", "original", 1.0) in games and ("pro", "natural", 1.0) in games


def test_bradley_terry_recovers_true_order():
    rng = random.Random(0)
    true = {"a": 4.0, "b": 2.0, "c": 1.0}
    games = []
    for _ in range(600):
        x, y = rng.sample(list(true), 2)
        games.append((x, y, 1.0 if rng.random() < true[x] / (true[x] + true[y]) else 0.0))
    p = bradley_terry(games, list(true))
    assert p["a"] > p["b"] > p["c"]
    assert p["a"] / p["c"] == pytest.approx(4.0, rel=0.35)


def test_leaderboard_without_ratings_is_neutral():
    board = leaderboard([], MANIFEST)
    assert board["ratings"] == 0
    assert all(v["preferred_over_original"] == pytest.approx(0.5) for v in board["variants"])


@pytest.fixture
def client(tmp_path, monkeypatch):
    import server.rating as rating

    (tmp_path / "rating").mkdir()
    (tmp_path / "rating" / "manifest.json").write_text(json.dumps(MANIFEST))
    monkeypatch.setattr(rating, "RATING_SET", tmp_path / "rating")
    monkeypatch.setattr(rating, "RATINGS_FILE", tmp_path / "ratings.jsonl")
    monkeypatch.setattr(rating, "FEEDBACK_FILE", tmp_path / "feedback.jsonl")
    from server.main import app

    return TestClient(app), tmp_path


def test_rate_roundtrip(client):
    c, tmp = client
    pair = c.get("/api/rate/next", params={"rater": "sam"}).json()
    assert {pair["left"], pair["right"]} <= set(MANIFEST["variants"]) and pair["left"] != pair["right"]
    res = c.post("/api/rate", json={"photo": pair["photo"], "left": pair["left"], "right": pair["right"],
                                    "choice": "left", "rater": "sam", "ms": 1200})
    assert res.status_code == 200
    assert len((tmp / "ratings.jsonl").read_text().splitlines()) == 1
    assert c.get("/api/rate/stats").json()["ratings"] == 1
    # The rater who already judged that pair is offered a different one next.
    nxt = c.get("/api/rate/next", params={"rater": "sam"}).json()
    assert (nxt["photo"], *sorted((nxt["left"], nxt["right"]))) != (pair["photo"], *sorted((pair["left"], pair["right"])))


def test_rate_rejects_bad_input(client):
    c, _ = client
    assert c.post("/api/rate", json={"photo": "p1", "left": "pro", "right": "pro", "choice": "left"}).status_code == 400
    assert c.post("/api/rate", json={"photo": "nope", "left": "pro", "right": "original",
                                     "choice": "left"}).status_code == 400
    assert c.post("/api/rate", json={"photo": "p1", "left": "pro", "right": "original",
                                     "choice": "maybe"}).status_code == 422


def test_feedback_logged(client):
    c, tmp = client
    res = c.post("/api/feedback", json={"style": "natural", "strength": 0.7, "defects": ["underexposure"]})
    assert res.status_code == 200
    assert json.loads((tmp / "feedback.jsonl").read_text())["strength"] == 0.7


def test_stats_endpoint_before_any_ratings(client):
    c, _ = client
    res = c.get("/api/rate/stats")  # used to fail: NaN confidence intervals aren't valid JSON
    assert res.status_code == 200
    assert all(v["ci90"] is None for v in res.json()["variants"])
