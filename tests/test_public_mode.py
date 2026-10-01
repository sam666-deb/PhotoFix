import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from skimage import data

from photofix.degradations import exposure
from photofix.imageio import encode_jpeg


@pytest.fixture
def public_client(monkeypatch):
    """The app as configured on the public demo (server settings are read at import time)."""
    import server.main

    monkeypatch.setenv("PHOTOFIX_PUBLIC", "1")
    monkeypatch.setenv("PHOTOFIX_MAX_SIDE", "256")
    yield TestClient(importlib.reload(server.main).app)
    monkeypatch.delenv("PHOTOFIX_PUBLIC")
    monkeypatch.delenv("PHOTOFIX_MAX_SIDE")
    importlib.reload(server.main)  # restore the local configuration for other tests


def test_health_reports_public_mode(public_client):
    body = public_client.get("/api/health").json()
    assert body["public"] is True and body["max_side"] == 256


def test_large_uploads_are_downscaled(public_client):
    img = encode_jpeg(exposure(data.astronaut().astype(np.float32) / 255.0, -2.0))  # 512×512
    body = public_client.post("/api/enhance", files={"file": ("a.jpg", img, "image/jpeg")}).json()
    assert (body["width"], body["height"]) == (256, 256)
    assert body["original_size"] == [512, 512]


def test_local_only_features_are_off(public_client):
    assert public_client.get("/api/rate/next").status_code == 404
    assert public_client.post("/api/feedback", json={"style": "natural", "strength": 1}).status_code in (404, 405)
    assert public_client.get("/rating/manifest.json").status_code == 404
