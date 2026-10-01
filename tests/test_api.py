import numpy as np
import pytest
from fastapi.testclient import TestClient
from skimage import data

from photofix.degradations import exposure
from photofix.imageio import encode_jpeg
from server.main import app

client = TestClient(app)


def test_enhance_endpoint():
    dark = exposure(data.astronaut().astype(np.float32) / 255.0, -2.5)
    res = client.post("/api/enhance", files={"file": ("dark.jpg", encode_jpeg(dark), "image/jpeg")})
    assert res.status_code == 200
    body = res.json()
    assert body["image"].startswith("data:image/jpeg;base64,")
    assert (body["width"], body["height"]) == (512, 512)
    assert "underexposure" in body["analysis"]["defects"]
    assert any("Exposure" in s for s in body["steps"])


def test_rejects_non_image():
    res = client.post("/api/enhance", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_serves_web_ui():
    res = client.get("/")
    assert res.status_code == 200
    assert "PhotoFix" in res.text


def test_heic_upload_returns_original_for_browser():
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(data.astronaut()).save(buf, "HEIF")
    res = client.post("/api/enhance", files={"file": ("photo.heic", buf.getvalue(), "image/heic")})
    assert res.status_code == 200
    assert res.json()["original"].startswith("data:image/jpeg;base64,")


def _post(style=None):
    img = encode_jpeg(exposure(data.astronaut().astype(np.float32) / 255.0, -2.5))
    form = {"style": style} if style else {}
    return client.post("/api/enhance", files={"file": ("dark.jpg", img, "image/jpeg")}, data=form)


def test_default_style_is_natural():
    body = _post().json()
    assert body["style"] == "natural"
    assert body["params"]["style"] == 0


def test_health_lists_styles():
    from server.main import STYLES

    assert client.get("/api/health").json()["styles"] == list(STYLES)
    assert "natural" in STYLES


def test_pro_style_uses_learned_lut():
    from server.main import STYLES

    if "pro" not in STYLES:
        pytest.skip("no trained LUT checkpoint on this machine")
    body = _post("pro").json()
    assert body["style"] == "pro"
    assert body["steps"][0] == "Expert color & tone (learned)"


def test_unknown_style_rejected():
    assert _post("vintage").status_code == 400
