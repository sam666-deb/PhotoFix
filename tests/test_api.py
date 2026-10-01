import numpy as np
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
