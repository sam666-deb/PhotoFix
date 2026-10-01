"""FastAPI app: serves the web UI and the enhancement API.

Run:  .venv/bin/uvicorn server.main:app --reload   ->  http://127.0.0.1:8000

Components load from checkpoints/ (or PHOTOFIX_CHECKPOINTS) when present. Switches for comparing
against baselines: PHOTOFIX_ANALYZER=classical, PHOTOFIX_LUT=off, PHOTOFIX_RESTORER=off,
PHOTOFIX_GUARDRAIL=off.
"""

import base64
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from photofix.imageio import BROWSER_FORMATS, encode_jpeg, image_format, load_image
from photofix.pipeline import CHECKPOINTS, Pipeline

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

pipeline = Pipeline.load(
    Path(os.environ.get("PHOTOFIX_CHECKPOINTS", CHECKPOINTS)),
    analyzer=os.environ.get("PHOTOFIX_ANALYZER") != "classical",
    lut=os.environ.get("PHOTOFIX_LUT") != "off",
    restorer=os.environ.get("PHOTOFIX_RESTORER") != "off",
    guardrail=os.environ.get("PHOTOFIX_GUARDRAIL") != "off",
)
STYLES = pipeline.styles
app = FastAPI(title="PhotoFix")


def _read_upload(file: UploadFile):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 50 MB.")
    try:
        rgb, icc = load_image(data)
        return rgb, icc, image_format(data) in BROWSER_FORMATS
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read that file as an image.")


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


@app.get("/api/health")
def health():
    return {"status": "ok", **pipeline.describe()}


# Sync handlers run in FastAPI's threadpool, so heavy image work doesn't block the event loop.
@app.post("/api/enhance")
def enhance_image(file: UploadFile = File(...), style: str = Form("natural")):
    if style not in STYLES:
        raise HTTPException(400, f"Unknown or unavailable style {style!r}; available: {', '.join(STYLES)}.")
    rgb, icc, browser_can_decode = _read_upload(file)
    start = time.perf_counter()
    result = pipeline.run(rgb, style)
    response = {
        **result.to_dict(),
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "elapsed_ms": round((time.perf_counter() - start) * 1000),
        "image": _data_url(encode_jpeg(result.image, icc=icc)),
    }
    if not browser_can_decode:  # e.g. HEIC in Chrome: send the original too, for the before view
        response["original"] = _data_url(encode_jpeg(rgb, icc=icc))
    return response


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
