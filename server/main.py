"""FastAPI app: serves the web UI and the enhancement API.

Run:  .venv/bin/uvicorn server.main:app --reload   ->  http://127.0.0.1:8000

Components load from checkpoints/ (or PHOTOFIX_CHECKPOINTS) when present. Switches for comparing
against baselines: PHOTOFIX_ANALYZER=classical, PHOTOFIX_LUT=off, PHOTOFIX_RESTORER=off,
PHOTOFIX_GUARDRAIL=off.

Public deployment (PHOTOFIX_PUBLIC=1): uploads are processed in memory and never stored, the local-only
rating/feedback features are switched off, photos are capped at PHOTOFIX_MAX_SIDE pixels (default 2048)
so a small CPU server stays responsive, and at most PHOTOFIX_CONCURRENCY photos (default 2) are
processed at once.
"""

import base64
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from photofix.imageio import BROWSER_FORMATS, encode_jpeg, image_format, load_image, resize_max_side
from photofix.pipeline import CHECKPOINTS, Pipeline
from server.rating import RATING_SET
from server.rating import router as rating_router

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
PUBLIC = os.environ.get("PHOTOFIX_PUBLIC") == "1"
MAX_SIDE = int(os.environ.get("PHOTOFIX_MAX_SIDE", 2048 if PUBLIC else 0))  # 0 = full resolution
_slots = threading.BoundedSemaphore(int(os.environ.get("PHOTOFIX_CONCURRENCY", 2 if PUBLIC else 8)))
BUSY_TIMEOUT_S = 60

pipeline = Pipeline.load(
    Path(os.environ.get("PHOTOFIX_CHECKPOINTS", CHECKPOINTS)),
    analyzer=os.environ.get("PHOTOFIX_ANALYZER") != "classical",
    lut=os.environ.get("PHOTOFIX_LUT") != "off",
    restorer=os.environ.get("PHOTOFIX_RESTORER") != "off",
    guardrail=os.environ.get("PHOTOFIX_GUARDRAIL") != "off",
)
STYLES = pipeline.styles
app = FastAPI(title="PhotoFix")
if not PUBLIC:  # rating writes files and serves the owner's photos: local use only
    app.include_router(rating_router)


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
    return {"status": "ok", "public": PUBLIC, "max_side": MAX_SIDE or None, **pipeline.describe()}


# Sync handlers run in FastAPI's threadpool, so heavy image work doesn't block the event loop.
@app.post("/api/enhance")
def enhance_image(file: UploadFile = File(...), style: str = Form("natural")):
    if style not in STYLES:
        raise HTTPException(400, f"Unknown or unavailable style {style!r}; available: {', '.join(STYLES)}.")
    rgb, icc, browser_can_decode = _read_upload(file)
    full_size = [rgb.shape[1], rgb.shape[0]]
    if MAX_SIDE:
        rgb = resize_max_side(rgb, MAX_SIDE)
    if not _slots.acquire(timeout=BUSY_TIMEOUT_S):
        raise HTTPException(503, "The server is busy with other photos. Please try again in a minute.")
    try:
        start = time.perf_counter()
        result = pipeline.run(rgb, style)
    finally:
        _slots.release()
    response = {
        **result.to_dict(),
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "original_size": full_size,
        "elapsed_ms": round((time.perf_counter() - start) * 1000),
        "image": _data_url(encode_jpeg(result.image, icc=icc)),
    }
    if not browser_can_decode:  # e.g. HEIC in Chrome: send the original too, for the before view
        response["original"] = _data_url(encode_jpeg(rgb, icc=icc))
    return response


if not PUBLIC:  # pre-rendered variants for /rate.html (built by scripts/build_rating_set.py)
    RATING_SET.mkdir(parents=True, exist_ok=True)
    app.mount("/rating", StaticFiles(directory=RATING_SET), name="rating")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
