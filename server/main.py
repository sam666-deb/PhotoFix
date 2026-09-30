"""FastAPI app: serves the web UI and the enhancement API.

Run:  .venv/bin/uvicorn server.main:app --reload   ->  http://127.0.0.1:8000
"""

import base64
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from photofix import enhance
from photofix.imageio import encode_jpeg, load_image

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="PhotoFix")


def _read_upload(file: UploadFile):
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 50 MB.")
    try:
        return load_image(data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read that file as an image.")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Sync handlers run in FastAPI's threadpool, so heavy image work doesn't block the event loop.
@app.post("/api/enhance")
def enhance_image(file: UploadFile = File(...)):
    rgb, icc = _read_upload(file)
    start = time.perf_counter()
    out, analysis, params = enhance(rgb)
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    jpeg = encode_jpeg(out, icc=icc)
    return {
        "analysis": analysis.to_dict(),
        "params": params.to_dict(),
        "steps": params.describe(),
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "elapsed_ms": elapsed_ms,
        "image": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
    }


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
