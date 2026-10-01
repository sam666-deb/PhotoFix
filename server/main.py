"""FastAPI app: serves the web UI and the enhancement API.

Run:  .venv/bin/uvicorn server.main:app --reload   ->  http://127.0.0.1:8000
"""

import base64
import os
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from photofix import enhance
from photofix.imageio import BROWSER_FORMATS, encode_jpeg, image_format, load_image

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
CHECKPOINT = Path(os.environ.get("PHOTOFIX_CHECKPOINT", ROOT / "checkpoints" / "analyzer.pt"))


def _load_net():
    """Use the trained analyzer when a checkpoint exists; PHOTOFIX_ANALYZER=classical forces the baseline."""
    if os.environ.get("PHOTOFIX_ANALYZER") == "classical" or not CHECKPOINT.exists():
        return None
    from photofix.net import DLAnalyzer

    return DLAnalyzer(CHECKPOINT)


net = _load_net()
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
    return {"status": "ok", "analyzer": net.name if net else "classical"}


# Sync handlers run in FastAPI's threadpool, so heavy image work doesn't block the event loop.
@app.post("/api/enhance")
def enhance_image(file: UploadFile = File(...)):
    rgb, icc, browser_can_decode = _read_upload(file)
    start = time.perf_counter()
    out, analysis, params = enhance(rgb, net=net)
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    response = {
        "analysis": analysis.to_dict(),
        "params": params.to_dict(),
        "steps": params.describe(),
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "elapsed_ms": elapsed_ms,
        "image": _data_url(encode_jpeg(out, icc=icc)),
    }
    if not browser_can_decode:  # e.g. HEIC in Chrome: send the original too, for the before view
        response["original"] = _data_url(encode_jpeg(rgb, icc=icc))
    return response


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
