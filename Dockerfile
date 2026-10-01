# PhotoFix public demo. CPU-only; runs on Google Cloud Run (or a Hugging Face Docker Space).
# Trained weights are pulled from the Hugging Face model repo at build time (they aren't in git).
FROM python:3.14-slim

ARG MODEL_REPO=Samdany/photofix-models

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    PHOTOFIX_PUBLIC=1 \
    PHOTOFIX_MAX_SIDE=2048 \
    PHOTOFIX_CONCURRENCY=2

# Non-root user (uid 1000 also matches what Hugging Face Spaces expects).
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch first (~200 MB instead of ~2 GB of CUDA libraries), then the rest.
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt huggingface_hub

COPY --chown=user photofix ./photofix
COPY --chown=user server ./server
COPY --chown=user web ./web
COPY --chown=user models ./models

RUN hf download "$MODEL_REPO" analyzer.pt lut.pt restorer.pt --local-dir checkpoints \
 && chown -R user checkpoints

USER user
# Cloud Run provides $PORT (8080); Hugging Face Spaces expects 7860.
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
