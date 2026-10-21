# CPU image for training/inference anywhere Docker runs.
# GPU: use --gpus all with a CUDA base (see README) or train on a machine with NVIDIA Container Toolkit.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY cxr_classifier.py .

# Mount your dataset at /data (CSV + images). Weights write to /app/best_model.pt
ENTRYPOINT ["python", "cxr_classifier.py"]
CMD ["--help"]
