# RTX 5090 (Blackwell sm_120) needs CUDA >= 12.8 and torch cu126/cu128.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ffmpeg libsndfile1 curl wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .

# torch first (cu128 wheel bundles its own CUDA libs), then the rest
RUN pip3 install --no-cache-dir --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/cu128 \
 && pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Bake the ByteDance checkpoint into the image (zenodo download, ~first-use otherwise)
RUN python3 -c "from piano_transcription_inference import PianoTranscription; PianoTranscription(device='cpu')"

COPY app.py .
VOLUME ["/data"]
EXPOSE 7777
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7777"]
