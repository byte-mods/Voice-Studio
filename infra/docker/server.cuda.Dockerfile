# Open Audio Studio — backend (CUDA 12.4). Run with `--gpus all`.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      build-essential git ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app
COPY packages/core /app/packages/core
COPY packages/sdk  /app/packages/sdk
COPY apps/server   /app/apps/server

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      -e packages/core \
      -e packages/sdk \
      -e apps/server

ENV OAS_DATA_DIR=/data OAS_SERVER_HOST=0.0.0.0 OAS_SERVER_PORT=8000 OAS_ENABLE_GPU=1
EXPOSE 8000
VOLUME ["/data"]

CMD ["uvicorn", "oas_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
