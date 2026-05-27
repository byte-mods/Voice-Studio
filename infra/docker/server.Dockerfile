# Open Audio Studio — backend (CPU). For GPU, use server.cuda.Dockerfile.
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY packages/core /app/packages/core
COPY packages/sdk  /app/packages/sdk
COPY apps/server   /app/apps/server

RUN pip install --no-cache-dir \
      -e packages/core \
      -e packages/sdk \
      -e apps/server

ENV OAS_DATA_DIR=/data OAS_SERVER_HOST=0.0.0.0 OAS_SERVER_PORT=8000
EXPOSE 8000
VOLUME ["/data"]

CMD ["uvicorn", "oas_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
