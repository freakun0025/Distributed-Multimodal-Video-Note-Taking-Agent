# syntax=docker/dockerfile:1
# Worker image: runs any stage of any job (ARCHITECTURE §2, §6.3). ffmpeg for frame extraction.
FROM ghcr.io/astral-sh/uv:0.11.8 AS uv

FROM python:3.12.13-slim-bookworm
COPY --from=uv /uv /bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

COPY common ./common
COPY worker ./worker

# /data is the shared artefact volume; a fresh named volume inherits this ownership.
RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir -p /data/jobs && chown -R app:app /data
USER app
CMD ["python", "-m", "worker.main"]
