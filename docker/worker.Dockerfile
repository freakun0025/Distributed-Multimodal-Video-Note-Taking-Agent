# syntax=docker/dockerfile:1
# Worker image: runs any stage of any job (ARCHITECTURE §2, §6.3). ffmpeg for frame extraction.
FROM ghcr.io/astral-sh/uv:0.11.8 AS uv

FROM denoland/deno:bin-2.9.8 AS deno

FROM python:3.12.13-slim-bookworm
COPY --from=uv /uv /bin/uv
# yt-dlp solves YouTube's JavaScript challenges in Deno (D32); found on PATH.
COPY --from=deno /deno /usr/local/bin/deno
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
# The app user has no home directory: Deno's and yt-dlp's caches go to /tmp. Set only now: set
# before `uv sync`, XDG_CACHE_HOME sent uv's cache to a root-owned /tmp/cache (138 MB in the
# image, the cache mount unused) that the app user then could not write, so yt-dlp lost its
# cache of solved YouTube player challenges and solved them again on every call.
ENV DENO_DIR=/tmp/deno \
    XDG_CACHE_HOME=/tmp/cache
USER app
CMD ["python", "-m", "worker.main"]
