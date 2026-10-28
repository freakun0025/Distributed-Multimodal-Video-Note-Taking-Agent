# syntax=docker/dockerfile:1
# GPU speech-to-text service, one process shared by all workers (profile `gpu`; ARCHITECTURE §2, O-2).
# The service is written in M4; base image, CUDA/cuDNN and faster-whisper are verified then.
FROM ghcr.io/astral-sh/uv:0.11.8 AS uv

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
COPY --from=uv /uv /bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 && uv sync --locked --no-dev --group stt --no-install-project

COPY common ./common
COPY stt ./stt

RUN useradd --system --uid 10001 --create-home app
USER app
EXPOSE 9000
CMD ["uvicorn", "stt.app:app", "--host", "0.0.0.0", "--port", "9000"]
