# syntax=docker/dockerfile:1
# Fake YouTube/LLM/Notion for failure injection (profile `fakes`; ARCHITECTURE §14, O-1).
# Happy path since M2 (D29); failure switches arrive in M3. ffmpeg only generates the fake video.
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

# The fake lecture video: 600 s like the fake metadata, a new colour every 20 s (scene cuts
# for CP7b), moving test pattern so frames are never blank. Built before the code is copied
# so code changes don't regenerate it.
RUN mkdir -p fakes && ffmpeg -nostdin -hide_banner -loglevel error \
    -f lavfi -i "testsrc2=size=640x360:rate=2:duration=600,hue=h='floor(t/20)*47'" \
    -c:v libx264 -preset veryfast -crf 30 -pix_fmt yuv420p -g 10 -y fakes/video.mp4

COPY common ./common
COPY fakes ./fakes

RUN useradd --system --uid 10001 --no-create-home app
USER app
EXPOSE 8090
CMD ["uvicorn", "fakes.app:app", "--host", "0.0.0.0", "--port", "8090"]
