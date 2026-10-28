# syntax=docker/dockerfile:1
# Fake YouTube/LLM/Notion for failure injection (profile `fakes`; ARCHITECTURE §14, O-1).
# The app itself is written in M3; this image builds now so the profile is wired end to end.
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
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

COPY common ./common
COPY fakes ./fakes

RUN useradd --system --uid 10001 --no-create-home app
USER app
EXPOSE 8090
CMD ["uvicorn", "fakes.app:app", "--host", "0.0.0.0", "--port", "8090"]
