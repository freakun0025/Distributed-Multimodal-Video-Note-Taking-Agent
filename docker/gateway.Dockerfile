# syntax=docker/dockerfile:1
# Gateway image (also runs the one-shot `migrate` service). ARCHITECTURE §2, §3, §10.
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

COPY alembic.ini ./
COPY migrations ./migrations
COPY common ./common
COPY gateway ./gateway

RUN useradd --system --uid 10001 --no-create-home app
USER app
EXPOSE 8000
CMD ["python", "-m", "gateway"]
