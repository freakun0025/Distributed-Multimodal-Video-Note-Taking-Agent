# Developer commands (CLAUDE.md §7). Run from a WSL/Linux shell.
COMPOSE ?= docker compose
UV ?= uv
# Separate virtualenv for Linux so it never clobbers the Windows .venv in the same folder.
export UV_PROJECT_ENVIRONMENT ?= .venv-linux

s ?=
n ?= 4
api ?= llm
stage ?= EXTRACT_FRAMES

.DEFAULT_GOAL := help
.PHONY: help up up-fakes up-gpu up-obs down logs migrate lint test itest scale \
        fake-429 kill-worker burst fernet-key

help: ## List commands
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

up: ## Core stack: postgres, rabbitmq, redis, migrate, gateway, 2 workers
	$(COMPOSE) up -d --build

up-fakes: ## Core + fake APIs (FAKE_APIS=true)
	FAKE_APIS=true $(COMPOSE) --profile fakes up -d --build

up-gpu: ## Core + GPU speech-to-text (STT_ENABLED=true)
	STT_ENABLED=true $(COMPOSE) --profile gpu up -d --build

up-obs: ## Prometheus + Grafana
	$(COMPOSE) --profile observability up -d

down: ## Stop everything (all profiles); volumes are kept
	$(COMPOSE) --profile "*" down

logs: ## Follow logs, e.g. make logs s=worker
	$(COMPOSE) logs -f $(s)

migrate: ## alembic upgrade head
	$(COMPOSE) run --rm migrate

lint: ## ruff + mypy (--strict on common/)
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run mypy --strict common
	$(UV) run mypy gateway worker stt fakes chaos eval migrations tests

test: ## Unit tests
	$(UV) run pytest tests/unit

itest: ## Integration tests against the fakes stack (make up-fakes first)
	$(UV) run pytest -m integration tests/integration

scale: ## Run n workers, e.g. make scale n=4
	$(COMPOSE) up -d --no-recreate --scale worker=$(n)

fernet-key: ## Print a new TOKEN_ENCRYPTION_KEY
	@$(UV) run python -m common.crypto

fake-429: ## Switch a fake API into 429 mode, e.g. make fake-429 api=llm  (M3)
	@echo "fake-429 arrives with the fakes service in M3 (docs/ROADMAP.md)"; exit 1

kill-worker: ## SIGKILL the worker running a stage, e.g. make kill-worker stage=EXTRACT_FRAMES  (M3)
	@echo "kill-worker arrives with chaos/ in M3 (docs/ROADMAP.md)"; exit 1

burst: ## Submit 6 jobs in a row  (M3)
	@echo "burst arrives with chaos/ in M3 (docs/ROADMAP.md)"; exit 1
