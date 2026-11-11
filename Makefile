# Developer commands (CLAUDE.md §7). Run from a WSL/Linux shell.
COMPOSE ?= docker compose
UV ?= uv
# Separate virtualenv for Linux so it never clobbers the Windows .venv in the same folder.
export UV_PROJECT_ENVIRONMENT ?= .venv-linux

s ?=
n ?= 4
api ?= llm
count ?= 1
status ?= 429
retry_after ?=
mode ?= partial
stage ?= EXTRACT_FRAMES
args ?=
# The fakes service's control endpoint, published on the host by docker-compose.yml.
FAKES ?= http://127.0.0.1:8090
FAKES_PUT = curl -fsS -X PUT -H 'content-type: application/json'

.DEFAULT_GOAL := help
.PHONY: help up up-fakes up-gpu up-obs down logs migrate lint test test-ext itest scale \
        fake-429 fake-frames fake-unavailable fake-reset fakes kill-worker burst fernet-key

help: ## List commands
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-17s %s\n", $$1, $$2}'

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

test-ext: ## Extension unit tests (Node's built-in runner, no npm); skipped without node
	@if command -v node >/dev/null 2>&1; then node --test tests/extension/*.test.mjs; \
	else echo "node not found: skipping extension tests (install Node 18+ to run them)"; fi

itest: ## Integration tests against the fakes stack (make up-fakes first); extra pytest args: args="-k dlq"
	$(UV) run pytest -m integration tests/integration $(args)

scale: ## Run n workers, e.g. make scale n=4
	$(COMPOSE) up -d --no-recreate --scale worker=$(n)

fernet-key: ## Print a new TOKEN_ENCRYPTION_KEY
	@$(UV) run python -m common.crypto

# --- Spec §6 failures on demand: the fakes stack (make up-fakes) ---------------------------------

fake-429: ## Next count calls to a fake API answer 429: make fake-429 api=youtube|llm|notion [count=2] [retry_after=20] [status=503]
	@body='{"api":"$(api)","status":$(status),"count":$(count)'; \
	if [ -n "$(retry_after)" ]; then body="$$body,\"retry_after\":$(retry_after)"; fi; \
	$(FAKES_PUT) $(FAKES)/_control/faults -d "$$body}"; echo

fake-frames: ## Break the frames download: make fake-frames mode=partial (cut at 5:00) | none (black) | ok
	@$(FAKES_PUT) $(FAKES)/_control/youtube -d '{"frames":"$(mode)"}'; echo

fake-unavailable: ## Every video is private: retries, then DEAD_LETTERED (make fake-reset ends it)
	@$(FAKES_PUT) $(FAKES)/_control/youtube -d '{"unavailable":true}'; echo

fake-reset: ## Every fake API works again; empties the fakes' call log
	@curl -fsS -X POST $(FAKES)/_control/reset; echo

fakes: ## Show the fakes' failure switches
	@curl -fsS $(FAKES)/_control/switches; echo

kill-worker: ## SIGKILL the worker running a stage, then follow the job: make kill-worker stage=EXTRACT_FRAMES
	$(UV) run python -m chaos.kill_worker --stage $(stage)

burst: ## Submit 6 jobs in a row  (CP11)
	@echo "burst arrives in CP11 (docs/ROADMAP.md)"; exit 1
