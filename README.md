# Distributed Multimodal Video Note-Taking Agent

Click **Take Notes** on a YouTube video (up to 20 minutes) and get a Notion page of study notes:
a summary, timestamped sections, self-test questions, and real frames from the video, each with
a caption saying what it shows and why it matters. A Chrome extension sends the job to a gateway;
a pool of workers does the work through a durable queue, within the rate limits of YouTube,
Gemini and Notion.

```
 Chrome extension ──POST /jobs──▶ gateway (FastAPI) ──▶ RabbitMQ ──▶ worker × N (≥ 2)
   popup ◀──GET /jobs/{id}──────┘     │                 jobs.work     TRANSCRIBE → SUMMARIZE →
                                       ▼                 retry 10/30/90 s   EXTRACT_FRAMES → PUBLISH
                                    Postgres (job state)  jobs.dlq        │ YouTube · Gemini · Notion
                                    Redis (shared rate limits) ◀──────────┘
```

Each box is its own container (Docker Compose), so any one can be killed or scaled on its own.
The gateway only validates and enqueues; all pipeline work happens in the workers.

## Contents

1. [Quick start](#quick-start) — run it from a fresh clone
2. [Using it](#using-it) — the extension, the base URL, where pages go
3. [REST API](#rest-api)
4. [Configuration](#configuration) — every environment variable
5. [Demo script for grading](#demo-script-for-grading) — every Spec §6 failure, the burst, a kill
6. [Design choices](#design-choices) — and why
7. [Development](#development) — tests and the `make` commands

## Quick start

**You need:** Docker with Compose v2 (Docker Desktop on Windows/macOS), and Chrome or another
Chromium browser (Brave, Edge). For the `make` shortcuts and the test/demo scripts also GNU make
and [uv](https://docs.astral.sh/uv/) (on Windows: from a WSL shell). Plain `docker compose`
commands are given for everything needed to run the stack.

### 1. Keys and IDs (once)

| What | Where | Goes into `.env` |
| --- | --- | --- |
| Gemini API key | [Google AI Studio](https://aistudio.google.com/apikey) → Create API key (the free tier is enough) | `LLM_API_KEY` |
| Notion public connection | [notion.so/profile/integrations](https://www.notion.so/profile/integrations) → New integration → type **Public**. Redirect URI: `https://hiimaheehbfpmgnjjpeoejngcmhjmjfi.chromiumapp.org/notion` (the extension's ID is pinned by `extension/manifest.json`, so this is the same on every machine). Capabilities: read, update and insert content. Copy the OAuth client ID and secret | `NOTION_CLIENT_ID`, `NOTION_CLIENT_SECRET` |
| Default destination page | Create a Notion page, e.g. "Video Notes". Its ID is the 32 hex characters at the end of its link | `NOTION_DEFAULT_PARENT_ID` |
| Token encryption key | `make fernet-key`, or `python3 -c "import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"` | `TOKEN_ENCRYPTION_KEY` |

### 2. Configure and start

```bash
git clone <this repository> && cd <its folder>
cp .env.example .env          # then fill the four rows above; every other value has a default
docker compose up -d --build  # or: make up
curl localhost:8000/healthz   # {"gateway":"ok","postgres":"ok","rabbitmq":"ok","redis":"ok"} once up
```

The first build takes a few minutes. `docker compose up` also runs the database migrations (a
one-shot `migrate` service) and starts **2 workers**. Scale with
`docker compose up -d --scale worker=4` (`make scale n=4`).

### 3. Load the extension

1. Open `chrome://extensions` (Brave: `brave://extensions`), turn on **Developer mode**, click
   **Load unpacked** and choose the `extension/` folder. There is no build step.
2. Open any YouTube video and click **Take Notes** in the player's toolbar (next to the settings
   gear). The first time, Notion's sign-in window opens: choose **Select pages**, tick the
   destination page from step 1, then **Allow access**. The extension keeps the token, so later
   clicks go straight through. (Or connect first: right-click the extension icon → **Options** →
   **Connect Notion**.)
3. Click the extension's toolbar icon to follow the job: Queued → Transcribing → Summarizing →
   Extracting Frames → Publishing → Done, then **View in Notion**.

**No keys at hand?** `make up-fakes` (or `FAKE_APIS=true docker compose --profile fakes up -d
--build`) runs the same stack against built-in fake YouTube, Gemini and Notion services. Then
only two values are needed: `TOKEN_ENCRYPTION_KEY`, and `NOTION_DEFAULT_PARENT_ID` set to any 32
hex characters (e.g. `0123456789abcdef0123456789abcdef`; the fake Notion accepts any page).
Everything works, including every failure demo; the "Notion page" then lives in the fake
service.

## Using it

- **Base URL:** the gateway listens on `http://localhost:8000` (`GATEWAY_BASE_URL`). The
  extension uses that by default; another address can be set on its Options page ("Notes server
  URL", with a **Test connection** button).
- **Where pages go (default Notion destination):** every job publishes a new child page under the
  page in `NOTION_DEFAULT_PARENT_ID`, unless the extension's Options page names another parent
  page (a link or an ID), which is then sent as `notion_parent_id` with each job. The parent must
  be one of the pages shared with the integration when you connected Notion; if it isn't, the
  job fails with "The Notion page to publish under wasn't found or isn't shared with the
  integration." Page title: the video title plus a job tag, e.g. `Raft explained · #1a2b3c4d`.
- **The token:** the extension sends the Notion token with every job; the gateway never stores
  it. It travels encrypted (Fernet, `TOKEN_ENCRYPTION_KEY`) inside the queue messages and is never
  logged.
- **Limits:** videos up to 20 minutes (`MAX_VIDEO_SECONDS`). A longer one fails at once with
  "This video is longer than 20 minutes" and nothing is processed. Live streams aren't supported.
- **Results:** a COMPLETED page has the video, a summary, timestamped sections (each heading links
  to that moment), frames placed in their sections with captions, and 5–8 "Test yourself"
  questions with hidden answers. Frames are real frames grabbed with ffmpeg from the video
  itself; a frame that fails is left out (never replaced) and the job ends
  `COMPLETED_WITH_WARNINGS`, listing what is missing.

## REST API

The extension uses these; you never need to call them by hand. `curl` examples:

```bash
# Submit (202 Accepted, Location: /jobs/{job_id})
curl -i -X POST localhost:8000/jobs -H 'content-type: application/json' \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ","notion_token":"<token>"}'
# {"job_id":"…","status":"QUEUED","video_id":"dQw4w9WgXcQ"}

curl localhost:8000/jobs/<job_id>          # status, progress, title, duration, warnings, error
curl localhost:8000/jobs/<job_id>/results  # 409 until the job is finished, then the page URL
curl localhost:8000/healthz                # postgres / rabbitmq / redis reachable (503 if not)
```

| Endpoint | Answers |
| --- | --- |
| `POST /jobs` `{youtube_url, notion_token, notion_parent_id?}` | `202` `{job_id, status: "QUEUED", video_id}` with `Location`; `400` `INVALID_URL` / `MISSING_NOTION_TOKEN`, `422` `VALIDATION_ERROR`, `503` `BROKER_UNAVAILABLE`. Errors are always `{"error": {"code", "message"}}`. An `Idempotency-Key` header makes a resubmission return the same job |
| `GET /jobs/{job_id}` | `200` `{job_id, status, progress, video_title, duration_sec, warnings, error, result, waiting}`; `404` for an unknown id. Read-only |
| `GET /jobs/{job_id}/results` | `409` `JOB_NOT_TERMINAL` until the job ends; then `200` `{status, result: {notion_page_url, frames_requested, frames_published}, warnings: [{code, timestamp_sec, message}], error}` |
| `GET /healthz` | `200` when Postgres, RabbitMQ and Redis answer, else `503` with the one that doesn't |
| `GET /auth/notion/config`, `POST /auth/notion/exchange`, `POST /auth/notion/refresh` | The extension's Notion OAuth: the gateway holds the client secret and swaps the code for a token, which it hands back without storing |

Statuses: `QUEUED`, `TRANSCRIBING`, `SUMMARIZING`, `EXTRACTING_FRAMES`, `PUBLISHING`, then
`COMPLETED`, `COMPLETED_WITH_WARNINGS`, `FAILED` (an error retrying cannot fix) or
`DEAD_LETTERED` (the retry budget ran out; the message waits in `jobs.dlq`). While a retry is
scheduled, `waiting` says why and until when (the popup shows "Waiting — …, retrying in 10 s").

## Configuration

All settings live in `.env` (copied from `.env.example`, which documents each one) and are read
by `common/config.py`. Only the first group must be filled in; everything else has a working
default.

**Required**

| Variable | Meaning |
| --- | --- |
| `LLM_API_KEY` | Gemini API key (not needed with `make up-fakes`) |
| `NOTION_CLIENT_ID`, `NOTION_CLIENT_SECRET` | The Notion public connection's OAuth credentials (the secret stays on the gateway) |
| `NOTION_DEFAULT_PARENT_ID` | The page new notes go under unless the extension names another |
| `TOKEN_ENCRYPTION_KEY` | Fernet key that encrypts the Notion token inside queue messages (gateway and workers share it) |

**Connections**

| Variable | Default | Meaning |
| --- | --- | --- |
| `GATEWAY_BASE_URL` | `http://localhost:8000` | The base URL the gateway is reached at |
| `RABBITMQ_URL` | `amqp://vna:vna@rabbitmq:5672/` | Broker connection string |
| `DATABASE_URL` | `postgresql+asyncpg://vna:vna@postgres:5432/vna` | Job state |
| `REDIS_URL` | `redis://redis:6379/0` | Shared rate limits and model health |
| `NOTION_REDIRECT_URI` | `https://hiimaheehbfpmgnjjpeoejngcmhjmjfi.chromiumapp.org/notion` | Must match the redirect URI registered in Notion |
| `NOTION_API_URL`, `NOTION_API_VERSION`, `NOTION_TIMEOUT_S` | `https://api.notion.com`, `2026-03-11`, `30` | Notion API |

**LLM (model and rate limit)**

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini` | `gemini`, `ollama` or `fake` |
| `LLM_CHAIN` | `gemini-3.6-flash:low:90,gemini-3.8-flash:low:120,gemini-3.5-flash-lite:medium:60` | **Model names**, in order, as `model:thinking:timeout_s`. A call goes to the first model no worker has benched (daily quota spent, or repeatedly overloaded) |
| `LLM_RPM` | `4` | **The configured LLM rate limit**: calls per minute for all workers together (one Redis token bucket). Keep `LLM_RPM + LLM_BURST` at or below the lowest RPM of the models in `LLM_CHAIN` (Gemini Flash free tier: 5) |
| `LLM_BURST` | `1` | Bucket capacity: calls allowed back to back |
| `LLM_TPM` | `200000` | Tokens per minute budget |
| `LIMITER_MAX_WAIT_S` | `20` | A wait for the limiter longer than this goes to a delay queue instead of holding the worker |
| `LLM_STEP_DOWN_FROM_ATTEMPT`, `LLM_MODEL_STRIKES`, `LLM_STRIKE_WINDOW_S`, `LLM_MODEL_COOLDOWN_S`, `LLM_QUOTA_RESET_TZ` | `3`, `2`, `600`, `600`, `America/Los_Angeles` | When the chain moves to the next model, and for how long a model is benched |
| `LLM_CHUNK_CHARS`, `LLM_MAX_REPAIRS`, `LLM_IMAGE_RESOLUTION` | auto, `1`, `medium` | Transcript chunk size (long transcripts are split and merged in order), re-prompts for invalid JSON, image detail |
| `OLLAMA_URL`, `OLLAMA_MODEL` | `http://ollama:11434`, `qwen3:8b` | Only with `LLM_PROVIDER=ollama` |

**Other rate limits**

| Variable | Default | Meaning |
| --- | --- | --- |
| `NOTION_RPS`, `NOTION_BURST` | `2.5`, `3` | Notion calls per second (Notion allows ~3) |
| `YOUTUBE_RPM`, `YOUTUBE_BURST`, `YOUTUBE_TIMEOUT_S` | `30`, `2`, `30` | YouTube calls (self-imposed) |
| `YOUTUBE_DOWNLOAD_TIMEOUT_S` | `300` | Longest a video download may take (YouTube throttles some streams); the next attempt resumes it, and the last one publishes without frames. Below `STAGE_TIMEOUT_S` |
| `REDIS_TIMEOUT_S` | `2` | A hung Redis makes the limiter fail closed (retry later), never open |

**Retries and backoff**

| Variable | Default | Meaning |
| --- | --- | --- |
| `RETRY_MAX_ATTEMPTS` | `4` | Attempts per stage: the first try + 3 retries; then the job is `DEAD_LETTERED` and the message goes to `jobs.dlq` |
| `RETRY_DELAYS` | `10,30,90` | Backoff in seconds between attempts (a longer `Retry-After` from the API wins). One delay queue per value. Quick DLQ demo: `2,4,8` |
| `STAGE_TIMEOUT_S` | `900` | Watchdog: a stage still running after this is stopped and retried, so a hung worker can't hold a job |
| `OUTBOX_TIMEOUT_S`, `OUTBOX_SWEEP_INTERVAL_S` | `300`, `30` | The gateway re-publishes a job whose first message never reached the broker |
| `BROKER_PUBLISH_TIMEOUT_S`, `HEALTHCHECK_TIMEOUT_S` | `5`, `2` | Publisher-confirm wait for `POST /jobs`; per-dependency timeout in `/healthz` |

**Pipeline**

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_VIDEO_SECONDS` | `1200` | Constraint A: longer videos fail with `VIDEO_TOO_LONG` before any work |
| `MIN_FRAMES`, `MAX_FRAMES`, `MIN_FRAME_EVERY_S`, `FRAME_MIN_GAP_S` | `3`, `8`, `180`, `10` | How many frames the notes may pick, and how spread out |
| `NOTES_SECTION_EVERY_S`, `NOTES_BULLETS_PER_MINUTE`, `NOTES_FLOOR_MAX_SECTIONS`, `NOTES_FLOOR_MAX_BULLETS`, `NOTES_QUESTIONS_MIN`, `NOTES_QUESTIONS_MAX`, `NOTES_LANGUAGE`, `TRANSCRIPT_LANGS` | see `.env.example` | Depth and language of the notes; caption languages tried, in order |
| `FRAMES_VIDEO_HEIGHT`, `FRAME_BURST`, `FRAME_BURST_CUT_SCORE`, `FRAME_JPEG_QUALITY`, `FRAME_BLANK_LUMA`, `FRAME_FLAT_RANGE`, `FFMPEG_TIMEOUT_S` | `720`, `3`, `1.0`, `3`, `24`, `24`, `60` | Frame extraction: download height, grabs per pick (sharpest kept), blank-frame rule |
| `CANDIDATES_VIDEO_HEIGHT`, `CANDIDATE_MAX`, `CANDIDATE_MIN_GAP_S`, `CANDIDATE_SETTLE_S`, `CANDIDATE_JPEG_QUALITY`, `CANDIDATE_SNAP_S`, `SCENE_THRESHOLD`, `SCENE_DETECT_TIMEOUT_S` | `480`, `36`, `15`, `1.5`, `5`, `2`, `10`, `180` | Candidate frames shown to the LLM so it picks moments it has seen |
| `STT_ENABLED`, `STT_URL`, `WHISPER_MODEL`, `WHISPER_DEVICE` | `false`, … | Optional speech-to-text service (profile `gpu`), not needed |
| `DATA_DIR`, `FAKE_APIS`, `FAKES_URL`, `LOG_LEVEL` | `/data`, `false`, `http://fakes:8090`, `INFO` | Shared volume for artefacts; fake APIs (`make up-fakes` sets `FAKE_APIS=true`) |

## Demo script for grading

From a WSL/Linux/macOS shell in the repository. `make help` lists every command.

**0. Sanity check.** `curl localhost:8000/healthz`, `docker compose ps` (gateway, postgres,
rabbitmq, redis, 2 workers). RabbitMQ's UI: <http://localhost:15672> (user and password `vna`)
shows `jobs.work`, the delay queues `jobs.retry.10s/30s/90s` and `jobs.dlq`.

**1. A real job.** Take Notes on a lecture of 15–20 minutes. Its stage timings are in the worker
logs (`make logs s=worker`).

**2. The four Spec §6 failures, on demand.** The 429 and frame switches live in the fake APIs, so
run `make up-fakes` first (`make up` brings the real APIs back; the Notion page is then in the
fake service, and the popup still shows every status).

| Failure | Command, then click Take Notes | What you see |
| --- | --- | --- |
| HTTP 429 from an external API | `make fake-429 api=youtube` (or `llm`, `notion`; `count=2` for two in a row, `retry_after=20`) | The popup shows "Waiting — YouTube is rate-limiting us, retrying in 10 s" (then 30 s, 90 s), never an error; the job completes |
| Worker crash | `make kill-worker stage=EXTRACT_FRAMES` (or `TRANSCRIBE`, `SUMMARIZE`, `PUBLISH`), started before the click | The worker running that stage is SIGKILLed (in `PUBLISH`, once the Notion page exists); the script prints which worker the broker gave the message to, starts the killed one again, and prints the recovery time; the job completes with exactly one Notion page |
| Some / all frames fail | `make fake-frames mode=partial` (or `none`) | `COMPLETED_WITH_WARNINGS`; the popup lists the missing frames; the page keeps the others (or is text-only) |
| Dead-letter queue | `make fake-unavailable` (on the real stack: a private video, or without one `curl -X POST localhost:8000/jobs -H 'Content-Type: application/json' -d '{"youtube_url":"https://www.youtube.com/watch?v=vnaDLQdemo1","notion_token":"unused"}'`, an ID no video can have; the job stops before Notion, so the token is never used) | Waiting through every retry, then `DEAD_LETTERED` about 2 minutes later (4 attempts, 10 + 30 + 90 s apart); the message is in `jobs.dlq` |

`make fake-reset` makes every fake API work again; `make fakes` shows the switches.
Without `make`: `docker kill vna-worker-1` kills a worker by hand (`docker start vna-worker-1`
brings it back), and `docker compose up -d --scale worker=1` / `=3` stops or adds workers.

**3. Burst (Spec §9 queue resilience).** Click Take Notes on several different videos in a row,
then run `make burst-report since=10`: each job's status and time, when the queue drained, and
every answer the LLM gave these jobs by HTTP status, read from the worker logs. The target is
`429s from the LLM: 0`. `make burst` does the same with 6 jobs on the fakes stack, where the fake
Gemini enforces the free tier's 5 calls per minute. Gemini's free tier also caps requests per
day (20 per Flash model), so on the real stack keep enough of the day's quota for the burst.

**4. Longer than 20 minutes.** Take Notes on a 25-minute video → `FAILED`, "This video is longer
than 20 minutes" (on fakes: `curl -X PUT localhost:8090/_control/youtube -H 'content-type:
application/json' -d '{"duration":1500}'`).

## Design choices

Spec §10 leaves these open; here is what we chose and why.

- **Broker: RabbitMQ (quorum queues).** Native acknowledgements and redelivery on a dead
  consumer; per-message TTL plus dead-lettering gives retry delay queues and the DLQ without any
  plugin or sleeping worker; the management UI shows queue depths live. Redis Streams would need
  our own pending-entry reclaiming; Kafka has no per-message ack.
- **One message per stage, acked only after the stage's work is durable.** A worker saves the
  stage's output (files written atomically, a checkpoint row in Postgres), publishes the next
  stage's message with a publisher confirm, and only then acks. A crash anywhere before the ack
  makes RabbitMQ hand the message to another worker, which repeats at most that one stage. Every
  stage is idempotent: a redelivered stage that already checkpointed only moves the job on.
- **No duplicate Notion pages.** The page is found by a job tag in its title through the parent
  page's children list (strongly consistent, unlike Notion search); its ID is saved on the job
  the moment it is created; blocks are appended from where the page left off; and a per-job
  Postgres advisory lock means two deliveries of one job never publish at the same time.
- **Job state: Postgres, one state machine.** Statuses change only through one function that
  locks the row and moves it from an allowed status, logging every transition. Transactions make
  "checkpoint + final status" atomic.
- **Global LLM rate limit: a token bucket in Redis**, updated atomically by a Lua script and
  shared by every worker, set below Gemini's real per-minute limit, so adding workers never adds
  LLM calls. A wait longer than 20 s goes to a delay queue instead of holding the worker. Each
  external API has its own bucket, and every call takes a token first. If Redis is down, the
  limiter fails closed (retry later), never open.
- **429s and other temporary errors: delay queues** (10 s, 30 s, 90 s, honouring `Retry-After`),
  with the job shown as waiting. Retries are bounded (`RETRY_MAX_ATTEMPTS`), then the DLQ. A
  private or removed video is treated as retryable, so it is the natural DLQ demo. A stage
  watchdog turns a hung stage into a normal retry.
- **Worker topology: one worker type that runs all four stages.** Per-stage messages already
  spread a job across workers and make retries fine-grained; one type keeps scaling to
  `--scale worker=N`.
- **LLM: Gemini, structured JSON output, a model chain.** The notes come back as JSON that matches
  a schema. The model sees candidate frames from the video next to the transcript, so it picks
  moments it has actually seen and captions them. Long transcripts are split into chunks and
  merged in order. If a model's daily quota is spent or it keeps failing, every worker moves to
  the next model in `LLM_CHAIN`.
- **Transcript:** YouTube captions (youtube-transcript-api / yt-dlp), which also give the true
  duration, so Constraint A is enforced before any LLM call.
- **Frames:** the video is downloaded once and ffmpeg grabs frames locally (stream URLs expire);
  three grabs within the second, the sharpest kept, blank frames rejected. Images are only ever
  real frames (Constraint B); a failed frame is omitted, never substituted.
- **Extension:** plain JavaScript, Manifest V3, no build step. The background script does the
  one-time Notion OAuth (the gateway swaps the code, so the client secret never ships in the
  extension), submits jobs and polls; the popup renders statuses, waiting reasons, warnings and
  errors as sentences, never raw JSON.

## Development

```bash
make lint       # ruff + mypy (--strict on common/)
make test       # unit tests (no network)
make test-ext   # extension tests (Node 18+, no npm)
make up-fakes && make itest   # integration tests against the running stack with fake APIs
```

The integration tests drive the real gateway, broker, Postgres, Redis and workers: every stage,
a 429 from each API, retries and backoff, the DLQ, frame degradation, Constraint A, killing a
worker in each stage, the publish lock, and a burst of 6 jobs with zero LLM 429s. `make down`
stops everything (data volumes are kept).
