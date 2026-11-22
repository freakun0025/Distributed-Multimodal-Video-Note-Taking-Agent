# Distributed Multimodal Video Note-Taking Agent

A distributed system that turns a YouTube video of up to 20 minutes into a Notion page of study
notes: a summary, timestamped sections, self-test questions, and real frames from the video with
captions explaining what each one shows and why it was chosen. The user clicks **Take Notes** on
the YouTube player; a browser extension hands the job to an API gateway, and a pool of workers
processes it through a durable message queue while respecting the rate limits of YouTube, Gemini
and Notion.

Course project for **Cloud Computing**.

| Contributor | Roll number |
| --- | --- |
| Kunal Mittal | IMT2023533 |
| Ishaan Sodhi | IMT2023090 |

## Contents

1. [Overview](#overview)
2. [System architecture](#system-architecture)
3. [How a job is processed](#how-a-job-is-processed)
4. [Specification compliance](#specification-compliance)
5. [Beyond the specification](#beyond-the-specification)
6. [Design decisions](#design-decisions)
7. [Measured results](#measured-results)
8. [Getting started](#getting-started)
9. [Configuration](#configuration)
10. [REST API](#rest-api)
11. [Testing and tooling](#testing-and-tooling)
12. [Repository layout](#repository-layout)
13. [Known limitations](#known-limitations)

## Overview

What the user does:

1. Opens a YouTube video and clicks **Take Notes** in the player's toolbar. The first time, the
   extension runs Notion's OAuth flow and keeps the token, so later clicks go straight through.
2. Opens the extension's popup to follow the job through Queued, Transcribing, Summarizing,
   Extracting Frames, Publishing and Done.
3. Clicks **View in Notion** when the job finishes.

What the Notion page contains:

- the embedded video and a short summary;
- one section per topic, in the video's order, each heading linking to the moment the topic
  starts;
- frames taken from the video with ffmpeg, placed in the section they illustrate, each with a
  caption and the reason it was picked;
- five to eight "Test yourself" questions with the answers in collapsed toggles.

If some frames cannot be extracted, the page is still published without them and the job ends as
`COMPLETED_WITH_WARNINGS`, listing what is missing. If no frame can be extracted, the page is
text-only.

## System architecture

Every component runs in its own container under Docker Compose, so any one of them can be
stopped, killed or scaled without touching the others.

```mermaid
flowchart LR
    subgraph EXT["Browser extension (Manifest V3)"]
        CS["Content script<br/>Take Notes button"]
        BG["Background service worker<br/>Notion OAuth, job submission, polling"]
        PU["Popup<br/>job progress"]
    end

    subgraph STACK["Docker Compose stack"]
        GW["API gateway<br/>FastAPI"]
        MQ[("RabbitMQ<br/>quorum queues")]
        W1["Worker 1"]
        W2["Worker 2 ... N"]
        PG[("PostgreSQL<br/>job state, events, checkpoints")]
        RD[("Redis<br/>shared rate limits, model health")]
        VOL[("Shared volume<br/>transcripts, notes, frames")]
    end

    YT["YouTube"]
    LLM["Gemini API"]
    NO["Notion API"]

    CS --> BG
    BG -- "POST /jobs" --> GW
    PU -- "GET /jobs/id" --> GW
    GW -- "insert job" --> PG
    GW -- "first stage message" --> MQ
    MQ --> W1
    MQ --> W2
    W1 & W2 -- "next stage, retries" --> MQ
    W1 & W2 --> PG
    W1 & W2 -- "token per call" --> RD
    W1 & W2 --> VOL
    W1 & W2 --> YT
    W1 & W2 --> LLM
    W1 & W2 --> NO
```

| Container | Role |
| --- | --- |
| `gateway` | Validates submissions, creates the job, publishes the first stage message, serves job status and results, and swaps Notion OAuth codes for tokens. It does no pipeline work. |
| `worker` (2 by default, `--scale worker=N`) | Consumes stage messages one at a time and runs whichever stage it receives. Any worker can run any stage of any job. |
| `rabbitmq` | Durable quorum queues with manual acknowledgements: the work queue, three delay queues for retries and the dead-letter queue. |
| `postgres` | The job table and its state machine, an event log per job, and a checkpoint per completed stage. |
| `redis` | Token buckets shared by all workers (one per external API) and the shared health of each Gemini model. |
| `migrate` | Runs the database migrations once at start-up. |

### Message flow

Each stage of a job is a separate message. A worker acknowledges a message only after the
stage's output is stored and the next stage's message has been confirmed by the broker.

```mermaid
flowchart LR
    GW["Gateway"] -- "TRANSCRIBE" --> EX(("exchange<br/>vna"))
    EX --> WQ["jobs.work"]
    WQ --> WK["Worker"]
    WK -- "next stage" --> EX
    WK -- "retryable error" --> RQ["jobs.retry.10s / 30s / 90s<br/>message TTL, no consumers"]
    RQ -- "TTL expires" --> EX
    WK -- "retry budget spent" --> DLQ["jobs.dlq"]
```

### Job lifecycle

The statuses and transitions are exactly those of the specification. Only one function in the
code base writes a job's status, and it only allows these transitions.

```mermaid
stateDiagram-v2
    [*] --> QUEUED
    QUEUED --> TRANSCRIBING
    TRANSCRIBING --> SUMMARIZING
    SUMMARIZING --> EXTRACTING_FRAMES
    EXTRACTING_FRAMES --> PUBLISHING
    PUBLISHING --> COMPLETED
    PUBLISHING --> COMPLETED_WITH_WARNINGS
    COMPLETED --> [*]
    COMPLETED_WITH_WARNINGS --> [*]
    note right of QUEUED
        Any non-terminal status can also end in
        FAILED (non-retryable error) or
        DEAD_LETTERED (retry budget spent)
    end note
```

## How a job is processed

**Submission.** `POST /jobs` checks the URL, encrypts the Notion token, inserts the job and
publishes the first message with a publisher confirm, then returns `202 Accepted`. The token is
never written to the database; it only travels, encrypted, inside the queue messages.

**1. Transcribe.** Fetches the video's metadata and timestamped captions. The duration is checked
here: a video longer than 20 minutes fails immediately with `VIDEO_TOO_LONG`, before any
download or LLM call.

**2. Summarize.** Downloads a 480p copy of the video and picks up to 36 candidate frames with
ffmpeg (scene changes plus evenly spaced moments). The transcript and the candidate frames,
interleaved at their timestamps, are sent to Gemini, which returns structured JSON: the summary,
sections with start times, self-test questions, and the frames to show with a caption and a
reason for each. The schema only allows timestamps of frames the model has actually seen. Long
transcripts are split into chunks whose results are merged in order.

**3. Extract frames.** Downloads the video at 720p once and grabs each chosen frame with ffmpeg.
Three grabs are taken within the same second and the sharpest one is kept; blank or flat frames
are rejected. A frame that fails is recorded as a warning and left out, never replaced.

**4. Publish.** Uploads the frames to Notion and builds the page under the chosen parent page.
The page is found or created by a job tag in its title, and its ID is saved on the job as soon as
it exists, so a repeated delivery of this stage continues the same page instead of creating a
second one.

## Specification compliance

How each requirement of the project specification is met.

### 1. System overview and required components

| Requirement | Implementation |
| --- | --- |
| Driven entirely from a browser extension | The extension submits jobs and shows their progress; the user never calls the API. |
| Browser extension | `extension/`, Manifest V3, plain JavaScript, no build step. |
| API gateway that does not perform pipeline work | `gateway/` (FastAPI) only validates, stores and enqueues, and serves status. |
| Message broker with durable queues, at-least-once delivery and redelivery on crash | RabbitMQ 4.3 quorum queues, persistent messages, manual acknowledgements, `prefetch_count=1`. An unacknowledged message is redelivered when its worker dies. |
| Worker pool of at least two independent processes | Two `worker` containers by default; `docker compose up -d --scale worker=N` for more. |
| Respect the rate limits of every external API | A Redis token bucket per API, shared by all workers; every external call takes a token first. |

### 2. Browser extension

| Requirement | Implementation |
| --- | --- |
| Content script injects Take Notes into the player toolbar and keeps working across navigation | `extension/content.js` re-adds the button on YouTube's `yt-navigate-finish` event and with a `MutationObserver`. |
| One-time Notion OAuth; token stored | `chrome.identity.launchWebAuthFlow`; the gateway exchanges the code (the client secret never ships in the extension); the token is kept in `chrome.storage`. |
| Background script submits jobs and relays status | `extension/background.js` posts to `/jobs`, polls `/jobs/{id}` and updates the badge. |
| Popup shows the human-readable sequence and a one-click View in Notion link | The popup lists each job with its stage, progress, waiting reason and countdown, warnings and errors as sentences. Raw JSON is never shown. |

### 3. Backend pipeline

| Step | Implementation |
| --- | --- |
| Transcript extraction, true duration, Constraint A | TRANSCRIBE stage (`worker/stages/transcribe.py`), YouTube captions via `youtube-transcript-api` and `yt-dlp`. |
| LLM synthesis with timestamps and a caption per frame; chunk long transcripts and keep section order | SUMMARIZE stage, Gemini structured output validated against a schema; chunking and in-order merge in `worker/llm/`. |
| Frame extraction directly from the video | EXTRACT_FRAMES stage, ffmpeg on the downloaded video. |
| Notion assembly with headings, bullets and captioned images | PUBLISH stage, Notion file uploads and block appends (`worker/notion_page.py`). |

### 4. Job lifecycle

All eight statuses and every transition of the specification are implemented as one state machine
in `common/state.py`. Each transition is recorded in `job_events`.

### 5. REST API

| Endpoint | Behaviour |
| --- | --- |
| `POST /jobs` | Validates the URL and returns `202 Accepted` with `Location: /jobs/{job_id}` and `{job_id, status, video_id}`. Invalid input returns 4xx with `{"error": {"code", "message"}}`. No pipeline work happens in the request. |
| `GET /jobs/{job_id}` | Read-only status with `status`, `progress`, `video_title`, `duration_sec`, `warnings`, `error`, `result`. Unknown IDs return 404. |
| `GET /jobs/{job_id}/results` | Returns 409 before the job is terminal, then the outcome with an absolute Notion page URL. |
| `notion_token` supplied on every submission | The gateway never stores it; it is encrypted (Fernet) inside the queue messages. |
| Default destination when `notion_parent_id` is omitted | The page set in `NOTION_DEFAULT_PARENT_ID` (see [Configuration](#configuration)). |

### 6. Required failure handling

| Failure mode | Implementation |
| --- | --- |
| External rate limits (HTTP 429) | A 429 from YouTube, Gemini or Notion is a retryable error. The message goes to a delay queue (10 s, 30 s, 90 s; a longer `Retry-After` is honoured) and the popup shows the job as waiting, never as failed, while retries remain. |
| Worker crashes | A message is acknowledged only after the stage's output is durable and the next message is confirmed. If a worker dies, RabbitMQ redelivers its message to another worker. Every stage is idempotent, and PUBLISH reuses the saved page under a per-job lock, so no job is lost or duplicated on Notion. |
| Graceful degradation | Failed frames are omitted with a warning and the job ends `COMPLETED_WITH_WARNINGS`; if every frame fails, a text-only page is published. |
| Dead-letter queue | At most `RETRY_MAX_ATTEMPTS` (4) attempts per stage with delays `RETRY_DELAYS` (10, 30, 90 s); then the job is `DEAD_LETTERED` and the message goes to `jobs.dlq`. A private video reaches the DLQ in about 2.5 minutes. Both values are configurable. |

### 7. Hard constraints

| Constraint | Implementation |
| --- | --- |
| A: videos up to 20 minutes, longer ones rejected deterministically | The true duration is checked in TRANSCRIBE against `MAX_VIDEO_SECONDS` (1200). A longer video fails with `VIDEO_TOO_LONG` before anything is downloaded or sent to the LLM. |
| B: real frames only, each with an LLM-written caption | Every image is a JPEG grabbed by ffmpeg from the source video at the chosen timestamp. The LLM writes a caption and a reason for each frame. A frame that cannot be extracted is omitted, never generated or substituted. |

### 8. Deployment and deliverables

| Deliverable | Where |
| --- | --- |
| Unpacked extension, no build step | Load the `extension/` folder with Load unpacked. |
| Gateway, broker and at least two workers from one Compose file | `docker-compose.yml`; scaling with `--scale worker=N`. |
| README with base URL, default Notion destination and every required environment variable | [Getting started](#getting-started) and [Configuration](#configuration). |
| Health check | `GET /healthz` reports the gateway, PostgreSQL, RabbitMQ and Redis. |

### 9. Evaluation criteria

| Criterion | How it is addressed |
| --- | --- |
| Client experience | Status, waiting reasons with a countdown, warnings and errors are shown as plain sentences in the popup. |
| End-to-end latency | A 19:30 video completes in about two minutes (see [Measured results](#measured-results)). |
| Queue resilience | A shared token bucket keeps all workers together under Gemini's per-minute limit; a burst of six real jobs finished with zero LLM 429s. |
| Fault tolerance | Workers killed in each of the four stages recover on another worker with exactly one Notion page. |
| Correctness of Constraints A and B | Checked automatically and with an LLM-as-judge by the scripts in `eval/`. |

## Beyond the specification

Additions we made that the specification does not ask for:

- **Frame-aware notes.** The model sees candidate frames next to the transcript, so it reads slide
  text, code and diagrams, and its captions describe what is actually on screen. Frame
  timestamps are limited to those candidates by the JSON schema.
- **Richer Notion pages.** Embedded video, headings that link to the moment in the video, frames
  placed in the section they illustrate, and self-test questions with hidden answers.
- **Gemini model chain.** If one model's daily quota is spent or it keeps returning errors, every
  worker moves to the next model in `LLM_CHAIN`. The state is shared through Redis.
- **Stage watchdog.** A stage that runs longer than `STAGE_TIMEOUT_S` is stopped and retried, so a
  hung worker cannot hold a message forever. Video downloads have their own time limit and resume
  where they stopped on the next attempt.
- **Outbox handling at the gateway.** A job is stored before it is published; if the broker is
  unreachable the request fails with 503, and a sweeper marks any job whose message never reached
  the broker as failed.
- **Idempotent submissions.** An `Idempotency-Key` header returns the existing job instead of
  creating a second one; the same key sent for a different video is refused with `409`.
- **Fake external APIs.** A `fakes` service imitates YouTube, Gemini and Notion, with switches to
  inject 429s, private videos, broken frame downloads and over-long videos. The whole system and
  every failure mode can be run without any API key.
- **Chaos and measurement tools.** `make kill-worker` kills the worker running a given stage and
  reports how the job recovered; `make burst-report` reports drain time and every LLM answer by
  HTTP status for a burst of jobs.
- **Evaluation scripts.** Automatic checks of Constraints A and B, an LLM-as-judge, and per-stage
  timings (see [Testing and tooling](#testing-and-tooling)).
- **Options page.** Server URL with a connection test, a custom Notion parent page, and a Connect
  Notion button.

## Design decisions

The specification leaves several choices open. These are ours and the reasons for them.

**RabbitMQ as the broker.** It has acknowledgements and redelivery built in, and per-queue
message TTL with dead-lettering gives us delayed retries and a DLQ without plugins or sleeping
workers. Quorum queues are durable and track redeliveries per message. Redis Streams would have required us to
reclaim pending entries ourselves, and Kafka has no per-message acknowledgement. The management
UI also makes queue depths visible during a demo.

**One message per stage.** Splitting a job into four messages means a crash repeats at most one
stage, different stages of a job can run on different workers, and retries are fine-grained. A
retry of the Notion step does not download the video again.

**Acknowledge only after the work is durable.** A worker writes the stage's files atomically
(temporary file, then rename), records a checkpoint row in PostgreSQL, publishes the next message
and waits for the broker's confirm, and only then acknowledges. A redelivered message whose stage
already has a checkpoint simply moves the job on. This gives at-least-once delivery with
idempotent stages.

**No duplicate Notion pages.** The page is located through its parent's list of children using a
job tag in the title, which is strongly consistent, unlike Notion's search. Its ID is stored on
the job the moment it is created, blocks are appended from where the page stopped, and a
PostgreSQL advisory lock per job prevents two workers from publishing the same job at once.

**Job state in PostgreSQL.** Transactions let a stage's checkpoint and the job's final status be
committed together. Each status change is a conditional update from an allowed previous status,
so two workers can never move a job in conflicting directions, and every change is logged.

**Global rate limiting in Redis.** Each external API has a token bucket implemented in a Lua
script, so updates are atomic across workers. The LLM bucket admits at most five calls in any
minute, the free-tier limit, regardless of how many workers run. A worker that would have to wait
more than 20 seconds for a token sends the stage to a delay queue instead of blocking. If Redis is
unavailable, the limiter refuses calls rather than allowing them.

**Retries through delay queues.** Retries never sleep inside a worker. A retryable failure
re-publishes the message to a queue whose TTL is the delay (10, 30 or 90 seconds), after which it
returns to the work queue. The number of attempts is bounded, then the job goes to the DLQ. A
private or removed video is treated as retryable, because YouTube's answer can be temporary.

**One worker type for all stages.** Per-stage messages already spread a job across the pool, so
specialised workers would add configuration without improving throughput. Scaling is a single
`--scale worker=N`. Our measurements also show that a burst is limited by the LLM's rate limit,
not by the number of workers.

**Gemini with structured output.** Gemini accepts the transcript and dozens of images in one
request and returns JSON that follows our schema. Invalid output is re-requested once before the
job fails.

**Captions from YouTube for the transcript.** They are fast, free, and come with the exact
duration needed for Constraint A before any download.

**Frames from a local download.** Stream URLs expire, so the video is downloaded once per job and
ffmpeg grabs frames locally. Taking the sharpest of three grabs within a second avoids motion
blur on slides and screen recordings.

**Plain JavaScript extension.** Manifest V3 with no framework and no build step, so it can be
loaded directly. The logic that can be tested without a browser lives in `extension/lib/` and is
unit tested with Node.

## Measured results

Measured on 14 November 2026 on a single laptop (Intel i7-12700H, 16 GB RAM, Docker Desktop), with
two workers and Gemini's free tier. The raw outputs are in
[`eval/results/2026-11-14/`](eval/results/2026-11-14/).

| Measurement | Result |
| --- | --- |
| End-to-end latency, 19:30 video, alone | 2 min 9 s from submission to finished page (transcript 7 s, notes 95 s including one 10 s retry, frames 18 s, publishing 10 s) |
| Burst of 6 different videos, real YouTube and Gemini | 6 of 6 completed, queue drained in 7 min 12 s, **0 LLM 429 responses** (one 503 from Gemini was retried) |
| Burst of 6 on the fake APIs (Gemini limited to 5 calls per minute) | 2 workers: 1 min 51 s; 4 workers: 2 min 13 s; 0 LLM 429s in both. The LLM limit, not the worker count, sets the pace |
| Worker killed (SIGKILL) during publishing, after the Notion page was created | Message redelivered to the other worker after 0.17 s, job finished 8.1 s after the kill, one Notion page |
| Dead-letter queue | A video that does not exist was retried after 10, 30 and 90 s and dead-lettered 144 s after submission |
| Constraint A | A real 45:38 lecture was rejected with `VIDEO_TOO_LONG` within seconds, without any LLM call |
| Constraint B, 8 videos (7 fixed, 1 held out), 38 frames | Every frame has a caption and a reason, decodes as a JPEG, and matches the frame the model saw at that second (perceptual hash within 0 to 11 of 64 bits, against a median of 16 to 36 bits for other moments of the same video) |
| LLM-as-judge on the same 8 videos | 37 of 38 captions match their picture; faithfulness, coverage, structure and question quality scored 5 of 5 on all eight |

The evaluation also led to changes. Two burst jobs had lost their candidate frames to a refused
YouTube download and written captions from the transcript alone, so that download is now retried
first. The judge found captions that completed half-typed code, described a console output that
was not yet on screen, or named a character who was not in the frame; the caption instructions
now rule out each of these.

## Getting started

### Prerequisites

- Docker with Compose v2 (Docker Desktop on Windows or macOS).
- Chrome or another Chromium-based browser (we tested in Brave).
- Optional: GNU make and [uv](https://docs.astral.sh/uv/) for the `make` shortcuts and the test
  and evaluation scripts (on Windows, from a WSL shell).

### 1. Keys and IDs

| Item | Where to get it | Variable in `.env` |
| --- | --- | --- |
| Gemini API key | [Google AI Studio](https://aistudio.google.com/apikey), Create API key (the free tier is enough) | `LLM_API_KEY` |
| Notion public integration | [notion.so/profile/integrations](https://www.notion.so/profile/integrations), New integration of type Public. Redirect URI: `https://hiimaheehbfpmgnjjpeoejngcmhjmjfi.chromiumapp.org/notion` (the extension ID is fixed by its manifest key, so it is the same on every machine). Capabilities: read, update and insert content | `NOTION_CLIENT_ID`, `NOTION_CLIENT_SECRET` |
| Default destination page | A Notion page such as "Video Notes"; its ID is the 32 hexadecimal characters at the end of its link | `NOTION_DEFAULT_PARENT_ID` |
| Token encryption key | `make fernet-key`, or `python3 -c "import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"` | `TOKEN_ENCRYPTION_KEY` |

### 2. Start the stack

```bash
git clone <repository-url> && cd <repository-folder>
cp .env.example .env            # fill in the four values above
docker compose up -d --build    # or: make up
curl localhost:8000/healthz     # {"gateway":"ok","postgres":"ok","rabbitmq":"ok","redis":"ok"}
```

The first build takes a few minutes. Compose also runs the database migrations and starts two
workers. To change the number of workers: `docker compose up -d --scale worker=4`.

### 3. Load the extension

1. Open `chrome://extensions` (or `brave://extensions`), enable Developer mode, click Load
   unpacked and select the `extension/` folder.
2. Open a YouTube video and click **Take Notes** next to the player's settings button. The first
   time, Notion asks for access: choose Select pages, tick the destination page, and allow.
3. Click the extension icon to follow the job and open the page when it is done.

### Base URL and default destination

- The gateway listens on `http://localhost:8000` (`GATEWAY_BASE_URL`). The extension uses this
  address by default; another one can be set on its Options page. The API has no authentication,
  so Compose publishes it on `127.0.0.1` only: other machines on the network cannot reach it.
- Every job publishes a new page under the page in `NOTION_DEFAULT_PARENT_ID`. If a different
  parent page is entered on the Options page, it is sent as `notion_parent_id` with each job. The
  parent must be one of the pages shared with the integration during the Notion connection. Page
  titles are the video title followed by a job tag, for example `Raft explained · #1a2b3c4d`.

### Running without API keys

`make up-fakes` (or `FAKE_APIS=true docker compose --profile fakes up -d --build`) starts the same
stack against the built-in fake YouTube, Gemini and Notion services. Only `TOKEN_ENCRYPTION_KEY`
and `NOTION_DEFAULT_PARENT_ID` (any 32 hexadecimal characters) are needed. The pages are then
stored in the fake Notion service. `make up` switches back to the real APIs.

## Configuration

All settings are read from `.env` by `common/config.py`. `.env.example` lists every variable with
a comment. Only the first group has to be filled in; all others have working defaults.

**Required**

| Variable | Purpose |
| --- | --- |
| `LLM_API_KEY` | Gemini API key |
| `NOTION_CLIENT_ID`, `NOTION_CLIENT_SECRET` | OAuth credentials of the Notion public integration (the secret stays on the gateway) |
| `NOTION_DEFAULT_PARENT_ID` | Page under which notes are published when the extension does not name another |
| `TOKEN_ENCRYPTION_KEY` | Fernet key used by the gateway and workers to encrypt the Notion token in queue messages |

**LLM model and rate limit**

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini` | LLM provider (Gemini is the implemented provider) |
| `LLM_CHAIN` | `gemini-3.6-flash:low:90,gemini-3.8-flash:low:120,gemini-3.5-flash-lite:medium:60` | Model names in order of preference, as `model:thinking-level:timeout-seconds` |
| `LLM_RPM`, `LLM_BURST` | `4`, `1` | The configured LLM rate limit for all workers together; together they admit at most five calls in any minute |
| `LLM_TPM` | `200000` | Token budget per minute |
| `LIMITER_MAX_WAIT_S` | `20` | Longest wait for a token before the stage is moved to a delay queue |

**Broker and other connections**

| Variable | Default | Purpose |
| --- | --- | --- |
| `RABBITMQ_URL` | `amqp://vna:vna@rabbitmq:5672/` | Broker connection string |
| `DATABASE_URL` | `postgresql+asyncpg://vna:vna@postgres:5432/vna` | Job state database |
| `REDIS_URL` | `redis://redis:6379/0` | Shared rate limits and model health |
| `GATEWAY_BASE_URL` | `http://localhost:8000` | Base URL of the gateway |
| `NOTION_REDIRECT_URI` | `https://hiimaheehbfpmgnjjpeoejngcmhjmjfi.chromiumapp.org/notion` | Must match the redirect URI registered with Notion |

**Retries and backoff**

| Variable | Default | Purpose |
| --- | --- | --- |
| `RETRY_MAX_ATTEMPTS` | `4` | Attempts per stage (the first try and three retries) before the job is dead-lettered |
| `RETRY_DELAYS` | `10,30,90` | Backoff in seconds between attempts; one delay queue per value |
| `STAGE_TIMEOUT_S` | `900` | Stage watchdog: a stage running longer is stopped and retried |
| `YOUTUBE_DOWNLOAD_TIMEOUT_S` | `300` | Longest single video download; the next attempt resumes it |
| `OUTBOX_TIMEOUT_S` | `300` | A job whose first message never reached the broker is marked failed after this |

**Other limits**

| Variable | Default | Purpose |
| --- | --- | --- |
| `MAX_VIDEO_SECONDS` | `1200` | Constraint A |
| `MIN_FRAMES`, `MAX_FRAMES` | `3`, `8` | Number of frames per page |
| `NOTION_RPS`, `NOTION_BURST` | `2.5`, `3` | Notion calls per second (Notion allows about three) |
| `YOUTUBE_RPM`, `YOUTUBE_BURST` | `30`, `2` | Self-imposed limit on YouTube requests |
| `NOTES_LANGUAGE`, `TRANSCRIPT_LANGS` | see `.env.example` | Language of the notes, and caption languages to try |

The remaining variables (frame extraction, candidate frames, note depth, model fallback timing,
and `EVAL_JUDGE_CHAIN` for the evaluation script) are documented in `.env.example`.

## REST API

The extension calls these endpoints; they are listed here for reference.

| Endpoint | Response |
| --- | --- |
| `POST /jobs` with `{youtube_url, notion_token, notion_parent_id?}` | `202` with `{job_id, status: "QUEUED", video_id}` and a `Location` header; `400` `INVALID_URL`, `MISSING_NOTION_TOKEN` or `NOTION_PARENT_NOT_FOUND` (no destination page, or one that is not a Notion page ID or link, refused before any work); `409` `IDEMPOTENCY_KEY_REUSED` when an `Idempotency-Key` header was already used for another video; `422` `VALIDATION_ERROR`; `503` `BROKER_UNAVAILABLE` |
| `GET /jobs/{job_id}` | `200` with `{job_id, status, progress, video_title, duration_sec, warnings, error, result, waiting}`; `404` for an unknown ID |
| `GET /jobs/{job_id}/results` | `409` `JOB_NOT_TERMINAL` until the job ends, then `200` with `{status, result: {notion_page_url, frames_requested, frames_published}, warnings, error}` |
| `GET /healthz` | `200` when PostgreSQL, RabbitMQ and Redis answer, otherwise `503` naming the one that does not |
| `GET /auth/notion/config`, `POST /auth/notion/exchange`, `POST /auth/notion/refresh` | Notion OAuth for the extension; the gateway exchanges the code and returns the token without storing it |

Errors always have the form `{"error": {"code": "...", "message": "..."}}`. While a retry is
scheduled, `waiting` gives the reason and the time of the next attempt.

```bash
curl -i -X POST localhost:8000/jobs -H 'content-type: application/json' \
  -d '{"youtube_url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ","notion_token":"<token>"}'
curl localhost:8000/jobs/<job_id>
curl localhost:8000/jobs/<job_id>/results
```

## Testing and tooling

| Command | What it runs |
| --- | --- |
| `make lint` | ruff and mypy (strict on `common/`) |
| `make test` | Python unit tests, no network |
| `make test-ext` | Extension unit tests with Node's built-in test runner |
| `make up-fakes && make itest` | Integration tests against the full stack with fake APIs |

GitHub Actions (`.github/workflows/ci.yml`) runs `make lint`, the unit tests and the extension
tests on every push and pull request. The integration tests need the Compose stack and run locally.

The integration tests run the real gateway, broker, database, Redis and workers. They cover every
stage, a 429 from each external API, retries and backoff, the DLQ, partial and total frame
failure, over-long videos, a worker killed in each stage, the publish lock, and a burst of six
jobs with zero LLM 429s. Current state: 572 unit tests, 72 extension tests and 64 integration
tests, all passing.

Failure injection on the fake APIs:

| Command | Effect |
| --- | --- |
| `make fake-429 api=llm` (or `youtube`, `notion`; `count=2`) | The next calls to that API answer 429 |
| `make fake-frames mode=partial` (or `none`) | The video download for frames breaks part-way or completely |
| `make fake-unavailable` | Every video behaves as private |
| `make fake-reset` | All fake APIs work normally again |
| `make kill-worker stage=EXTRACT_FRAMES` | Kills the worker running that stage and reports the recovery |
| `make burst` / `make burst-report since=10` | Submits a burst on the fakes, or reports on recent jobs: drain time and LLM answers by status |

Evaluation, run on the jobs of the videos in `eval/videos.json` (or `ids="..."`):

| Command | What it checks |
| --- | --- |
| `make check-constraints` | Constraint A over every job; for each job, that every frame has a caption and reason, is a valid JPEG, matches the candidate frame the model saw, and that status and warnings agree. No LLM calls |
| `make judge` | An LLM-as-judge (`EVAL_JUDGE_CHAIN`, a different model from the note writer) that checks each caption against its picture and scores faithfulness, coverage, structure and questions |
| `make bench-report` | Queue wait, time per stage, retries and total time, from each job's events |

## Repository layout

```
├── docker-compose.yml     gateway, workers, broker, database, Redis; fakes profile
├── Makefile               shortcuts for the commands above
├── .env.example           every setting, documented
├── common/                shared by gateway and workers: config, models, state machine,
│                          broker topology, rate limiter, errors, encryption
├── gateway/               FastAPI app: jobs, health and Notion OAuth routes, outbox sweeper
├── worker/                worker runtime, the four stages, YouTube/Gemini/Notion clients,
│                          prompts, frame extraction, Notion page builder
├── extension/             Manifest V3 extension: content script, background, popup, options
├── migrations/            Alembic database migrations
├── fakes/                 fake YouTube, Gemini and Notion with failure switches
├── chaos/                 kill-worker and burst tools
├── eval/                  constraint checks, LLM-as-judge, benchmarks, results
├── docker/                Dockerfiles
└── tests/                 unit, integration and extension tests
```

## Known limitations

- Only the Gemini provider is implemented. The Compose file contains placeholder profiles for a
  speech-to-text service, a local LLM and Prometheus/Grafana; these were planned extras and are
  not part of this submission.
- Videos without captions cannot be processed (`NO_TRANSCRIPT`), since speech-to-text was not
  built.
- Gemini's free tier allows about 20 requests per model per day. One job uses one request, so the
  model chain gives roughly 60 jobs a day across its three models.
- YouTube sometimes throttles or refuses downloads. Both are retried; if every attempt fails,
  frames are dropped (text-only page) or, on the last attempt of the summary stage, captions are
  written from the transcript alone.
- The evaluation scripts read the files each job keeps on the shared volume, which are the inputs
  the Notion page was built from, because reading the page itself would require the user's
  Notion token.
