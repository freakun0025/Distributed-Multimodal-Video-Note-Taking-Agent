# Distributed Multimodal Video Note-Taking Agent

Chrome extension → FastAPI gateway → RabbitMQ → worker pool that turns a YouTube video
(≤ 20 minutes) into a Notion page of notes with real, captioned video frames.

> Work in progress. Setup, base URL, default Notion destination, every environment variable,
> the demo script and benchmarks will be documented here (see `.env.example` for now).

## Chrome extension (development)

No build step. With the stack running (`make up`, gateway on `http://localhost:8000`):

1. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and pick the
   `extension/` folder.
2. Open a YouTube video and click **Take Notes** in the player's toolbar. The first time, Notion's
   sign-in window opens: choose **Select pages**, pick the page to publish under (the default
   destination, `NOTION_DEFAULT_PARENT_ID`), then **Allow access**. The token is kept by the
   extension, so later clicks don't ask again. (Or connect first: extension menu → **Options** →
   **Connect Notion**.)
3. Follow the job from the toolbar icon's popup; when it is done, **View in Notion** opens the page.

Extension unit tests: `make test-ext` (Node 18+, no npm packages).

## Failure demos (Spec §6)

Run from a WSL/Linux shell. The switches live in the fake APIs, so start the stack with
`make up-fakes` (fake YouTube, Gemini and Notion; no real quota is spent). Then click
**Take Notes** on any video and watch the popup.

| Failure | Command | What you see |
| --- | --- | --- |
| HTTP 429 from an external API | `make fake-429 api=youtube` (or `llm`, `notion`; `count=2` for two in a row, `retry_after=20`) | The popup shows "Waiting … retrying in 10 s" (then 30 s, 90 s), never an error; the job completes |
| Worker crash | `make kill-worker stage=EXTRACT_FRAMES` (any stage), then Take Notes | The worker running that stage is SIGKILLed; the script prints which worker got the message and how long recovery took; one Notion page |
| Some / all frames fail | `make fake-frames mode=partial` (or `none`) | `COMPLETED_WITH_WARNINGS`; the popup lists the missing frames; the page keeps the others (or is text only) |
| Dead-letter queue | `make fake-unavailable` (or a private video on the real stack) | Waiting through every retry, then `DEAD_LETTERED` after about 4 minutes (10+30+90+90 s); the message is in `jobs.dlq` (RabbitMQ UI, `localhost:15672`) |

`make fake-reset` makes every fake API work again; `make fakes` shows the switches.
`make itest` runs all of these as integration tests.
