# Distributed Multimodal Video Note-Taking Agent

Chrome extension → FastAPI gateway → RabbitMQ → worker pool that turns a YouTube video
(≤ 20 minutes) into a Notion page of notes with real, captioned video frames.

> Work in progress. Setup, base URL, default Notion destination, every environment variable,
> the demo script and benchmarks will be documented here (see `.env.example` for now).

## Chrome extension (development)

No build step. With the stack running (`make up`, gateway on `http://localhost:8000`):

1. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and pick the
   `extension/` folder.
2. The settings page opens on your first Take Notes click (or: extension menu → **Options**).
   Paste a Notion token there (until the Notion sign-in flow lands, M2) and click **Test connection**.
3. Open a YouTube video and click **Take Notes** in the player's toolbar. Follow the job from the
   toolbar icon's popup.

Extension unit tests: `make test-ext` (Node 18+, no npm packages).
