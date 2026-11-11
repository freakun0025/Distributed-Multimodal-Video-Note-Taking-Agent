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
