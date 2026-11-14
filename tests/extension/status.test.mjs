import assert from "node:assert/strict";
import { test } from "node:test";

import { loadExtensionScripts, plain } from "./helpers.mjs";

const { VNA_STATUS: status } = loadExtensionScripts(["lib/status.js"]);
const NOW = Date.parse("2026-11-02T12:00:00Z");

function job(overrides) {
  return {
    job_id: "6f1c0000-0000-4000-8000-000000000001",
    video_id: "dQw4w9WgXcQ",
    url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    status: "QUEUED",
    progress: 0,
    video_title: null,
    warnings: [],
    error: null,
    result: null,
    waiting: null,
    gone: false,
    ...overrides,
  };
}

const states = (view) => plain(view.steps).map((step) => step.state);

test("the stepper is the spec's sequence", () => {
  assert.deepEqual(plain(status.STEPS), [
    "Queued",
    "Transcribing",
    "Summarizing",
    "Extracting Frames",
    "Publishing",
    "Done",
  ]);
});

test("an active job marks earlier steps done and its own step current", () => {
  const view = status.describeJob(job({ status: "EXTRACTING_FRAMES", progress: 63 }), NOW);
  assert.deepEqual(states(view), ["done", "done", "done", "current", "todo", "todo"]);
  assert.equal(view.headline, "Extracting frames…");
  assert.equal(view.progress, 63);
  assert.equal(view.tone, "active");
  assert.equal(view.terminal, false);
});

test("a queued job is current on the first step", () => {
  const view = status.describeJob(job({}), NOW);
  assert.deepEqual(states(view), ["current", "todo", "todo", "todo", "todo", "todo"]);
  assert.equal(view.title, "YouTube video dQw4w9WgXcQ");
});

test("a completed job is done everywhere, full bar, and links to Notion", () => {
  const view = status.describeJob(
    job({
      status: "COMPLETED",
      progress: 100,
      video_title: "Lecture 3: Consensus",
      result: { notion_page_url: "https://www.notion.so/Lecture-3-abc", frames_requested: 4, frames_published: 4 },
    }),
    NOW,
  );
  assert.deepEqual(states(view), Array(6).fill("done"));
  assert.equal(view.headline, "Done");
  assert.equal(view.title, "Lecture 3: Consensus");
  assert.equal(view.notionUrl, "https://www.notion.so/Lecture-3-abc");
  assert.equal(view.tone, "success");
  assert.equal(view.errorText, null);
});

test("completed with warnings says which frames were skipped, in plain words", () => {
  const view = status.describeJob(
    job({
      status: "COMPLETED_WITH_WARNINGS",
      progress: 100,
      warnings: [
        { code: "FRAME_EXTRACTION_FAILED", timestamp_sec: 461 },
        { code: "FRAME_EXTRACTION_FAILED", timestamp_sec: 723 },
      ],
      result: { notion_page_url: "https://www.notion.so/x", frames_requested: 6, frames_published: 4 },
    }),
    NOW,
  );
  assert.equal(view.headline, "Done, with warnings");
  assert.equal(view.tone, "warning");
  assert.equal(view.warningText, "2 of 6 frames could not be captured (at 7:41, 12:03).");
});

test("a single skipped frame without a result still reads naturally", () => {
  const text = status.warningText([{ code: "FRAME_EXTRACTION_FAILED", timestamp_sec: 5 }], null);
  assert.equal(text, "1 frame could not be captured (at 0:05).");
  assert.equal(status.warningText([], null), null);
});

test("a failed job shows the step it stopped at, from its last progress", () => {
  const cases = [
    [0, 0],
    [5, 1],
    [19, 1],
    [20, 2],
    [49, 2],
    [50, 3],
    [80, 4],
    [97, 4],
  ];
  for (const [progress, failedAt] of cases) {
    const view = status.describeJob(job({ status: "FAILED", progress, error: { code: "INTERNAL_ERROR", message: "x" } }), NOW);
    const expected = Array(6).fill("todo");
    for (let i = 0; i < failedAt; i += 1) expected[i] = "done";
    expected[failedAt] = "failed";
    assert.deepEqual(states(view), expected, `progress ${progress}`);
  }
});

test("errors are one sentence from the code, never the server's raw message", () => {
  const view = status.describeJob(
    job({ status: "FAILED", progress: 5, error: { code: "VIDEO_TOO_LONG", message: "duration 1500s > 1200s" } }),
    NOW,
  );
  assert.equal(view.errorText, "This video is longer than 20 minutes, so it can't be turned into notes.");
  assert.equal(view.errorCode, "VIDEO_TOO_LONG");
  assert.equal(view.tone, "error");
  assert.equal(view.headline, "Failed");
});

test("dead-lettered jobs say the system gave up after retrying", () => {
  const view = status.describeJob(
    job({ status: "DEAD_LETTERED", progress: 5, error: { code: "VIDEO_UNAVAILABLE", message: "private" } }),
    NOW,
  );
  assert.equal(view.headline, "Stopped after repeated failures");
  assert.equal(view.errorText, "This video is private, removed or otherwise unavailable.");
});

test("an unknown error code falls back to a generic sentence", () => {
  assert.equal(status.errorMessage("SOMETHING_NEW"), "Something went wrong.");
});

test("a scheduled retry reads as waiting with a countdown, not as an error", () => {
  const waiting = { until: "2026-11-02T12:00:12Z", reason: "RATE_LIMITED:notion" };
  const view = status.describeJob(job({ status: "PUBLISHING", progress: 85, waiting }), NOW);
  assert.equal(view.tone, "waiting");
  assert.equal(view.waitingText, "Waiting — Notion is rate-limiting us, retrying in 12 s.");
  assert.equal(view.errorText, null);
});

test("waiting text covers other reasons, unknown apis and an elapsed deadline", () => {
  assert.equal(
    status.waitingText({ until: "2026-11-02T12:00:30Z", reason: "UPSTREAM_ERROR:llm" }, NOW),
    "Waiting — the AI model had a temporary problem, retrying in 30 s.",
  );
  assert.equal(
    status.waitingText({ until: "2026-11-02T11:59:00Z", reason: "CONNECTION_ERROR" }, NOW),
    "Waiting — an external service had a temporary problem, retrying now.",
  );
  assert.equal(status.waitingText(null, NOW), null);
});

test("waiting text names our own reasons instead of blaming an external service", () => {
  assert.equal(
    status.waitingText({ until: "2026-11-02T12:00:10Z", reason: "STAGE_TIMEOUT" }, NOW),
    "Waiting — a step took too long, retrying in 10 s.",
  );
  assert.equal(
    status.waitingText({ until: "2026-11-02T12:00:10Z", reason: "PUBLISH_IN_PROGRESS" }, NOW),
    "Waiting — another worker is already publishing this job, retrying in 10 s.",
  );
});

test("a terminal job never shows a stale waiting line", () => {
  const waiting = { until: "2026-11-02T12:00:12Z", reason: "RATE_LIMITED:llm" };
  const view = status.describeJob(job({ status: "COMPLETED", progress: 100, waiting }), NOW);
  assert.equal(view.waitingText, null);
});

test("only an https Notion url becomes a link", () => {
  for (const url of ["javascript:alert(1)", "http://www.notion.so/x", "notion.so/x", null]) {
    const view = status.describeJob(job({ status: "COMPLETED", result: { notion_page_url: url } }), NOW);
    assert.equal(view.notionUrl, null, String(url));
  }
});

test("a job the server no longer knows is shown as gone and stops counting as active", () => {
  const view = status.describeJob(job({ status: "SUMMARIZING", progress: 30, gone: true }), NOW);
  assert.equal(view.terminal, true);
  assert.equal(view.headline, "Not found");
  assert.equal(view.errorText, "This job no longer exists on the notes server.");
  assert.deepEqual(states(view), Array(6).fill("todo"));
});

test("progress is clamped to 0..100", () => {
  assert.equal(status.describeJob(job({ status: "SUMMARIZING", progress: 140 }), NOW).progress, 100);
  assert.equal(status.describeJob(job({ status: "SUMMARIZING", progress: -3 }), NOW).progress, 0);
});
