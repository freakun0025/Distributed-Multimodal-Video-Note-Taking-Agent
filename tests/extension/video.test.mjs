import assert from "node:assert/strict";
import { test } from "node:test";

import { loadExtensionScripts } from "./helpers.mjs";

const { VNA_VIDEO: video } = loadExtensionScripts(["lib/video.js"]);

test("watch page urls yield the video id, whatever else is in the query", () => {
  for (const href of [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123&index=4&t=61s",
    "https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
    "https://youtube.com/watch?v=dQw4w9WgXcQ",
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
  ]) {
    assert.equal(video.videoIdFromUrl(href), "dQw4w9WgXcQ", href);
  }
});

test("pages that are not a watch page have no video id", () => {
  for (const href of [
    "https://www.youtube.com/",
    "https://www.youtube.com/results?search_query=lecture",
    "https://www.youtube.com/@channel/videos",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://www.youtube.com/watch",
    "https://www.youtube.com/watch?v=short",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ<script>",
    "https://evil.example/watch?v=dQw4w9WgXcQ",
    "not a url",
  ]) {
    assert.equal(video.videoIdFromUrl(href), null, href);
  }
});

test("the url sent to the gateway is canonical", () => {
  assert.equal(video.watchUrl("dQw4w9WgXcQ"), "https://www.youtube.com/watch?v=dQw4w9WgXcQ");
});
