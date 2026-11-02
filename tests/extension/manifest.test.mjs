// The unpacked extension loads with no build step (F-22): every file the manifest and pages
// reference exists, MV3 rules hold, and the host permissions cover the default gateway.
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { test } from "node:test";

import { ROOT, loadExtensionScripts, readRepoFile } from "./helpers.mjs";

const manifest = JSON.parse(readRepoFile("extension/manifest.json"));
const exists = (file) => existsSync(`${ROOT}extension/${file}`);

test("it is a Manifest V3 extension with the least permissions it needs today", () => {
  assert.equal(manifest.manifest_version, 3);
  // identity is added with the Notion OAuth flow (ROADMAP M2).
  assert.deepEqual(manifest.permissions, ["storage", "alarms"]);
});

test("every file the manifest names exists", () => {
  const files = [
    manifest.background.service_worker,
    manifest.action.default_popup,
    manifest.options_ui.page,
    ...manifest.content_scripts.flatMap((cs) => [...cs.js, ...(cs.css ?? [])]),
  ];
  for (const file of files) assert.ok(exists(file), file);
});

test("the service worker imports scripts that exist, shared libraries first", () => {
  const match = readRepoFile("extension/background.js").match(/importScripts\(([^)]*)\)/);
  const files = [...match[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(files, ["config.js", "lib/status.js", "lib/jobs.js", "lib/api.js"]);
  for (const file of files) assert.ok(exists(file), file);
});

test("content scripts load config and the video helper before content.js", () => {
  assert.deepEqual(manifest.content_scripts[0].js, ["config.js", "lib/video.js", "content.js"]);
  assert.deepEqual(manifest.content_scripts[0].matches, ["https://www.youtube.com/*"]);
});

test("pages use only external scripts (MV3 forbids inline script) that exist", () => {
  for (const page of ["popup.html", "options.html"]) {
    const html = readRepoFile(`extension/${page}`);
    const tags = [...html.matchAll(/<script\b([^>]*)>/g)].map((m) => m[1]);
    assert.ok(tags.length > 0, page);
    for (const attrs of tags) {
      const src = attrs.match(/src="([^"]+)"/)?.[1];
      assert.ok(src, `${page}: inline <script>`);
      assert.ok(exists(src), `${page}: ${src}`);
    }
    for (const [, href] of html.matchAll(/<link [^>]*href="([^"]+)"/g)) assert.ok(exists(href), `${page}: ${href}`);
  }
});

test("host permissions cover the default gateway url", () => {
  const { VNA_CONFIG: config } = loadExtensionScripts(["config.js"]);
  const gateway = new URL(config.defaultGatewayUrl);
  assert.ok(manifest.host_permissions.includes(`${gateway.protocol}//${gateway.hostname}/*`));
});
