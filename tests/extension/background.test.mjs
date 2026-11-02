// background.js against a fake chrome.* API and a fake gateway: what the service worker stores,
// what it sends, and what it answers the content script, popup and options page.
import assert from "node:assert/strict";
import { test } from "node:test";

import { createContext, plain, runScript } from "./helpers.mjs";

const EXTENSION_ID = "vna-test-extension";
const TOKEN = "ntn-secret-token-value";
const VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ";

/** A gateway answering from `routes`: { "POST /jobs": (init) => [status, body], ... }. */
function fakeGateway(routes) {
  const calls = [];
  async function fetch(url, init) {
    const { pathname } = new URL(url);
    calls.push({ method: init.method, path: pathname, init });
    const route = routes[`${init.method} ${pathname}`];
    if (!route) throw new TypeError("Failed to fetch");
    const [status, body] = await route(init);
    return { status, json: async () => body };
  }
  return { calls, fetch };
}

function fakeChrome(initialStore = {}) {
  const store = structuredClone(initialStore);
  const listeners = { message: [], alarm: [] };
  const alarms = new Map();
  const state = { badge: "", optionsOpened: 0 };
  const keysOf = (keys) => (typeof keys === "string" ? [keys] : keys);
  const chrome = {
    runtime: {
      id: EXTENSION_ID,
      onMessage: { addListener: (fn) => listeners.message.push(fn) },
      onInstalled: { addListener: () => {} },
      onStartup: { addListener: () => {} },
      openOptionsPage: async () => {
        state.optionsOpened += 1;
      },
    },
    storage: {
      local: {
        get: async (keys) =>
          Object.fromEntries(keysOf(keys).filter((k) => k in store).map((k) => [k, structuredClone(store[k])])),
        set: async (items) => Object.assign(store, structuredClone(items)),
        remove: async (keys) => keysOf(keys).forEach((k) => delete store[k]),
      },
    },
    action: {
      setBadgeText: async ({ text }) => {
        state.badge = text;
      },
      setBadgeBackgroundColor: async () => {},
    },
    alarms: {
      get: async (name) => alarms.get(name),
      create: async (name, info) => alarms.set(name, { name, ...info }),
      clear: async (name) => alarms.delete(name),
      onAlarm: { addListener: (fn) => listeners.alarm.push(fn) },
    },
  };
  return { chrome, store, listeners, alarms, state };
}

function startBackground({ store = {}, routes = {} } = {}) {
  const chromeFake = fakeChrome(store);
  const gateway = fakeGateway(routes);
  const context = createContext({ chrome: chromeFake.chrome, fetch: gateway.fetch });
  runScript(context, "extension/background.js");
  const send = (message, sender = { id: EXTENSION_ID }) =>
    new Promise((resolve) => {
      const isAsync = chromeFake.listeners.message[0](message, sender, (response) => resolve(plain(response)));
      if (!isAsync) resolve(undefined);
    });
  return { ...chromeFake, gateway, send };
}

const accepted = (jobId) => () => [202, { job_id: jobId, status: "QUEUED", video_id: "dQw4w9WgXcQ" }];
const withToken = { notionToken: TOKEN };

test("submit without a token opens the options page and calls nothing", async () => {
  const bg = startBackground({ routes: { "POST /jobs": accepted("j1") } });
  const response = await bg.send({ type: "SUBMIT", url: VIDEO_URL });
  assert.equal(response.ok, false);
  assert.equal(response.code, "NO_TOKEN");
  assert.equal(bg.state.optionsOpened, 1);
  assert.equal(bg.gateway.calls.length, 0);
});

test("submit posts the job, stores it as queued, counts it on the badge and starts the alarm", async () => {
  const bg = startBackground({ store: withToken, routes: { "POST /jobs": accepted("j1") } });
  const response = await bg.send({ type: "SUBMIT", url: VIDEO_URL });

  assert.equal(response.ok, true);
  assert.equal(response.jobId, "j1");
  const [call] = bg.gateway.calls;
  assert.equal(call.path, "/jobs");
  assert.deepEqual(JSON.parse(call.init.body), { youtube_url: VIDEO_URL, notion_token: TOKEN });
  assert.match(call.init.headers["Idempotency-Key"], /^[0-9a-f-]{36}$/);
  assert.equal(bg.store.jobs[0].job_id, "j1");
  assert.equal(bg.store.jobs[0].status, "QUEUED");
  assert.equal(bg.state.badge, "1");
  assert.equal(bg.alarms.get("vna-refresh").periodInMinutes, 0.5);
});

test("each click gets its own Idempotency-Key, and the token is never copied into the job list", async () => {
  let n = 0;
  const bg = startBackground({
    store: { ...withToken, notionParentId: "parent-1" },
    routes: { "POST /jobs": () => accepted(`j${(n += 1)}`)() },
  });
  await bg.send({ type: "SUBMIT", url: VIDEO_URL });
  await bg.send({ type: "SUBMIT", url: VIDEO_URL });
  const keys = bg.gateway.calls.map((c) => c.init.headers["Idempotency-Key"]);
  assert.notEqual(keys[0], keys[1]);
  assert.equal(JSON.parse(bg.gateway.calls[0].init.body).notion_parent_id, "parent-1");
  assert.deepEqual(bg.store.jobs.map((j) => j.job_id), ["j2", "j1"]);
  assert.equal(JSON.stringify(bg.store.jobs).includes(TOKEN), false);
});

test("a rejected submission answers with a plain sentence and stores nothing", async () => {
  const bg = startBackground({
    store: withToken,
    routes: { "POST /jobs": () => [400, { error: { code: "INVALID_URL", message: "bad" } }] },
  });
  const response = await bg.send({ type: "SUBMIT", url: "https://www.youtube.com/watch?v=x" });
  assert.deepEqual(response, { ok: false, code: "INVALID_URL", message: "That is not a YouTube video link." });
  assert.equal(bg.store.jobs, undefined);
});

test("an unreachable gateway on submit says so", async () => {
  const bg = startBackground({ store: withToken });
  const response = await bg.send({ type: "SUBMIT", url: VIDEO_URL });
  assert.equal(response.code, "GATEWAY_UNREACHABLE");
});

const queued = (jobId, overrides = {}) => ({
  job_id: jobId,
  video_id: "dQw4w9WgXcQ",
  url: VIDEO_URL,
  submitted_at: "2026-11-02T12:00:00.000Z",
  status: "QUEUED",
  progress: 0,
  video_title: null,
  duration_sec: null,
  warnings: [],
  error: null,
  result: null,
  waiting: null,
  gone: false,
  ...overrides,
});

const statusBody = (jobId, status, progress, extra = {}) => () => [
  200,
  { job_id: jobId, status, progress, video_title: "Raft", duration_sec: 600, warnings: [], error: null, result: null, waiting: null, ...extra },
];

test("refresh polls only running jobs and relays their status into storage", async () => {
  const bg = startBackground({
    store: { jobs: [queued("j1"), queued("j2", { status: "COMPLETED", progress: 100 })] },
    routes: { "GET /jobs/j1": statusBody("j1", "SUMMARIZING", 31) },
  });
  const response = await bg.send({ type: "REFRESH" });
  assert.equal(response.ok, true);
  assert.deepEqual(bg.gateway.calls.map((c) => c.path), ["/jobs/j1"]);
  assert.equal(bg.store.jobs[0].status, "SUMMARIZING");
  assert.equal(bg.store.jobs[0].video_title, "Raft");
  assert.deepEqual(bg.store.connection, { ok: true, gatewayUrl: "http://localhost:8000" });
});

test("when the last job finishes the badge clears and the alarm stops", async () => {
  const bg = startBackground({
    store: { jobs: [queued("j1", { status: "PUBLISHING", progress: 90 })] },
    routes: { "GET /jobs/j1": statusBody("j1", "COMPLETED", 100) },
  });
  await bg.chrome.alarms.create("vna-refresh", { periodInMinutes: 0.5 });
  await bg.send({ type: "REFRESH" });
  assert.equal(bg.store.jobs[0].status, "COMPLETED");
  assert.equal(bg.state.badge, "");
  assert.equal(bg.alarms.has("vna-refresh"), false);
});

test("a 404 marks the job gone instead of polling it forever", async () => {
  const bg = startBackground({
    store: { jobs: [queued("j1")] },
    routes: { "GET /jobs/j1": () => [404, { error: { code: "JOB_NOT_FOUND", message: "x" } }] },
  });
  await bg.send({ type: "REFRESH" });
  assert.equal(bg.store.jobs[0].gone, true);
  assert.equal(bg.state.badge, "");
});

test("an unreachable gateway keeps the last known status and flags the connection", async () => {
  const bg = startBackground({ store: { jobs: [queued("j1", { status: "TRANSCRIBING", progress: 9 })] } });
  const response = await bg.send({ type: "REFRESH" });
  assert.equal(response.ok, false);
  assert.equal(bg.store.jobs[0].status, "TRANSCRIBING");
  assert.equal(bg.store.connection.ok, false);
});

test("simultaneous refresh requests share one round of requests", async () => {
  const bg = startBackground({
    store: { jobs: [queued("j1"), queued("j2")] },
    routes: { "GET /jobs/j1": statusBody("j1", "TRANSCRIBING", 8), "GET /jobs/j2": statusBody("j2", "QUEUED", 0) },
  });
  await Promise.all([bg.send({ type: "REFRESH" }), bg.send({ type: "REFRESH" }), bg.send({ type: "REFRESH" })]);
  assert.equal(bg.gateway.calls.length, 2);
});

test("a job submitted during a refresh is not overwritten by it", async () => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const bg = startBackground({
    store: { ...withToken, jobs: [queued("j1")] },
    routes: {
      // The GET is held until the submit below has stored its job.
      "GET /jobs/j1": async () => {
        await gate;
        return statusBody("j1", "TRANSCRIBING", 8)();
      },
      "POST /jobs": accepted("j2"),
    },
  });
  const refreshing = bg.send({ type: "REFRESH" });
  while (bg.gateway.calls.length === 0) await new Promise((resolve) => setImmediate(resolve));
  await bg.send({ type: "SUBMIT", url: VIDEO_URL });
  release();
  await refreshing;
  assert.deepEqual(bg.store.jobs.map((j) => [j.job_id, j.status]), [
    ["j2", "QUEUED"],
    ["j1", "TRANSCRIBING"],
  ]);
});

test("clear finished keeps only running jobs", async () => {
  const bg = startBackground({
    store: { jobs: [queued("j1"), queued("j2", { status: "FAILED" }), queued("j3", { status: "COMPLETED" })] },
  });
  await bg.send({ type: "CLEAR_FINISHED" });
  assert.deepEqual(bg.store.jobs.map((j) => j.job_id), ["j1"]);
});

test("health checks the url typed on the options page, and rejects other hosts", async () => {
  const bg = startBackground({
    routes: { "GET /healthz": () => [503, { gateway: "ok", postgres: "ok", rabbitmq: "down", redis: "ok" }] },
  });
  const response = await bg.send({ type: "HEALTH", gatewayUrl: "http://localhost:8000" });
  assert.deepEqual(response, { ok: true, health: { gateway: "ok", postgres: "ok", rabbitmq: "down", redis: "ok" } });
  const rejected = await bg.send({ type: "HEALTH", gatewayUrl: "http://example.com" });
  assert.equal(rejected.code, "INVALID_GATEWAY_URL");
});

test("messages from other extensions and unknown types are ignored", async () => {
  const bg = startBackground({ store: withToken, routes: { "POST /jobs": accepted("j1") } });
  assert.equal(await bg.send({ type: "SUBMIT", url: VIDEO_URL }, { id: "someone-else" }), undefined);
  assert.equal(await bg.send({ type: "NOPE" }), undefined);
  assert.equal(bg.gateway.calls.length, 0);
});
