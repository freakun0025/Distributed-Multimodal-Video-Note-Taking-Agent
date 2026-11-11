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

// What Notion's consent window does: answer(url) -> the URL it redirects to, or throws (closed).
const REDIRECT = `https://${EXTENSION_ID}.chromiumapp.org/notion`;
const approve = (url) => `${REDIRECT}?code=the-code&state=${new URL(url).searchParams.get("state")}`;

function fakeChrome(initialStore = {}, authAnswer = approve) {
  const store = structuredClone(initialStore);
  const listeners = { message: [], alarm: [] };
  const alarms = new Map();
  const state = { badge: "", optionsOpened: 0, authUrls: [] };
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
    identity: {
      getRedirectURL: (path) => `https://${EXTENSION_ID}.chromiumapp.org/${path}`,
      launchWebAuthFlow: async ({ url, interactive }) => {
        assert.equal(interactive, true);
        state.authUrls.push(url);
        return authAnswer(url);
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

function startBackground({ store = {}, routes = {}, authAnswer } = {}) {
  const chromeFake = fakeChrome(store, authAnswer);
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

const notionConfig = () => [200, { client_id: "client-1", redirect_uri: REDIRECT }];
const notionToken =
  (access = TOKEN) =>
  () => [
    200,
    { access_token: access, refresh_token: "refresh-1", workspace_name: "My workspace", bot_id: "b", expires_in: null },
  ];
const signIn = { "GET /auth/notion/config": notionConfig, "POST /auth/notion/exchange": notionToken() };
const rejectedAt = (code) => () => [400, { error: { code, message: "x" } }];

test("the first click connects Notion, then submits with the new token", async () => {
  const bg = startBackground({ routes: { ...signIn, "POST /jobs": accepted("j1") } });
  const response = await bg.send({ type: "SUBMIT", url: VIDEO_URL });

  assert.equal(response.ok, true);
  const authUrl = new URL(bg.state.authUrls[0]);
  assert.equal(`${authUrl.origin}${authUrl.pathname}`, "https://api.notion.com/v1/oauth/authorize");
  assert.equal(authUrl.searchParams.get("client_id"), "client-1");
  assert.equal(authUrl.searchParams.get("redirect_uri"), REDIRECT);
  assert.equal(authUrl.searchParams.get("owner"), "user");
  assert.equal(authUrl.searchParams.get("response_type"), "code");
  assert.match(authUrl.searchParams.get("state"), /^[0-9a-f-]{36}$/);
  assert.deepEqual(
    bg.gateway.calls.map((c) => `${c.method} ${c.path}`),
    ["GET /auth/notion/config", "POST /auth/notion/exchange", "POST /jobs"],
  );
  assert.deepEqual(JSON.parse(bg.gateway.calls[1].init.body), { code: "the-code" });
  assert.equal(JSON.parse(bg.gateway.calls[2].init.body).notion_token, TOKEN);
  assert.equal(bg.store.notionToken, TOKEN);
  assert.equal(bg.store.notionRefreshToken, "refresh-1");
  assert.equal(bg.store.notionWorkspace, "My workspace");
});

test("a later click reuses the stored token without asking Notion again", async () => {
  const bg = startBackground({ store: withToken, routes: { ...signIn, "POST /jobs": accepted("j1") } });
  await bg.send({ type: "SUBMIT", url: VIDEO_URL });
  assert.equal(bg.state.authUrls.length, 0);
  assert.deepEqual(bg.gateway.calls.map((c) => c.path), ["/jobs"]);
});

for (const [name, authAnswer] of [
  [
    "the window is closed",
    () => {
      throw new Error("The user did not approve access.");
    },
  ],
  ["Cancel is pressed", (url) => `${REDIRECT}?error=access_denied&state=${new URL(url).searchParams.get("state")}`],
]) {
  test(`when ${name}, nothing is submitted or stored`, async () => {
    const bg = startBackground({ routes: { ...signIn, "POST /jobs": accepted("j1") }, authAnswer });
    const response = await bg.send({ type: "SUBMIT", url: VIDEO_URL });
    assert.deepEqual(response, {
      ok: false,
      code: "OAUTH_CANCELLED",
      message: "Notion sign-in was cancelled, so nothing was sent.",
    });
    assert.deepEqual(bg.gateway.calls.map((c) => c.path), ["/auth/notion/config"]);
    assert.equal(bg.store.notionToken, undefined);
  });
}

test("an answer that does not carry our state is ignored, never exchanged", async () => {
  const bg = startBackground({ routes: signIn, authAnswer: () => `${REDIRECT}?code=stolen&state=someone-elses` });
  const response = await bg.send({ type: "CONNECT_NOTION" });
  assert.equal(response.code, "OAUTH_STATE_MISMATCH");
  assert.deepEqual(bg.gateway.calls.map((c) => c.path), ["/auth/notion/config"]);
});

test("a server whose redirect URI names another extension never opens Notion", async () => {
  const other = () => [200, { client_id: "c", redirect_uri: "https://other.chromiumapp.org/notion" }];
  const bg = startBackground({ routes: { "GET /auth/notion/config": other } });
  const response = await bg.send({ type: "CONNECT_NOTION" });
  assert.equal(response.code, "OAUTH_REDIRECT_MISMATCH");
  assert.equal(bg.state.authUrls.length, 0);
});

test("a rejected code is reported and nothing is stored", async () => {
  const bg = startBackground({
    routes: { "GET /auth/notion/config": notionConfig, "POST /auth/notion/exchange": rejectedAt("OAUTH_CODE_INVALID") },
  });
  const response = await bg.send({ type: "CONNECT_NOTION" });
  assert.equal(response.code, "OAUTH_CODE_INVALID");
  assert.equal(bg.store.notionToken, undefined);
});

test("connect answers with the workspace name for the options page", async () => {
  const bg = startBackground({ routes: signIn });
  assert.deepEqual(await bg.send({ type: "CONNECT_NOTION" }), { ok: true, workspace: "My workspace" });
});

test("disconnect forgets the token, its refresh token and the workspace", async () => {
  const bg = startBackground({
    store: { ...withToken, notionRefreshToken: "r", notionWorkspace: "w", gatewayUrl: "http://localhost:8000" },
  });
  assert.deepEqual(await bg.send({ type: "DISCONNECT_NOTION" }), { ok: true });
  assert.deepEqual(Object.keys(bg.store), ["gatewayUrl"]);
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

const rejectedByNotion = (jobId) =>
  statusBody(jobId, "FAILED", 90, { error: { code: "NOTION_UNAUTHORIZED", message: "x" } });

test("a job Notion rejected renews the token with the refresh token", async () => {
  const bg = startBackground({
    store: { ...withToken, notionRefreshToken: "refresh-1", jobs: [queued("j1")] },
    routes: { "GET /jobs/j1": rejectedByNotion("j1"), "POST /auth/notion/refresh": notionToken("ntn-renewed") },
  });
  await bg.send({ type: "REFRESH" });
  assert.equal(bg.store.notionToken, "ntn-renewed");
  assert.deepEqual(JSON.parse(bg.gateway.calls.at(-1).init.body), { refresh_token: "refresh-1" });
});

test("without a working refresh token a rejected token is forgotten, so the next click reconnects", async () => {
  const bg = startBackground({
    store: { ...withToken, notionRefreshToken: "refresh-1", jobs: [queued("j1")] },
    routes: { "GET /jobs/j1": rejectedByNotion("j1"), "POST /auth/notion/refresh": rejectedAt("OAUTH_CODE_INVALID") },
  });
  await bg.send({ type: "REFRESH" });
  assert.equal(bg.store.notionToken, undefined);
  assert.equal(bg.store.notionRefreshToken, undefined);
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
