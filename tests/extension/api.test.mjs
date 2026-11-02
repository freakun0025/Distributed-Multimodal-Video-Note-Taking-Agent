import assert from "node:assert/strict";
import { test } from "node:test";

import { loadExtensionScripts, plain } from "./helpers.mjs";

const { VNA_API: api } = loadExtensionScripts(["lib/api.js"]);
const BASE = "http://localhost:8000";

function fakeFetch(status, body) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, init });
    return { status, json: async () => (body === undefined ? Promise.reject(new SyntaxError()) : body) };
  };
  return { calls, fetchImpl };
}

const client = (fetchImpl) => api.createClient({ baseUrl: BASE, timeoutMs: 1000, fetchImpl });

test("submit posts the spec body with an Idempotency-Key and returns the 202 body", async () => {
  const accepted = { job_id: "j1", status: "QUEUED", video_id: "dQw4w9WgXcQ" };
  const { calls, fetchImpl } = fakeFetch(202, accepted);
  const result = await client(fetchImpl).submitJob({
    youtubeUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    notionToken: "secret-token",
    notionParentId: "",
    idempotencyKey: "k-1",
  });
  assert.deepEqual(plain(result), accepted);
  assert.equal(calls[0].url, `${BASE}/jobs`);
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers["Idempotency-Key"], "k-1");
  assert.equal(calls[0].init.headers["Content-Type"], "application/json");
  // No notion_parent_id when none is set: the gateway then uses the README's default page.
  assert.deepEqual(JSON.parse(calls[0].init.body), {
    youtube_url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    notion_token: "secret-token",
  });
});

test("submit includes notion_parent_id when one is configured", async () => {
  const { calls, fetchImpl } = fakeFetch(202, { job_id: "j1", status: "QUEUED", video_id: "x" });
  await client(fetchImpl).submitJob({ youtubeUrl: "u", notionToken: "t", notionParentId: "p-1", idempotencyKey: "k" });
  assert.equal(JSON.parse(calls[0].init.body).notion_parent_id, "p-1");
});

test("an error body's code becomes the GatewayError code", async () => {
  const { fetchImpl } = fakeFetch(400, { error: { code: "INVALID_URL", message: "not a video" } });
  await assert.rejects(
    client(fetchImpl).submitJob({ youtubeUrl: "u", notionToken: "t", idempotencyKey: "k" }),
    (error) => error.code === "INVALID_URL" && error.httpStatus === 400,
  );
});

test("a 404 on GET /jobs/{id} is JOB_NOT_FOUND", async () => {
  const { calls, fetchImpl } = fakeFetch(404, { error: { code: "JOB_NOT_FOUND", message: "no such job" } });
  await assert.rejects(client(fetchImpl).getJob("j/1"), (error) => error.code === "JOB_NOT_FOUND");
  assert.equal(calls[0].url, `${BASE}/jobs/j%2F1`);
  assert.equal(calls[0].init.cache, "no-store");
});

test("a network failure is GATEWAY_UNREACHABLE", async () => {
  const fetchImpl = async () => {
    throw new TypeError("Failed to fetch");
  };
  await assert.rejects(client(fetchImpl).getJob("j1"), (error) => error.code === "GATEWAY_UNREACHABLE");
});

test("a non-JSON or unexpected response is GATEWAY_ERROR", async () => {
  await assert.rejects(client(fakeFetch(502).fetchImpl).getJob("j1"), (error) => error.code === "GATEWAY_ERROR");
  await assert.rejects(client(fakeFetch(500, {}).fetchImpl).getJob("j1"), (error) => error.code === "GATEWAY_ERROR");
});

test("health returns the component body for 200 and for 503", async () => {
  const down = { gateway: "ok", postgres: "ok", rabbitmq: "down", redis: "ok" };
  assert.deepEqual(plain(await client(fakeFetch(503, down).fetchImpl).health()), down);
});

test("gateway urls are limited to http on localhost/127.0.0.1, reduced to an origin", () => {
  assert.deepEqual(plain(api.normalizeGatewayUrl(" http://localhost:8000/ ")), { url: "http://localhost:8000" });
  assert.deepEqual(plain(api.normalizeGatewayUrl("http://127.0.0.1:9000/x")), { url: "http://127.0.0.1:9000" });
  for (const bad of ["https://localhost:8000", "http://example.com", "ftp://localhost", "localhost:8000", ""]) {
    assert.ok(api.normalizeGatewayUrl(bad).error, bad);
  }
});
