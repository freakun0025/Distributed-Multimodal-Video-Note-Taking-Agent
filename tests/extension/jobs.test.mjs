import assert from "node:assert/strict";
import { test } from "node:test";

import { loadExtensionScripts, plain } from "./helpers.mjs";

const { VNA_JOBS: jobs } = loadExtensionScripts(["lib/status.js", "lib/jobs.js"]);
const URL_A = "https://www.youtube.com/watch?v=aaaaaaaaaaa";

const created = (id) => ({ job_id: id, status: "QUEUED", video_id: "aaaaaaaaaaa" });
const record = (id, overrides = {}) => ({ ...plain(jobs.newJob(created(id), URL_A, 0)), ...overrides });

test("a new job starts queued at 0% with nothing else known", () => {
  const job = plain(jobs.newJob(created("j1"), URL_A, Date.parse("2026-11-02T12:00:00Z")));
  assert.deepEqual(job, {
    job_id: "j1",
    video_id: "aaaaaaaaaaa",
    url: URL_A,
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
  });
});

test("jobs are kept newest first, once each, up to the limit", () => {
  let list = [];
  for (const id of ["j1", "j2", "j3"]) list = jobs.addJob(list, record(id), 2);
  assert.deepEqual(plain(list).map((j) => j.job_id), ["j3", "j2"]);
  list = jobs.addJob(list, record("j2"), 2); // an Idempotency-Key replay returns the same job
  assert.deepEqual(plain(list).map((j) => j.job_id), ["j2", "j3"]);
});

test("a status body updates only the matching job and only known fields", () => {
  const list = [record("j1"), record("j2")];
  const body = { job_id: "j2", status: "SUMMARIZING", progress: 31, video_title: "Raft", extra: "ignored" };
  const [first, second] = plain(jobs.applyStatus(list, "j2", body));
  assert.equal(first.status, "QUEUED");
  assert.equal(second.status, "SUMMARIZING");
  assert.equal(second.progress, 31);
  assert.equal(second.video_title, "Raft");
  assert.equal("extra" in second, false);
  assert.equal(second.url, URL_A); // fields the gateway doesn't send are kept
});

test("a job the server no longer has is marked gone and is not refreshed again", () => {
  const list = jobs.markGone([record("j1", { waiting: { until: "x", reason: "RATE_LIMITED" } })], "j1");
  assert.equal(list[0].gone, true);
  assert.equal(list[0].waiting, null);
  assert.equal(jobs.needsRefresh(list[0]), false);
});

test("only running jobs count on the badge and survive clear-finished", () => {
  const list = [
    record("a", { status: "TRANSCRIBING" }),
    record("b", { status: "COMPLETED" }),
    record("c", { status: "PUBLISHING" }),
    record("d", { status: "DEAD_LETTERED" }),
    record("e", { status: "QUEUED", gone: true }),
  ];
  assert.equal(jobs.badgeText(list), "2");
  assert.deepEqual(plain(jobs.clearFinished(list)).map((j) => j.job_id), ["a", "c"]);
  assert.equal(jobs.badgeText([record("b", { status: "FAILED" })]), "");
});
