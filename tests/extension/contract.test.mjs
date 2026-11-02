// The extension mirrors a few backend contracts by hand (plain JS, no build step). These tests read
// the Python source of truth so a change on either side fails here instead of in the popup.
import assert from "node:assert/strict";
import { test } from "node:test";

import { loadExtensionScripts, plain, readRepoFile } from "./helpers.mjs";

const { VNA_STATUS: status } = loadExtensionScripts(["lib/status.js"]);

/** Member values of a `class Name(StrEnum):` block in a Python file. */
function enumValues(path, className) {
  const source = readRepoFile(path);
  const start = source.indexOf(`class ${className}(StrEnum):`);
  assert.ok(start >= 0, `${className} not found in ${path}`);
  const body = source.slice(start).split(/\n(?=\S)/)[0];
  return [...body.matchAll(/^\s+[A-Z_]+ = "([^"]+)"/gm)].map((m) => m[1]);
}

// Codes that only ever appear in REST responses the popup handles itself, never in job.error.
const API_ONLY_CODES = ["JOB_NOT_FOUND", "JOB_NOT_TERMINAL"];

test("every job error code has a plain-words message", () => {
  const codes = enumValues("common/errors.py", "ErrorCode").filter((c) => !API_ONLY_CODES.includes(c));
  assert.ok(codes.length > 10);
  const missing = codes.filter((code) => !(code in status.ERROR_MESSAGES));
  assert.deepEqual(missing, []);
});

test("the extension knows exactly the backend's job statuses, in pipeline order", () => {
  const backend = enumValues("common/state.py", "JobStatus");
  assert.deepEqual([...plain(status.ACTIVE_STATUSES), ...plain(status.TERMINAL_STATUSES)].sort(), [...backend].sort());
  assert.deepEqual(plain(status.ACTIVE_STATUSES), backend.slice(0, 5));
});

test("stage start progress matches PROGRESS_RANGE in common/state.py", () => {
  const source = readRepoFile("common/state.py");
  const starts = Object.fromEntries(
    [...source.matchAll(/JobStatus\.([A-Z_]+): \((\d+), (\d+)\)/g)].map((m) => [m[1], Number(m[2])]),
  );
  for (const [stageStatus, start] of plain(status.STAGE_START_PROGRESS)) {
    assert.equal(start, starts[stageStatus], stageStatus);
  }
});

test("warning codes are the ones the popup words specifically", () => {
  // A new warning code would only show as "1 other issue was noted." — give it its own words.
  assert.deepEqual(enumValues("common/errors.py", "WarningCode"), ["FRAME_EXTRACTION_FAILED"]);
});
