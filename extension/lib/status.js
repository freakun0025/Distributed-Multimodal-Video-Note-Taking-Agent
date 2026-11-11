// Job status → plain words for the popup and the on-page toast (F-9, ARCHITECTURE §11).
// Pure: no chrome.* calls, so tests/extension/ runs it under Node. Raw JSON is never shown.
(() => {
  "use strict";

  // Spec §4 status names, in pipeline order.
  const ACTIVE_STATUSES = ["QUEUED", "TRANSCRIBING", "SUMMARIZING", "EXTRACTING_FRAMES", "PUBLISHING"];
  const SUCCESS_STATUSES = ["COMPLETED", "COMPLETED_WITH_WARNINGS"];
  const ERROR_STATUSES = ["FAILED", "DEAD_LETTERED"];
  const TERMINAL_STATUSES = [...SUCCESS_STATUSES, ...ERROR_STATUSES];

  // The stepper the spec asks for: Queued → … → Publishing → Done.
  const STEPS = ["Queued", "Transcribing", "Summarizing", "Extracting Frames", "Publishing", "Done"];

  // Progress at which each stage starts (common/state.py PROGRESS_RANGE). A failed job keeps its
  // last progress, so this shows which step it stopped at.
  const STAGE_START_PROGRESS = [
    ["TRANSCRIBING", 5],
    ["SUMMARIZING", 20],
    ["EXTRACTING_FRAMES", 50],
    ["PUBLISHING", 80],
  ];

  const HEADLINES = {
    QUEUED: "Queued — waiting for a worker",
    TRANSCRIBING: "Transcribing…",
    SUMMARIZING: "Summarizing…",
    EXTRACTING_FRAMES: "Extracting frames…",
    PUBLISHING: "Publishing to Notion…",
    COMPLETED: "Done",
    COMPLETED_WITH_WARNINGS: "Done, with warnings",
    FAILED: "Failed",
    DEAD_LETTERED: "Stopped after repeated failures",
  };

  // One sentence per error code (common/errors.py ErrorCode), plus the extension's own codes.
  const ERROR_MESSAGES = {
    RATE_LIMITED: "An external service kept rate-limiting us, so we stopped retrying.",
    RATE_LIMITED_LOCAL: "Our shared rate limit stayed full for too long.",
    LIMITER_UNAVAILABLE: "The rate-limit service was down, so the job could not continue safely.",
    UPSTREAM_ERROR: "An external service kept failing with server errors.",
    UPSTREAM_TIMEOUT: "An external service kept timing out.",
    CONNECTION_ERROR: "We could not connect to an external service.",
    VIDEO_UNAVAILABLE: "This video is private, removed or otherwise unavailable.",
    STT_UNAVAILABLE: "This video has no captions and speech-to-text was unavailable.",
    INTERNAL_ERROR: "Something went wrong on our side.",
    INVALID_URL: "That is not a YouTube video link.",
    MISSING_NOTION_TOKEN: "Connect Notion first: open the extension settings and click Connect Notion.",
    VIDEO_TOO_LONG: "This video is longer than 20 minutes, so it can't be turned into notes.",
    VIDEO_UNSUPPORTED: "This kind of video (for example a live stream) isn't supported.",
    NO_TRANSCRIPT: "This video has no transcript we could use.",
    NOTION_UNAUTHORIZED: "Notion rejected our access. Reconnect Notion and try again.",
    NOTION_PARENT_NOT_FOUND: "The Notion page to publish under wasn't found or isn't shared with the integration.",
    LLM_INVALID_OUTPUT: "The AI model returned notes we couldn't use.",
    UPSTREAM_REJECTED: "An external service rejected the request.",
    TOKEN_DECRYPT_FAILED: "The job's Notion credentials couldn't be read. Please try again.",
    ENQUEUE_FAILED: "The job couldn't be started. Please try again.",
    OAUTH_CODE_INVALID: "Connecting to Notion failed. Please try again.",
    BROKER_UNAVAILABLE: "The notes server couldn't queue the job right now. Try again in a moment.",
    VALIDATION_ERROR: "The notes server didn't accept the request.",
    // Extension-side codes.
    NO_TOKEN: "Connect Notion first: open the extension settings and click Connect Notion.",
    OAUTH_CANCELLED: "Notion sign-in was cancelled, so nothing was sent.",
    OAUTH_STATE_MISMATCH: "Notion's answer didn't match our sign-in request, so it was ignored. Try again.",
    OAUTH_FAILED: "Connecting to Notion failed. Please try again.",
    OAUTH_REDIRECT_MISMATCH:
      "This extension's ID doesn't match the server's Notion redirect URI (NOTION_REDIRECT_URI in .env).",
    GATEWAY_UNREACHABLE: "Can't reach the notes server. Is the stack running?",
    GATEWAY_ERROR: "The notes server returned an unexpected response.",
    JOB_GONE: "This job no longer exists on the notes server.",
  };
  const FALLBACK_ERROR = "Something went wrong.";

  const API_NAMES = { youtube: "YouTube", llm: "the AI model", notion: "Notion" };

  function isTerminal(status) {
    return TERMINAL_STATUSES.includes(status);
  }

  function errorMessage(code) {
    return ERROR_MESSAGES[code] ?? FALLBACK_ERROR;
  }

  /** Index into STEPS of the step a failed job stopped at, from its last progress. */
  function failedStepIndex(progress) {
    let index = 0;
    STAGE_START_PROGRESS.forEach(([status, start]) => {
      if ((progress ?? 0) >= start) index = ACTIVE_STATUSES.indexOf(status);
    });
    return index;
  }

  function stepStates(job) {
    const states = STEPS.map(() => "todo");
    const fill = (until, last) => {
      for (let i = 0; i < until; i += 1) states[i] = "done";
      states[until] = last;
    };
    if (job.gone) return states;
    if (SUCCESS_STATUSES.includes(job.status)) return states.map(() => "done");
    if (ERROR_STATUSES.includes(job.status)) {
      fill(failedStepIndex(job.progress), "failed");
      return states;
    }
    const index = ACTIVE_STATUSES.indexOf(job.status);
    fill(Math.max(index, 0), "current");
    return states;
  }

  /** "Waiting — Notion is rate-limiting us, retrying in 12 s" while a retry is scheduled (F-18). */
  function waitingText(waiting, nowMs) {
    if (!waiting) return null;
    const [code, api] = String(waiting.reason ?? "").split(":");
    const who = API_NAMES[api] ?? "an external service";
    const why = code.startsWith("RATE_LIMITED") ? "is rate-limiting us" : "had a temporary problem";
    const seconds = Math.ceil((Date.parse(waiting.until) - nowMs) / 1000);
    const when = Number.isFinite(seconds) && seconds > 0 ? `retrying in ${seconds} s` : "retrying now";
    return `Waiting — ${who} ${why}, ${when}.`;
  }

  function clock(totalSeconds) {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = String(totalSeconds % 60).padStart(2, "0");
    return `${minutes}:${seconds}`;
  }

  /** "2 of 6 frames could not be captured (at 7:41, 12:03)." */
  function warningText(warnings, result) {
    const frames = (warnings ?? []).filter((w) => w.code === "FRAME_EXTRACTION_FAILED");
    const other = (warnings ?? []).length - frames.length;
    const parts = [];
    if (frames.length > 0) {
      const total = result?.frames_requested;
      const noun = (total ?? frames.length) === 1 ? "frame" : "frames";
      const count = total ? `${frames.length} of ${total} ${noun}` : `${frames.length} ${noun}`;
      const times = frames
        .filter((w) => Number.isInteger(w.timestamp_sec))
        .map((w) => clock(w.timestamp_sec));
      parts.push(`${count} could not be captured${times.length ? ` (at ${times.join(", ")})` : ""}.`);
    }
    if (other > 0) parts.push(other === 1 ? "1 other issue was noted." : `${other} other issues were noted.`);
    return parts.length ? parts.join(" ") : null;
  }

  /** Only an absolute https URL becomes the View in Notion link. */
  function safeLink(url) {
    try {
      return new URL(url).protocol === "https:" ? url : null;
    } catch {
      return null;
    }
  }

  function tone(job) {
    if (job.gone || ERROR_STATUSES.includes(job.status)) return "error";
    if (job.status === "COMPLETED_WITH_WARNINGS") return "warning";
    if (job.status === "COMPLETED") return "success";
    return job.waiting ? "waiting" : "active";
  }

  /** Everything the popup renders for one stored job, as plain strings and flags. */
  function describeJob(job, nowMs) {
    const terminal = isTerminal(job.status) || Boolean(job.gone);
    const errorCode = job.gone ? "JOB_GONE" : job.error?.code;
    const states = stepStates(job);
    return {
      title: job.video_title || `YouTube video ${job.video_id}`,
      headline: job.gone ? "Not found" : HEADLINES[job.status] ?? job.status,
      tone: tone(job),
      terminal,
      progress: SUCCESS_STATUSES.includes(job.status) ? 100 : Math.max(0, Math.min(100, job.progress ?? 0)),
      steps: STEPS.map((label, i) => ({ label, state: states[i] })),
      waitingText: terminal ? null : waitingText(job.waiting, nowMs),
      warningText: warningText(job.warnings, job.result),
      errorText: errorCode ? errorMessage(errorCode) : null,
      errorCode: errorCode ?? null,
      notionUrl: safeLink(job.result?.notion_page_url),
    };
  }

  globalThis.VNA_STATUS = Object.freeze({
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    STEPS,
    STAGE_START_PROGRESS,
    ERROR_MESSAGES,
    isTerminal,
    errorMessage,
    waitingText,
    warningText,
    describeJob,
  });
})();
