// The recent-jobs list kept in chrome.storage.local (ARCHITECTURE §11). Pure functions that
// return a new list; only the background service worker writes the list. Needs lib/status.js.
(() => {
  "use strict";

  const { isTerminal } = globalThis.VNA_STATUS;

  // Fields copied from GET /jobs/{id} (Spec §5 plus our additive `waiting`).
  const STATUS_FIELDS = ["status", "progress", "video_title", "duration_sec", "warnings", "error", "result", "waiting"];

  /** The stored record for a job the gateway just accepted (202 body of POST /jobs). */
  function newJob(created, url, nowMs) {
    return {
      job_id: created.job_id,
      video_id: created.video_id,
      url,
      submitted_at: new Date(nowMs).toISOString(),
      status: created.status,
      progress: 0,
      video_title: null,
      duration_sec: null,
      warnings: [],
      error: null,
      result: null,
      waiting: null,
      gone: false,
    };
  }

  /** Newest first, one entry per job_id, at most `max` entries. */
  function addJob(jobs, job, max) {
    return [job, ...jobs.filter((j) => j.job_id !== job.job_id)].slice(0, max);
  }

  function applyStatus(jobs, jobId, body) {
    return jobs.map((job) => {
      if (job.job_id !== jobId) return job;
      const update = {};
      for (const field of STATUS_FIELDS) if (field in body) update[field] = body[field];
      return { ...job, ...update };
    });
  }

  /** The gateway answered 404: the row was deleted (e.g. a database reset). Stop polling it. */
  function markGone(jobs, jobId) {
    return jobs.map((job) => (job.job_id === jobId ? { ...job, gone: true, waiting: null } : job));
  }

  function needsRefresh(job) {
    return !job.gone && !isTerminal(job.status);
  }

  function clearFinished(jobs) {
    return jobs.filter(needsRefresh);
  }

  /** Number of jobs still running, or "" when none (an empty badge is hidden). */
  function badgeText(jobs) {
    const active = jobs.filter(needsRefresh).length;
    return active > 0 ? String(active) : "";
  }

  globalThis.VNA_JOBS = Object.freeze({
    newJob,
    addJob,
    applyStatus,
    markGone,
    needsRefresh,
    clearFinished,
    badgeText,
  });
})();
