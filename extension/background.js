// Background service worker (ARCHITECTURE §11, F-8): submits jobs to the gateway and relays their
// status to the popup by refreshing it into chrome.storage.local, which the popup renders. MV3
// stops idle workers, so no state lives in memory between events; listeners register at top level.
importScripts("config.js", "lib/status.js", "lib/jobs.js", "lib/api.js");

const CONFIG = globalThis.VNA_CONFIG;
const JOBS = globalThis.VNA_JOBS;
const { errorMessage } = globalThis.VNA_STATUS;
const { createClient, normalizeGatewayUrl } = globalThis.VNA_API;

const ALARM = "vna-refresh";
const BADGE_COLOR = "#2563eb";

async function settings() {
  const stored = await chrome.storage.local.get(["gatewayUrl", "notionToken", "notionParentId"]);
  return {
    gatewayUrl: stored.gatewayUrl || CONFIG.defaultGatewayUrl,
    notionToken: stored.notionToken || "",
    notionParentId: stored.notionParentId || "",
  };
}

function gateway(baseUrl) {
  return createClient({ baseUrl, timeoutMs: CONFIG.requestTimeoutMs });
}

function failure(code) {
  return { ok: false, code, message: errorMessage(code) };
}

// Writes to the jobs list are chained, so a refresh never overwrites a job submitted meanwhile.
let writes = Promise.resolve();
function updateJobs(mutate) {
  const run = writes.then(async () => {
    const { jobs = [] } = await chrome.storage.local.get("jobs");
    const next = mutate(jobs);
    await chrome.storage.local.set({ jobs: next });
    await syncBadgeAndAlarm(next);
    return next;
  });
  writes = run.catch(() => {});
  return run;
}

// The badge counts running jobs; the alarm keeps it fresh only while there are some.
async function syncBadgeAndAlarm(jobs) {
  await chrome.action.setBadgeBackgroundColor({ color: BADGE_COLOR });
  await chrome.action.setBadgeText({ text: JOBS.badgeText(jobs) });
  const active = jobs.some(JOBS.needsRefresh);
  const alarm = await chrome.alarms.get(ALARM);
  if (active && !alarm) await chrome.alarms.create(ALARM, { periodInMinutes: CONFIG.alarmPeriodMinutes });
  if (!active && alarm) await chrome.alarms.clear(ALARM);
}

/** SUBMIT from the content script: POST /jobs with the stored token, then track the job. */
async function submit({ url }) {
  const { gatewayUrl, notionToken, notionParentId } = await settings();
  if (!notionToken) {
    await chrome.runtime.openOptionsPage();
    return failure("NO_TOKEN");
  }
  let created;
  try {
    created = await gateway(gatewayUrl).submitJob({
      youtubeUrl: url,
      notionToken,
      notionParentId,
      idempotencyKey: crypto.randomUUID(),
    });
  } catch (error) {
    return failure(error.code ?? "GATEWAY_ERROR");
  }
  await updateJobs((jobs) => JOBS.addJob(jobs, JOBS.newJob(created, url, Date.now()), CONFIG.recentJobsMax));
  return { ok: true, jobId: created.job_id, message: "Queued! Open Video Notes in the toolbar to follow it." };
}

// The popup timer and the alarm can ask at the same time; they share one round of requests.
let refreshing = null;
function refresh() {
  refreshing ??= refreshOnce().finally(() => {
    refreshing = null;
  });
  return refreshing;
}

/** GET /jobs/{id} for every running job; store the answers and whether the gateway answered. */
async function refreshOnce() {
  const { gatewayUrl } = await settings();
  const { jobs = [] } = await chrome.storage.local.get("jobs");
  const active = jobs.filter(JOBS.needsRefresh);
  if (active.length === 0) {
    await chrome.storage.local.remove("connection");
    await syncBadgeAndAlarm(jobs);
    return { ok: true };
  }
  const api = gateway(gatewayUrl);
  const results = await Promise.allSettled(active.map((job) => api.getJob(job.job_id)));
  await updateJobs((current) =>
    results.reduce((list, result, i) => {
      const jobId = active[i].job_id;
      if (result.status === "fulfilled") return JOBS.applyStatus(list, jobId, result.value);
      if (result.reason?.code === "JOB_NOT_FOUND") return JOBS.markGone(list, jobId);
      return list; // keep the last known status; the banner says the server is unreachable
    }, current),
  );
  const reachable = results.some((r) => r.status === "fulfilled" || r.reason?.code !== "GATEWAY_UNREACHABLE");
  await chrome.storage.local.set({ connection: { ok: reachable, gatewayUrl } });
  return { ok: reachable };
}

async function clearFinished() {
  await updateJobs(JOBS.clearFinished);
  return { ok: true };
}

/** HEALTH from the options page: GET /healthz on the URL typed there (saved or not). */
async function health({ gatewayUrl }) {
  const { url, error } = normalizeGatewayUrl(gatewayUrl);
  if (error) return { ok: false, code: "INVALID_GATEWAY_URL", message: error };
  try {
    return { ok: true, health: await gateway(url).health() };
  } catch (err) {
    return failure(err.code ?? "GATEWAY_ERROR");
  }
}

const HANDLERS = { SUBMIT: submit, REFRESH: refresh, CLEAR_FINISHED: clearFinished, HEALTH: health };

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const handler = sender.id === chrome.runtime.id ? HANDLERS[message?.type] : undefined;
  if (!handler) return false;
  handler(message).then(sendResponse, () => sendResponse(failure("EXTENSION_ERROR")));
  return true; // the response is sent asynchronously
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === ALARM) refresh();
});
chrome.runtime.onInstalled.addListener(() => refresh());
chrome.runtime.onStartup.addListener(() => refresh());
