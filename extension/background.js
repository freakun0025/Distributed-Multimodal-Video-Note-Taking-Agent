// Background service worker (ARCHITECTURE §11, F-7, F-8): connects Notion (OAuth), submits jobs to
// the gateway and relays their status to the popup by refreshing it into chrome.storage.local,
// which the popup renders. MV3 stops idle workers, so no state lives in memory between events;
// listeners register at top level.
importScripts("config.js", "lib/status.js", "lib/jobs.js", "lib/api.js", "lib/oauth.js");

const CONFIG = globalThis.VNA_CONFIG;
const JOBS = globalThis.VNA_JOBS;
const { errorMessage } = globalThis.VNA_STATUS;
const { createClient, normalizeGatewayUrl } = globalThis.VNA_API;
const { authorizeUrl, readRedirect } = globalThis.VNA_OAUTH;
const NOTION_KEYS = ["notionToken", "notionRefreshToken", "notionWorkspace"];

const ALARM = "vna-refresh";
const BADGE_COLOR = "#2563eb";

async function settings() {
  const stored = await chrome.storage.local.get(["gatewayUrl", "notionParentId", ...NOTION_KEYS]);
  return {
    gatewayUrl: stored.gatewayUrl || CONFIG.defaultGatewayUrl,
    notionToken: stored.notionToken || "",
    notionRefreshToken: stored.notionRefreshToken || "",
    notionParentId: stored.notionParentId || "",
  };
}

async function storeToken(token) {
  await chrome.storage.local.set({
    notionToken: token.access_token,
    notionRefreshToken: token.refresh_token ?? "",
    notionWorkspace: token.workspace_name ?? "",
  });
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

/**
 * CONNECT_NOTION (and the first Take Notes click): Notion's consent page opens in a browser
 * window, the user picks the pages to share, and Notion redirects to
 * https://<extension-id>.chromiumapp.org/notion with a code. The gateway swaps the code for a
 * token, because only it holds the client secret (D10). The token stays in this browser.
 */
async function connectNotion() {
  const api = gateway((await settings()).gatewayUrl);
  let config;
  try {
    config = await api.notionConfig();
  } catch (error) {
    return failure(error.code ?? "GATEWAY_ERROR");
  }
  if (config.redirect_uri !== chrome.identity.getRedirectURL(CONFIG.notionRedirectPath)) {
    return failure("OAUTH_REDIRECT_MISMATCH"); // Notion would refuse to send the code here
  }
  const state = crypto.randomUUID();
  const url = authorizeUrl({
    base: CONFIG.notionAuthorizeUrl,
    clientId: config.client_id,
    redirectUri: config.redirect_uri,
    state,
  });
  let redirectedTo;
  try {
    redirectedTo = await chrome.identity.launchWebAuthFlow({ url, interactive: true });
  } catch {
    return failure("OAUTH_CANCELLED"); // the window was closed
  }
  const { code, error } = readRedirect(redirectedTo, state);
  if (error) return failure(error);
  let token;
  try {
    token = await api.exchangeNotionCode(code);
  } catch (err) {
    return failure(err.code ?? "OAUTH_FAILED");
  }
  await storeToken(token);
  return { ok: true, workspace: token.workspace_name ?? "" };
}

/** DISCONNECT_NOTION: forget the token; the next click signs in again. */
async function disconnectNotion() {
  await chrome.storage.local.remove(NOTION_KEYS);
  return { ok: true };
}

/** A job failed with NOTION_UNAUTHORIZED: try the refresh token once, else forget the token. */
async function renewNotionToken() {
  const { gatewayUrl, notionRefreshToken } = await settings();
  if (notionRefreshToken) {
    try {
      await storeToken(await gateway(gatewayUrl).refreshNotionToken(notionRefreshToken));
      return;
    } catch {
      // fall through: sign in again on the next click
    }
  }
  await chrome.storage.local.remove(NOTION_KEYS);
}

/** SUBMIT from the content script: POST /jobs with the stored token, then track the job. */
async function submit({ url }) {
  if (!(await settings()).notionToken) {
    const connected = await connectNotion();
    if (!connected.ok) return connected;
  }
  const { gatewayUrl, notionToken, notionParentId } = await settings();
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
  const rejected = results.some(
    (r) => r.status === "fulfilled" && r.value?.error?.code === "NOTION_UNAUTHORIZED",
  );
  if (rejected) await renewNotionToken();
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

const HANDLERS = {
  SUBMIT: submit,
  REFRESH: refresh,
  CLEAR_FINISHED: clearFinished,
  HEALTH: health,
  CONNECT_NOTION: connectNotion,
  DISCONNECT_NOTION: disconnectNotion,
};

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
