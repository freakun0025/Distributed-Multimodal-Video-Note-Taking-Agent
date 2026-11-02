// Popup (F-9, ARCHITECTURE §11): recent jobs with a stage stepper, waiting/warnings/errors in plain
// words and View in Notion. It renders chrome.storage.local and asks the background to refresh
// every CONFIG.popupPollMs while open. All text goes through textContent; JSON is never shown.
(() => {
  "use strict";

  const CONFIG = globalThis.VNA_CONFIG;
  const { describeJob, errorMessage } = globalThis.VNA_STATUS;
  const RENDER_TICK_MS = 1000; // keeps "retrying in N s" counting down between refreshes

  const state = { jobs: [], hasToken: true, connection: null };
  let rendered = null; // what is on screen; unchanged views are not rebuilt (a click isn't lost)
  const $ = (id) => document.getElementById(id);

  function renderJob(job, view) {
    const item = $("job-template").content.firstElementChild.cloneNode(true);
    item.dataset.tone = view.tone;

    const title = item.querySelector(".job-title");
    title.textContent = view.title;
    title.href = job.url;
    item.querySelector(".job-headline").textContent = view.headline;

    const steps = item.querySelector(".steps");
    for (const step of view.steps) {
      const li = document.createElement("li");
      li.className = `step step-${step.state}`;
      li.textContent = step.label;
      if (step.state === "current") li.setAttribute("aria-current", "step");
      steps.append(li);
    }

    const bar = item.querySelector(".progress");
    bar.setAttribute("aria-valuenow", String(view.progress));
    bar.querySelector(".progress-fill").style.width = `${view.progress}%`;

    show(item.querySelector(".note-waiting"), view.waitingText);
    show(item.querySelector(".note-warning"), view.warningText);
    const error = item.querySelector(".note-error");
    show(error, view.errorText);
    if (view.errorCode) error.title = view.errorCode;

    const notion = item.querySelector(".notion");
    if (view.notionUrl) {
      notion.href = view.notionUrl;
      notion.hidden = false;
    }
    return item;
  }

  function show(element, text) {
    element.textContent = text ?? "";
    element.hidden = !text;
  }

  function render() {
    const now = Date.now();
    const views = state.jobs.map((job) => describeJob(job, now));
    let banner = null;
    if (!state.hasToken) banner = errorMessage("NO_TOKEN");
    else if (state.connection && !state.connection.ok) {
      banner = `${errorMessage("GATEWAY_UNREACHABLE")} (${state.connection.gatewayUrl})`;
    }
    const key = JSON.stringify([views, banner, state.jobs.map((job) => job.url)]);
    if (key === rendered) return;
    rendered = key;

    $("jobs").replaceChildren(...state.jobs.map((job, i) => renderJob(job, views[i])));
    $("empty").hidden = state.jobs.length > 0;
    $("clear").hidden = !views.some((view) => view.terminal);
    show($("banner"), banner);
  }

  function refresh() {
    chrome.runtime.sendMessage({ type: "REFRESH" }).catch(() => {});
  }

  async function load() {
    const stored = await chrome.storage.local.get(["jobs", "notionToken", "connection"]);
    state.jobs = stored.jobs ?? [];
    state.hasToken = Boolean(stored.notionToken);
    state.connection = stored.connection ?? null;
    render();
  }

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local") return;
    if (changes.jobs) state.jobs = changes.jobs.newValue ?? [];
    if (changes.notionToken) state.hasToken = Boolean(changes.notionToken.newValue);
    if (changes.connection) state.connection = changes.connection.newValue ?? null;
    render();
  });

  $("settings").addEventListener("click", () => chrome.runtime.openOptionsPage());
  $("clear").addEventListener("click", () => {
    chrome.runtime.sendMessage({ type: "CLEAR_FINISHED" }).catch(() => {});
  });

  load();
  refresh();
  setInterval(refresh, CONFIG.popupPollMs);
  setInterval(render, RENDER_TICK_MS);
})();
