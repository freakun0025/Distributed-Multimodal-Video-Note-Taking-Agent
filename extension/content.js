// Content script (ARCHITECTURE §11): a Take Notes button in the YouTube player toolbar (F-5) that
// survives SPA navigation between videos (F-6). The video ID is read from the URL at click time.
// Classic script after config.js and lib/video.js (manifest.json content_scripts).
(() => {
  "use strict";

  const CONFIG = globalThis.VNA_CONFIG;
  const { videoIdFromUrl, watchUrl } = globalThis.VNA_VIDEO;
  const BUTTON_ID = "vna-take-notes";
  const TOAST_ID = "vna-toast";
  const OBSERVER_THROTTLE_MS = 300;

  function toolbar() {
    for (const selector of CONFIG.toolbarSelectors) {
      const element = document.querySelector(selector);
      if (element) return element;
    }
    return null;
  }

  function createButton() {
    const button = document.createElement("button");
    button.id = BUTTON_ID;
    button.type = "button";
    button.title = "Turn this video into Notion notes";
    button.setAttribute("aria-label", "Take Notes");
    const label = document.createElement("span");
    label.className = "vna-pill";
    label.textContent = "Take Notes";
    button.append(label);
    button.addEventListener("click", onClick);
    return button;
  }

  /** Idempotent: run on every navigation and DOM change; adds, moves or removes the button. */
  function ensureButton() {
    const existing = document.getElementById(BUTTON_ID);
    if (!videoIdFromUrl(location.href)) {
      existing?.remove();
      return;
    }
    const bar = toolbar();
    if (!bar || (existing && bar.contains(existing))) return;
    existing?.remove();
    bar.prepend(createButton());
  }

  function toast(message, kind) {
    document.getElementById(TOAST_ID)?.remove();
    const element = document.createElement("div");
    element.id = TOAST_ID;
    element.className = `vna-toast vna-toast-${kind}`;
    element.setAttribute("role", "status");
    element.textContent = message;
    (document.getElementById("movie_player") ?? document.body).append(element);
    setTimeout(() => element.remove(), CONFIG.toastMs);
  }

  async function onClick(event) {
    event.preventDefault();
    event.stopPropagation(); // keep the click away from the player (play/pause)
    const button = event.currentTarget;
    if (button.disabled) return;
    const videoId = videoIdFromUrl(location.href); // now, not at injection: SPA navigation (F-6)
    if (!videoId) {
      toast("Open a YouTube video first.", "error");
      return;
    }
    button.disabled = true;
    setTimeout(() => {
      button.disabled = false;
    }, CONFIG.clickDebounceMs);
    let response;
    try {
      response = await chrome.runtime.sendMessage({ type: "SUBMIT", url: watchUrl(videoId) });
    } catch {
      // The extension was reloaded or updated; this page still runs the old content script.
      toast("Video Notes was updated. Reload this page and try again.", "error");
      return;
    }
    toast(response?.message ?? "Something went wrong.", response?.ok ? "ok" : "error");
  }

  // YouTube fires yt-navigate-finish after each in-app navigation; the observer covers the
  // player toolbar being rebuilt and any navigation the event misses.
  document.addEventListener("yt-navigate-finish", ensureButton);
  let pending = false;
  new MutationObserver(() => {
    if (pending) return;
    pending = true;
    setTimeout(() => {
      pending = false;
      ensureButton();
    }, OBSERVER_THROTTLE_MS);
  }).observe(document.documentElement, { childList: true, subtree: true });
  ensureButton();
})();
