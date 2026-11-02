// Options page (ARCHITECTURE §11): gateway URL, Notion token and optional parent page ID in
// chrome.storage.local. Until the Notion OAuth flow lands (ROADMAP M2) the token is pasted here.
// The saved token is never put back into the page; an empty field keeps it.
(() => {
  "use strict";

  const CONFIG = globalThis.VNA_CONFIG;
  const { normalizeGatewayUrl } = globalThis.VNA_API;
  const $ = (id) => document.getElementById(id);

  function say(message, kind = "info") {
    $("status").textContent = message;
    $("status").dataset.kind = kind;
  }

  function showTokenState(saved) {
    $("token").value = "";
    $("token").placeholder = saved ? "Saved — paste a new one to replace it" : "Paste your token";
    $("forget").disabled = !saved;
  }

  async function load() {
    const stored = await chrome.storage.local.get(["gatewayUrl", "notionToken", "notionParentId"]);
    $("gateway").value = stored.gatewayUrl || CONFIG.defaultGatewayUrl;
    $("parent").value = stored.notionParentId || "";
    showTokenState(Boolean(stored.notionToken));
  }

  async function save(event) {
    event.preventDefault();
    const { url, error } = normalizeGatewayUrl($("gateway").value);
    if (error) {
      say(error, "error");
      return;
    }
    const update = { gatewayUrl: url, notionParentId: $("parent").value.trim() };
    const token = $("token").value.trim();
    if (token) update.notionToken = token;
    await chrome.storage.local.set(update);
    const { notionToken } = await chrome.storage.local.get("notionToken");
    $("gateway").value = url;
    showTokenState(Boolean(notionToken));
    say(notionToken ? "Saved." : "Saved. Add a Notion token before taking notes.", notionToken ? "ok" : "error");
  }

  async function test() {
    say("Checking…");
    let response;
    try {
      response = await chrome.runtime.sendMessage({ type: "HEALTH", gatewayUrl: $("gateway").value });
    } catch {
      say("The extension's background worker did not answer. Reload the extension.", "error");
      return;
    }
    if (!response?.ok) {
      say(response?.message ?? "Something went wrong.", "error");
      return;
    }
    const down = Object.entries(response.health)
      .filter(([, value]) => value !== "ok")
      .map(([name]) => name);
    if (down.length === 0) say("Connected: gateway, database, broker and Redis are all up.", "ok");
    else say(`The server answered, but these are down: ${down.join(", ")}.`, "error");
  }

  async function forget() {
    await chrome.storage.local.remove("notionToken");
    showTokenState(false);
    say("Token removed.", "ok");
  }

  $("form").addEventListener("submit", save);
  $("test").addEventListener("click", test);
  $("forget").addEventListener("click", forget);
  load();
})();
