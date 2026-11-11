// Options page (ARCHITECTURE §11): gateway URL, the Notion connection (OAuth, F-7) and an optional
// parent page, in chrome.storage.local. Connecting runs in the background worker, the same code
// path as the first Take Notes click; the token is never shown here.
(() => {
  "use strict";

  const CONFIG = globalThis.VNA_CONFIG;
  const { normalizeGatewayUrl } = globalThis.VNA_API;
  const $ = (id) => document.getElementById(id);

  function say(message, kind = "info") {
    $("status").textContent = message;
    $("status").dataset.kind = kind;
  }

  async function showNotionState() {
    const { notionToken, notionWorkspace } = await chrome.storage.local.get(["notionToken", "notionWorkspace"]);
    $("notion-state").textContent = notionToken
      ? `Connected${notionWorkspace ? ` to ${notionWorkspace}` : ""}.`
      : "Not connected. Take Notes will ask you to connect on the first click.";
    $("connect").textContent = notionToken ? "Reconnect" : "Connect Notion";
    $("disconnect").disabled = !notionToken;
  }

  async function load() {
    const stored = await chrome.storage.local.get(["gatewayUrl", "notionParentId"]);
    $("gateway").value = stored.gatewayUrl || CONFIG.defaultGatewayUrl;
    $("parent").value = stored.notionParentId || "";
    await showNotionState();
  }

  /** Saves the gateway URL and parent page; false (with a message) if the URL is invalid. */
  async function saveFields() {
    const { url, error } = normalizeGatewayUrl($("gateway").value);
    if (error) {
      say(error, "error");
      return false;
    }
    await chrome.storage.local.set({ gatewayUrl: url, notionParentId: $("parent").value.trim() });
    $("gateway").value = url;
    return true;
  }

  async function ask(message) {
    try {
      return await chrome.runtime.sendMessage(message);
    } catch {
      return { ok: false, message: "The extension's background worker did not answer. Reload the extension." };
    }
  }

  async function save(event) {
    event.preventDefault();
    if (await saveFields()) say("Saved.", "ok");
  }

  async function connect() {
    if (!(await saveFields())) return; // the connection goes through the gateway typed here
    say("Waiting for Notion…");
    const response = await ask({ type: "CONNECT_NOTION" });
    await showNotionState();
    if (response?.ok) say(`Connected${response.workspace ? ` to ${response.workspace}` : ""}.`, "ok");
    else say(response?.message ?? "Something went wrong.", "error");
  }

  async function disconnect() {
    await ask({ type: "DISCONNECT_NOTION" });
    await showNotionState();
    say("Disconnected. The next Take Notes click asks you to connect again.", "ok");
  }

  async function test() {
    say("Checking…");
    const response = await ask({ type: "HEALTH", gatewayUrl: $("gateway").value });
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

  $("form").addEventListener("submit", save);
  $("test").addEventListener("click", test);
  $("connect").addEventListener("click", connect);
  $("disconnect").addEventListener("click", disconnect);
  load();
})();
