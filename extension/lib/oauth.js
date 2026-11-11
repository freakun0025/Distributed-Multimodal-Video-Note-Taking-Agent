// Notion OAuth helpers (ARCHITECTURE §11, F-7): the authorize URL and reading the redirect Notion
// sends back. Pure functions, so the Node tests run them; background.js does the chrome.identity
// call. Classic script: sets globalThis.VNA_OAUTH.
(() => {
  "use strict";

  /** Notion's consent page for this extension, with a fresh `state` to match on the way back. */
  function authorizeUrl({ base, clientId, redirectUri, state }) {
    const url = new URL(base);
    const params = { client_id: clientId, response_type: "code", owner: "user", redirect_uri: redirectUri, state };
    for (const [name, value] of Object.entries(params)) url.searchParams.set(name, value);
    return url.toString();
  }

  /**
   * The URL Notion redirected to → `{code}`, or `{error}` with an extension error code:
   * OAUTH_CANCELLED (the user pressed Cancel), OAUTH_STATE_MISMATCH (not the answer to our
   * request, so it is ignored), OAUTH_FAILED (anything else).
   */
  function readRedirect(redirectedTo, expectedState) {
    let params;
    try {
      params = new URL(redirectedTo).searchParams;
    } catch {
      return { error: "OAUTH_FAILED" };
    }
    if (params.get("state") !== expectedState) return { error: "OAUTH_STATE_MISMATCH" };
    const error = params.get("error");
    if (error) return { error: error === "access_denied" ? "OAUTH_CANCELLED" : "OAUTH_FAILED" };
    const code = params.get("code");
    return code ? { code } : { error: "OAUTH_FAILED" };
  }

  globalThis.VNA_OAUTH = Object.freeze({ authorizeUrl, readRedirect });
})();
