// Extension settings in one place (ARCHITECTURE §11). The gateway URL can be changed on the
// options page; everything else is fixed here. Classic script: loaded by every extension context.
(() => {
  "use strict";

  globalThis.VNA_CONFIG = Object.freeze({
    // Default gateway (GATEWAY_BASE_URL in .env). host_permissions cover localhost/127.0.0.1 on any port.
    defaultGatewayUrl: "http://localhost:8000",
    // The popup asks the background to refresh job status this often while it is open.
    popupPollMs: 2000,
    // Badge refresh for active jobs while the popup is closed (chrome.alarms period).
    alarmPeriodMinutes: 0.5,
    // Recent jobs kept in chrome.storage.local, newest first.
    recentJobsMax: 20,
    // The Take Notes button stays disabled this long after a click.
    clickDebounceMs: 3000,
    // How long the on-page confirmation stays visible.
    toastMs: 5000,
    // Gateway requests give up after this long.
    requestTimeoutMs: 10000,
    // Player toolbar, most specific first (YouTube changes its DOM; the first match wins).
    toolbarSelectors: ["#movie_player .ytp-right-controls", ".ytp-right-controls"],
  });
})();
