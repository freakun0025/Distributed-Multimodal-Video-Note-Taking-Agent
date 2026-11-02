// Gateway REST client (Spec §5, ARCHITECTURE §10). Every failure becomes a GatewayError with a
// code that lib/status.js turns into a sentence. `fetchImpl` is injectable for the Node tests.
(() => {
  "use strict";

  class GatewayError extends Error {
    constructor(code, httpStatus = null) {
      super(code);
      this.name = "GatewayError";
      this.code = code;
      this.httpStatus = httpStatus;
    }
  }

  // Must match manifest.json host_permissions (http, any port), or the browser blocks the request.
  const ALLOWED_HOSTS = new Set(["localhost", "127.0.0.1"]);

  /** Validates a gateway URL typed on the options page: `{url}` (an origin) or `{error}`. */
  function normalizeGatewayUrl(raw) {
    let url;
    try {
      url = new URL(String(raw ?? "").trim());
    } catch {
      return { error: "Enter a full URL such as http://localhost:8000." };
    }
    if (url.protocol !== "http:") {
      return { error: "The URL must start with http://." };
    }
    if (!ALLOWED_HOSTS.has(url.hostname)) {
      return { error: "Only localhost or 127.0.0.1 is allowed (the extension's host permissions)." };
    }
    return { url: url.origin };
  }

  function createClient({ baseUrl, timeoutMs, fetchImpl }) {
    const doFetch = fetchImpl ?? ((...args) => globalThis.fetch(...args));

    async function call(path, { method = "GET", body, headers = {}, okStatuses = [200] } = {}) {
      let response;
      try {
        response = await doFetch(`${baseUrl}${path}`, {
          method,
          headers: body === undefined ? headers : { "Content-Type": "application/json", ...headers },
          body: body === undefined ? undefined : JSON.stringify(body),
          cache: "no-store",
          signal: AbortSignal.timeout(timeoutMs),
        });
      } catch {
        throw new GatewayError("GATEWAY_UNREACHABLE");
      }
      const data = await response.json().catch(() => null);
      if (okStatuses.includes(response.status) && data !== null) return data;
      throw new GatewayError(data?.error?.code ?? "GATEWAY_ERROR", response.status);
    }

    return {
      /** POST /jobs → `{job_id, status, video_id}`. One Idempotency-Key per click. */
      submitJob({ youtubeUrl, notionToken, notionParentId, idempotencyKey }) {
        const body = { youtube_url: youtubeUrl, notion_token: notionToken };
        if (notionParentId) body.notion_parent_id = notionParentId;
        return call("/jobs", {
          method: "POST",
          headers: { "Idempotency-Key": idempotencyKey },
          body,
          okStatuses: [202],
        });
      },
      /** GET /jobs/{id} → the status body; 404 throws GatewayError("JOB_NOT_FOUND"). */
      getJob(jobId) {
        return call(`/jobs/${encodeURIComponent(jobId)}`);
      },
      /** GET /healthz → `{gateway, postgres, rabbitmq, redis}`; a 503 still carries the body. */
      health() {
        return call("/healthz", { okStatuses: [200, 503] });
      },
    };
  }

  globalThis.VNA_API = Object.freeze({ GatewayError, normalizeGatewayUrl, createClient });
})();
