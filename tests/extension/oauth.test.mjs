// lib/oauth.js: Notion's consent URL and reading the redirect (ARCHITECTURE §11, F-7).
import assert from "node:assert/strict";
import { test } from "node:test";

import { loadExtensionScripts, plain } from "./helpers.mjs";

const { VNA_OAUTH: oauth } = loadExtensionScripts(["lib/oauth.js"]);
const REDIRECT = "https://hiimaheehbfpmgnjjpeoejngcmhjmjfi.chromiumapp.org/notion";

test("the consent URL carries the client, our redirect, owner=user and the state", () => {
  const url = new URL(
    oauth.authorizeUrl({
      base: "https://api.notion.com/v1/oauth/authorize",
      clientId: "client-1",
      redirectUri: REDIRECT,
      state: "s-1",
    }),
  );
  assert.equal(`${url.origin}${url.pathname}`, "https://api.notion.com/v1/oauth/authorize");
  assert.deepEqual(Object.fromEntries(url.searchParams), {
    client_id: "client-1",
    response_type: "code",
    owner: "user",
    redirect_uri: REDIRECT,
    state: "s-1",
  });
});

test("the redirect gives the code only when the state is ours", () => {
  assert.deepEqual(plain(oauth.readRedirect(`${REDIRECT}?code=abc&state=s-1`, "s-1")), { code: "abc" });
  assert.deepEqual(plain(oauth.readRedirect(`${REDIRECT}?code=abc&state=other`, "s-1")), {
    error: "OAUTH_STATE_MISMATCH",
  });
  assert.deepEqual(plain(oauth.readRedirect(`${REDIRECT}?code=abc`, "s-1")), { error: "OAUTH_STATE_MISMATCH" });
});

test("cancel, other errors and garbage are told apart", () => {
  assert.deepEqual(plain(oauth.readRedirect(`${REDIRECT}?error=access_denied&state=s-1`, "s-1")), {
    error: "OAUTH_CANCELLED",
  });
  assert.deepEqual(plain(oauth.readRedirect(`${REDIRECT}?error=server_error&state=s-1`, "s-1")), {
    error: "OAUTH_FAILED",
  });
  assert.deepEqual(plain(oauth.readRedirect(`${REDIRECT}?state=s-1`, "s-1")), { error: "OAUTH_FAILED" });
  assert.deepEqual(plain(oauth.readRedirect("not a url", "s-1")), { error: "OAUTH_FAILED" });
});
