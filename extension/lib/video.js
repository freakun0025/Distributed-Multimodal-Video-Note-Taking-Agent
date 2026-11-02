// Video ID of a YouTube watch page (F-5). Pure; the content script calls it at click time.
(() => {
  "use strict";

  // Same rule as common/youtube_url.py.
  const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
  const HOSTS = new Set(["www.youtube.com", "youtube.com", "m.youtube.com"]);

  /** The `v` parameter of a youtube.com/watch URL, or null for any other page. */
  function videoIdFromUrl(href) {
    let url;
    try {
      url = new URL(href);
    } catch {
      return null;
    }
    if (!HOSTS.has(url.hostname) || url.pathname !== "/watch") return null;
    const id = url.searchParams.get("v");
    return id && VIDEO_ID.test(id) ? id : null;
  }

  /** Canonical URL sent to the gateway: no playlist, timestamp or tracking parameters. */
  function watchUrl(videoId) {
    return `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}`;
  }

  globalThis.VNA_VIDEO = Object.freeze({ videoIdFromUrl, watchUrl });
})();
