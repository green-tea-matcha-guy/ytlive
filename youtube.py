"""YouTube Data API helpers: OAuth refresh, broadcast lifecycle and ingest details."""

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

API_ROOT = "https://www.googleapis.com/youtube/v3"
TOKEN_URL = "https://oauth2.googleapis.com/token"


def required(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class OAuthToken:
    """Mints a fresh access token from the refresh token whenever the last one is about to expire."""

    def __init__(self):
        self.access_token = None
        self.expires_at = 0
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            if self.access_token and self.expires_at > time.time() + 60:
                return self.access_token
            request = urllib.request.Request(
                TOKEN_URL,
                data=urllib.parse.urlencode(
                    {
                        "client_id": required("YOUTUBE_CLIENT_ID"),
                        "client_secret": required("YOUTUBE_CLIENT_SECRET"),
                        "refresh_token": required("YOUTUBE_REFRESH_TOKEN"),
                        "grant_type": "refresh_token",
                    }
                ).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            self.access_token = payload["access_token"]
            self.expires_at = time.time() + int(payload.get("expires_in", 3600))
            return self.access_token


oauth = OAuthToken()


def youtube_api(method, path, params=None, body=None):
    query = urllib.parse.urlencode(params or {})
    url = API_ROOT + path + ("?" + query if query else "")
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": "Bearer " + oauth.get(),
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"YouTube API {method} {path} failed ({exc.code}): {detail}") from exc


def get_broadcast():
    result = youtube_api(
        "GET",
        "/liveBroadcasts",
        {"part": "snippet,status,contentDetails", "id": required("YOUTUBE_BROADCAST_ID")},
    )
    items = result.get("items", [])
    if not items:
        raise RuntimeError("Configured YouTube broadcast no longer exists.")
    return items[0]


def get_stream():
    result = youtube_api(
        "GET",
        "/liveStreams",
        {"part": "cdn,status,contentDetails", "id": required("YOUTUBE_STREAM_ID")},
    )
    items = result.get("items", [])
    if not items:
        raise RuntimeError("Configured YouTube ingest stream no longer exists.")
    return items[0]


def transition(status):
    return youtube_api(
        "POST",
        "/liveBroadcasts/transition",
        {
            "broadcastStatus": status,
            "id": required("YOUTUBE_BROADCAST_ID"),
            "part": "id,status",
        },
    )


def ingest_url(stream):
    info = stream["cdn"]["ingestionInfo"]
    base = info.get("rtmpsIngestionAddress") or info["ingestionAddress"]
    return base.rstrip("/") + "/" + info["streamName"]


def check_resumable():
    """Validate the configured broadcast and stream; return the stream resource."""
    lifecycle = get_broadcast().get("status", {}).get("lifeCycleStatus")
    if lifecycle not in ("ready", "testing", "liveStarting", "live"):
        raise RuntimeError(f"Broadcast is not resumable from state: {lifecycle}")
    stream = get_stream()
    if not stream.get("contentDetails", {}).get("isReusable"):
        raise RuntimeError("Configured YouTube stream is not reusable.")
    return stream


def go_live(stop):
    """Wait for YouTube to accept ingest, then walk the broadcast ready -> testing -> live."""
    deadline = time.time() + 75
    while not stop.is_set():
        if time.time() > deadline:
            raise RuntimeError("YouTube did not accept ingest within 75 seconds.")
        if get_stream().get("status", {}).get("streamStatus") == "active":
            break
        stop.wait(3)
    deadline = time.time() + 90
    requested = set()
    while not stop.is_set():
        lifecycle = get_broadcast().get("status", {}).get("lifeCycleStatus")
        if lifecycle == "live":
            return
        if lifecycle == "complete":
            raise RuntimeError("YouTube completed the broadcast.")
        if time.time() > deadline:
            raise RuntimeError("YouTube did not finish the live transition within 90 seconds.")
        # each step can only be requested once
        target = {"ready": "testing", "testing": "live"}.get(lifecycle)
        if target and target not in requested:
            transition(target)
            requested.add(target)
        stop.wait(2)
