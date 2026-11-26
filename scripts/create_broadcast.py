#!/usr/bin/env python3
"""One-off setup: create a reusable ingest stream and an unlisted broadcast bound to it.

Needs YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET and YOUTUBE_REFRESH_TOKEN in the environment.
Prints the YOUTUBE_STREAM_ID and YOUTUBE_BROADCAST_ID to configure in CircleCI.
BROADCAST_PRIVACY (unlisted by default) can be set to public or private.
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from youtube import youtube_api  # noqa: E402

TITLE = os.environ.get("BROADCAST_TITLE", "Tiny Pen Stories")
DESCRIPTION = (
    "Short stories, sketched live by a small AI that holds a pen.\n\n"
    "AI-generated: stories are from the SimpleStories dataset (which is itself AI-written), scenes and keywords "
    "are planned by a language model, drawings come from PenGPT (https://github.com/greydanus/pengpt, "
    "Sam Greydanus, Apache-2.0), and the piano is synthesised."
)


def create_stream():
    return youtube_api(
        "POST",
        "/liveStreams",
        {"part": "snippet,cdn,contentDetails,status"},
        {
            "snippet": {"title": f"{TITLE} ingest"},
            "cdn": {"frameRate": "variable", "ingestionType": "rtmp", "resolution": "variable"},
            "contentDetails": {"isReusable": True},
        },
    )


def create_broadcast():
    start = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)
    status = {"privacyStatus": os.environ.get("BROADCAST_PRIVACY", "unlisted"), "selfDeclaredMadeForKids": False}
    return youtube_api(
        "POST",
        "/liveBroadcasts",
        {"part": "snippet,status,contentDetails"},
        {
            "snippet": {
                "title": TITLE,
                "description": DESCRIPTION,
                "scheduledStartTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "status": status,
            "contentDetails": {
                "enableAutoStart": False,
                "enableAutoStop": False,
                "enableDvr": True,
                "enableEmbed": True,
                "latencyPreference": "low",
                "monitorStream": {"enableMonitorStream": False},
            },
        },
    )


def main():
    stream_id = os.environ.get("YOUTUBE_STREAM_ID") or create_stream()["id"]
    broadcast = create_broadcast()
    youtube_api(
        "POST",
        "/liveBroadcasts/bind",
        {"id": broadcast["id"], "part": "id,contentDetails", "streamId": stream_id},
    )
    # The live-broadcast resource has no synthetic-media field; the video resource does (write-only in the API).
    video = youtube_api("GET", "/videos", {"part": "status", "id": broadcast["id"]})["items"][0]["status"]
    keep = {k: video[k] for k in ("privacyStatus", "license", "embeddable", "publicStatsViewable", "selfDeclaredMadeForKids")}
    youtube_api(
        "PUT", "/videos", {"part": "status"},
        {"id": broadcast["id"], "status": {**keep, "containsSyntheticMedia": True}},
    )
    status = broadcast.get("status", {})
    print(f"YOUTUBE_STREAM_ID={stream_id}")
    print(f"YOUTUBE_BROADCAST_ID={broadcast['id']}")
    print(f"privacy={status.get('privacyStatus')} (synthetic-media flag set on the video; confirm it in Studio)")
    print(f"studio: https://studio.youtube.com/video/{broadcast['id']}/livestreaming")


if __name__ == "__main__":
    main()
