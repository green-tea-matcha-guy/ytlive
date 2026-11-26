# YouTube Live chat AI on CircleCI

A long-running CircleCI job that streams a generated 720p video feed to one
reusable YouTube ingest stream, reads the broadcast's live chat, answers each
new message through the Exa demo endpoint, and shows the question and answer on
screen.

The job holds one ffmpeg connection open for its whole run. Chat is polled in
a loop, messages are answered one at a time, and each answer stays on screen
until the next one replaces it. There is no database or other persisted state:
the chat page token, the pending queue and the on-screen card live in memory.
Messages already in chat when a run starts are skipped.

## Running continuously

CircleCI jobs have a maximum duration, so the process exits cleanly after
`MAX_RUNTIME_SECONDS` (default 4 hours) without completing the YouTube
broadcast. A scheduled pipeline trigger every 4 hours starts the next run, and
`serial-group` makes it wait until the previous run has finished, so only one
connection to the stream key is open at a time. Messages sent in the gap
between runs are not answered.

## Environment variables

Set these as project environment variables in CircleCI.

Required:

- `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN` — OAuth
  credentials with access to the YouTube Data API
- `YOUTUBE_BROADCAST_ID` — a non-completed broadcast
- `YOUTUBE_STREAM_ID` — the reusable stream bound to that broadcast

Optional:

- `CARD_SECONDS` — minimum seconds each answer stays up (8–20, default 10)
- `CHAT_POLL_SECONDS` — minimum seconds between chat polls (default 30)
- `MAX_RUNTIME_SECONDS` — seconds before the run exits (default 14400)
- `EXA_ENDPOINT`, `EXA_MODEL` — answer backend and model

Each chat poll costs YouTube API quota, so lowering `CHAT_POLL_SECONDS` can
exhaust the daily quota.
