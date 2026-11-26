# Tiny Pen Stories

A continuous YouTube Live stream of short stories, sketched by a small AI that holds a pen. Runs as a
self-restarting CircleCI job.

## What it does

1. Picks a random story from the [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories)
   dataset (read through the Hugging Face datasets-server; no download).
2. Asks the Exa demo LLM to split it into scenes, keeping the story's own sentences, and to give 2 to 4
   drawable keywords per scene. The reply is parsed defensively: any trailing `followups` block or code
   fence is ignored, the scenes must match the original story, and if the LLM fails the story is split
   locally and keywords fall back to heuristics (`llm.py`, covered by `tests/`).
3. Gets each keyword's sketch from a pre-drawn library (`sketch_library.json.gz`, about 560 common nouns, two
   variants each, built by `scripts/build_library.py`; the LLM is steered towards those words) or, for rarer
   words, draws it live with the [PenGPT](https://github.com/greydanus/pengpt) drawing model (SigLIP 2 text
   vector in, pen strokes out; live drawing is slow on a small CI machine, which is why the library exists).
   The strokes are animated as if being drawn. Drawing settings are the demo's: guidance halfway between
   loose and literal (2.5), and no ink request (the sparse end).
4. Renders the sentence on a paper page with the sketches below it, and streams it with a synthesised,
   looping piano track. Scenes are prepared a few ahead while the current one is on screen.

Nothing is persisted: stories are random and there is no state between runs.

## Running continuously

CircleCI jobs have a time limit, so the process exits cleanly after `MAX_RUNTIME_SECONDS` (default 4 hours)
without completing the broadcast, and the last step of each run starts the next one through the CircleCI
API (scheduled pipelines are not available for standalone projects). `serial-group` keeps a single
connection to the stream key open at a time.

To stop: set `RESTART_ENABLED=false` first, then cancel the workflow. Set it back to `true` and start a
pipeline to resume.

## Setup

`scripts/create_broadcast.py` creates a reusable ingest stream and an unlisted broadcast (auto start and
auto stop off) and prints the two IDs. Project environment variables:

- `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN`: an access token is minted from the
  refresh token on every run
- `YOUTUBE_BROADCAST_ID`, `YOUTUBE_STREAM_ID`
- `CIRCLE_TOKEN`, `RESTART_PROJECT_SLUG`, `RESTART_DEFINITION_ID`, `RESTART_ENABLED`: for chaining runs
- Optional: `MAX_RUNTIME_SECONDS`, `EXA_ENDPOINT`, `EXA_MODEL`

For a local test, `LOCAL_OUTPUT=out.mkv MAX_RUNTIME_SECONDS=120 python app.py` writes a file instead of
streaming (needs `ffmpeg`, `torch`, `transformers`, `numpy`, `Pillow`).

## Credits

`pengpt_inference/` is copied from PenGPT (Sam Greydanus, Apache-2.0); the model weights are downloaded
from its releases at run time. Stories are from SimpleStories.
