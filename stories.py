"""Random stories from the SimpleStories dataset, read through the Hugging Face datasets-server."""

import json
import random
import sys
import time
import urllib.parse
import urllib.request

DATASET = "SimpleStories/SimpleStories"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
FALLBACK_TOTAL = 2_000_000  # train split has about 2.1M rows
MIN_WORDS, MAX_WORDS = 80, 180
BATCH = 40

_total = None


def _fetch_rows(offset, length):
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": "train", "offset": offset, "length": length}
    )
    request = urllib.request.Request(ROWS_URL + "?" + query, headers={"User-Agent": "ytlive/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def random_story(rng=random, retries=5):
    """One random story of a watchable length, as a dict with story, topic and theme."""
    global _total
    last = None
    for attempt in range(retries):
        try:
            offset = rng.randrange(0, (_total or FALLBACK_TOTAL) - BATCH)
            data = _fetch_rows(offset, BATCH)
            _total = int(data.get("num_rows_total") or _total or FALLBACK_TOTAL)
            rows = [r["row"] for r in data.get("rows", [])]
            rows = [r for r in rows if MIN_WORDS <= len(r["story"].split()) <= MAX_WORDS]
            if rows:
                row = rng.choice(rows)
                return {"story": row["story"], "topic": row.get("topic", ""), "theme": row.get("theme", "")}
        except Exception as exc:
            last = exc
            print(f"story fetch attempt {attempt + 1} failed: {exc}", file=sys.stderr, flush=True)
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"could not fetch a story: {last}")
