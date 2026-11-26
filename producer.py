"""Story, scene and sketch production, run in its own process.

Drawing inference is CPU-heavy; keeping it out of the streaming process stops it from starving the
frame renderer and the encoder on a small machine.
"""

import os

# one thread per math library: the machine has two cores and the encoder needs one of them
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import queue  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402


def produce(scenes, stop):
    """Endlessly pick a story, plan its scenes, sketch each one and put it on `scenes`."""
    import time

    started = time.monotonic()
    from llm import plan_scenes
    from sketch import Sketcher
    from stories import random_story

    sketcher = Sketcher()  # loads the pre-drawn library now and the drawing models in the background
    print(f"producer: library of {len(sketcher.library)} words ready after {time.monotonic() - started:.1f}s", flush=True)
    rng = random.Random()
    while not stop.is_set():
        try:
            t0 = time.monotonic()
            story = random_story(rng)
            plan = plan_scenes(story["story"])
            print(f"story: {len(plan)} scenes, topic {story['topic']!r} (planned in {time.monotonic() - t0:.0f}s)", flush=True)
            for index, scene in enumerate(plan):
                t0 = time.monotonic()
                seeds = [rng.randrange(10**6) for _ in scene["keywords"]]
                # while the drawing models are still loading, leave out keywords that would need them, so the
                # first scenes go out at once; if nothing is left, wait for the models
                usable = [(w, s) for w, s in zip(scene["keywords"], seeds) if sketcher.available(w, s)]
                keywords, strokes = [], []
                for word, seed in usable or list(zip(scene["keywords"], seeds)):
                    keywords.append(word)
                    strokes.append(sketcher.draw(word, seed=seed))
                live = sum(1 for w, s in zip(keywords, seeds) if sketcher.from_library(w, s) is None)
                print(f"producer: scene {index + 1}/{len(plan)} drawn in {time.monotonic() - t0:.0f}s ({live} live)", flush=True)
                item = {
                    "text": scene["text"], "keywords": keywords, "strokes": strokes,
                    "index": index, "total": len(plan), "topic": story["topic"],
                }
                while not stop.is_set():
                    try:
                        scenes.put(item, timeout=1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except Exception as exc:
            print(f"producer error: {exc}", file=sys.stderr, flush=True)
            stop.wait(10)
