#!/usr/bin/env python3
"""Pre-draw a library of sketches for the common vocabulary, so most scenes need no live drawing.

  python scripts/build_library.py shard I N     # draw every N-th word starting at I -> /tmp/library_I.json
  python scripts/build_library.py merge N       # merge the shards into sketch_library.json.gz
"""

import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

VARIANTS = 2


def shard(index, count):
    import numpy as np

    from sketch import Sketcher
    from vocabulary import WORDS

    sketcher = Sketcher(library=False)
    out = {}
    for n, word in enumerate(WORDS):
        if n % count != index:
            continue
        drawings = []
        for variant in range(VARIANTS):
            strokes = sketcher.draw(word, seed=1000 + variant)
            if strokes:
                drawings.append([np.round(s, 3).tolist() for s in strokes])
        out[word] = drawings
        print(f"{index}: {word} ({len(drawings)})", flush=True)
    with open(f"/tmp/library_{index}.json", "w") as f:
        json.dump(out, f)


def merge(count):
    library = {}
    for index in range(count):
        with open(f"/tmp/library_{index}.json") as f:
            library.update(json.load(f))
    library = {w: d for w, d in library.items() if d}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sketch_library.json.gz")
    with gzip.open(path, "wt", compresslevel=9) as f:
        json.dump(library, f, separators=(",", ":"))
    print(f"{len(library)} words -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    if sys.argv[1] == "shard":
        shard(int(sys.argv[2]), int(sys.argv[3]))
    else:
        merge(int(sys.argv[2]))
