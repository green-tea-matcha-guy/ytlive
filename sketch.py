"""Keyword sketches from the PenGPT drawing model."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pengpt_inference"))
from pen import Pen, strokes as split_strokes  # noqa: E402

MODEL_DIR = os.environ.get("PENGPT_MODEL_DIR", os.path.expanduser("~/models"))
SKETCH_MODEL = "sketch.npz"
TEXT_MODEL = "siglip-text.npz"


def ensure_models():
    """Download the model files into MODEL_DIR if they are missing (also warms the CI cache)."""
    from pen import fetch

    os.makedirs(MODEL_DIR, exist_ok=True)
    return [fetch(os.path.join(MODEL_DIR, name)) for name in (SKETCH_MODEL, TEXT_MODEL)]


class Sketcher:
    def __init__(self):
        from words import Words

        sketch_path, text_path = ensure_models()
        self.pen = Pen(sketch_path)
        self.pen.encoder = Words(text_path)

    def draw(self, keyword, seed=0):
        """A list of strokes, each an (n, 2) array scaled so the drawing fits a unit box, y pointing down."""
        prompt = keyword if keyword.split()[0] in ("a", "an", "the") else f"a {keyword}"
        best = None
        for attempt in range(3):
            rng = np.random.default_rng(seed * 7 + attempt)
            points = self.pen.draw(prompt, temperature=0.6, rng=rng)
            if best is None or len(points) > len(best):
                best = points
            if len(points) >= 40:
                break
        return normalise_strokes(best)


def normalise_strokes(points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return []
    lo, hi = points[:, :2].min(0), points[:, :2].max(0)
    scale = max(hi - lo)
    if scale <= 0:
        return []
    centre = (lo + hi) / 2
    out = []
    for a, b in split_strokes(points):
        xy = (points[a:b, :2] - centre) / scale
        if len(xy) == 1:
            xy = np.concatenate([xy, xy + 0.002])
        out.append(xy)
    return out
