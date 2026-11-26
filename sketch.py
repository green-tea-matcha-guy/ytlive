"""Keyword sketches from the PenGPT drawing model."""

import gzip
import json
import os
import random
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pengpt_inference"))
from pen import Pen, strokes as split_strokes  # noqa: E402

MODEL_DIR = os.environ.get("PENGPT_MODEL_DIR", os.path.expanduser("~/models"))
# the demo's sliders: loose<->literal is guidance 1..4 (halfway = 2.5); sparse<->dense is ink, where the sparse
# end (0) means "let the model choose", i.e. no ink request
GUIDANCE = 2.5
SKETCH_MODEL = "sketch.npz"
TEXT_MODEL = "siglip-text.npz"


def ensure_models():
    """Download the model files into MODEL_DIR if they are missing (also warms the CI cache)."""
    from pen import fetch

    os.makedirs(MODEL_DIR, exist_ok=True)
    return [fetch(os.path.join(MODEL_DIR, name)) for name in (SKETCH_MODEL, TEXT_MODEL)]


class LeanWords:
    """The SigLIP 2 text tower from siglip-text.npz, held in bfloat16 and filled tensor by tensor.

    pengpt's own Words loads every weight as float32 twice over (about 5 GB at peak), which is more than a
    4 GB CI machine has; this needs about 1.5 GB and gives the same vectors to within bfloat16 rounding.
    """

    def __init__(self, path):
        import tempfile

        import torch
        from transformers import PreTrainedTokenizerFast, SiglipTextConfig, SiglipTextModel

        z = np.load(path)
        meta = json.loads(str(z["meta"]))
        # Every weight is overwritten below, so skip random initialisation, which is very slow for 536M
        # parameters on one thread. Newer transformers has its own init functions, so patch the model's
        # _init_weights as well as torch's in-place fills.
        fills = {n: getattr(torch.Tensor, n) for n in ("normal_", "uniform_")}
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        SiglipTextModel._init_weights = lambda self, module: None
        try:
            for n in fills:
                setattr(torch.Tensor, n, lambda t, *a, **k: t)
            self.model = SiglipTextModel(SiglipTextConfig(**meta["config"]))
        finally:
            torch.set_default_dtype(previous)
            for n, f in fills.items():
                setattr(torch.Tensor, n, f)
            del SiglipTextModel._init_weights
        files = set(z.files)
        with torch.no_grad():
            for name, param in self.model.state_dict().items():
                key = name if name in files else "text_model." + name
                if key not in files:
                    raise RuntimeError(f"siglip-text.npz has no weight for {name}")
                raw = z[key]
                scale = z[key + ".scale"] if key + ".scale" in files else None
                step = 2048  # dequantise a slab of rows at a time to keep the transient small
                if raw.ndim == 0:
                    raw, scale = raw.reshape(1), None if scale is None else scale.reshape(1)
                for i in range(0, raw.shape[0], step):
                    slab = raw[i:i + step].astype(np.float32)
                    if scale is not None:
                        slab = slab * scale[i:i + step]
                    param.reshape(-1, *param.shape[1:])[i:i + step].copy_(torch.from_numpy(slab))
        self.model.eval()
        self.remap = torch.from_numpy(z["remap"]).long() if "remap" in files else None
        with tempfile.TemporaryDirectory() as d:
            path_json = os.path.join(d, "tokenizer.json")
            open(path_json, "w", encoding="utf-8").write(meta["tokenizer"])
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=path_json, pad_token="<pad>")
        self.torch = torch

    def __call__(self, text):
        ids = self.tokenizer([text.lower()], padding="max_length", truncation=True, max_length=64, return_tensors="pt")["input_ids"]
        if self.remap is not None:
            ids = self.remap[ids]
        with self.torch.inference_mode():
            out = self.model(input_ids=ids).pooler_output[0].float()
        return (out / out.norm()).numpy().astype(np.float32)


LIBRARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sketch_library.json.gz")


def load_library(path=LIBRARY_PATH):
    """Pre-drawn sketches: {word: [drawing, ...]}, each drawing a list of (n, 2) stroke arrays."""
    if not os.path.exists(path):
        return {}
    with gzip.open(path, "rt") as f:
        raw = json.load(f)
    return {word: [[np.array(stroke) for stroke in drawing] for drawing in drawings] for word, drawings in raw.items()}


def singular_forms(word):
    yield word
    if word.endswith("ies"):
        yield word[:-3] + "y"
    if word.endswith("es"):
        yield word[:-2]
    if word.endswith("s"):
        yield word[:-1]


class Sketcher:
    """Sketches for keywords: from the pre-drawn library when the word is in it, else drawn live.

    The models load in a background thread, so scenes made only of library words can go out at once.
    """

    def __init__(self, library=True, preload=True):
        self.library = load_library() if library else {}
        self.pen = None
        self.lock = threading.Lock()
        if preload:
            threading.Thread(target=self.load_models, daemon=True).start()

    def load_models(self):
        with self.lock:
            if self.pen is not None:
                return
            import torch

            torch.set_num_threads(1)
            started = time.monotonic()
            sketch_path, text_path = ensure_models()
            pen = Pen(sketch_path)
            pen.encoder = LeanWords(text_path)
            self.pen = pen
            print(f"sketch models ready after {time.monotonic() - started:.0f}s", flush=True)

    def from_library(self, keyword, seed):
        word = keyword.lower().strip()
        for form in singular_forms(word):
            if form in self.library:
                drawings = self.library[form]
                return drawings[random.Random(seed).randrange(len(drawings))]
        return None

    def draw(self, keyword, seed=0):
        """A list of strokes, each an (n, 2) array scaled so the drawing fits a unit box, y pointing down."""
        cached = self.from_library(keyword, seed)
        if cached is not None:
            return cached
        self.load_models()
        prompt = keyword if keyword.split()[0] in ("a", "an", "the") else f"a {keyword}"
        best = None
        for attempt in range(3):
            rng = np.random.default_rng(seed * 7 + attempt)
            points = self.pen.draw(prompt, temperature=0.6, rng=rng, guidance=GUIDANCE)
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
