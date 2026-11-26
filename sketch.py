"""Keyword sketches from the PenGPT drawing model."""

import json
import os
import sys

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


class Sketcher:
    def __init__(self):
        sketch_path, text_path = ensure_models()
        self.pen = Pen(sketch_path)
        self.pen.encoder = LeanWords(text_path)

    def draw(self, keyword, seed=0):
        """A list of strokes, each an (n, 2) array scaled so the drawing fits a unit box, y pointing down."""
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
