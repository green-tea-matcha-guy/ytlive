"""Frames for the story video: a sentence on a paper page with pen sketches drawn below it."""

import math
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1280, 720, 15
SS = 2  # supersampling factor for the sketch layer

PAPER = (243, 238, 228)
INK = (44, 48, 74)
MUTED = (132, 124, 112)
ACCENT = (176, 132, 100)

SERIF_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
)
SANS_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/SFNS.ttf",
)

TEXT_BOX = (90, 62, 1190, 300)  # sentence area
DIVIDER_Y = 322
SKETCH_TOP, SKETCH_HEIGHT = 345, 285  # sketch layer region
SKETCH_SIZE = 255
CAPTION_Y = 640
FADE = 0.5
DRAW_START = 0.6


def first_font(candidates):
    for path in candidates:
        if Path(path).exists():
            return path
    raise RuntimeError("no usable font found; install fonts-dejavu-core")


SERIF = first_font(SERIF_CANDIDATES)
SANS = first_font(SANS_CANDIDATES)


def paper_texture():
    """Warm paper with fine grain and a soft vignette, made once."""
    rng = np.random.default_rng(7)
    grain = rng.normal(0, 2.2, (HEIGHT, WIDTH, 1))
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    d = np.sqrt(((x - WIDTH / 2) / (WIDTH / 2)) ** 2 + ((y - HEIGHT / 2) / (HEIGHT / 2)) ** 2)
    vignette = 1 - 0.07 * np.clip(d - 0.55, 0, 1) ** 1.5
    img = (np.array(PAPER, dtype=float)[None, None, :] + grain) * vignette[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


PAPER_IMAGE = paper_texture()


def wrap(draw, text, font, max_width):
    lines, current = [], ""
    for word in text.split():
        candidate = word if not current else current + " " + word
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_sentence(draw, text, box=TEXT_BOX, start=44, minimum=28):
    """Centre the text in box, shrinking the font until it fits."""
    left, top, right, bottom = box
    for size in range(start, minimum - 1, -2):
        font = ImageFont.truetype(SERIF, size)
        lines = wrap(draw, text, font, right - left)
        spacing = int(size * 0.38)
        block = "\n".join(lines)
        bbox = draw.multiline_textbbox((0, 0), block, font=font, spacing=spacing, align="center")
        if bbox[3] - bbox[1] <= bottom - top or size == minimum:
            break
    y = top + (bottom - top - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.multiline_text(
        ((left + right) / 2, y), block, font=font, fill=INK, spacing=spacing, align="center", anchor="ma"
    )


def ornament(draw, y):
    cx = WIDTH // 2
    draw.line((cx - 44, y, cx - 8, y), fill=ACCENT, width=1)
    draw.line((cx + 8, y, cx + 44, y), fill=ACCENT, width=1)
    draw.ellipse((cx - 2, y - 2, cx + 2, y + 2), fill=ACCENT)


def centred(draw, xy, text, font, fill):
    draw.text(xy, text, font=font, fill=fill, anchor="ma")


class Sketch:
    """One keyword's strokes laid out in pixels (at SS scale) and drawn incrementally onto a mask."""

    def __init__(self, strokes, cx, size):
        self.segments = []  # (x0, y0, x1, y1, cum_start, cum_end), in layer pixels
        total = 0.0
        for stroke in strokes:
            pts = stroke * size * SS + np.array([cx * SS, SKETCH_HEIGHT * SS / 2])
            for a, b in zip(pts[:-1], pts[1:]):
                length = float(np.hypot(*(b - a)))
                self.segments.append((a[0], a[1], b[0], b[1], total, total + length))
                total += length
        self.length = max(total, 1e-6)
        self.done = 0

    def advance(self, mask_draw, fraction):
        """Draw every segment up to `fraction` of the total length."""
        target = fraction * self.length
        width = int(2.6 * SS)
        radius = width / 2
        i = self.done
        while i < len(self.segments):
            x0, y0, x1, y1, c0, c1 = self.segments[i]
            if c0 >= target:
                break
            if c1 > target:  # partial segment: stop part-way, redraw next frame
                t = (target - c0) / max(c1 - c0, 1e-9)
                x1, y1 = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                mask_draw.line((x0, y0, x1, y1), fill=255, width=width)
                break
            mask_draw.line((x0, y0, x1, y1), fill=255, width=width)
            mask_draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill=255)
            i += 1
        self.done = i
        return fraction >= 1.0


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class SceneView:
    """A sentence with its keyword sketches. frame(t) is the page t seconds after the scene started."""

    def __init__(self, text, keywords, strokes_by_keyword, index, total, topic="", duration=12.0):
        self.duration = duration
        self.draw_time = max(4.0, duration * 0.62)
        self.base = PAPER_IMAGE.copy()
        draw = ImageDraw.Draw(self.base)
        if topic:
            centred(draw, (WIDTH / 2, 28), f"a story about {topic}", ImageFont.truetype(SANS, 17), MUTED)
        draw_sentence(draw, text)
        ornament(draw, DIVIDER_Y)

        caption_font = ImageFont.truetype(SANS, 21)
        n = max(len(keywords), 1)
        cell = 1120 / n
        size = min(cell * 0.82, SKETCH_SIZE)
        self.sketches = []
        for k, (word, strokes) in enumerate(zip(keywords, strokes_by_keyword)):
            cx = 80 + cell * (k + 0.5)
            centred(draw, (cx, CAPTION_Y), word, caption_font, MUTED)
            if strokes:
                self.sketches.append(Sketch(strokes, cx, size))

        dot_y, gap = 692, 16
        x0 = WIDTH / 2 - gap * (total - 1) / 2
        for i in range(total):
            fill = ACCENT if i <= index else (205, 196, 182)
            r = 3 if i == index else 2
            draw.ellipse((x0 + i * gap - r, dot_y - r, x0 + i * gap + r, dot_y + r), fill=fill)

        self.mask = Image.new("L", (WIDTH * SS, SKETCH_HEIGHT * SS), 0)
        self.mask_draw = ImageDraw.Draw(self.mask)
        self.finished = not self.sketches
        self._last_image = None

    def frame(self, t):
        if self.finished and self._last_image is not None:
            return self._last_image
        fraction = ease((t - DRAW_START) / self.draw_time)
        done = [s.advance(self.mask_draw, fraction) for s in self.sketches]
        image = self.base.copy()
        if self.sketches:
            image.paste(INK, (0, SKETCH_TOP), self.mask.reduce(SS))
        self.finished = all(done)
        self._last_image = image if self.finished else None
        return image


class TitleView:
    def __init__(self):
        image = PAPER_IMAGE.copy()
        draw = ImageDraw.Draw(image)
        centred(draw, (WIDTH / 2, 250), "Tiny Pen Stories", ImageFont.truetype(SERIF, 66), INK)
        ornament(draw, 345)
        centred(draw, (WIDTH / 2, 380), "short stories, sketched by a small AI that holds a pen", ImageFont.truetype(SANS, 24), MUTED)
        centred(draw, (WIDTH / 2, 640), "drawings by PenGPT · Sam Greydanus", ImageFont.truetype(SANS, 18), MUTED)
        self.image = image
        self.finished = True

    def frame(self, t):
        return self.image


class Player:
    """Holds the current view and cross-fades to the next one; render() is called once per video frame."""

    def __init__(self, now):
        self.lock = threading.Lock()
        self.current = TitleView()
        self.started = now
        self.pending = None
        self.fade_out_at = None
        self.fade_in_at = None

    def show(self, view, now):
        with self.lock:
            self.pending = view
            self.fade_out_at = now

    def scene_age(self, now):
        with self.lock:
            return now - self.started

    def render(self, now):
        with self.lock:
            if self.pending is not None and now - self.fade_out_at >= FADE:
                self.current, self.pending = self.pending, None
                self.started = self.fade_in_at = now
                self.fade_out_at = None
            if self.pending is not None:
                alpha = 1 - (now - self.fade_out_at) / FADE
            elif self.fade_in_at is not None and now - self.fade_in_at < FADE:
                alpha = (now - self.fade_in_at) / FADE
            else:
                alpha = 1.0
            image = self.current.frame(now - self.started)
        if alpha < 1.0:
            image = Image.blend(PAPER_IMAGE, image, max(alpha, 0.0))
        return image.tobytes()


def scene_seconds(text):
    """How long a scene stays up: reading time plus time to watch the drawing finish."""
    return min(max(len(text.split()) * 0.42 + 4.0, 10.0), 24.0)
