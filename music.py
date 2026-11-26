"""A calm, piano-like background loop synthesised with numpy (no samples, no licensing)."""

import wave

import numpy as np

SR = 44100
BPM = 60
STEP = 60 / BPM / 2  # an eighth note
STEPS_PER_CHORD = 16
# (bass root, chord tones) as MIDI notes; diatonic to C major, slow and open
CHORDS = (
    (36, (60, 64, 67, 71)),  # Cmaj7
    (33, (57, 60, 64, 67)),  # Am7
    (41, (60, 64, 65, 69)),  # Fmaj7
    (31, (59, 62, 64, 67)),  # G6
    (40, (59, 62, 64, 67)),  # Em7
    (33, (57, 60, 64, 67)),  # Am7
    (38, (57, 60, 62, 65)),  # Dm7
    (31, (59, 62, 67, 71)),  # G
)
PENTATONIC = (72, 74, 76, 79, 81, 84)  # C D E G A C
ARPEGGIO = (0, 2, 1, 3, 2, 1, 3, 2)


def midi_hz(note):
    return 440.0 * 2 ** ((note - 69) / 12)


def piano_note(freq, seconds, velocity, rng):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    tau = np.clip(4.2 - 0.0021 * freq, 0.9, 4.0)
    for k in range(1, 11):
        f = k * freq * np.sqrt(1 + 0.0004 * k * k)
        if f > 9000:
            break
        decay = np.exp(-t / (tau / k ** 0.55))
        out += (1 / k ** 1.15) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * decay
    attack = np.minimum(t / 0.006, 1.0)
    release = np.minimum((seconds - t) / 0.4, 1.0)
    hammer = rng.normal(0, 1, n) * np.exp(-t / 0.012) * 0.05
    return (out * attack + hammer) * release * velocity


def add_wrapped(buffer, segment, start, pan):
    """Mix a mono segment into a stereo loop buffer, wrapping the tail around to the start."""
    idx = (start + np.arange(len(segment))) % len(buffer)
    left, right = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    delay = int(0.011 * SR)  # a touch of width
    buffer[idx, 0] += segment * left
    buffer[(idx + delay) % len(buffer), 1] += segment * right


def make_bgm(path, seed=3):
    rng = np.random.default_rng(seed)
    total_steps = STEPS_PER_CHORD * len(CHORDS)
    buffer = np.zeros((int(total_steps * STEP * SR), 2))
    for c, (root, tones) in enumerate(CHORDS):
        base = c * STEPS_PER_CHORD
        for step, note in ((0, root), (8, root + 7)):  # bass: root, then the fifth
            seg = piano_note(midi_hz(note), 6.0, 0.55, rng)
            add_wrapped(buffer, seg, int((base + step) * STEP * SR), -0.25)
        for i in range(STEPS_PER_CHORD):
            if rng.random() < 0.12:  # an occasional rest
                continue
            note = tones[ARPEGGIO[i % len(ARPEGGIO)]] + (12 if i % 8 == 7 else 0)
            seg = piano_note(midi_hz(note), 3.2, rng.uniform(0.28, 0.42), rng)
            add_wrapped(buffer, seg, int((base + i) * STEP * SR + rng.integers(0, 600)), 0.1 + 0.2 * (note - 60) / 24)
        for step in (0, 8):  # a slow melody note, sometimes
            if rng.random() < 0.7:
                note = int(rng.choice(PENTATONIC))
                seg = piano_note(midi_hz(note), 5.0, rng.uniform(0.35, 0.5), rng)
                add_wrapped(buffer, seg, int((base + step + 1) * STEP * SR), 0.3)
    buffer = np.tanh(buffer / (np.abs(buffer).max() / 1.4))  # soft limit, keeps loop edges continuous
    buffer *= 0.6 / np.abs(buffer).max()
    pcm = (buffer * 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(pcm.tobytes())
    return path
