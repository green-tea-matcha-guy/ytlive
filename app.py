#!/usr/bin/env python3
"""Continuous YouTube Live stream of short illustrated stories.

Random stories are split into scenes by an LLM, each scene's keywords are sketched by the PenGPT
drawing model, and the result is streamed as video with a calm piano loop. Nothing is persisted.
"""

import multiprocessing
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import music
import producer
import youtube
from render import FPS, HEIGHT, WIDTH, Player, SceneView, scene_seconds

BGM_PATH = "/tmp/bgm.wav"
AUDIO_FILTER = "[1:a]aecho=0.7:0.6:70|150|260:0.3|0.22|0.14,lowpass=f=6500,volume=0.9[a]"


def ffmpeg_command(output, local=False):
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-hide_banner", "-loglevel", "warning",
        "-use_wallclock_as_timestamps", "1",
        "-thread_queue_size", "64", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0",
        "-re", "-stream_loop", "-1", "-i", BGM_PATH,
        "-filter_complex", AUDIO_FILTER,
        "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-preset", "superfast", "-profile:v", "high", "-level:v", "4.0",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-b:v", "1500k", "-minrate", "1500k", "-maxrate", "1500k", "-bufsize", "3000k",
        # constant bitrate with filler: a still page would otherwise send almost nothing and YouTube
        # reports the ingest as starved
        "-x264-params", "nal-hrd=cbr:force-cfr=1",
        "-g", str(2 * FPS), "-keyint_min", str(2 * FPS), "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
    ]
    if local:
        return command + ["-f", "matroska", "-y", output]
    return command + ["-flvflags", "no_duration_filesize", "-f", "flv", output]


def feed_video(process, player, stop):
    deadline = time.monotonic()
    try:
        while not stop.is_set() and process.poll() is None:
            process.stdin.write(player.render(time.monotonic()))
            process.stdin.flush()
            deadline += 1 / FPS
            delay = deadline - time.monotonic()
            if delay < -1:  # fell far behind; resynchronise instead of racing to catch up
                deadline = time.monotonic()
            stop.wait(max(0, delay))
    except (BrokenPipeError, ValueError):
        stop.set()
    finally:
        try:
            process.stdin.close()
        except (BrokenPipeError, ValueError):
            pass


class Encoder:
    """One long-lived ffmpeg process fed with rendered frames."""

    def __init__(self, output, local, player):
        self.output, self.local, self.player = output, local, player
        self.process = None
        self.stop = threading.Event()
        self.feeder = None

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def start(self):
        self.close()
        self.stop = threading.Event()
        self.process = subprocess.Popen(ffmpeg_command(self.output, self.local), stdin=subprocess.PIPE)
        self.feeder = threading.Thread(
            target=feed_video, args=(self.process, self.player, self.stop), daemon=True
        )
        self.feeder.start()

    def close(self):
        self.stop.set()
        if self.feeder:
            self.feeder.join(timeout=3)
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        self.process = None


def main():
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    deadline = time.monotonic() + int(os.environ.get("MAX_RUNTIME_SECONDS", "14400"))
    local_output = os.environ.get("LOCAL_OUTPUT")  # write a file instead of streaming (for testing)

    output = local_output
    if not local_output:
        stream = youtube.check_resumable()
        output = youtube.ingest_url(stream)

    music.make_bgm(BGM_PATH)
    player = Player(time.monotonic())
    ctx = multiprocessing.get_context("spawn")
    scenes, producer_stop = ctx.Queue(maxsize=3), ctx.Event()
    worker = ctx.Process(target=producer.produce, args=(scenes, producer_stop), daemon=True)
    worker.start()

    encoder = Encoder(output, bool(local_output), player)
    next_scene_at = 0.0
    live = bool(local_output)
    print("Streaming started", flush=True)
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            try:
                if not encoder.alive():
                    encoder.start()
                    live = bool(local_output)
                if not live:  # retried on the next pass if YouTube is not ready yet
                    youtube.go_live(stop)
                    live = True
                now = time.monotonic()
                if now >= next_scene_at:
                    try:
                        item = scenes.get_nowait()
                    except queue.Empty:
                        pass  # the next scene is still being drawn; keep the current page up
                    else:
                        view = SceneView(
                            item["text"], item["keywords"], item["strokes"], item["index"], item["total"],
                            topic=item["topic"], duration=scene_seconds(item["text"]),
                        )
                        player.show(view, now)
                        next_scene_at = now + view.duration + 0.6
                if not worker.is_alive():
                    raise RuntimeError("scene producer process exited")
            except Exception as exc:
                print(f"Loop error: {exc}", file=sys.stderr, flush=True)
                stop.wait(10)
            stop.wait(0.25)
    finally:
        stop.set()
        producer_stop.set()
        encoder.close()
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()


if __name__ == "__main__":
    main()
