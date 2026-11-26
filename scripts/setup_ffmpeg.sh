#!/usr/bin/env bash
# Install a pinned static ffmpeg into $FFMPEG_DIR (default ~/ffmpeg, cached by CircleCI).
# Falls back to the distro package if the download or its checksum fails.
set -uo pipefail

DIR="${FFMPEG_DIR:-$HOME/ffmpeg}"
TAG=autobuild-2026-11-25-12-41
FILE=ffmpeg-n8.1.3-41-ga8029b6528-linux64-gpl-8.1.tar.xz
SHA256=52e5d6c80ef213861dcdbdf5f742c551469ee7de91b3a846256660c9d0943997

if [ -x "$DIR/bin/ffmpeg" ] && "$DIR/bin/ffmpeg" -version >/dev/null 2>&1; then
  echo "ffmpeg already installed in $DIR"
  exit 0
fi

rm -rf "$DIR"
mkdir -p "$DIR"
tmp="$(mktemp -d)"
if curl -fsSL --retry 3 -o "$tmp/ffmpeg.tar.xz" "https://github.com/BtbN/FFmpeg-Builds/releases/download/$TAG/$FILE" \
  && echo "$SHA256  $tmp/ffmpeg.tar.xz" | sha256sum -c - \
  && tar -xJf "$tmp/ffmpeg.tar.xz" --strip-components=1 -C "$DIR" \
  && "$DIR/bin/ffmpeg" -version >/dev/null 2>&1; then
  echo "installed pinned static ffmpeg"
  exit 0
fi

echo "static ffmpeg unavailable; falling back to apt" >&2
rm -rf "$DIR"
for attempt in 1 2 3 4; do
  if sudo apt-get update && sudo apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core; then
    mkdir -p "$DIR/bin"
    ln -sf "$(command -v ffmpeg)" "$DIR/bin/ffmpeg"
    exit 0
  fi
  sleep 5
done
exit 1
