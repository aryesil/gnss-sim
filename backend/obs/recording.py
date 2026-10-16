# backend/recording.py
"""Session recording & replay: capture the exact SSE payload sequence a
live transmit already emits (progress, spectrogram, cn0, timeline_step,
finished) to a JSONL file, then replay it later as the same SSE shape --
so the frontend needs no new message types to render a replay, just a
different source.

Recording is opt-in per live session (a channel's "Record" checkbox);
replay is read-only and never touches _tx_slots or real hardware -- it
is a pure playback of what was already recorded, at a chosen speed.
"""
from __future__ import annotations

import json
import pathlib
import time

from backend import config

_DIR_NAME = "recordings"


def _dir() -> pathlib.Path:
    d = config.OUT_DIR / _DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


class RecordingWriter:
    """One file per recorded session. append() is called once per SSE
    payload the live session already emits; each line gets a `t` field
    (seconds since the recording started) so replay can reproduce pacing."""

    def __init__(self, slot: str):
        self._started = time.monotonic()
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.name = f"{slot}-{ts}"
        self.path = _dir() / f"{self.name}.jsonl"
        self._f = open(self.path, "w")

    def append(self, event: dict) -> None:
        # Kayıt yardımcı bir özelliktir: disk dolu/dosya hatası canlı yayını
        # (SSE üreticisini) düşürmemeli. İlk hatada kayıt kapatılır, yayın sürer.
        if self._f is None:
            return
        row = {"t": time.monotonic() - self._started, **event}
        try:
            self._f.write(json.dumps(row) + "\n")
            self._f.flush()
        except OSError:
            self.close()
            self._f = None

    def close(self) -> None:
        try:
            if self._f is not None:
                self._f.close()
        except OSError:
            pass


def list_names() -> list[str]:
    # Arayüz adları innerHTML'e basıyor; geçersiz adlı dosyalar listelenmez.
    return sorted(p.stem for p in _dir().glob("*.jsonl") if _NAME_RE.match(p.stem))


import re

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InvalidRecordingNameError(FileNotFoundError, ValueError):
    pass


def read_events(name: str) -> list[dict]:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise InvalidRecordingNameError(f"invalid recording name: {name!r}")
    path = _dir() / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(name)
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue    # oturum yarıda kesildiyse son satır yarım olabilir
    return out
