"""Shared event log and per-tile load status.

Network threads, mesh workers and the UI all report here, so the screen (and a log file)
can show exactly what a slow travel is waiting on instead of a silent "Travelling...".
Everything is thread-safe and cheap enough to call from hot paths.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from pathlib import Path

# Pipeline stages of one tile, in order. "queued" waits for a worker thread.
STAGES = ("queued", "cache", "download", "mesh", "upload", "done")
STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}
STAGE_LABEL = {
    "queued": "Waiting for a worker",
    "cache": "Check local cache",
    "download": "Download map data",
    "mesh": "Build 3D geometry",
    "upload": "Upload to GPU",
    "done": "Ready",
    "failed": "Failed",
}
# rough share of total time per stage, used for the overall progress bar
_BASE = {"queued": 0.00, "cache": 0.03, "download": 0.08, "mesh": 0.72,
         "upload": 0.93, "done": 1.00, "failed": 0.00}


def fmt_bytes(n: float) -> str:
    n = float(n)
    if n < 1024: return f"{int(n)} B"
    if n < 1024 ** 2: return f"{n / 1024:.0f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


class EventLog:
    LEVEL_TAG = {"info": "    ", "ok": " OK ", "warn": "WARN", "error": "FAIL"}

    def __init__(self, maxlen: int = 400):
        self._lock = threading.Lock()
        self._lines = collections.deque(maxlen=maxlen)
        self._file = None
        self.echo = True          # mirror to the console

    def attach_file(self, path):
        try:
            p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(p, "a", encoding="utf-8", buffering=1)
            self._file.write(f"\n==== session {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        except OSError:
            self._file = None

    def add(self, level: str, msg: str):
        now = time.time()
        stamp = time.strftime("%H:%M:%S", time.localtime(now))
        with self._lock:
            self._lines.append((now, level, msg))
            line = f"[{stamp}] {self.LEVEL_TAG.get(level, '    ')} {msg}"
            if self._file is not None:
                try: self._file.write(line + "\n")
                except (OSError, ValueError): self._file = None
        if self.echo:
            try: print(line, flush=True)
            except Exception: pass

    def info(self, msg): self.add("info", msg)
    def ok(self, msg): self.add("ok", msg)
    def warn(self, msg): self.add("warn", msg)
    def error(self, msg): self.add("error", msg)

    def tail(self, n: int):
        with self._lock:
            return list(self._lines)[-n:]

    def close(self):
        if self._file is not None:
            try: self._file.close()
            except OSError: pass
            self._file = None


class TileStatus:
    __slots__ = ("stage", "detail", "t0", "t_stage", "nbytes", "host", "error")

    def __init__(self):
        self.stage, self.detail = "queued", ""
        self.t0 = self.t_stage = time.time()
        self.nbytes, self.host, self.error = 0, "", ""

    def fraction(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        base = _BASE.get(self.stage, 0.0)
        if self.stage == "download":
            # total size is unknown (Overpass streams gzip), so ease towards 0.62 over time
            return base + 0.54 * (1.0 - math.exp(-(now - self.t_stage) / 22.0))
        if self.stage == "mesh":
            return base + 0.17 * (1.0 - math.exp(-(now - self.t_stage) / 4.0))
        return base


class TileBoard:
    def __init__(self):
        self._lock = threading.Lock()
        self._tiles: dict = {}

    def set(self, key, stage: str, detail: str = ""):
        with self._lock:
            st = self._tiles.get(key)
            if st is None or (stage == "queued" and st.stage in ("done", "failed")):
                st = self._tiles[key] = TileStatus()
            if st.stage != stage:
                st.stage, st.t_stage = stage, time.time()
                if stage == "download": st.nbytes = 0
            st.detail = detail

    def update(self, key, detail=None, nbytes=None, host=None, error=None):
        with self._lock:
            st = self._tiles.get(key)
            if st is None: return
            if detail is not None: st.detail = detail
            if nbytes is not None: st.nbytes = max(st.nbytes, nbytes)
            if host is not None: st.host = host
            if error is not None: st.error = error

    def get(self, key):
        with self._lock:
            st = self._tiles.get(key)
            if st is None: return None
            c = TileStatus()
            for a in TileStatus.__slots__: setattr(c, a, getattr(st, a))
            return c

    def drop(self, key):
        with self._lock: self._tiles.pop(key, None)

    def prune(self, keep):
        with self._lock:
            for k in [k for k in self._tiles if not keep(k)]: del self._tiles[k]


LOG = EventLog()
BOARD = TileBoard()
