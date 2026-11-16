"""Typed, persisted user settings.

A ``Setting`` is a small adapter between the menu and one live value somewhere in the
app: ``get`` reads the value from its owner (so keyboard shortcuts and the menu can
never disagree) and ``set`` applies it.  The store remembers every setting's start-up
value as its default, saves only what differs from those defaults, and re-applies the
saved values on the next launch.

The file is plain JSON (``settings.json`` in the cache directory).  A corrupt or
partly unknown file is ignored value-by-value rather than failing the launch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from ..core import telemetry as tm


@dataclass(eq=False)
class Setting:
    id: str
    label: str
    kind: str                                   # toggle | slider | choice | action | info
    get: Callable[[], Any] | None = None
    set: Callable[[Any], None] | None = None
    desc: str = ""
    lo: float = 0.0
    hi: float = 1.0
    step: float = 0.05
    fmt: Callable[[Any], str] | None = None
    choices: tuple = ()                         # ((value, label), ...)
    action: Callable[[], Any] | None = None
    button: str = "Run"
    enabled: Callable[[], bool] | None = None
    persist: bool = True
    note: str = ""                              # short badge, e.g. "next travel"
    default: Any = None
    icon: str | None = None

    # ------------------------------------------------------------------ helpers
    def is_enabled(self) -> bool:
        if self.enabled is None:
            return True
        try:
            return bool(self.enabled())
        except Exception:
            return False

    def value(self):
        try:
            return self.get() if self.get is not None else None
        except Exception:
            return None

    def text(self) -> str:
        v = self.value()
        if self.fmt is not None:
            try:
                return self.fmt(v)
            except Exception:
                return str(v)
        if self.kind == "choice":
            return dict(self.choices).get(v, str(v))
        return "" if v is None else str(v)

    def choice_index(self) -> int:
        v = self.value()
        best, gap = 0, math.inf
        for i, (cv, _) in enumerate(self.choices):
            if cv == v:
                return i
            if isinstance(cv, (int, float)) and isinstance(v, (int, float)) and abs(cv - v) < gap:
                best, gap = i, abs(cv - v)
        return best

    def fraction(self) -> float:
        v = self.value()
        if not isinstance(v, (int, float)) or self.hi <= self.lo:
            return 0.0
        return max(0.0, min(1.0, (float(v) - self.lo) / (self.hi - self.lo)))

    def quantize(self, v: float) -> float:
        v = max(self.lo, min(self.hi, float(v)))
        if self.step > 0:
            v = self.lo + round((v - self.lo) / self.step) * self.step
        return round(max(self.lo, min(self.hi, v)), 4)


@dataclass(eq=False)
class Section:
    title: str
    rows: list = field(default_factory=list)


@dataclass(eq=False)
class Category:
    id: str
    label: str
    icon: str
    blurb: str
    sections: list = field(default_factory=list)
    custom: str | None = None                   # a page-drawn category (e.g. "controls")

    def settings(self):
        return [r for s in self.sections for r in s.rows]


def _same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    return a == b


class SettingsStore:
    VERSION = 1

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.categories: list[Category] = []
        self.by_id: dict[str, Setting] = {}
        self.dirty = False

    # -------------------------------------------------------------- registry
    def register(self, categories):
        self.categories = list(categories)
        self.by_id = {s.id: s for c in self.categories for s in c.settings()}

    def capture_defaults(self):
        for s in self.by_id.values():
            if s.kind not in ("action", "info"):
                s.default = s.value()

    # ----------------------------------------------------------- persistence
    def load(self) -> dict:
        if self.path is None or not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            values = data.get("values", {}) if isinstance(data, dict) else {}
            return values if isinstance(values, dict) else {}
        except (OSError, ValueError) as exc:
            tm.LOG.warn(f"settings: ignoring unreadable {self.path.name} ({exc})")
            return {}

    def _coerce(self, s: Setting, v):
        if s.kind == "toggle":
            return bool(v) if isinstance(v, (bool, int)) else None
        if s.kind == "slider":
            return s.quantize(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None
        if s.kind == "choice":
            for cv, _ in s.choices:
                if _same(cv, v):
                    return cv
        return None

    def apply_stored(self) -> int:
        applied = 0
        for sid, raw in self.load().items():
            s = self.by_id.get(sid)
            if s is None or not s.persist or s.set is None or s.kind in ("action", "info"):
                continue
            value = self._coerce(s, raw)
            if value is None:
                continue
            try:
                s.set(value)
                applied += 1
            except Exception as exc:
                tm.LOG.warn(f"settings: could not apply {sid}={raw!r} ({exc})")
        return applied

    def snapshot(self) -> dict:
        out = {}
        for sid, s in self.by_id.items():
            if not s.persist or s.kind in ("action", "info") or s.default is None:
                continue
            v = s.value()
            if v is not None and not _same(v, s.default):
                out[sid] = v
        return out

    def save(self, force=False) -> bool:
        if self.path is None or not (self.dirty or force):
            return False
        payload = {"version": self.VERSION, "values": self.snapshot()}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix="settings.", suffix=".tmp", dir=str(self.path.parent))
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError as exc:
            tm.LOG.warn(f"settings: could not save ({exc})")
            return False
        self.dirty = False
        return True

    # --------------------------------------------------------------- changes
    def change(self, s: Setting, value):
        """Apply a new value from the UI."""
        if s.set is None or not s.is_enabled():
            return
        s.set(value)
        if s.persist:
            self.dirty = True

    def reset(self, s: Setting):
        if s.kind in ("action", "info") or s.default is None or s.set is None:
            return
        s.set(s.default)                        # resets ignore "enabled": order must not matter
        if s.persist:
            self.dirty = True

    def reset_category(self, cat: Category):
        for s in cat.settings():
            self.reset(s)

    def reset_all(self):
        for s in self.by_id.values():
            self.reset(s)
