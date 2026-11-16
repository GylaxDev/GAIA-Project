"""Now-playing data from the Windows System Media Transport Controls (SMTC).

Windows exposes what Spotify, a browser or Media Player is playing to desktop
applications through WinRT.  Those calls are asynchronous and can take a few
milliseconds, so a daemon thread owns them and the render thread only reads an
immutable snapshot.  The integration is optional: without the ``winrt`` (or
legacy ``winsdk``) packages, or on a non-Windows machine, ``create_media_session``
returns ``None`` and the HUD omits the card.
"""
from __future__ import annotations

import asyncio
import datetime
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass

PLAYING, PAUSED, STOPPED = "playing", "paused", "stopped"

# Friendly names for the app ids Windows reports.  The fallback keeps the last
# component of the id, which is readable for most players.
_SOURCE_NAMES = (
    ("spotify", "Spotify"),
    ("zunemusic", "Media Player"),
    ("mediaplayer", "Media Player"),
    ("groove", "Media Player"),
    ("msedge", "Edge"),
    ("chrome", "Chrome"),
    ("firefox", "Firefox"),
    ("opera", "Opera"),
    ("vivaldi", "Vivaldi"),
    ("vlc", "VLC"),
    ("foobar", "foobar2000"),
    ("musicbee", "MusicBee"),
    ("tidal", "TIDAL"),
    ("deezer", "Deezer"),
    ("amazonmusic", "Amazon Music"),
    ("youtubemusic", "YouTube Music"),
    ("youtube-music", "YouTube Music"),
)


@dataclass(frozen=True)
class NowPlaying:
    """One sample of the active media session.  Immutable and thread-safe."""

    title: str
    artist: str = ""
    album: str = ""
    source: str = ""
    status: str = PLAYING
    position: float = 0.0          # seconds at ``received_at``
    duration: float = 0.0          # 0 when the stream has no known end
    art: bytes | None = None       # encoded PNG/JPEG cover, decoded on the UI thread
    track_key: tuple = ()
    received_at: float = 0.0

    @property
    def playing(self) -> bool:
        return self.status == PLAYING

    @property
    def paused(self) -> bool:
        return self.status == PAUSED

    def position_at(self, now: float) -> float:
        """Extrapolate between polls so the progress bar moves smoothly."""
        duration = max(0.0, self.duration)
        if self.playing and duration > 0.0:
            position = self.position + max(0.0, now - self.received_at)
            return min(duration, position)
        return max(0.0, min(duration, self.position)) if duration > 0.0 else max(0.0, self.position)

    @property
    def byline(self) -> str:
        return "  \u00b7  ".join(part for part in (self.artist, self.album) if part)


def timeline_progress(position: float, start: float, end: float, updated,
                      now: datetime.datetime, playing: bool) -> tuple[float, float]:
    """Return ``(elapsed, duration)`` for an SMTC timeline.

    ``Position`` is only the position *at* ``LastUpdatedTime``; many players
    (YouTube Music in particular) refresh that pair rarely.  Adding the age of
    the sample keeps the clock correct between updates.  Paused sessions must
    not advance, and players with a missing or nonsensical timestamp fall back
    to the raw position.
    """
    start = max(0.0, float(start))
    elapsed = max(0.0, float(position) - start)
    duration = max(0.0, float(end) - start)
    if playing and updated is not None:
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=datetime.timezone.utc)
        age = (now - updated).total_seconds()
        if 0.0 <= age <= 86_400.0:
            elapsed += age
    if duration > 0.0:
        elapsed = min(elapsed, duration)
    return elapsed, duration


def friendly_source(app_id: str) -> str:
    """Turn ``SpotifyAB.SpotifyMusic_zpd!Spotify`` into ``Spotify``."""
    if not app_id:
        return ""
    lowered = app_id.lower()
    for needle, name in _SOURCE_NAMES:
        if needle in lowered:
            return name
    name = app_id.rsplit("!", 1)[-1]
    name = name.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if name.lower().endswith(".exe"):
        name = name[:-4]
    name = name.rsplit(".", 1)[-1]
    return name.replace("_", " ").strip()


class MediaSession:
    """Poll the active Windows media session on a background thread."""

    def __init__(self, interval: float = 0.75):
        self.interval = max(0.25, float(interval))
        self._snapshot: NowPlaying | None = None
        self._lock = threading.Lock()
        self._commands: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self._session = None
        self._art_key: tuple | None = None
        self._art: bytes | None = None

    # --------------------------------------------------------------- lifecycle
    def start(self) -> "MediaSession":
        if self._thread is None:
            self._thread = threading.Thread(target=self._thread_main, name="now-playing",
                                            daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._commands.put(None)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.5)

    @property
    def available(self) -> bool:
        """True once the WinRT manager is ready (an empty session is still ready)."""
        return self._ready.is_set() and self._error is None

    @property
    def error(self) -> str | None:
        return self._error

    # ---------------------------------------------------------------- main thread
    def snapshot(self) -> NowPlaying | None:
        with self._lock:
            return self._snapshot

    def toggle_play_pause(self) -> None:
        self._commands.put("toggle")

    def next_track(self) -> None:
        self._commands.put("next")

    def previous_track(self) -> None:
        self._commands.put("previous")

    def _publish(self, value: NowPlaying | None) -> None:
        with self._lock:
            self._snapshot = value

    # ------------------------------------------------------------------- worker
    def _thread_main(self) -> None:
        try:
            self._init_apartment()
            asyncio.run(self._run())
        except Exception as exc:                     # pragma: no cover - platform dependent
            self._error = f"{type(exc).__name__}: {exc}"
            try:
                from ..core import telemetry as tm
                tm.LOG.warn(f"now playing unavailable: {self._error}")
            except Exception:
                pass
        finally:
            self._ready.set()
            try:
                import winrt.runtime as runtime
                runtime.uninit_apartment()
            except Exception:
                pass

    @staticmethod
    def _init_apartment() -> None:
        try:
            import winrt.runtime as runtime
        except ImportError:
            return
        try:
            runtime.init_apartment()
        except Exception:
            pass

    async def _run(self) -> None:
        try:
            from winrt.windows.media.control import (
                GlobalSystemMediaTransportControlsSessionManager as Manager,
            )
        except ImportError:                          # legacy PyWinRT package
            from winsdk.windows.media.control import (
                GlobalSystemMediaTransportControlsSessionManager as Manager,
            )
        manager = await asyncio.wait_for(Manager.request_async(), timeout=8.0)
        self._ready.set()
        last_poll = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            acted = await self._run_commands()
            if acted or now - last_poll >= self.interval:
                await self._refresh(manager)
                last_poll = time.monotonic()
            await asyncio.sleep(0.15)

    async def _run_commands(self) -> bool:
        acted = False
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            session = self._session
            if command is None or session is None:
                continue
            acted = True
            try:
                if command == "toggle":
                    await session.try_toggle_play_pause_async()
                elif command == "next":
                    await session.try_skip_next_async()
                elif command == "previous":
                    await session.try_skip_previous_async()
            except Exception:
                pass
        return acted

    async def _refresh(self, manager) -> None:
        session = manager.get_current_session()
        if session is None:
            self._session = None
            self._art_key, self._art = None, None
            self._publish(None)
            return
        self._session = session
        try:
            info = session.get_playback_info()
            status = self._status(info)
            # Only a real PLAYING sample may age the timeline; "opened" and
            # "changing" can still carry the previous track's stale timestamp.
            advancing = int(info.playback_status) == 4
            if status == STOPPED:
                self._publish(None)
                return
            props = await asyncio.wait_for(session.try_get_media_properties_async(), timeout=3.0)
            title = (props.title or props.subtitle or "").strip()
            if not title:
                self._publish(None)
                return
            artist = (props.artist or "").strip()
            album = (props.album_title or "").strip()
            source_id = session.source_app_user_model_id or ""
            key = (source_id, title, artist, album)
            if key != self._art_key:
                self._art_key = key
                self._art = await self._read_thumbnail(props)
            timeline = session.get_timeline_properties()
            position, duration = timeline_progress(
                timeline.position.total_seconds(),
                timeline.start_time.total_seconds(),
                timeline.end_time.total_seconds(),
                timeline.last_updated_time,
                datetime.datetime.now(datetime.timezone.utc),
                advancing)
            self._publish(NowPlaying(
                title=title, artist=artist, album=album,
                source=friendly_source(source_id), status=status,
                position=position, duration=duration, art=self._art,
                track_key=key, received_at=time.monotonic()))
        except Exception:
            # A player changing tracks can invalidate properties mid-read; keep
            # the previous sample and try again on the next poll.
            pass

    @staticmethod
    def _status(info) -> str:
        # 0 closed, 1 opened, 2 changing, 3 stopped, 4 playing, 5 paused.
        # Players briefly report "changing"/"opened" between tracks; keep the
        # card visible through that instead of flickering away.
        value = int(info.playback_status)
        if value == 5:
            return PAUSED
        if value in (1, 2, 4):
            return PLAYING
        return STOPPED

    @staticmethod
    async def _read_thumbnail(props) -> bytes | None:
        thumbnail = props.thumbnail
        if thumbnail is None:
            return None
        stream = None
        try:
            stream = await asyncio.wait_for(thumbnail.open_read_async(), timeout=3.0)
            size = int(stream.size)
            if size <= 0 or size > 8_000_000:
                return None
            try:
                from winrt.windows.storage.streams import DataReader
            except ImportError:
                from winsdk.windows.storage.streams import DataReader
            reader = DataReader(stream)
            await asyncio.wait_for(reader.load_async(size), timeout=3.0)
            buffer = reader.read_buffer(size)
            return bytes(memoryview(buffer))
        except Exception:
            return None
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


def create_media_session(interval: float = 0.75) -> MediaSession | None:
    """Start the Windows media reader, or return None when it is unavailable."""
    if sys.platform != "win32":
        return None
    if os.environ.get("OSM3D_NOW_PLAYING", "1").strip().lower() in ("0", "off", "false", "no"):
        return None
    try:
        import winrt.windows.media.control  # noqa: F401
    except ImportError:
        try:
            import winsdk.windows.media.control  # noqa: F401
        except ImportError:
            return None
    except Exception:
        return None
    return MediaSession(interval).start()
