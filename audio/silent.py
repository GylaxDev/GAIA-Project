"""No-op audio backend.

Sound (engine, tyres, wind, impacts and interface clicks) has been removed from
the game, so the app installs this backend instead of opening a mixer.  It keeps
the small ``CarAudio`` surface the render loop, garage and menus already call, so
those call sites stay honest and nothing has to special-case "sound is off".
"""
from __future__ import annotations


class SilentAudio:
    """Accepts every audio call and does nothing; never touches a device."""

    enabled = False
    _closed = False
    _rate = 44100

    def __getattr__(self, name):
        # Any other playback/mixer helper the app might call is a no-op too.
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None

    def update(self, dt, car=None, active=False, blocked=False, throttle=0.0,
               wetness=0.0, handbrake=False):
        return None

    def enter(self):
        return None

    def leave(self):
        return None

    def close(self):
        self._closed = True
