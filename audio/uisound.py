"""Interface sounds: soft synthesized clicks, toggles, panel open/close and an error buzz.

Borrows the car audio's mixer, so it is silent (every call a no-op) whenever audio is
unavailable or disabled (``OSM3D_AUDIO=0``).
"""
from __future__ import annotations

import math

import numpy as np

from . import car as _audio
from .car import _lowpass


def _norm(w):
    return w / max(float(np.max(np.abs(w))), 1e-9)


def _env(n, attack, rate):
    a = max(2, int(attack * rate))
    e = np.ones(n)
    e[:a] = np.linspace(0., 1., a)
    return e


def _note(rate, dur, f0, f1, decay, harmonics=(1., .25)):
    n = int(rate * dur)
    t = np.arange(n) / rate
    f = f0 + (f1 - f0) * np.clip(t / dur, 0., 1.)
    phase = 2. * math.pi * np.cumsum(f) / rate
    wave = sum(a * np.sin((i + 1) * phase) for i, a in enumerate(harmonics))
    return _norm(wave * np.exp(-t * decay) * _env(n, .002, rate))


def make_sounds(rate):
    rng = np.random.default_rng(7)
    tick_n = int(rate * .02)
    tick = _lowpass(rng.standard_normal(tick_n), rate, 6000.) * np.exp(-np.arange(tick_n) / rate * 400.)
    click = _norm(np.pad(.45 * _norm(tick), (0, int(rate * .055)))[:int(rate * .075)]
                  + np.pad(.8 * _note(rate, .06, 1850., 1350., 60.), (0, int(rate * .015)))[:int(rate * .075)])
    gap = np.zeros(int(rate * .012))
    on = _norm(np.concatenate([_note(rate, .07, 720., 740., 38.), gap, _note(rate, .10, 1080., 1100., 30.)]))
    off = _norm(np.concatenate([_note(rate, .07, 1000., 980., 38.), gap, _note(rate, .10, 640., 620., 30.)]))
    # panel open / close: a short filtered-noise sweep under a soft interval
    n = int(rate * .16)
    t = np.arange(n) / rate
    sweep = _lowpass(rng.standard_normal(n), rate, 3200.) * np.sin(math.pi * t / .16) ** 2 * .35
    opened = _norm(sweep + .8 * np.pad(_note(rate, .13, 560., 840., 18.), (0, max(0, n - int(rate * .13))))[:n])
    closed = _norm(sweep[::-1] * .8 + .8 * np.pad(_note(rate, .13, 840., 520., 22.), (0, max(0, n - int(rate * .13))))[:n])
    m = int(rate * .20)
    tt = np.arange(m) / rate
    buzz = _norm((np.sign(np.sin(2. * math.pi * 150. * tt)) * .6 + np.sin(2. * math.pi * 300. * tt) * .4)
                 * np.exp(-tt * 12.) * (np.sin(2. * math.pi * 12. * tt) > -.2))
    return {'click': click, 'on': on, 'off': off, 'open': opened, 'close': closed, 'error': _lowpass(buzz, rate, 1400.)}


class UiSound:
    VOLUME = {'click': .50, 'on': .50, 'off': .45, 'open': .55, 'close': .50, 'error': .55}

    def __init__(self, audio):
        self.sounds = {}
        self._clock = {}
        self.enabled = False
        self.muted = False          # settings: interface sounds on / off
        self.volume = 1.0           # settings: loudness relative to the master volume
        if audio is None or not getattr(audio, 'enabled', False):
            return
        try:
            for name, wave in make_sounds(audio._rate).items():
                self.sounds[name] = audio._sound(wave)
        except Exception:
            self.sounds = {}
            return
        self.audio, self.enabled = audio, True

    def play(self, name):
        if not self.enabled or self.muted or getattr(self.audio, '_closed', False):
            return
        sound = self.sounds.get(name)
        if sound is None:
            return
        try:
            sound.set_volume(min(1., _audio.MASTER * self.volume * self.VOLUME.get(name, .5)))
            sound.play()
        except Exception:
            pass

    def click(self): self.play('click')
    def toggle(self, on=True): self.play('on' if on else 'off')
    def open(self): self.play('open')
    def close(self): self.play('close')
    def error(self): self.play('error')
