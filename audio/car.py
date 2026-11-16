"""Recorded V8 engine and procedural tyres, impacts, wind and indicators.

The engine follows a sound-only automatic gearbox and is continuously resampled
from real combustion cycles. An audio worker keeps it playing through frame
hitches. Construction is safe when no audio device exists (headless
runs, CI, ``OSM3D_AUDIO=0``): ``enabled`` stays ``False`` and every method is a
no-op.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .engine_audio import (ASSETS, EngineBank, EngineState, IDLE_RPM, RATE, REDLINE_RPM,
                           procedural_engine_loop, _read_wave)
from .engine_stream import EngineStream

BUFFER = 512                 # ~12 ms at 44.1 kHz
OUT_CHANNELS = 2             # keep the ambient soundscape spatial/stereo, too
TOP_SPEED = 160. / 3.6       # m/s, the arcade cap in driving.py
MASTER = .55
_BASE_MASTER = MASTER        # the authored level; the settings slider scales it (1.0 = authored)
INDICATOR_PERIOD = .42
CHANNELS = 8
RESERVED = 4                 # engine/fallback, spare, wind, tyres; ambience follows
SKID_PITCH = 1.5             # playback ratio of the recorded skid loop (1.5 = a fifth higher)
SHOCK_MIN = 2.0              # m/s of wheel vertical speed below which a bump stays silent
ECHO = 1.0                   # master scale of the street echo on car one-shots / slides (0 = dry)


def set_master_volume(scale):
    """Scale every car and interface sound (1.0 = the authored mix, 0 = silent).

    ``MASTER`` is read at call time by every sound path, so a change is heard on the
    very next engine update or one-shot.
    """
    global MASTER
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return
    if np.isfinite(scale):
        MASTER = _BASE_MASTER * max(0., min(1.5, scale))


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0., 1.)
    return t * t * (3. - 2. * t)


def _shaped_noise(rate, length, low, high, tilt=0., seed=0):
    """Band-limited white noise shaped in the frequency domain.

    The inverse FFT is circular, so the buffer loops without a seam.
    """
    n = int(rate * length)
    n -= n % 2
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n, 1. / rate)
    spec = np.fft.rfft(rng.standard_normal(n))
    mask = _smoothstep(low * .55, low, f) * (1. - _smoothstep(high, high * 1.4, f))
    spec *= mask * np.maximum(f, 30.) ** tilt
    wave = np.fft.irfft(spec, n)
    peak = float(np.max(np.abs(wave))) if n else 0.
    return wave / peak if peak > 1e-9 else wave


def _lowpass(wave, rate, cutoff):
    n = len(wave)
    f = np.fft.rfftfreq(n, 1. / rate)
    spec = np.fft.rfft(wave) * (1. - _smoothstep(cutoff * .5, cutoff, f))
    return np.fft.irfft(spec, n)


def _engine_loop(rate, rpm):
    """Asset-free exhaust pulses for passing traffic (not the player's engine)."""
    return procedural_engine_loop(rate, rpm)


def _wind_loop(rate):
    wave = _shaped_noise(rate, 1., 180., 5200., tilt=-.35, seed=11)
    return wave * .9


def _tyre_loop(rate):
    """Squeal: narrow-band noise with a slow, seamless amplitude wobble."""
    n = int(rate)
    n -= n % 2
    wave = _shaped_noise(rate, 1., 780., 4200., tilt=-.1, seed=29)
    t = np.arange(n, dtype=np.float64) / n
    wave *= .78 + .22 * np.sin(2. * math.pi * 7. * t) + .10 * np.sin(2. * math.pi * 19. * t)
    peak = float(np.max(np.abs(wave)))
    return wave / peak if peak > 1e-9 else wave


def _seamless(wave, rate, fade=.012):
    """Equal-power cross-fade of the tail into the head, so a short loop has no click."""
    n = len(wave)
    k = min(int(rate * fade), n // 4)
    if k < 2:
        return wave
    ramp = np.linspace(0., 1., k)
    out = wave[:n - k].copy()
    out[:k] = wave[:k] * np.sqrt(ramp) + wave[n - k:] * np.sqrt(1. - ramp)
    return out


def _pitch_loop(wave, ratio):
    """Resample a loop by ``ratio`` (>1 = higher and shorter) without breaking its seam."""
    n = len(wave)
    m = max(64, int(round(n / ratio)))
    pos = np.arange(m) * (n / m)
    i = pos.astype(int)
    frac = pos - i
    return wave[i % n] * (1. - frac) + wave[(i + 1) % n] * frac


def _street_ir(rate, rt60, seed):
    """Impulse response of a city street: canyon slap-backs plus a dark, diffuse tail."""
    n = int(rate * rt60 * 1.05)
    t = np.arange(n, dtype=np.float64) / rate
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    # highs die faster than lows, as in a real reverberant street
    tail = (.65 * _lowpass(noise, rate, 6500.) * 10. ** (-3. * t / (rt60 * .55))
            + .35 * _lowpass(noise, rate, 1600.) * 10. ** (-3. * t / rt60))
    tail *= np.clip((t - .012) / .03, 0., 1.)                   # pre-delay and soft onset
    ir = tail / max(float(np.sqrt(np.sum(tail ** 2))), 1e-9)
    for delay, gain in ((.037, .50), (.071, .40), (.113, .32), (.171, .24), (.248, .16)):
        if int(delay * rate) < n:
            ir[int(delay * rate)] += gain * .35               # building-face slap-backs
    return ir


def _reverb(wave, rate, wet=.35, rt60=1.0, circular=False, seed=17):
    """Mix a street echo into ``wave``. ``circular`` keeps a loop seamless (no tail added)."""
    wet *= ECHO
    if wet <= 1e-4:
        return wave
    n = len(wave)
    ir = _street_ir(rate, rt60, seed)
    if circular:
        ir = ir[:n]
        conv = np.fft.irfft(np.fft.rfft(wave) * np.fft.rfft(ir, n), n)
        out = wave + wet * conv * (float(np.sqrt(np.mean(wave ** 2))) / max(float(np.sqrt(np.mean(conv ** 2))), 1e-9))
    else:
        size = n + len(ir)
        conv = np.fft.irfft(np.fft.rfft(wave, size) * np.fft.rfft(ir, size), size)
        wet_part = wet * conv * (float(np.sqrt(np.mean(wave ** 2))) / max(float(np.sqrt(np.mean(conv[:n] ** 2))), 1e-9))
        out = np.zeros(size)
        out[:n] += wave
        out += wet_part
    peak = float(np.max(np.abs(out)))
    return out / peak * min(peak, 1.) if peak > 1. else out


def _squeal_loop(rate):
    """Tyre squeal: a few warbling stick-slip tones plus a hiss of band noise.

    Integer frequencies and warble rates over a one-second buffer make it loop seamlessly.
    """
    n = int(rate)
    t = np.arange(n, dtype=np.float64) / n
    wave = np.zeros(n)
    for freq, amp, warble, depth, phase in ((1210., 1., 3., 24., .3), (1830., .75, 5., 34., 1.7),
                                            (2480., .5, 7., 44., 2.9), (905., .45, 2., 16., 4.1)):
        wave += amp * np.sin(2. * math.pi * freq * t + phase
                             + (depth / warble) * np.sin(2. * math.pi * warble * t))
    wave /= max(float(np.max(np.abs(wave))), 1e-9)
    wave += .35 * _shaped_noise(rate, 1., 1000., 4500., tilt=0., seed=31)
    wave *= .80 + .20 * np.sin(2. * math.pi * 11. * t) + .08 * np.sin(2. * math.pi * 23. * t)
    return wave / max(float(np.max(np.abs(wave))), 1e-9)


def _scrub_loop(rate):
    """Rubber on asphalt: broadband roar for rolling at speed, and the scrub of a slide."""
    n = int(rate)
    t = np.arange(n, dtype=np.float64) / n
    wave = _shaped_noise(rate, 1., 110., 1400., tilt=-.5, seed=53)
    wave += .25 * _shaped_noise(rate, 1., 1600., 5200., tilt=-.2, seed=59)
    wave *= .88 + .08 * np.sin(2. * math.pi * 3. * t) + .06 * np.sin(2. * math.pi * 11. * t)
    return wave / max(float(np.max(np.abs(wave))), 1e-9)


def _pop_sound(rate, seed):
    """An overrun exhaust pop: a short pressure thump with a ragged burst of noise."""
    n = int(rate * .14)
    t = np.arange(n, dtype=np.float64) / rate
    rng = np.random.default_rng(seed)
    thump = np.sin(2. * math.pi * (150. - 700. * t) * t) * np.exp(-t * (40. + 8. * (seed % 3)))
    burst = _lowpass(rng.standard_normal(n), rate, 2600. + 400. * (seed % 3)) * np.exp(-t * 65.)
    wave = .7 * thump + .55 * burst
    return wave / max(float(np.max(np.abs(wave))), 1e-9)


def _clunk_sound(rate):
    """A gearbox shift: soft driveline thud."""
    n = int(rate * .10)
    t = np.arange(n, dtype=np.float64) / rate
    body = np.sin(2. * math.pi * (88. - 160. * t) * t) * np.exp(-t * 38.)
    tick = _lowpass(np.random.default_rng(5).standard_normal(n), rate, 1800.) * np.exp(-t * 120.)
    wave = .8 * body + .25 * tick
    return wave / max(float(np.max(np.abs(wave))), 1e-9)


def _ratchet_sound(rate):
    """Handbrake lever: a run of ratchet clicks ending in a clack."""
    n = int(rate * .22)
    wave = np.zeros(n)
    rng = np.random.default_rng(83)
    for i in range(6):
        at = int(rate * (.012 + .016 * i + .0006 * i * i))
        k = int(rate * .012)
        tt = np.arange(k) / rate
        click = _lowpass(rng.standard_normal(k), rate, 4200.) * np.exp(-tt * 480.)
        wave[at:at + k] += click * (.5 + .08 * i)
    at = int(rate * .15)
    tt = np.arange(n - at) / rate
    wave[at:] += (np.sin(2. * math.pi * 520. * tt) * np.exp(-tt * 70.)
                  + .4 * np.sin(2. * math.pi * 1250. * tt) * np.exp(-tt * 110.)) * .8
    return wave / max(float(np.max(np.abs(wave))), 1e-9)


def _impact_variant(rate, weight):
    """Collision of weight 0 (bump) .. 1 (hard crash): bumper thud, then body crunch and debris."""
    n = int(rate * (.30 + .45 * weight))
    t = np.arange(n, dtype=np.float64) / rate
    rng = np.random.default_rng(41 + int(weight * 10))
    thump = np.sin(2. * math.pi * ((96. - 20. * weight) - 34. * t) * t) * np.exp(-t * (11. - 4. * weight))
    crunch = _lowpass(rng.standard_normal(n), rate, 1200. + 1400. * weight) * np.exp(-t * (30. - 12. * weight))
    knock = np.sin(2. * math.pi * (230. - 40. * weight) * t) * np.exp(-t * 30.)
    wave = .85 * thump + (.30 + .45 * weight) * crunch + .30 * knock
    if weight > .45:                                 # broken trim / glass tinkle
        for i in range(10):
            at = int(rate * (.03 + .35 * rng.random() * weight))
            freq = 2600. + 3200. * rng.random()
            k = min(int(rate * .06), n - at)
            if k > 8:
                tt = np.arange(k) / rate
                wave[at:at + k] += .10 * weight * np.sin(2. * math.pi * freq * tt) * np.exp(-tt * 90.)
    return wave / max(float(np.max(np.abs(wave))), 1e-9)


def _impact_sound(rate):
    n = int(rate * .55)
    t = np.arange(n, dtype=np.float64) / rate
    thump = np.sin(2. * math.pi * (74. - 30. * t) * t) * np.exp(-t * 7.5)
    crunch = _lowpass(np.random.default_rng(41).standard_normal(n), rate, 1800.) * np.exp(-t * 22.)
    knock = np.sin(2. * math.pi * 190. * t) * np.exp(-t * 26.)
    wave = .85 * thump + .55 * crunch + .35 * knock
    peak = float(np.max(np.abs(wave)))
    return wave / peak if peak > 1e-9 else wave


def _door_sound(rate):
    n = int(rate * .26)
    t = np.arange(n, dtype=np.float64) / rate
    body = np.sin(2. * math.pi * (96. - 44. * t) * t) * np.exp(-t * 17.)
    latch = _lowpass(np.random.default_rng(73).standard_normal(n), rate, 2600.) * np.exp(-t * 55.)
    wave = .8 * body + .45 * latch
    peak = float(np.max(np.abs(wave)))
    return wave / peak if peak > 1e-9 else wave


def _tick_sound(rate):
    n = int(rate * .06)
    t = np.arange(n, dtype=np.float64) / rate
    click = np.sin(2. * math.pi * 1450. * t) * np.exp(-t * 190.)
    snap = _lowpass(np.random.default_rng(97).standard_normal(n), rate, 5200.) * np.exp(-t * 320.)
    wave = .7 * click + .5 * snap
    peak = float(np.max(np.abs(wave)))
    return wave / peak if peak > 1e-9 else wave


class CarAudio:
    """Engine and road sounds for the driven car; a no-op without an audio device."""

    def __init__(self, enabled=None):
        self.enabled = False
        self.pg = None
        self._out_channels = OUT_CHANNELS
        self._engine = None
        self.engine_state = EngineState()
        self._bank = None
        self._wind = self._tyre = self._scrub = None
        self._wind_ch = self._tyre_ch = self._road_ch = None
        self._impact = self._door = self._tick = None
        self._impacts = []
        self._suspension = []
        self._shock_clock = 0.
        self._pops = []
        self._clunk = self._ratchet = None
        self._gear, self._prev_power = 1, 0.
        self._pop_clock, self._pop_queue, self._pop_i = 0., [], 0
        self._handbrake = False
        self._speed = 0.
        self._tick_clock = 0.
        self._closed = False
        if enabled is None:
            enabled = os.environ.get('OSM3D_AUDIO', '1').strip().lower() not in (
                '0', 'false', 'no', 'off')
        if not enabled:
            return
        try:
            import pygame
        except ImportError:
            return
        self.pg = pygame
        try:
            if not self._open_mixer():
                self.pg = None
                return
            self._build(pygame)
        except (pygame.error, AttributeError, ValueError, MemoryError):
            self._stop_channels()
            self.pg = None
        self.enabled = self.pg is not None

    # -- setup -----------------------------------------------------------
    def _open_mixer(self):
        """Reuse pygame's mixer when its format suits int16 arrays, else re-open it."""
        pg = self.pg
        info = pg.mixer.get_init()
        if info is not None and (info[1] != -16 or info[2] not in (1, 2)):
            pg.mixer.quit()
            info = None
        if info is None:
            try:
                pg.mixer.init(RATE, -16, OUT_CHANNELS, BUFFER)
            except pg.error:
                return False
            info = pg.mixer.get_init()
        if info is None:
            return False
        self._rate, self._out_channels = int(info[0]), int(info[2])
        return True

    def _sound(self, wave):
        pcm = (np.clip(wave, -1., 1.) * 32000.).astype(np.int16)
        if self._out_channels == 2:
            pcm = np.column_stack((pcm, pcm))
        return self.pg.sndarray.make_sound(np.ascontiguousarray(pcm))

    def _build(self, pg):
        pg.mixer.set_num_channels(CHANNELS)
        pg.mixer.set_reserved(RESERVED)
        self._bank = EngineBank.load(self._rate)
        try:                                        # recorded in-car wind, a seamless 10 s loop
            wind = _read_wave(ASSETS / 'car_wind.wav', self._rate).astype(np.float64)
        except (OSError, ValueError, EOFError):
            wind = _wind_loop(self._rate)
        self._wind = self._sound(wind)
        self._suspension = []
        for index in range(3):                      # light / medium / heavy suspension hit
            try:
                clip = _read_wave(ASSETS / f'car_suspension_{index}.wav', self._rate).astype(np.float64)
            except (OSError, ValueError, EOFError):
                continue
            self._suspension.append(self._sound(_reverb(clip, self._rate, .35, .9, seed=60 + index)))
        r = self._rate
        # Recorded brake squeal (seamless loop) and crash, both with a street echo; fall back
        # to the synthesized versions if an asset is missing or unreadable.
        try:
            skid = _read_wave(ASSETS / 'car_skid.wav', r).astype(np.float64)
        except (OSError, ValueError, EOFError):
            skid = _squeal_loop(r)
        try:
            crash = _read_wave(ASSETS / 'car_crash.wav', r).astype(np.float64)
        except (OSError, ValueError, EOFError):
            crash = _impact_variant(r, .95)
        skid = _pitch_loop(skid, SKID_PITCH)           # a higher, tighter squeal
        self._tyre = self._sound(_reverb(skid, r, .30, .8, circular=True, seed=3))
        self._scrub = self._sound(_scrub_loop(r))
        heavy = _impact_variant(r, .95)
        mixed = np.zeros(len(crash)); mixed[:len(heavy)] += .45 * heavy[:len(mixed)]
        self._impacts = [self._sound(_reverb(_impact_variant(r, .1), r, .45, .9, seed=5)),
                         self._sound(_reverb(crash, r, .55, 1.6, seed=7)),
                         self._sound(_reverb(crash * .85 + mixed, r, .65, 2.0, seed=9))]
        self._pops = [self._sound(_reverb(_pop_sound(r, s), r, .55, 1.1, seed=20 + s)) for s in (1, 2, 3)]
        self._clunk = self._sound(_reverb(_clunk_sound(r), r, .25, .6, seed=31))
        self._ratchet = self._sound(_reverb(_ratchet_sound(r), r, .40, .9, seed=33))
        self._impact = self._impacts[1]
        self._door = self._sound(_reverb(_door_sound(self._rate), self._rate, .45, 1.0, seed=37))
        self._tick = self._sound(_tick_sound(self._rate))
        self._wind_ch, self._tyre_ch = pg.mixer.Channel(2), pg.mixer.Channel(3)
        self._road_ch = pg.mixer.Channel(1)       # the "spare" reserved channel: road roar / slide scrub
        self._wind_ch.play(self._wind, loops=-1)
        self._tyre_ch.play(self._tyre, loops=-1)
        self._road_ch.play(self._scrub, loops=-1)
        self._wind_ch.set_volume(0.)
        self._tyre_ch.set_volume(0.)
        self._road_ch.set_volume(0.)
        self._engine = EngineStream(pg, pg.mixer.Channel(0), self._bank, self._out_channels)

    # -- events ----------------------------------------------------------
    def enter(self):
        if not self.enabled:
            return
        self._speed = 0.
        self._tick_clock = 0.
        self._gear, self._prev_power, self._pop_queue = 1, 0., []
        self._handbrake = False
        self._shock_clock = .6
        self.engine_state.reset()
        # The starter and live idle share the audio sample clock/output. A
        # one-shot can no longer finish before a delayed frame unmutes the engine.
        self._engine.set_state(IDLE_RPM, 0., MASTER * .5, active=True, reset=True, startup=True)
        self._play(self._door, MASTER * .8)

    def leave(self):
        if not self.enabled:
            return
        self._play(self._door, MASTER * .55)
        self._silence()

    def close(self):
        if not self.enabled or self._closed:
            return
        self._closed = True
        self._stop_channels()
        self.enabled = False

    def _stop_channels(self):
        if self._engine is not None:
            self._engine.close()
        for channel in (self._wind_ch, self._tyre_ch, self._road_ch):
            if channel is not None:
                channel.stop()
        self._engine, self._wind_ch, self._tyre_ch, self._road_ch = None, None, None, None

    # -- per-frame -------------------------------------------------------
    def _play(self, sound, volume):
        if sound is None:
            return
        sound.set_volume(float(np.clip(volume, 0., 1.)))
        sound.play()

    def _silence(self):
        if self._closed:
            return
        self._engine.set_state(IDLE_RPM, 0., 0., active=False)
        self._wind_ch.set_volume(0.)
        self._tyre_ch.set_volume(0.)
        self._road_ch.set_volume(0.)

    def update(self, dt, car, *, active=True, blocked=False, throttle=0., wetness=0., handbrake=False):
        if not self.enabled or self._closed:
            return
        try:
            dt = float(dt)
        except (TypeError, ValueError):
            return
        if not math.isfinite(dt) or dt < 0.:
            return
        if car is None or not active:
            self._speed, self._tick_clock = 0., 0.
            self.engine_state.reset()
            self._silence()
            return
        try:
            signed_speed, steer = float(car.speed), abs(float(car.steering))
        except (TypeError, ValueError):
            self._silence()
            return
        if not math.isfinite(signed_speed) or not math.isfinite(steer):
            self._silence()
            return
        speed = abs(signed_speed)
        previous, self._speed = self._speed, speed
        if blocked and previous > 1.:
            # The fixed step zeroes speed on impact, so the pre-collision speed
            # (tracked here) is what sets the thump's weight and which crash is heard.
            index = 0 if previous < 5. else (1 if previous < 12. else 2)
            self._play(self._impacts[index] if self._impacts else self._impact,
                       .16 + min(previous / 18., 1.) * .78)
        state = self.engine_state
        rpm, power, shift_gain = state.update(dt, signed_speed, throttle,
                                              braking=car.braking, blocked=blocked)
        gain = MASTER * (.48 + .36 * power + .10 * min(rpm / REDLINE_RPM, 1.))
        self._engine.set_state(rpm, power, gain * shift_gain)
        # Driveline: a soft thud on each gear change; overrun pops when the throttle is lifted
        # at high revs (a short, randomly spaced burst, rate limited).
        if state.gear != self._gear:
            if not state.reverse and speed > 2.:
                self._play(self._clunk, MASTER * (.16 + .20 * power))
            self._gear = state.gear
        self._pop_clock = max(0., self._pop_clock - dt)
        # the driver's own throttle input (the engine's smoothed `power` lags a lift by ~0.5 s)
        pedal = max(0., float(throttle)) if signed_speed > 0. else 0.
        if (self._prev_power > .45 and pedal < .1 and rpm > 3300. and speed > 6.
                and self._pop_clock <= 0. and self._pops):
            count = 1 + int(rpm > 4300.) + int(rpm > 5000.)
            gap = 0.
            for _ in range(count):
                self._pop_queue.append((gap, MASTER * (.14 + .34 * min(rpm / REDLINE_RPM, 1.))))
                gap += .045 + .06 * ((self._pop_i * 7) % 5) / 4.
                self._pop_i += 1
            self._pop_clock = 1.2
        self._prev_power = pedal
        if self._pop_queue:
            due = [p for p in self._pop_queue if p[0] - dt <= 0.]
            self._pop_queue = [(p[0] - dt, p[1]) for p in self._pop_queue if p[0] - dt > 0.]
            for _, volume in due:
                self._play(self._pops[self._pop_i % len(self._pops)], volume)
                self._pop_i += 1
        # Suspension: wheel vertical speed is ~0 on a smooth road and spikes on a bump, kerb or
        # landing. Class by size (small hit / big hit), cool down so one event is one sound.
        self._shock_clock = max(0., self._shock_clock - dt)
        if self._suspension and getattr(car, 'suspension_ready', False) and self._shock_clock <= 0.:
            try:
                shock = float(np.max(np.abs(car.wheel_rate)))
            except (TypeError, ValueError):
                shock = 0.
            if math.isfinite(shock) and shock > SHOCK_MIN and speed > 3.:
                index = min(len(self._suspension) - 1, 0 if shock < 3.2 else (1 if shock < 4.8 else 2))
                # quiet for a small bump, rising steeply only for a real kerb / landing
                level = float(_smoothstep(SHOCK_MIN, 6.5, shock)) ** 1.5
                self._play(self._suspension[index],
                           MASTER * (.05 + .65 * level) * (.4 + .6 * min(speed / 12., 1.)))
                self._shock_clock = .6
        if handbrake and not self._handbrake and speed > 2.:
            self._play(self._ratchet, MASTER * .55)
        self._handbrake = bool(handbrake)
        try:
            wetness = float(wetness)
        except (TypeError, ValueError):
            wetness = 0.
        wetness = float(np.clip(wetness, 0., 1.)) if math.isfinite(wetness) else 0.
        # Rolling noise and wind rise with speed; the road roar is louder when it is wet.
        road = min(speed / TOP_SPEED, 1.)
        self._wind_ch.set_volume(MASTER * (.42 * road ** 1.3 + .05 * wetness * road))
        # A real slide (handbrake, wet, over the grip limit) drives squeal and scrub; hard
        # cornering near the limit howls before it lets go. Steering alone does not.
        slide = max(float(getattr(car, 'skid_front', 0.)), float(getattr(car, 'skid_rear', 0.)))
        slide = min(max(slide, 0.), 1.) if math.isfinite(slide) else 0.
        try:
            lateral_g = abs(speed * float(getattr(car, 'yaw_rate', 0.))) / 9.3
        except (TypeError, ValueError):
            lateral_g = 0.
        howl = float(_smoothstep(.62, 1.0, lateral_g)) * .45 if math.isfinite(lateral_g) else 0.
        squeal = max(slide, howl) * min(speed / 6., 1.)
        self._tyre_ch.set_volume(MASTER * .5 * squeal * (1.15 - .3 * wetness))
        self._road_ch.set_volume(min(1., MASTER * (.30 * road ** .8 * (1. + .7 * wetness)
                                                   + .50 * slide * min(speed / 5., 1.))))
        if car.indicator:
            self._tick_clock -= dt
            if self._tick_clock <= 0.:
                self._play(self._tick, MASTER * .6)
                self._tick_clock = INDICATOR_PERIOD
        else:
            self._tick_clock = 0.
