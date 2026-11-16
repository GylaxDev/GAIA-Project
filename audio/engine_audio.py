"""Recorded V8 loops, a sound-only gearbox, and continuous-rate engine playback.

The renderer is numpy-only. Pygame cannot change a Sound's playback rate, so we
render short, phase-continuous buffers instead of cross-fading musical notes.
All recordings contain whole 720-degree engine cycles and share one crank clock.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import wave

import numpy as np

RATE = 44100
IDLE_RPM = 780.
REDLINE_RPM = 6200.
FIRING_ORDER = 4.             # four-stroke V8: four firings per crank revolution
ASSETS = Path(__file__).resolve().parents[1] / 'assets' / 'audio'
GEAR_RATIOS = (3.40, 2.38, 1.72, 1.30, 1.00, .80)
FINAL_DRIVE = 4.10
WHEEL_RADIUS = .31


def _finite(value, default=0.):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _filter(wave_data, rate, cutoff):
    f = np.fft.rfftfreq(len(wave_data), 1. / rate)
    return np.fft.irfft(np.fft.rfft(wave_data) / (1. + (f / cutoff) ** 4),
                        len(wave_data)).astype(np.float32)


def procedural_engine_loop(rate, rpm, cycles=8):
    """Asset-free fallback: uneven exhaust banks and filtered combustion impulses.

    Unlike a harmonic oscillator bank, the exhaust's resonances stay at fixed
    frequencies while the firing rate changes. Also used by distant AI traffic.
    """
    rpm = max(150., _finite(rpm, IDLE_RPM))
    n = max(256, int(round(rate * 120. * cycles / rpm)))
    rng = np.random.default_rng(101 + int(rpm))
    left, right = np.zeros(n), np.zeros(n)
    # Cross-plane banks have unequal gaps even though the combined engine fires
    # every 90 degrees. Alternate cylinder strengths keep the exhaust lumpy.
    banks = (0, 1, 0, 0, 1, 0, 1, 1)
    strength = (1., .93, 1.08, .96, 1.03, .91, 1.06, .98)
    for cycle in range(cycles):
        for cylinder, bank in enumerate(banks):
            p = (cycle + cylinder / 8.) * n / cycles
            i, part = int(p), p % 1.
            out = right if bank else left
            gain = strength[cylinder] * rng.uniform(.90, 1.10)
            out[i % n] += gain * (1. - part)
            out[(i + 1) % n] += gain * part
    f = np.fft.rfftfreq(n, 1. / rate)
    # Pipe/body resonances, not a scale of sine-wave partials.
    def resonance(hz, damping):
        q = f / hz
        return 1. / (1. - q * q + 2j * damping * q)
    response = (resonance(92., .52) + .30 * resonance(310., .42)
                + .12 * resonance(920., .65)) / (1. + (f / 3200.) ** 2)
    response *= f / np.maximum(f, 25.)
    exhaust = np.fft.irfft((np.fft.rfft(left) + .72 * np.fft.rfft(right)
                           * np.exp(-2j * math.pi * f * .0045)) * response, n)
    exhaust /= max(float(np.sqrt(np.mean(exhaust ** 2))), 1e-9)
    noise = rng.standard_normal(n)
    noise = _filter(noise, rate, 2700.) - _filter(noise, rate, 180.)
    noise /= max(float(np.sqrt(np.mean(noise ** 2))), 1e-9)
    envelope = _filter(left + right, rate, 240.)
    envelope /= max(float(np.max(np.abs(envelope))), 1e-9)
    out = .78 * exhaust + .16 * noise * (.4 + .6 * np.abs(envelope))
    out = np.tanh(out * .8)
    out -= out.mean()
    return (out * (.88 / max(float(np.max(np.abs(out))), 1e-9))).astype(np.float32)


def _read_wave(path, rate):
    with wave.open(str(path), 'rb') as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise ValueError('engine assets must be mono PCM16')
        src_rate, count = source.getframerate(), source.getnframes()
        if not 8000 <= src_rate <= 192000 or not 64 <= count <= src_rate * 12:
            raise ValueError('invalid engine asset length or rate')
        data = np.frombuffer(source.readframes(count), dtype='<i2').astype(np.float32) / 32768.
    if len(data) != count or not np.isfinite(data).all():
        raise ValueError('incomplete engine asset')
    if src_rate != rate:
        length = max(64, int(round(len(data) * rate / src_rate)))
        data = np.interp(np.arange(length) * src_rate / rate,
                         np.arange(len(data) + 1), np.r_[data, data[0]]).astype(np.float32)
    return data


@dataclass(frozen=True)
class EngineLayer:
    name: str
    rpm: float
    cycles: int
    wave: np.ndarray
    coast: np.ndarray


class EngineBank:
    """Small immutable recordings; missing/broken assets fall back safely."""

    def __init__(self, rate, layers, startup=None, recorded=False):
        self.rate = int(rate)
        layers = sorted(layers, key=lambda layer: layer.rpm)
        if not layers:
            raise ValueError('an engine needs at least one layer')
        prepared = []
        for i, layer in enumerate(layers):
            highest = layers[i + 1].rpm if i + 1 < len(layers) else REDLINE_RPM
            cutoff = .46 * self.rate * min(1., layer.rpm / highest)
            f = np.fft.rfftfreq(len(layer.wave), 1. / self.rate)
            fade = np.clip((cutoff - f) / max(cutoff * .15, 1.), 0., 1.)
            fade = fade * fade * (3. - 2. * fade)
            data = np.fft.irfft(np.fft.rfft(layer.wave) * fade, len(layer.wave)).astype(np.float32)
            if prepared:
                # Positive carrier phase alone does not align different exhaust
                # recordings. Align their mean combustion cycle, so blends do not
                # cancel the low engine note or pump as RPM crosses a layer.
                size = 2048
                def mean_cycle(wave_data, count):
                    positions = ((np.arange(count)[:, None] + np.arange(size) / size)
                                 * len(wave_data) / count)
                    indices = positions.astype(np.int64)
                    part = positions - indices
                    sampled = (wave_data[indices % len(wave_data)] * (1. - part)
                               + wave_data[(indices + 1) % len(wave_data)] * part)
                    return sampled.mean(axis=0)
                previous = prepared[-1]
                a, b = mean_cycle(previous.wave, previous.cycles), mean_cycle(data, layer.cycles)
                correlation = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), size)
                offset = int(np.argmax(correlation)) * len(data) / (size * layer.cycles)
                positions = (np.arange(len(data)) - offset) % len(data)
                indices = positions.astype(np.int64)
                part = positions - indices
                data = (data[indices] * (1. - part) + data[(indices + 1) % len(data)] * part).astype(np.float32)
            coast = _filter(data, self.rate, 1050.)
            data.setflags(write=False)
            coast.setflags(write=False)
            prepared.append(EngineLayer(layer.name, layer.rpm, layer.cycles, data, coast))
        self.layers = tuple(prepared)
        self.startup, self.recorded = startup, bool(recorded)
        self.log_rpm = np.log([layer.rpm for layer in self.layers])

    @classmethod
    def load(cls, rate=RATE, directory=ASSETS):
        directory = Path(directory)
        try:
            manifest = json.loads((directory / 'v8.json').read_text(encoding='utf-8'))
            if manifest['version'] not in (1, 2) or not 1 <= len(manifest['layers']) <= 8:
                raise ValueError('unsupported engine bank')
            layers = []
            for item in manifest['layers']:
                filename = item['file']
                if Path(filename).name != filename:
                    raise ValueError('engine files must be in the asset directory')
                data = _read_wave(directory / filename, rate)
                cycles = int(item['cycles'])
                if not 1 <= cycles <= 100:
                    raise ValueError('invalid engine cycle count')
                # Derive the exact rate from the quantised sample length, so every
                # layer runs at the same pitch even with a different mixer rate.
                rpm = 120. * cycles * rate / len(data)
                if not 150. <= rpm <= 10000.:
                    raise ValueError('invalid reference RPM')
                layers.append(EngineLayer(str(item['name']), rpm, cycles, data,
                                          _filter(data, rate, 1050.)))
            startup_file = manifest.get('startup')
            if startup_file and Path(startup_file).name != startup_file:
                raise ValueError('invalid startup filename')
            startup = _read_wave(directory / startup_file, rate) if startup_file else None
            return cls(rate, layers, startup, recorded=True)
        except (OSError, ValueError, TypeError, KeyError, wave.Error, EOFError):
            layers = []
            for rpm in (IDLE_RPM, 1700., 3100., 5200.):
                data = procedural_engine_loop(rate, rpm)
                actual = 120. * 8 * rate / len(data)
                layers.append(EngineLayer('fallback', actual, 8, data,
                                          _filter(data, rate, 1050.)))
            return cls(rate, layers)


class EngineState:
    """An automatic gearbox for sound only; never changes driving physics or HUD."""

    STEP = 1. / 120.

    def __init__(self):
        self.reset()

    def reset(self):
        self.rpm, self.load, self.shift_gain = IDLE_RPM, 0., 1.
        self.gear, self.reverse = 1, False
        self._hold = self._shift = 0.

    def update(self, dt, speed, throttle=0., *, braking=False, blocked=False):
        dt = min(max(_finite(dt), 0.), .25)
        speed, throttle = _finite(speed), float(np.clip(_finite(throttle), -1., 1.))
        reverse = speed < -.15 or (abs(speed) <= .15 and throttle < 0.)
        if reverse != self.reverse:
            self.gear, self._hold, self._shift = 1, .35, 0.
            self.reverse = reverse
        # S while travelling forward is a brake, not a second accelerator.
        power = abs(throttle) if ((reverse and throttle < 0.)
                                  or (not reverse and throttle > 0.)) else 0.
        if braking or blocked:
            power = 0.
        road = abs(speed) * 60. * FINAL_DRIVE / (2. * math.pi * WHEEL_RADIUS)
        remaining = dt
        while remaining > 1e-9:
            step = min(remaining, self.STEP)
            remaining -= step
            self._hold, self._shift = max(0., self._hold - step), max(0., self._shift - step)
            self.load += (power - self.load) * (1. - math.exp(-step / (.065 if power > self.load else .20)))
            ratio = 3.15 if reverse else GEAR_RATIOS[self.gear - 1]
            wheel_rpm = road * ratio
            if not reverse and self._hold <= 0.:
                up = 2600. + 2600. * self.load
                down = 1200. + 700. * self.load
                if self.gear < len(GEAR_RATIOS) and wheel_rpm > up:
                    self.gear += 1
                    self._hold, self._shift = .65, .24
                elif self.gear > 1 and wheel_rpm < down:
                    self.gear -= 1
                    self._hold, self._shift = .65, .24
                ratio = GEAR_RATIOS[self.gear - 1]
                wheel_rpm = road * ratio
            # Torque-converter slip lets the engine rev on launch, then locks up
            # naturally. Coasting stays coupled to the wheels for engine braking.
            launch = IDLE_RPM + self.load * 1050. * math.exp(-abs(speed) / 5.)
            target = min(REDLINE_RPM, max(IDLE_RPM, wheel_rpm, launch))
            tau = .19 if self._shift > 0. else (.22 if target > self.rpm else .32)
            self.rpm += (target - self.rpm) * (1. - math.exp(-step / tau))
            # A gearbox interrupts torque, not the entire exhaust sound.
            self.shift_gain = 1. - .12 * math.sin(math.pi * self._shift / .24) ** 2
        return self.rpm, self.load, self.shift_gain


class EngineVoice:
    """Resample real combustion cycles at any RPM, without pitch buckets or seams."""

    def __init__(self, bank):
        self.bank, self.rate = bank, bank.rate
        self.reset()

    def reset(self, *, startup=False):
        self.phase = 0.
        self.rpm, self.load, self.gain = IDLE_RPM, 0., 0.
        self._primed = False
        self._startup_position = 0 if startup and self.bank.startup is not None else None

    @staticmethod
    def _sample(data, positions):
        indices = positions.astype(np.int64)
        part = positions - indices
        return data[indices % len(data)] * (1. - part) + data[(indices + 1) % len(data)] * part

    def render(self, frames, rpm, load, gain=1., *, rpm_end=None):
        frames = int(frames)
        if frames <= 0:
            return np.empty(0, np.float32)
        rpm = float(np.clip(_finite(rpm, IDLE_RPM), IDLE_RPM, REDLINE_RPM))
        load, gain = float(np.clip(_finite(load), 0., 1.)), float(np.clip(_finite(gain), 0., 1.))
        if not self._primed:
            self.rpm, self.load, self._primed = rpm, load, True
        t = np.arange(frames, dtype=np.float64) / self.rate
        seconds = frames / self.rate
        end = rpm if rpm_end is None else float(np.clip(_finite(rpm_end, rpm), IDLE_RPM, REDLINE_RPM))
        velocity = (end - rpm) / seconds
        tau = .055
        # Follow a continuous RPM trajectory inside each output buffer, rather
        # than holding one pitch for a block and jumping to the next frame's RPM.
        decay = np.exp(-t / tau)
        rpms = rpm + velocity * (t - tau) + (self.rpm - rpm + velocity * tau) * decay
        rpms = np.clip(rpms, IDLE_RPM, REDLINE_RPM)
        loads = load + (self.load - load) * np.exp(-t / .065)
        gains = gain + (self.gain - gain) * np.exp(-t / .045)
        advance = rpms / (120. * self.rate)
        phases = self.phase + np.r_[0., np.cumsum(advance[:-1])]
        self.phase += float(np.sum(advance))
        # Periodic reduction avoids losing interpolation precision after long runs.
        period = math.lcm(*(layer.cycles for layer in self.bank.layers))
        self.phase %= period
        log_rpm = np.log(rpms)
        references = self.bank.log_rpm
        out = np.zeros(frames)
        for i, layer in enumerate(self.bank.layers):
            if len(references) == 1:
                weight = np.ones(frames)
            else:
                lo = np.ones(frames) if i == 0 else np.clip(
                    (log_rpm - references[i - 1]) / max(references[i] - references[i - 1], 1e-9), 0., 1.)
                hi = np.ones(frames) if i == len(references) - 1 else np.clip(
                    (references[i + 1] - log_rpm) / max(references[i + 1] - references[i], 1e-9), 0., 1.)
                # Equal-power cosine/sine fades have bounded slopes at layer
                # boundaries; sqrt ramps make a new timbre jump in too sharply.
                weight = np.sin(lo * math.pi / 2.) * np.sin(hi * math.pi / 2.)
            if not np.any(weight > 0.):
                continue
            pos = np.remainder(phases, layer.cycles) * len(layer.wave) / layer.cycles
            full = self._sample(layer.wave, pos)
            coast = self._sample(layer.coast, pos)
            # Throttle opens up the recorded exhaust/intake texture; lifting off
            # leaves a softer low-frequency engine-braking growl, not just less gain.
            brightness = .24 + .76 * loads
            out += (coast + (full - coast) * brightness) * weight
        self.rpm = float(np.clip(end - velocity * tau + (self.rpm - rpm + velocity * tau)
                                * math.exp(-seconds / tau), IDLE_RPM, REDLINE_RPM))
        self.load = load + (self.load - load) * math.exp(-seconds / .065)
        self.gain = gain + (self.gain - gain) * math.exp(-seconds / .045)
        out *= gains
        if self._startup_position is not None:
            # Use the audio sample clock for the starter-to-idle hand-off. Render
            # hitches must not let a one-shot finish while the engine stays muted.
            startup = self.bank.startup
            position = self._startup_position
            sample_positions = position + np.arange(frames)
            fade_length = min(int(self.rate * .65), len(startup))
            blend = np.clip((sample_positions - (len(startup) - fade_length)) / fade_length, 0., 1.)
            blend = blend * blend * (3. - 2. * blend)
            out *= .08 + .92 * blend
            count = min(frames, max(0, len(startup) - position))
            if count:
                out[:count] += startup[position:position + count] * .45 * (1. - .30 * blend[:count])
            self._startup_position += frames
            if self._startup_position >= len(startup):
                self._startup_position = None
        return np.clip(out, -.98, .98).astype(np.float32)
