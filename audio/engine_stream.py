"""Keep the recorded engine fed independently of rendering/frame-rate hitches."""
from __future__ import annotations

import threading
import time

import numpy as np

from .audio_output import MixerOutput, open_engine_output
from .engine_audio import EngineVoice, IDLE_RPM, _finite


class EngineStream:
    """A continuous engine voice with a ~240 ms native PCM safety reservoir.

    Only the worker queues audio. The render thread publishes small immutable
    snapshots; it never decodes, resamples, or waits for an engine buffer.
    """

    def __init__(self, pg, channel, bank, out_channels=2, *, start=True, output=None):
        self.pg, self.channel, self.voice = pg, channel, EngineVoice(bank)
        self.out_channels = out_channels
        self.output = output if output is not None else open_engine_output(pg, channel, bank.rate, out_channels)
        self.frames = self.output.frames
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._state = (False, IDLE_RPM, 0., 0., 0, time.perf_counter(), 0., False)
        self._generation, self._running = -1, False
        self._had_sound = False
        self.underruns = 0
        self.error = None
        self.failed = False
        self._thread = None
        self.channel.set_volume(0.)
        if start:
            self._thread = threading.Thread(target=self._run, name='osm3d-engine', daemon=True)
            try:
                self._thread.start()
            except (RuntimeError, MemoryError):
                self.output.close()
                raise

    def set_state(self, rpm, load, gain, *, active=True, reset=False, startup=False):
        now = time.perf_counter()
        rpm = _finite(rpm, IDLE_RPM)
        with self._lock:
            if self._stop.is_set():
                return
            generation = self._state[4] + int(reset or active != self._state[0])
            interval = now - self._state[5]
            velocity = 0.
            if generation == self._state[4] and active and interval > .001:
                raw = float(np.clip((rpm - self._state[1]) / interval, -6000., 6000.))
                alpha = 1. - np.exp(-interval / .08)
                velocity = self._state[6] + (raw - self._state[6]) * alpha
            starting = bool(startup or (generation == self._state[4] and self._state[7]))
            if not active and (self._state[0] or reset):
                # Flush all PCM, not just the currently playing mixer block.
                try:
                    self.output.clear()
                except OSError as exc:
                    self.error = str(exc)
            self._state = (bool(active), rpm, load, gain, generation, now, velocity, starting)

    def _pump(self):
        with self._lock:
            active, rpm, load, gain, generation, stamp, velocity, startup = self._state
        if not active:
            if self._running:
                self.output.clear()
                self.voice.reset()
            self._running, self._generation = False, generation
            return False
        if generation != self._generation:
            self.output.clear()
            self.voice.reset(startup=startup)
            self._generation = generation
            self._had_sound = False
        self._running = True
        queued = self.output.queued_frames
        if queued >= self.output.target_frames or not self.output.writable:
            return False
        if queued == 0 and self._had_sound:
            self.underruns += 1
            self.output.target_frames = min(self.output.max_frames,
                                            self.output.target_frames + int(self.voice.rate * .06))
        # Compensate bounded output look-ahead with a smoothed RPM trajectory.
        # Larger safety buffers must not turn acceleration into held-pitch steps.
        age = max(0., time.perf_counter() - stamp) + queued / self.voice.rate
        begin = rpm + velocity * min(age, .20)
        end = rpm + velocity * min(age + self.frames / self.voice.rate, .20)
        data = self.voice.render(self.frames, begin, load, gain, rpm_end=end)
        pcm = (data * 32000.).astype(np.int16)
        if self.out_channels == 2:
            pcm = np.column_stack((pcm, pcm))
        # A pause, exit or new-car entry may have arrived during resampling.
        with self._lock:
            if not self._state[0] or self._state[4] != generation:
                return False
            written = self.output.write(pcm)
            self._had_sound |= written
        return written

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    # Prime/fill the entire reservoir before yielding. A slow
                    # frame must not drain a single queued successor.
                    while not self._stop.is_set() and self._pump():
                        pass
                except OSError as exc:
                    if self.output.name != 'waveout':
                        raise
                    # A native driver failure should not permanently kill the
                    # engine worker. Preserve its crank/startup clock and fall
                    # back once to the still-running pygame mixer.
                    with self._lock:
                        self.error = str(exc)
                        self.output.close()
                        self.output = MixerOutput(self.pg, self.channel, self.voice.rate, self.out_channels)
                        self.frames = self.output.frames
                        self._had_sound = False
                self._stop.wait(.004)
        except (self.pg.error, OSError, ValueError, AttributeError, MemoryError) as exc:
            self.error = str(exc)
            self.failed = True
        finally:
            try:
                self.output.clear()
            except (self.pg.error, OSError):
                pass

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=.5)
        self.output.close()
