"""Buffered PCM output without Python callbacks between Windows engine blocks.

Pygame's Channel.queue only holds one successor, and its completion callback
needs the GIL. WinMM can keep a whole PCM reservoir playing natively while the
renderer is busy. No extra package, codec, DLL asset or compiler is required.
Other platforms/headless runs retain the pygame mixer with longer buffers.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading

import numpy as np

# If an audio driver refuses teardown, keep its buffers pinned rather than
# releasing memory that a native device could still be reading.
_RETIRED_OUTPUTS = []


class _WaveFormat(ctypes.Structure):
    _pack_ = 2
    _fields_ = [('format', ctypes.c_uint16), ('channels', ctypes.c_uint16),
                ('rate', ctypes.c_uint32), ('bytes_per_second', ctypes.c_uint32),
                ('block_align', ctypes.c_uint16), ('bits', ctypes.c_uint16),
                ('extra', ctypes.c_uint16)]


class _WaveHeader(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('length', ctypes.c_uint32),
                ('recorded', ctypes.c_uint32), ('user', ctypes.c_size_t),
                ('flags', ctypes.c_uint32), ('loops', ctypes.c_uint32),
                ('next', ctypes.c_void_p), ('reserved', ctypes.c_size_t)]


class WaveOutOutput:
    """Pinned prepared buffers, consumed by Windows without invoking Python."""

    name = 'waveout'
    SLOTS = 20
    DONE = 1

    def __init__(self, rate, channels, *, library=None):
        self.rate, self.channels = int(rate), int(channels)
        self.frames = max(256, int(rate * .02))
        self.max_frames = self.SLOTS * self.frames
        self.target_frames = int(rate * .24)
        self._lock = threading.RLock()
        self._handle, self._slots = ctypes.c_void_p(), []
        self._api = library if library is not None else ctypes.WinDLL('winmm')
        if library is None:
            self._api.waveOutOpen.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32,
                                              ctypes.POINTER(_WaveFormat), ctypes.c_size_t,
                                              ctypes.c_size_t, ctypes.c_uint32]
            self._api.waveOutOpen.restype = ctypes.c_uint32
            for name in ('waveOutPrepareHeader', 'waveOutUnprepareHeader', 'waveOutWrite'):
                function = getattr(self._api, name)
                function.argtypes = [ctypes.c_void_p, ctypes.POINTER(_WaveHeader), ctypes.c_uint32]
                function.restype = ctypes.c_uint32
            for name in ('waveOutReset', 'waveOutClose'):
                function = getattr(self._api, name)
                function.argtypes, function.restype = [ctypes.c_void_p], ctypes.c_uint32
        fmt = _WaveFormat(1, channels, rate, rate * channels * 2, channels * 2, 16, 0)
        self._check(self._api.waveOutOpen(ctypes.byref(self._handle), 0xFFFFFFFF,
                                         ctypes.byref(fmt), 0, 0, 0), 'open engine output')
        try:
            for _ in range(self.SLOTS):
                memory = ctypes.create_string_buffer(self.frames * channels * 2)
                header = _WaveHeader(ctypes.cast(memory, ctypes.c_void_p), len(memory), 0, 0, 0, 0, None, 0)
                self._check(self._api.waveOutPrepareHeader(self._handle, ctypes.byref(header),
                                                         ctypes.sizeof(header)), 'prepare engine buffer')
                # Keep both the native header and sample memory alive until reset,
                # unprepare and close. Windows reads them asynchronously.
                self._slots.append([memory, header, False, 0])
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(result, operation):
        if result:
            raise OSError(f'WinMM could not {operation} (error {result})')

    def _reap(self):
        for slot in self._slots:
            if slot[2] and slot[1].flags & self.DONE:
                slot[2], slot[3] = False, 0

    @property
    def queued_frames(self):
        with self._lock:
            self._reap()
            return sum(slot[3] for slot in self._slots if slot[2])

    @property
    def writable(self):
        with self._lock:
            self._reap()
            return bool(self._handle.value and any(not slot[2] for slot in self._slots))

    def write(self, pcm):
        data = np.ascontiguousarray(pcm, dtype='<i2')
        if data.nbytes > self.frames * self.channels * 2 or data.nbytes % (self.channels * 2):
            raise ValueError('PCM buffer does not match the engine output')
        with self._lock:
            if not self._handle.value:
                return False
            self._reap()
            slot = next((slot for slot in self._slots if not slot[2]), None)
            if slot is None:
                return False
            memory, header = slot[:2]
            ctypes.memmove(memory, data.ctypes.data, data.nbytes)
            header.length = data.nbytes
            self._check(self._api.waveOutWrite(self._handle, ctypes.byref(header),
                                             ctypes.sizeof(header)), 'queue engine buffer')
            slot[2], slot[3] = True, data.nbytes // (2 * self.channels)
            return True

    def clear(self):
        with self._lock:
            if self._handle.value:
                self._check(self._api.waveOutReset(self._handle), 'mute engine output')
                for slot in self._slots:
                    slot[2], slot[3] = False, 0

    def close(self):
        with self._lock:
            if not self._handle.value:
                return
            reset = self._api.waveOutReset(self._handle)
            unprepared = True
            for _, header, _, _ in self._slots:
                result = self._api.waveOutUnprepareHeader(self._handle, ctypes.byref(header), ctypes.sizeof(header))
                unprepared &= result == 0
            closed = self._api.waveOutClose(self._handle)
            if reset or not unprepared or closed:
                if self not in _RETIRED_OUTPUTS:
                    _RETIRED_OUTPUTS.append(self)
                return
            self._handle = ctypes.c_void_p()
            self._slots.clear()


class MixerOutput:
    """Portable/headless fallback with a larger margin than 40 ms handoffs."""

    name = 'mixer'

    def __init__(self, pg, channel, rate, channels):
        self.pg, self.channel, self.channels = pg, channel, channels
        self.frames = max(256, int(rate * .12))
        self.target_frames = self.max_frames = self.frames * 2
        self._sounds = []
        channel.set_volume(0.)

    @property
    def queued_frames(self):
        return self.frames * (int(self.channel.get_busy()) + int(self.channel.get_queue() is not None))

    @property
    def writable(self):
        return self.channel.get_queue() is None

    def write(self, pcm):
        if not self.writable:
            return False
        sound = self.pg.sndarray.make_sound(np.ascontiguousarray(pcm))
        if self.channel.get_busy():
            self.channel.queue(sound)
        else:
            self.channel.play(sound)
        self._sounds = (self._sounds + [sound])[-2:]
        self.channel.set_volume(1.)
        return True

    def clear(self):
        self.channel.set_volume(0.)
        self.channel.stop()
        self._sounds = []

    def close(self):
        try:
            self.clear()
        except self.pg.error:
            pass


def open_engine_output(pg, channel, rate, channels):
    backend = os.environ.get('OSM3D_ENGINE_BACKEND', 'auto').strip().lower()
    driver = os.environ.get('SDL_AUDIODRIVER', '').strip().lower()
    if (sys.platform == 'win32' and backend != 'mixer' and driver not in ('dummy', 'disk')
            and getattr(pg, '__file__', None)):
        try:
            return WaveOutOutput(rate, channels)
        except (OSError, AttributeError, ValueError):
            pass
    return MixerOutput(pg, channel, rate, channels)
