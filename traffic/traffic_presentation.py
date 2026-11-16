"""Bounded, identity-safe presentation of asynchronous traffic snapshots.

Worker time can fall behind render time when bounded IPC drops excess elapsed
time. It is not a valid prediction clock. Blend confirmed poses over the observed
publication interval instead; never extrapolate a queued car past its latest pose.
The authoritative frame, traffic physics and 100 ms legacy prediction stay intact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..vehicles.vehicle_instances import VehicleFrame, predict_instances


@dataclass
class _Blend:
    start: np.ndarray
    target: np.ndarray
    ids: np.ndarray
    distance: np.ndarray
    travel: np.ndarray
    yaw: np.ndarray
    turn: np.ndarray
    pitch: np.ndarray
    tilt: np.ndarray
    wheel: np.ndarray


class TrafficPresentation:
    """One latest authority plus one aligned presentation start, never a queue."""
    MIN_INTERVAL = 1 / 60.
    MAX_INTERVAL = 1.
    INTERVAL_MARGIN = 1.15

    def __init__(self):
        self.reset()

    def reset(self):
        self.source = None
        self._blends = None
        self._received = self._started = self._sampled = 0.
        self._interval = self.MIN_INTERVAL
        self._duration = self.MIN_INTERVAL
        self._start_clock = 0.
        self._output = None
        self._updates = 0
        self._now = 0.

    @staticmethod
    def _identified(frame):
        ids, distances = getattr(frame, 'ids', None), getattr(frame, 'distances', None)
        if ids is None or distances is None or len(ids) != len(frame) or len(distances) != len(frame):
            return False
        return all(rows is None or (identity is not None and distance is not None
                                   and identity.shape == distance.shape == (len(rows),))
                   for rows, identity, distance in zip(frame, ids, distances))

    @staticmethod
    def _align(current, target):
        blends = []
        for kind, rows in enumerate(target):
            if rows is None:
                blends.append(None)
                continue
            end = rows.copy()
            start = end.copy()
            identity = target.ids[kind].copy()
            distance = target.distances[kind].copy()
            # A birth has no previous body to blend from. Fade at its confirmed
            # location; never borrow a retired car's row or recycled slot's paint.
            start[:, 23] = 0.
            old = current[kind] if kind < len(current) else None
            if old is not None and TrafficPresentation._identified(current):
                _, before, after = np.intersect1d(current.ids[kind], identity,
                                                 assume_unique=True, return_indices=True)
                start[after, :16] = old[before, :16]
                start[after, 20:22] = old[before, 20:22]
                start[after, 23] = old[before, 23]
                distance[after] = current.distances[kind][before]
            travel = target.distances[kind] - distance
            yaw = np.arctan2(start[:, 1], start[:, 0])
            turn = np.arctan2(end[:, 1], end[:, 0]) - yaw
            turn = np.arctan2(np.sin(turn), np.cos(turn))
            pitch = np.arcsin(np.clip(start[:, 2], -1., 1.))
            tilt = np.arcsin(np.clip(end[:, 2], -1., 1.)) - pitch
            wheel = travel / np.maximum(end[:, 11], .05)
            for array in (start, end, identity, distance, travel, yaw, turn, pitch, tilt, wheel):
                array.flags.writeable = False
            blends.append(_Blend(start, end, identity, distance, travel, yaw, turn, pitch, tilt, wheel))
        return blends

    def _sample(self, now):
        if self._output is not None and now == self._sampled:
            return self._output
        source = self.source
        if self._blends is None:
            # Old callers/fixtures without identity metadata cannot be matched
            # safely. Retain their bounded predictor, timed from receipt rather
            # than the render clock's accumulated (possibly dropped) worker lag.
            prediction = float(np.clip(now - self._received, 0., .1))
            rows = [predict_instances(a, prediction) if a is not None else None for a in source]
            output = VehicleFrame(rows, clock=source.clock + prediction, generation=source.generation,
                                  sequence=source.sequence, counts=source.counts, signals=source.signals)
        else:
            alpha = float(np.clip((now - self._started) / self._duration, 0., 1.))
            arrays, identities, distances = [], [], []
            for blend in self._blends:
                if blend is None:
                    arrays.append(None); identities.append(None); distances.append(None)
                    continue
                a, b = blend.start, blend.target
                rows = b.copy()
                rows[:, 12:15] = a[:, 12:15] + (b[:, 12:15] - a[:, 12:15]) * alpha
                yaw, pitch = blend.yaw + blend.turn * alpha, blend.pitch + blend.tilt * alpha
                c, s, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
                rows[:, 0], rows[:, 1], rows[:, 2] = c * cp, s * cp, sp
                rows[:, 4], rows[:, 5], rows[:, 6] = -s, c, 0.
                rows[:, 8], rows[:, 9], rows[:, 10] = -c * sp, -s * sp, cp
                # Odometer deltas preserve forward wheel rotation even when a
                # slow publication spans several complete wheel revolutions.
                rows[:, 20] = np.mod(a[:, 20] + blend.wheel * alpha, 2 * np.pi)
                rows[:, 21] = a[:, 21] + (b[:, 21] - a[:, 21]) * alpha
                rows[:, 23] = np.clip(a[:, 23] + (b[:, 23] - a[:, 23]) * alpha, 0., 1.)
                # Paint, discrete lamps and the latest safe speed are not lerped.
                distance = blend.distance + blend.travel * alpha
                rows.flags.writeable = distance.flags.writeable = False
                arrays.append(rows); identities.append(blend.ids); distances.append(distance)
            clock = self._start_clock + (source.clock - self._start_clock) * alpha
            output = VehicleFrame(arrays, clock=clock, generation=source.generation,
                                  sequence=source.sequence, counts=source.counts, signals=source.signals,
                                  ids=identities, distances=distances)
        self._output, self._sampled = output, now
        return output

    def sample(self, frame, now):
        now = max(0., float(now))
        self._now = now
        previous = self.source
        changed = (previous is None or frame.sequence != previous.sequence
                   or frame.generation != previous.generation or frame.clock != previous.clock)
        if changed:
            current = self._sample(now) if previous is not None else frame
            interval = now - self._received
            if previous is not None and interval > 0.:
                # Fast attack, slow decay keeps jitter from exhausting the blend
                # just before the next publication. A dead worker still stops.
                self._interval = float(np.clip(max(interval, self._interval * .9),
                                               self.MIN_INTERVAL, self.MAX_INTERVAL))
            self._duration = min(self.MAX_INTERVAL, self._interval * self.INTERVAL_MARGIN)
            self._blends = self._align(current, frame) if self._identified(frame) else None
            self._start_clock = current.clock
            self.source = frame
            self._received = self._started = now
            self._output = None
            self._updates += 1
        return self._sample(now)

    def stats(self):
        clock = self._output.clock if self._output is not None else 0.
        return dict(presentation='interpolation' if self._blends is not None else 'prediction',
                    presentation_interval_ms=self._duration * 1000.,
                    presentation_age_ms=max(0., self._now - self._received) * 1000.,
                    presentation_delay_seconds=max(0., self.source.clock - clock) if self.source else 0.,
                    presentation_updates=self._updates)
