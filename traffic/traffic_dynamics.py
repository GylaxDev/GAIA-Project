"""IDM longitudinal physics and MOBIL lane-change decisions, independent of GL."""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left, bisect_right
import math

import numpy as np

from ..vehicles.vehicle_instances import LENGTHS, HEAVY


# Cities: Skylines style driving. Vehicles pull away from a stop quickly, hold a
# short following headway and take corners briskly instead of crawling at the
# gentle limits of a driving-test model. Every braking envelope in the traffic
# package derives from COMFORT_BRAKING so these stay consistent with each other.
COMFORT_BRAKING = 3.0
LATERAL_ACCELERATION = 3.4        # m/s^2 sustained in a bend (curve_speed = sqrt(a / curvature))
MIN_CURVE_SPEED = 4.5             # m/s: no bend or junction turn is taken slower than ~16 km/h
BRAKE_ENVELOPE = 2. * COMFORT_BRAKING  # v^2 = v_end^2 + BRAKE_ENVELOPE * distance


@dataclass(frozen=True)
class DriverModel:
    acceleration: float = 2.8
    comfortable_braking: float = COMFORT_BRAKING
    emergency_braking: float = 9.0
    time_headway: float = 1.0
    minimum_gap: float = 1.6
    exponent: float = 4.0
    politeness: float = .25
    lane_threshold: float = .15
    safe_lane_braking: float = 4.0


DEFAULT_DRIVER = DriverModel()
HEAVY_ACCELERATION = 1.7


# Road hierarchy in km/h, the way Cities: Skylines tiers its road types:
# local streets are slow, avenues faster, highways fastest.
_HIERARCHY_KMH = {
    'motorway': 110., 'trunk': 90., 'primary': 70., 'secondary': 60., 'tertiary': 50.,
    'motorway_link': 70., 'trunk_link': 60., 'primary_link': 50., 'secondary_link': 45.,
    'tertiary_link': 40., 'unclassified': 45., 'residential': 40., 'living_street': 20.,
    'service': 25.,
}


def speed_limit(highway, tagged=0.):
    """m/s; a valid OSM maxspeed overrides the hierarchy defaults."""
    if tagged and math.isfinite(float(tagged)) and float(tagged) > 0:
        return float(tagged)
    return _HIERARCHY_KMH.get(highway, 40.) / 3.6


def sample_temperament(rng, heavy):
    """(speed factor, following headway scale, acceleration scale) for one driver.

    Real traffic is a spread, not a single speed: cautious drivers sit well under
    the limit and keep a long gap, most hug it, and a minority run well over it
    and tailgate. Trucks and buses are steadier and slower. MOBIL lane changes
    then sort the mix, so fast drivers overtake slow ones.
    """
    if heavy:
        return float(np.clip(rng.normal(.90, .04), .80, .98)), float(rng.uniform(1.1, 1.5)), 1.
    roll = rng.random()
    if roll < .20:      # cautious
        return float(rng.uniform(.80, .95)), float(rng.uniform(1.2, 1.5)), float(rng.uniform(.8, .95))
    if roll < .80:      # ordinary
        return float(rng.uniform(.96, 1.07)), float(rng.uniform(.9, 1.15)), float(rng.uniform(.95, 1.1))
    return float(rng.uniform(1.08, 1.22)), float(rng.uniform(.65, .85)), float(rng.uniform(1.1, 1.3))  # aggressive


def road_priority(highway):
    return {'motorway': 5, 'trunk': 4, 'primary': 3, 'secondary': 3,
            'tertiary': 2, 'unclassified': 1, 'residential': 1,
            'service': 0}.get(highway.replace('_link', ''), 0)


def desired_gap(velocity, leader_velocity=0., *, acceleration=None, model=DEFAULT_DRIVER, time_headway=None):
    """IDM standstill/headway/closing gap, also used for safe warm-start spawns."""
    v = np.maximum(np.asarray(velocity, np.float64), 0.)
    a = np.asarray(model.acceleration if acceleration is None else acceleration, np.float64)
    dv = v - np.maximum(np.asarray(leader_velocity, np.float64), 0.)
    headway = model.time_headway if time_headway is None else np.asarray(time_headway, np.float64)
    return model.minimum_gap + np.maximum(0., v * headway
                + v * dv / (2 * np.sqrt(a * model.comfortable_braking)))


def idm_acceleration(velocity, desired, gap=np.inf, leader_velocity=0., *,
                     acceleration=None, model=DEFAULT_DRIVER, time_headway=None):
    """IDM dv/dt with a bumper-to-bumper gap and a positive closing velocity.

    s* = s0 + max(0, v T + v (v-v_leader) / (2 sqrt(a b))). The singular
    emergency term is bounded by physical braking, not a linear velocity servo.
    Infinite gaps are genuine free flow; stopped cars never acquire negative v.
    """
    v = np.maximum(np.asarray(velocity, np.float64), 0.)
    v0 = np.maximum(np.asarray(desired, np.float64), .1)
    a = np.asarray(model.acceleration if acceleration is None else acceleration, np.float64)
    star = desired_gap(v, leader_velocity, acceleration=a, model=model, time_headway=time_headway)
    spacing = np.maximum(np.asarray(gap, np.float64), .05)
    result = a * (1. - (v / v0) ** model.exponent - (star / spacing) ** 2)
    return np.clip(result, -model.emergency_braking, a)


def _scalar_idm(velocity, desired, gap, leader_velocity, model):
    """The same IDM equation for MOBIL's scalar probes, without NumPy dispatch."""
    v, v0 = max(0., float(velocity)), max(.1, float(desired))
    closing = v - max(0., float(leader_velocity))
    star = model.minimum_gap + max(0., v * model.time_headway
                                   + v * closing / (2. * math.sqrt(model.acceleration * model.comfortable_braking)))
    result = model.acceleration * (1. - (v / v0) ** model.exponent - (star / max(.05, float(gap))) ** 2)
    return min(model.acceleration, max(-model.emergency_braking, result))


def integrate_idm(velocity, acceleration, dt, clearance=np.inf):
    """Ballistic integration, including the sub-tick time at which v reaches zero."""
    v = np.maximum(np.asarray(velocity, np.float64), 0.)
    a = np.asarray(acceleration, np.float64)
    dt = max(0., float(dt))
    end = np.maximum(0., v + a * dt)
    braking = (a < 0.) & (v + a * dt < 0.)
    t = np.where(braking, v / np.maximum(-a, 1e-9), dt)
    distance = np.maximum(0., v * t + .5 * a * t * t)
    cap = np.maximum(0., np.asarray(clearance, np.float64))
    limited = distance > cap
    return np.where(limited, 0., end), np.minimum(distance, cap)


class LaneOccupancy:
    """Sorted centre positions, including both lanes occupied during a merge.

    Construction is O(n log n), once per behaviour tick. Queries are binary
    searches; no all-pairs vehicle scan is needed for IDM or MOBIL.
    """
    def __init__(self, slots, lanes, positions, speeds, kinds, extra=()):
        self._entries = {}
        self._entry_ranges = {}
        self.heads = {}  # immutable per-lane first-neighbour tuples for sparse route probes
        columns = [np.asarray(c) for c in (slots, lanes, positions, speeds, kinds)]
        if extra:
            columns = [np.concatenate((a, np.asarray(b))) for a, b in zip(columns, extra)]
        ids, lane, pos, vel, kind = columns
        order = np.lexsort((ids, pos, lane))
        ids, lane, pos, vel, kind = [a[order] for a in (ids, lane, pos, vel, kind)]
        lengths = LENGTHS[kind.astype(int)]
        boundaries = np.r_[0, np.flatnonzero(np.diff(lane)) + 1, len(lane)] if len(lane) else [0]
        self._entry_columns = ids, pos, vel, lengths
        if len(lane):
            starts = boundaries[:-1]
            keys = lane[starts].astype(int, copy=False).tolist()
            self._entry_ranges = dict(zip(keys, zip(starts.tolist(), boundaries[1:].tolist())))
            self.heads = dict(zip(keys, zip(ids[starts].astype(int, copy=False).tolist(),
                                pos[starts].astype(float, copy=False).tolist(),
                                vel[starts].astype(float, copy=False).tolist(), lengths[starts].tolist())))
        self._batch_columns = ids, lane, pos, vel, lengths
        self._batch_heads = None

    def _entry(self, lane):
        data = self._entries.get(lane)
        if data is None:
            bounds = self._entry_ranges.get(lane)
            if bounds is not None:
                a, b = bounds
                data = self._entries[lane] = tuple(column[a:b] for column in self._entry_columns)
        return data

    @property
    def entries(self):
        # Most ticks only need vector tables and first occupants. Build the four
        # per-lane array views lazily for scalar probes or explicit diagnostics.
        for lane in self._entry_ranges:
            self._entry(lane)
        return self._entries

    def _batch_tables(self, lanes):
        if self._batch_columns is None:
            entries = self.entries
            keys = sorted(entries)
            if keys:
                columns = [np.concatenate([entries[k][i] for k in keys]) for i in range(4)]
                resident = np.repeat(keys, [len(entries[k][0]) for k in keys])
                self._batch_columns = columns[0], resident, *columns[1:]
            else:
                self._batch_columns = tuple(np.empty(0) for _ in range(5))
        _, resident, _, _, _ = self._batch_columns
        size = max(int(np.max(lanes, initial=-1)), int(np.max(resident, initial=-1))) + 1
        if self._batch_heads is None or len(self._batch_heads) < size:
            self._batch_heads = np.full(size, -1, np.int64)
            if len(resident):
                starts = np.r_[0, np.flatnonzero(np.diff(resident)) + 1]
                self._batch_heads[resident[starts].astype(np.int64)] = starts

    def _batch_records(self, lanes, indices, exclude):
        slots, resident, position, speed, length = self._batch_columns
        if not len(slots):
            return (np.full(lanes.shape, -1, np.int64), np.full(lanes.shape, np.inf),
                    np.zeros(lanes.shape), np.zeros(lanes.shape))
        indices = indices.copy()
        while True:
            safe = np.clip(indices, 0, len(slots) - 1)
            valid = (indices >= 0) & (indices < len(slots)) & (resident[safe] == lanes)
            skip = valid & (slots[safe] == exclude)
            if not skip.any():
                break
            indices[skip] += 1
        return (np.where(valid, slots[safe], -1).astype(np.int64),
                np.where(valid, position[safe], np.inf),
                np.where(valid, speed[safe], 0.), np.where(valid, length[safe], 0.))

    def leaders(self, lanes, positions, exclude=-1):
        """Vector equivalent of neighbours(...)[0], without per-driver searches."""
        lanes = np.asarray(lanes, np.int64)
        positions = np.asarray(positions, np.float64)
        self._batch_tables(lanes)
        _, resident, position, _, _ = self._batch_columns
        # Integer lane bounds avoid a structured (lane, float) comparison for
        # every binary-search step. Most city lanes then need only one position
        # probe; denser queues retain an exact batched upper-bound search.
        lower = np.searchsorted(resident, lanes, side='left')
        upper = np.searchsorted(resident, lanes, side='right')
        if len(position):
            while np.any(lower < upper):
                middle = (lower + upper) // 2
                live = lower < upper
                advance = live & ((position[np.minimum(middle, len(position) - 1)] <= positions)
                                  | np.isnan(positions))
                lower = np.where(advance, middle + 1, lower)
                upper = np.where(live & ~advance, middle, upper)
        return self._batch_records(lanes, lower, exclude)

    def first_batch(self, lanes, exclude=-1):
        """First occupants of route splines, including exclusion/empty-lane rules."""
        lanes = np.asarray(lanes, np.int64)
        self._batch_tables(lanes)
        indices = np.full(lanes.shape, -1, np.int64)
        valid = (lanes >= 0) & (lanes < len(self._batch_heads))
        indices[valid] = self._batch_heads[lanes[valid]]
        return self._batch_records(lanes, indices, exclude)

    def neighbours(self, lane, position, exclude=-1):
        """Return (leader, follower) as (slot, centre, v, length), or None."""
        data = self._entry(int(lane))
        if data is None:
            return None, None
        ids, pos, vel, length = data
        # Most city lanes have zero or one resident car. bisect avoids a NumPy
        # scalar-dispatch wrapper while retaining sorted binary-search semantics.
        k = bisect_right(pos, position)
        before, after = k - 1, k
        while before >= 0 and ids[before] == exclude:
            before -= 1
        while after < len(pos) and ids[after] == exclude:
            after += 1
        leader = (int(ids[after]), float(pos[after]), float(vel[after]), float(length[after])) if after < len(pos) else None
        follower = (int(ids[before]), float(pos[before]), float(vel[before]), float(length[before])) if before >= 0 else None
        return leader, follower

    def first(self, lane, exclude=-1):
        head = self.heads.get(int(lane))
        if head is None or head[0] != exclude:
            return head
        data = self._entry(int(lane))
        if data is not None:
            for k, slot in enumerate(data[0]):
                if slot != exclude:
                    return int(slot), float(data[1][k]), float(data[2][k]), float(data[3][k])
        return None

    def last(self, lane, exclude=-1):
        lane = int(lane)
        head = self.heads.get(lane)
        if head is None: return None
        data = self._entries.get(lane)
        if data is None:
            a, b = self._entry_ranges[lane]
            if b - a == 1:
                return head if head[0] != exclude else None
            data = self._entry(lane)
        if data is not None:
            for k in range(len(data[0]) - 1, -1, -1):
                if data[0][k] != exclude:
                    return int(data[0][k]), float(data[1][k]), float(data[2][k]), float(data[3][k])
        return None

    def reserve(self, slot, lane, position, speed, length):
        """Make this tick's accepted merge visible to subsequent decisions."""
        self._batch_columns = self._batch_heads = None
        data = self._entry(int(lane))
        if data is None:
            self._entries[int(lane)] = tuple(np.array([v]) for v in (slot, position, speed, length))
            self.heads[int(lane)] = int(slot), float(position), float(speed), float(length)
            return
        k = bisect_left(data[1], position)
        self._entries[int(lane)] = tuple(np.insert(a, k, v) for a, v in zip(data, (slot, position, speed, length)))
        if k == 0:
            self.heads[int(lane)] = int(slot), float(position), float(speed), float(length)


def mobil_decision(*, velocity, desired, length, old_leader=None, old_follower=None,
                   new_leader=None, new_follower=None, position=0., mandatory=False,
                   follower_desired=None, model=DEFAULT_DRIVER):
    """MOBIL safety AND incentive, including both affected followers.

    Mandatory turn-lane preparation relaxes the incentive only, never the rear
    braking/gap test. The returned incentive is useful for choosing the best
    adjacent lane. All neighbour positions use the candidate lane's coordinates.
    """
    def ego_acc(leader):
        gap = np.inf if leader is None else leader[1] - position - .5 * (length + leader[3])
        return _scalar_idm(velocity, desired, gap, 0. if leader is None else leader[2], model)

    new_gap = np.inf if new_leader is None else new_leader[1] - position - .5 * (length + new_leader[3])
    rear_gap = np.inf if new_follower is None else position - new_follower[1] - .5 * (length + new_follower[3])
    if new_gap < model.minimum_gap or rear_gap < model.minimum_gap:
        return False, -np.inf
    new_acceleration = ego_acc(new_leader)
    if new_acceleration < -model.safe_lane_braking:
        return False, -np.inf

    follower_delta = 0.
    if new_follower is not None:
        fd = desired if follower_desired is None else follower_desired(new_follower[0])
        after = _scalar_idm(new_follower[2], fd, rear_gap, velocity, model)
        if after < -model.safe_lane_braking:
            return False, -np.inf
        before_gap = np.inf if new_leader is None else new_leader[1] - new_follower[1] - .5 * (new_leader[3] + new_follower[3])
        before = _scalar_idm(new_follower[2], fd, before_gap, 0. if new_leader is None else new_leader[2], model)
        follower_delta += after - before
    if old_follower is not None:
        fd = desired if follower_desired is None else follower_desired(old_follower[0])
        before_gap = position - old_follower[1] - .5 * (length + old_follower[3])
        before = _scalar_idm(old_follower[2], fd, before_gap, velocity, model)
        after_gap = np.inf if old_leader is None else old_leader[1] - old_follower[1] - .5 * (old_leader[3] + old_follower[3])
        after = _scalar_idm(old_follower[2], fd, after_gap, 0. if old_leader is None else old_leader[2], model)
        follower_delta += after - before
    incentive = new_acceleration - ego_acc(old_leader) + model.politeness * follower_delta
    return bool(mandatory or incentive > model.lane_threshold), incentive
