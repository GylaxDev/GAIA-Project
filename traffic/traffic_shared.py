"""Direction admission for two routed directions sharing one physical lane."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .road_graph import STOP_MARGIN
from ..vehicles.vehicle_instances import LENGTHS


@dataclass(frozen=True, slots=True)
class SharedApproach:
    slot: int
    group: int
    direction: int
    remaining: float             # centre allowance to the front-body stop bound
    origin: float                # current-centre distance to corridor spline origin
    reservations: tuple         # a connector may touch two different corridors


@dataclass(slots=True)
class _Lease:
    direction: int
    entry: float                 # odometer at corridor origin, possibly still ahead
    entered: bool = False
    exit: float | None = None
    admission: float | None = None  # odometer when the front body consumes entry


class SharedLaneManager:
    """Live occupants/reservations only; topology stays in the immutable graph."""
    def __init__(self, graph):
        self.graph = graph
        self._leases = {}         # slot -> corridor -> lease, including rear clearance
        self._groups = {}         # corridor -> its slots/directions only
        self._life = {}           # leases never transfer to a recycled fleet slot

    def owns(self, slot, group, direction):
        lease = self._leases.get(int(slot), {}).get(int(group))
        return lease is not None and lease.direction == direction

    def can_enter(self, group, direction, slot=-1):
        # A slot cannot reverse while its own rear still occupies the corridor.
        return all(own == direction for own in self._groups.get(int(group), {}).values())

    def can_receive(self, connection, slot=-1):
        return all(self.can_enter(group, direction, slot)
                   for group, direction in self.graph.shared_memberships[connection])

    def available(self, approach):
        return all(self.can_enter(group, direction, approach.slot) for group, direction in approach.reservations)

    def admitted(self, approach):
        return all(self.owns(approach.slot, group, direction) for group, direction in approach.reservations)

    def commit_approach(self, approach, odometer, *, entered=False, life=None):
        for group, direction in approach.reservations:
            self.commit(approach.slot, group, direction, odometer, entered=entered, life=life,
                        admission=odometer + approach.remaining + STOP_MARGIN - approach.origin)

    def commit(self, slot, group, direction, odometer, *, entered=False, life=None, admission=None):
        slot, group, direction = int(slot), int(group), int(direction)
        if life is not None:
            if slot in self._life and self._life[slot] != life: self.release(slot)
            self._life[slot] = int(life)
        bucket = self._leases.setdefault(slot, {})
        if group not in bucket:
            bucket[group] = _Lease(direction, float(odometer), entered, admission=admission)
            self._groups.setdefault(group, {})[slot] = direction
        elif entered:
            bucket[group].entered = True

    def _release(self, slot, group):
        del self._leases[slot][group]
        if not self._leases[slot]:
            del self._leases[slot]
            self._life.pop(slot, None)
        del self._groups[group][slot]
        if not self._groups[group]: del self._groups[group]

    def release(self, slot):
        for group in tuple(self._leases.get(int(slot), ())): self._release(int(slot), group)

    def sync_slots(self, fleet):
        for slot in tuple(self._leases):
            if (not fleet.active[slot] or fleet.lane[slot] < 0
                    or self._life.get(slot, int(fleet.life[slot])) != fleet.life[slot]):
                self.release(slot)
            else:
                self._life[slot] = int(fleet.life[slot])

    def rewind(self, slot, odometer):
        for group, lease in tuple(self._leases.get(int(slot), {}).items()):
            if lease.admission is not None and odometer < lease.admission - 1e-9:
                self._release(int(slot), group)

    def begin_tick(self, active):
        for slot, leases in tuple(self._leases.items()):
            car = active.get(slot)
            if car is None or (len(car) > 4 and self._life.get(slot, car[4]) != car[4]):
                self.release(slot); continue
            if len(car) > 4: self._life[slot] = int(car[4])
            for group, lease in tuple(leases.items()):
                lane, progress, length, distance = car[:4]
                memberships = self.graph.shared_memberships[lane]
                present = (group, lease.direction) in memberships
                if not present and any(g == group for g, _ in memberships):
                    self._release(slot, group); continue
                if present:
                    lease.entered, lease.exit = True, None
                elif lease.entered or distance >= lease.entry:
                    if lease.exit is None: lease.exit = distance - progress
                    if distance - lease.exit >= .5 * length + 2.: self._release(slot, group)
        # Remapping and diagnostic fixtures may install a body on a shared path.
        # Recover it, but never treat an upstream early reservation as an exit.
        for slot, car in active.items():
            lane, progress, _, distance = car[:4]
            for group, direction in self.graph.shared_memberships[lane]:
                self.commit(slot, group, direction, distance - progress, entered=True,
                            life=car[4] if len(car) > 4 else None)

    def remap(self, previous, fleet):
        """Map corridor orientation through surviving lane keys, including splits.

        Group integers and source-way direction are not stable across graph
        builds. Preserve early origins and rear-clearance odometers separately
        from recovering the newly installed centre positions.
        """
        mapping = {}
        for lane in previous.graph.lanes:
            old_group, old_direction = previous.graph.shared_group[lane.id], previous.graph.shared_direction[lane.id]
            if old_group < 0: continue
            target = self.graph.spline_by_key.get(lane.key, -1)
            if target < 0: continue
            for group, direction in self.graph.shared_memberships[target]:
                mapping.setdefault(int(old_group), set()).add((group, int(old_direction) * direction))
        for slot, leases in previous._leases.items():
            if (not fleet.active[slot] or fleet.lane[slot] < 0
                    or previous._life.get(slot, int(fleet.life[slot])) != fleet.life[slot]): continue
            for old_group, lease in leases.items():
                for group, sign in mapping.get(old_group, ()):
                    existing = self._leases.get(slot, {}).get(group)
                    old_exit = existing.exit if existing is not None else lease.exit
                    self.commit(slot, group, lease.direction * sign, lease.entry,
                                entered=lease.entered, life=int(fleet.life[slot]), admission=lease.admission)
                    target = self._leases[slot][group]
                    target.entry = min(target.entry, lease.entry)
                    target.entered |= lease.entered
                    target.exit = max(old_exit, lease.exit) if old_exit is not None and lease.exit is not None else None
        self.begin_tick({int(i): (int(fleet.lane[i]), float(fleet.progress[i]), float(LENGTHS[fleet.kind[i]]),
                                  float(fleet.distance[i]), int(fleet.life[i]))
                         for i in np.flatnonzero(fleet.active & (fleet.lane >= 0))})

    def event(self, core, route, cursor, lane, progress, length, horizon, slot=-1):
        if route is None or lane < 0 or not self.graph.shared_corridors: return None
        offset = 0.
        candidates = ((lane, 0.),)
        while True:
            for spline, start in candidates:
                for group, direction in self.graph.shared_memberships[spline]:
                    if self.owns(slot, group, direction): continue
                    origin = offset + start - progress
                    # A shared receiving corridor must be reserved before the
                    # junction body enters, even if its trimmed spline starts late.
                    entry = min(0., float(self.graph.entry_progress[spline]))
                    remaining = origin + entry - .5 * length - STOP_MARGIN
                    if remaining <= horizon:
                        return SharedApproach(int(slot), group, direction, remaining, origin, self.graph.shared_memberships[spline])
                    return None
            trace = core._route_trace(route, cursor, lane)
            candidates = trace.following
            for spline, start in candidates:
                if offset + start - progress > horizon + .5 * length + self.graph.max_stop_back:
                    return None
                for group, direction in self.graph.shared_memberships[spline]:
                    if self.owns(slot, group, direction): continue
                    origin = offset + start - progress
                    entry = min(0., float(self.graph.entry_progress[spline]))
                    remaining = origin + entry - .5 * length - STOP_MARGIN
                    return (SharedApproach(int(slot), group, direction, remaining, origin, self.graph.shared_memberships[spline])
                            if remaining <= horizon else None)
            if trace.continuation is None: return None
            cursor, lane, advance = trace.continuation
            offset += advance
            if offset - progress > horizon + .5 * length + self.graph.max_stop_back: return None
            candidates = ()

    def approaches(self, core, ids):
        if not self.graph.shared_corridors or not len(ids): return {}
        f, graph = core.fleet, self.graph
        batch = core._road_batch(ids)
        dependencies = batch.dependency_lanes[ids]
        shared = graph.shared_group[np.maximum(dependencies, 0)] >= 0
        candidates = ((graph.shared_group[f.lane[ids]] >= 0)
                      | np.any(shared & (dependencies >= 0), axis=1)
                      | (batch.dependency_continuation[ids] - f.progress[ids] <= np.maximum(80., f.speed[ids]**2 / 4. + 8.)
                         + graph.max_stop_back + .5 * LENGTHS[f.kind[ids]]))
        result = {}
        for i in ids[candidates].tolist():
            event = self.event(core, f.routes[i], int(f.cursor[i]), int(f.lane[i]), float(f.progress[i]),
                               float(LENGTHS[f.kind[i]]), max(80., f.speed[i]**2 / 4. + 8.), i)
            if event is not None: result[i] = event
        return result

    def signature(self):
        return tuple((slot, group, lease.direction, lease.entry, lease.entered, lease.exit, lease.admission,
                      self._life.get(slot))
                     for slot, leases in self._leases.items() for group, lease in leases.items())

    def conflicts(self):
        return sum(len(set(slots.values())) > 1 for slots in self._groups.values())
