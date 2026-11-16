"""Protected signals, uncontrolled priorities and receiving-lane box protection."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
import math
import zlib

from .road_graph import TURN_LEFT, UTURN
from .traffic_dynamics import COMFORT_BRAKING
from ..vehicles.vehicle_instances import LENGTHS


class SignalState(IntEnum):
    GREEN = 0
    YELLOW = 1
    RED = 2
    DEDICATED_LEFT = 3


class SignalDemand:
    """What each approach group of one junction wants right now (built per tick)."""
    __slots__ = ('through', 'left', 'arriving', 'left_arriving', 'starved')

    def __init__(self):
        self.through, self.left = [False, False], [False, False]
        self.arriving, self.left_arriving = [False, False], [False, False]
        self.starved = [False, False]   # somebody has been held at this red too long

    def waiting(self, group):
        return self.through[group] or self.left[group]


class SignalController:
    """Demand-actuated two-group signal, in the style of Cities: Skylines.

    A phase is shared by the entire junction. Instead of a blind fixed timer, the
    light serves the traffic that is actually there: a group with nobody queued is
    skipped, a protected-left phase only runs when a left turner is waiting, a green
    is extended while cars keep arriving (up to MAX_GREEN) and ends as soon as the
    queue has discharged and somebody else is waiting. With no demand anywhere the
    light rests on its current green. Vehicles already committed to a connector
    finish on yellow/red.

    PHASES keeps the fixed eight-slot layout (the lamp renderer indexes into it);
    ``index`` is simply the slot that is live right now.
    """
    PHASES = ((0, SignalState.DEDICATED_LEFT, 4.), (0, SignalState.GREEN, 12.),
              (0, SignalState.YELLOW, 3.), (-1, SignalState.RED, 1.5),
              (1, SignalState.DEDICATED_LEFT, 4.), (1, SignalState.GREEN, 12.),
              (1, SignalState.YELLOW, 3.), (-1, SignalState.RED, 1.5))
    PERIOD = sum(p[2] for p in PHASES)
    LEFT, GREEN, YELLOW, RED = (0, 4), (1, 5), (2, 6), (3, 7)
    MIN_GREEN, MAX_GREEN = 7., 22.
    FLUSH_GREEN = 3.         # a green nobody is using only needs to flush the junction
    MIN_LEFT, MAX_LEFT = 3., 9.
    GAP_OUT = 2.2            # seconds with no arriving car before a green may be ended
    LEFT_GAP_OUT = 1.6

    def __init__(self, graph, node, offset=None):
        self.node = node.id
        strongest = max(node.incoming, key=lambda e: (graph.edges[e].priority, graph.edges[e].lane_count), default=None)
        axis = graph.edges[strongest].heading_end if strongest is not None else 0.
        self.groups = {e: int(abs(math.cos(graph.edges[e].heading_end - axis)) < .7071) for e in node.incoming}
        seed = (zlib.crc32(repr(node.key).encode()) % 10000) / 10000.
        # Junctions start in different phases so a whole city does not pulse in step.
        self.offset = seed * self.PERIOD if offset is None else float(offset)
        self.index = self.GREEN[int(self.offset // (self.PERIOD / 2.)) % 2]
        self.elapsed = (self.offset % 5.) if offset is None else 0.
        self.clock = None
        self.last_extend = 0.

    # -- state machine -----------------------------------------------------
    def _enter(self, index):
        self.index, self.elapsed = index, 0.
        self.last_extend = self.clock if self.clock is not None else 0.

    def _choose(self, after_group, demand):
        other = 1 - after_group
        for group in (other, after_group):
            if demand.waiting(group):
                return self.LEFT[group] if demand.left[group] else self.GREEN[group]
        return self.GREEN[other]

    def update(self, clock, demand):
        clock = float(clock)
        dt = 0. if self.clock is None else max(0., min(clock - self.clock, 1.))
        self.clock = clock
        self.elapsed += dt
        group, state, duration = self.PHASES[self.index]
        if state == SignalState.DEDICATED_LEFT:
            if demand.left_arriving[group]: self.last_extend = clock
            if self.elapsed >= self.MIN_LEFT and (clock - self.last_extend >= self.LEFT_GAP_OUT
                                                  or self.elapsed >= self.MAX_LEFT):
                self._enter(self.GREEN[group])
        elif state == SignalState.GREEN:
            if demand.arriving[group]: self.last_extend = clock
            minimum = self.MIN_GREEN if (demand.through[group] or demand.arriving[group]) else self.FLUSH_GREEN
            others = demand.waiting(1 - group) or demand.left[group]
            held = demand.starved[1 - group] or demand.starved[group]
            if others and self.elapsed >= minimum and (clock - self.last_extend >= self.GAP_OUT or held
                                                       or self.elapsed >= self.MAX_GREEN):
                self._enter(self.YELLOW[group])
        elif state == SignalState.YELLOW:
            if self.elapsed >= duration: self._enter(self.RED[group])
        elif self.elapsed >= duration:
            previous = 0 if self.index == self.RED[0] else 1
            self._enter(self._choose(previous, demand))

    def signature(self):
        return self.index

    def state(self, incoming_edge, turn, clock=None):
        group, phase, _ = self.PHASES[self.index]
        own = group == self.groups.get(int(incoming_edge), 0)
        left = turn in (TURN_LEFT, UTURN)
        if not own:
            return SignalState.RED
        if phase == SignalState.DEDICATED_LEFT:
            return phase if left else SignalState.RED
        return SignalState.RED if left else phase

    def permits(self, incoming_edge, turn, clock, remaining, velocity):
        state = self.state(incoming_edge, turn, clock)
        if state in (SignalState.GREEN, SignalState.DEDICATED_LEFT):
            return True
        if state == SignalState.YELLOW:
            # Stop when comfortable braking is possible; otherwise clear the
            # junction. Never snap a fast car to zero on a green/yellow edge.
            return float(remaining) < velocity * velocity / (2. * COMFORT_BRAKING) + velocity * .5
        return False


@dataclass(frozen=True, slots=True)
class Approach:
    slot: int
    connection: int
    remaining: float
    velocity: float
    length: float
    waiting: float = 0.


class IntersectionManager:
    NEAR_LINE = 14.   # metres beyond the braking distance at which line tests begin
    RECEIVING_STEPS = 32
    RECEIVING_CACHE_SIZE = 4096

    def __init__(self, graph, shared=None):
        self.graph = graph
        self.shared = shared
        self.signals = {n.id: SignalController(graph, n) for n in graph.nodes if n.signal}
        self._conflicts = {}
        self._lines = {}
        self._leases = {}         # stable instance slot -> committed connector
        self._node_leases = {}    # node -> its committed slots/connectors only
        self._lease_distance = {} # odometer at connector origin; may still be ahead of the centre
        self._lease_life = {}     # a recycled slot cannot inherit consumed permission
        self._approaches = {}
        self._receiving_paths = OrderedDict()
        self.clock = 0.

    def begin_tick(self, clock, active, approaches):
        """active maps slot -> (spline, progress, length[, odometer])."""
        self.clock = float(clock)
        for slot, cid in tuple(self._leases.items()):
            car = active.get(slot)
            if car is not None and len(car) > 4 and self._lease_life.get(slot, car[4]) != car[4]:
                self._release_lease(slot); continue
            if car is not None and len(car) > 4: self._lease_life[slot] = int(car[4])
            connection = self.graph.splines[cid]
            distance = self._lease_distance.get(slot)
            if car is not None and len(car) > 3 and distance is not None:
                clear = car[3] - distance >= connection.length + .5 * car[2] + 2.
            else:
                clear = car is not None and car[0] != cid and (
                        car[0] != connection.outgoing or car[1] >= .5 * car[2] + 2.)
            if car is None or clear:
                self._release_lease(slot)
        # Recover occupancy after graph remapping or a test-injected vehicle.
        for slot, car in active.items():
            lane = car[0]
            if (lane >= self.graph.lane_count and self.graph.controlled[self.graph.splines[lane].node]
                    and car[1] + .5 * car[2] >= self.graph.entry_progress[lane]
                    and slot not in self._leases):
                # A short receiving road can end before the rear clears the box.
                # Traversing its ordinary seam must not replace that older lease.
                self.commit(slot, lane, car[3] - car[1] if len(car) > 3 else None,
                            life=car[4] if len(car) > 4 else None)
        closest = {}
        for approach in approaches:
            # A committed car is already being served. Pick the first waiting
            # car behind it, rather than hiding that lane's demand until the
            # committed body has left the approach spline.
            if self._leases.get(approach.slot) == approach.connection:
                continue
            c = self.graph.splines[approach.connection]
            current = closest.get(c.incoming)
            if current is None or approach.remaining < current.remaining:
                closest[c.incoming] = approach
        self._approaches = {}
        for approach in closest.values():
            c = self.graph.splines[approach.connection]
            self._approaches.setdefault(c.node, []).append(approach)
        self._actuate(clock)

    DETECTOR_RANGE = 70.      # metres of approach the loop detectors can see
    ARRIVAL_SECONDS = 3.2     # a car this close to the line extends its own green
    STARVATION_SECONDS = 18.  # a car held this long at a red forces the light to change

    def _actuate(self, clock):
        """Feed each light the traffic that is really waiting or about to arrive."""
        graph = self.graph
        for node, signal in self.signals.items():
            demand = SignalDemand()
            for a in self._approaches.get(node, ()):
                # A car stopped at the line sits at ~0 m; one already inside the box owns a lease.
                if a.remaining < -2. or a.slot in self._leases or a.remaining > self.DETECTOR_RANGE:
                    continue
                c = graph.splines[a.connection]
                group = signal.groups.get(graph.lanes[c.incoming].edge, 0)
                left = c.turn in (TURN_LEFT, UTURN)
                # A car is "arriving" when it will reach the line within a few seconds
                # or is queued at it, so a green is held open while a platoon passes.
                coming = (a.velocity > 1.5 and a.remaining / a.velocity <= self.ARRIVAL_SECONDS) or a.remaining < 10.
                if a.waiting > self.STARVATION_SECONDS:
                    demand.starved[group] = True
                if left:
                    demand.left[group] = True
                    demand.left_arriving[group] |= coming
                else:
                    demand.through[group] = True
                    demand.arriving[group] |= coming
            signal.update(clock, demand)

    def inherit(self, previous):
        """Carry live signal phases across a graph rebuild (tiles streaming in/out)."""
        if previous is None:
            return
        old = {previous.graph.nodes[n].key: c for n, c in previous.signals.items()}
        for node, signal in self.signals.items():
            before = old.get(self.graph.nodes[node].key)
            if before is not None and before.groups.keys() == signal.groups.keys() and before.groups == signal.groups:
                signal.index, signal.elapsed = before.index, before.elapsed
                signal.clock, signal.last_extend = before.clock, before.last_extend

    def _receiving_path(self, lane):
        """Bounded geometry only; a second box, branch or cycle ends the path."""
        cached = self._receiving_paths.get(lane)
        if cached is not None:
            self._receiving_paths.move_to_end(lane)
            return cached
        graph, origin = self.graph, lane
        records, seen, distance, continuation = [], set(), 0., -1
        while lane not in seen and len(records) < self.RECEIVING_STEPS:
            seen.add(lane)
            end = distance + graph.splines[lane].length
            if lane < graph.lane_count and graph.controlled[graph.edges[graph.lanes[lane].edge].end]:
                # The next box's stop bound can lie BEFORE a short final link.
                # A trimmed centre endpoint is not usable front-bumper storage.
                records.append((lane, distance, end + graph.stop_offsets[lane]))
                break
            records.append((lane, distance, end))
            distance = end
            following = graph.successors.get(lane, ())
            if len(following) != 1:
                break
            lane = following[0]
        else:
            if lane not in seen:
                continuation = lane
        cached = tuple(records), continuation
        self._receiving_paths[origin] = cached
        if len(self._receiving_paths) > self.RECEIVING_CACHE_SIZE:
            self._receiving_paths.popitem(last=False)
        return cached

    def receiving_clear(self, connection, length, occupancy, slot=-1):
        """A whole vehicle plus its standstill gap must fit beyond the box."""
        if self.shared is not None and self.graph.shared_corridors and not self.shared.can_receive(connection, slot): return False
        records, lane = self._receiving_path(self.graph.splines[connection].outgoing)
        required = float(length) + 2.
        rear_reach = .5 * float(LENGTHS.max())
        probe = required + max(self.graph.max_stop_back, rear_reach)
        # Probe far enough that an unseen negative stop offset cannot eat the
        # claimed receiving space. Ordinary cached prefixes need no extra walk.
        end = records[-1][2] if records else 0.
        if lane >= 0 and end < probe:
            extended, seen, distance = [], {previous for previous, _, _ in records}, end
            while lane >= 0 and lane not in seen:
                seen.add(lane)
                end = distance + self.graph.splines[lane].length
                controlled = lane < self.graph.lane_count and self.graph.controlled_ends[lane]
                extended.append((lane, distance, end + self.graph.stop_offsets[lane] if controlled else end))
                if controlled: break
                distance = end
                if end >= probe: break
                following = self.graph.successors.get(lane, ())
                lane = following[0] if len(following) == 1 else -1
            records = (*records, *extended)
        if not records or records[-1][2] < required: return False
        for previous, start, end in records:
            if start > required + rear_reach: break
            first = occupancy.first(previous, exclude=slot)
            if first is not None and start + first[1] - .5 * first[3] < required: return False
        return True

    def conflicts(self, a, b):
        if a == b:
            return True
        ca, cb = self.graph.splines[a], self.graph.splines[b]
        if ca.node != cb.node:
            return False  # unrelated city junctions need no cached conflict pair
        key = (min(a, b), max(a, b))
        cached = self._conflicts.get(key)
        if cached is not None:
            return cached
        if ca.outgoing == cb.outgoing or ca.incoming == cb.incoming:
            result = True
        else:
            from shapely.geometry import LineString
            for cid, c in ((a, ca), (b, cb)):
                if cid not in self._lines:
                    self._lines[cid] = LineString(c.points[:, :2])
            # Physical vehicle widths, not just infinitesimal crossing lines.
            result = self._lines[a].distance(self._lines[b]) < 1.95
        self._conflicts[key] = result
        return result

    def _yield_required(self, own, occupancy):
        c = self.graph.splines[own.connection]
        edge = self.graph.edges[self.graph.lanes[c.incoming].edge]
        for other in self._approaches.get(c.node, ()):
            if other.slot == own.slot:
                continue
            oc = self.graph.splines[other.connection]
            oe = self.graph.edges[self.graph.lanes[oc.incoming].edge]
            if not self.conflicts(own.connection, other.connection):
                continue
            if other.remaining > max(15., other.velocity * 4.):
                continue
            if not self.receiving_clear(other.connection, other.length, occupancy, other.slot):
                continue  # a blocked main road cannot claim the box indefinitely
            if oe.priority > edge.priority:
                return True
            if oe.priority < edge.priority:
                continue
            if own.waiting > 1. and other.velocity < .3 and other.remaining < 5.:
                # Deterministic arrival-order arbitration breaks four-way-yield
                # cycles without taking priority away from the main road.
                if (other.waiting, -other.slot) > (own.waiting, -own.slot):
                    return True
                continue
            cross = math.sin(oe.heading_end - edge.heading_end)
            if cross > .2:
                return True  # approaching from the driver's right
            if abs(cross) < .2 and c.turn in (TURN_LEFT, UTURN) and oc.turn not in (TURN_LEFT, UTURN):
                return True  # an unprotected left yields to opposing through traffic
        return False

    def blocked(self, approach, occupancy):
        """Return the stop reason, or None. Box checks apply even on green.

        Cars already following a leader through the same connector are the one
        exception: IDM keeps their bumper gap, while cross traffic, merges and
        other turns still require a clear box instead of serialising every car.

        Cheap signal state is tested first. The expensive receiving-lane, priority
        and conflict tests only matter once the car is close enough that it would
        have to start braking for them; far upstream it just keeps rolling, which
        is what keeps a street flowing the way Cities: Skylines traffic does.
        """
        # Permission has already been consumed at the actual entry bound. A
        # changing light/receiving queue must not strand that body in the box.
        if self._leases.get(approach.slot) == approach.connection:
            return None
        c = self.graph.splines[approach.connection]
        edge = self.graph.lanes[c.incoming].edge
        signal = self.signals.get(c.node)
        if signal is not None and not signal.permits(edge, c.turn, self.clock, approach.remaining, approach.velocity):
            return 'signal'
        if self.shared is not None and self.graph.shared_corridors and not self.shared.can_receive(approach.connection, approach.slot):
            return 'shared'
        v = approach.velocity
        if approach.remaining > v * v / (2. * COMFORT_BRAKING) + self.NEAR_LINE:
            return None
        if not self.receiving_clear(approach.connection, approach.length, occupancy, approach.slot):
            return 'box'
        if signal is None and self.graph.nodes[c.node].junction and self._yield_required(approach, occupancy):
            return 'yield'
        for slot, leased in self._node_leases.get(c.node, {}).items():
            # Cars on the very same connector follow their leader through IDM.
            # Reserving it exclusively makes every follower brake again on green.
            # Other turns, crossing paths and merges still require a clear box.
            if leased == approach.connection:
                continue
            if slot != approach.slot and self.conflicts(approach.connection, leased):
                return 'conflict'
        return None

    def _release_lease(self, slot):
        cid = self._leases.pop(slot)
        self._lease_distance.pop(slot, None)
        self._lease_life.pop(slot, None)
        node = self.graph.splines[cid].node
        bucket = self._node_leases[node]
        del bucket[slot]
        if not bucket:
            del self._node_leases[node]

    def sync_slots(self, fleet):
        for slot in tuple(self._leases):
            if (not fleet.active[slot] or fleet.lane[slot] < 0
                    or self._lease_life.get(slot, int(fleet.life[slot])) != fleet.life[slot]):
                self._release_lease(slot)
            else:
                self._lease_life[slot] = int(fleet.life[slot])

    def commit(self, slot, connection, odometer=None, *, life=None):
        slot, connection = int(slot), int(connection)
        if slot in self._leases and self._leases[slot] != connection:
            self._release_lease(slot)
        self._leases[slot] = connection
        self._node_leases.setdefault(self.graph.splines[connection].node, {})[slot] = connection
        if odometer is not None:
            self._lease_distance[slot] = float(odometer)
        if life is not None: self._lease_life[slot] = int(life)
