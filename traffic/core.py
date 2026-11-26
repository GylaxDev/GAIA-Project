"""City traffic: worker-owned IDM/MOBIL drivers and a bounded instance mailbox.

The render thread submits elapsed time/tile commands and consumes packed arrays.
Graph builds and A* searches have their own bounded workers. Behaviour, lane
sorting, intersection arbitration and instance packing run on the traffic
process; only bounded vectorized pose presentation/culling remains on the caller.
In-process thread and synchronous modes are retained for diagnostic fixtures.
"""
from __future__ import annotations

from collections import deque, OrderedDict
from concurrent.futures import CancelledError
from dataclasses import dataclass
import heapq
import math
import threading
import time
from types import SimpleNamespace

import numpy as np

from ..core.config import DEFAULT_TRAFFIC_COUNT
from .road_graph import GraphTile, STRAIGHT, TURN_LEFT, TURN_RIGHT, UTURN, STOP_MARGIN
from .traffic_dynamics import (BRAKE_ENVELOPE, DEFAULT_DRIVER, HEAVY_ACCELERATION, LaneOccupancy, idm_acceleration,
                              integrate_idm, mobil_decision, desired_gap, sample_temperament)
from .traffic_intersections import Approach, IntersectionManager
from .traffic_lookahead import RoadLookahead
from .traffic_routing import GraphBuilder, RoutingService
from .traffic_signals import SignalStream
from .traffic_shared import SharedLaneManager
from .traffic_presentation import TrafficPresentation
from ..vehicles.vehicle_instances import (VEHICLE_STRIDE, LAMPS, BRAKE, LEFT, RIGHT,
                                LENGTHS, WHEELBASES, WHEEL_RADII, HEAVY,
                                VehicleFrame, predict_instances)

KINDS = 9
PAINTS = np.array([(.90, .90, .90), (.90, .90, .90), (.68, .69, .72),
                   (.68, .69, .72), (.04, .04, .045), (.04, .04, .045),
                   (.30, .31, .33), (.10, .16, .34), (.62, .07, .07),
                   (.08, .22, .14), (.55, .45, .30)], np.float32)
KIND_PROBABILITIES = [.30, .12, .10, .04, .04, .10, .08, .15, .07]


@dataclass(frozen=True, slots=True)
class _RouteTrace:
    route: object  # Retain identity: an evicted route's id must not alias a new one.
    roads: tuple
    following: tuple
    continuation: tuple | None
    occupied_lanes: frozenset
    preparation: tuple | None
    approach: tuple | None


class DriverFleet:
    """Fixed-capacity structure-of-arrays; despawning recycles a stable slot."""
    def __init__(self, capacity):
        self.capacity = int(capacity)
        n = self.capacity
        self.active = np.zeros(n, bool)
        self.lane = np.full(n, -1, np.int64)
        self.progress = np.zeros(n)
        self.speed = np.zeros(n)
        self.acceleration = np.zeros(n)
        self.factor = np.ones(n)
        self.headway = np.full(n, DEFAULT_DRIVER.time_headway)   # per-driver following time gap
        self.accel_scale = np.ones(n)
        self.kind = np.zeros(n, np.int64)
        self.color = np.ones((n, 3), np.float32)
        self.cursor = np.zeros(n, np.int64)
        self.routes = [None] * n
        self.origin = [None] * n
        self.destination = [None] * n
        self.source = [None] * n
        self.life = np.zeros(n, np.int64)
        self.lane_from = np.full(n, -1, np.int64)
        self.planned = np.full(n, -1, np.int64)
        self.intent_distance = np.zeros(n)
        self.merge_elapsed = np.zeros(n)
        self.cooldown = np.zeros(n)
        self.distance = np.zeros(n)
        self.wheel = np.zeros(n)
        self.steer = np.zeros(n)
        self.stationary = np.zeros(n)
        self.junction_wait = np.zeros(n)
        self.age = np.zeros(n)
        self.retiring = np.zeros(n, bool)
        self.retire_elapsed = np.zeros(n)
        self.visibility = np.ones(n)
        self.flags = np.zeros(n, np.int32)
        self.phase = np.zeros(n, np.int32)
        self.pose = np.zeros((n, 5))  # xyz, yaw, pitch; also preserves orphan fade poses
        self.yaw_rate = np.zeros(n)
        self.pitch_rate = np.zeros(n)
        self.stop_reason = [''] * n
        self._free = list(range(n))
        heapq.heapify(self._free)

    @property
    def n(self):
        return self.capacity - len(self._free)

    def allocate(self):
        if not self._free:
            return None
        i = heapq.heappop(self._free)
        self.active[i] = True
        self.life[i] += 1
        return i

    def release(self, i):
        if not self.active[i]:
            return
        self.active[i] = False
        self.routes[i] = None
        self.origin[i] = self.destination[i] = self.source[i] = None
        self.lane[i] = self.lane_from[i] = self.planned[i] = -1
        heapq.heappush(self._free, int(i))


class TrafficCore:
    STEP = 1 / 30.
    MAX_STEP = 1 / 10.
    MERGE_SECONDS = 3.0
    MIN_MERGE_SPEED = 4.
    INDICATOR_DISTANCE = 30.
    PREPARE_DISTANCE = 180.
    STUCK_SECONDS = 45.
    FADE_SECONDS = 2.
    TRACE_STEPS = 32
    TRACE_CACHE_SIZE = 16384
    UPSTREAM_STEPS = 32
    UPSTREAM_CACHE_SIZE = 4096
    ROUTE_POOL_MAX = 192
    ROUTE_POOL_MIN = 8
    ROUTE_POOL_CARS_PER_ROUTE = 16

    def __init__(self, target_count=DEFAULT_TRAFFIC_COUNT, *, threaded_services=True, seed=0xDEADBEEF, notify=None):
        self.target_count = max(0, int(target_count))
        self.fleet = DriverFleet(self.target_count)
        self.rng = np.random.default_rng(seed)
        self.builder = GraphBuilder(threaded=threaded_services, notify=notify)
        self.router = RoutingService(threaded=threaded_services, notify=notify)
        self.tiles = {}
        self.generation = 0
        self.graph = None
        self.intersections = None
        self.shared = None
        self.clock = self.accum = self._decisions = 0.
        self._spawn_jobs = []
        self._spawn_wait_until = 0.
        self._ready_routes = deque(maxlen=64)
        self._route_pool = []
        self._seed_target = 0
        self._seeding = False
        self._reroutes = {}
        self._route_traces = OrderedDict()
        self._upstream_chains = OrderedDict()
        self._road_lookahead = None
        self._prediction_clearance = np.zeros(self.target_count)
        self._prediction_state = None
        self._prediction_routes = ()
        self._prediction_controls = None
        self._publication_spawns = set()
        self._packed = np.zeros((self.target_count, VEHICLE_STRIDE), np.float32)
        self._sequence = 0
        self.frame = VehicleFrame([None] * KINDS)
        self.recycled_stuck = self.completed_trips = 0
        self.error = None
        self.last_tick_ms = 0.
        self.last_publish_ms = 0.
        self._tick_cost = 0.
        self.tick_count = 0
        self._tick_history = deque(maxlen=4096)
        self._publish_history = deque(maxlen=4096)
        self._clearance_history = deque(maxlen=4096)
        self.closed = False

    def set_tiles(self, tiles):
        self.tiles = dict(tiles)
        self.generation += 1
        self.builder.request(self.tiles, self.generation)

    def _retire(self, i, reason):
        f = self.fleet
        if not f.retiring[i]:
            f.retiring[i] = True
            f.retire_elapsed[i] = 0.
            f.planned[i] = -1
            f.stop_reason[i] = reason
            if reason == 'stuck': self.recycled_stuck += 1
            if reason == 'destination': self.completed_trips += 1

    def _seed_pool_size(self):
        """Enough route diversity to fill the missing fleet without one A* per car."""
        remaining = max(0, self.target_count - self.fleet.n)
        if not remaining:
            return 0
        return max(1, min(self.ROUTE_POOL_MAX, remaining,
                          max(self.ROUTE_POOL_MIN, remaining // self.ROUTE_POOL_CARS_PER_ROUTE)))

    def install_graph(self, graph):
        """Remap surviving roads/routes; never attach a stale route to a recycled car."""
        old, previous_intersections, previous_shared, f = self.graph, self.intersections, self.shared, self.fleet
        self.graph = graph
        self._route_traces.clear()
        self._upstream_chains.clear()
        self._road_lookahead = None
        self._prediction_state = None
        self._prediction_controls = None
        self._publication_spawns.clear()
        self.shared = SharedLaneManager(graph)
        self.intersections = IntersectionManager(graph, self.shared)
        self.intersections.clock = self.clock
        self.intersections.inherit(previous_intersections)
        self._signal_stream = SignalStream(graph, self.intersections)
        self.router.set_graph(graph)
        self._spawn_jobs.clear(); self._ready_routes.clear(); self._reroutes.clear()
        self._route_pool.clear()
        self._seed_target = self._seed_pool_size()
        self._seeding = bool(self._seed_target)
        self._spawn_wait_until = 0.
        if old is None:
            return
        for slot, cid in previous_intersections._leases.items():
            remapped = graph.spline_by_key.get(old.splines[cid].key, -1)
            if (remapped >= graph.lane_count and f.active[slot]
                    and previous_intersections._lease_life.get(slot, int(f.life[slot])) == f.life[slot]):
                self.intersections.commit(slot, remapped, previous_intersections._lease_distance.get(slot), life=f.life[slot])
        for i in np.flatnonzero(f.active):
            previous = int(f.lane[i])
            if previous < 0:
                continue
            old_spline = old.splines[previous]
            lane = graph.spline_by_key.get(old_spline.key, -1)
            if lane < 0 and previous < old.lane_count:
                edge = old.edges[old_spline.edge]
                candidates = []
                for eid in graph.source_edges.get((*edge.source, edge.forward), ()):
                    target = graph.edges[eid].lanes[min(old_spline.index, graph.edges[eid].lane_count - 1)]
                    p, d = graph.lanes[target].project(f.pose[i, :3])
                    candidates.append((d, target, p))
                if candidates:
                    distance, lane, progress = min(candidates)
                    if distance > 6.:
                        lane = -1
            if lane < 0:
                f.lane[i] = -1; f.speed[i] = 0.; f.yaw_rate[i] = 0.
                self._retire(i, 'unloaded')
                continue
            f.lane[i] = lane
            f.progress[i] = graph.splines[lane].project(f.pose[i, :3])[0]
            if f.lane_from[i] >= 0:
                f.lane_from[i] = graph.spline_by_key.get(old.splines[f.lane_from[i]].key, -1)
            if f.planned[i] >= 0:
                f.planned[i] = graph.spline_by_key.get(old.splines[f.planned[i]].key, -1)
            route = f.routes[i]
            tail = route.edges[int(f.cursor[i]):] if route is not None else ()
            remapped = [graph.edge_by_key.get(old.edges[e].key, -1) for e in tail]
            new_route = graph.compile_route(remapped) if remapped and min(remapped) >= 0 else None
            f.cursor[i] = 0
            if new_route is not None:
                f.routes[i] = new_route
            elif lane < graph.lane_count:
                current = graph.lanes[lane].edge
                f.routes[i] = graph.compile_route((current,))
                self._request_reroute(int(i), current)
            else:
                # A preserved connector's receiving road supplies the new tail.
                c = graph.splines[lane]
                current = graph.lanes[c.incoming].edge
                following = graph.lanes[c.outgoing].edge
                f.routes[i] = graph.compile_route((current, following))
                self._request_reroute(int(i), following, on_connector=True)
        self.shared.remap(previous_shared, f)
        self.intersections.sync_slots(f)

    def _request_reroute(self, i, edge, on_connector=False):
        if i in self._reroutes or self.fleet.retiring[i]:
            return
        future = self.router.random_request(int(self.rng.integers(0, 2**63)),
                                            origin=self.graph.edges[edge].end, incoming=edge)
        if future is not None:
            prefix = self.fleet.routes[i].edges if on_connector else (edge,)
            self._reroutes[i] = (int(self.fleet.life[i]), future, prefix)

    def _collect_services(self):
        graph = self.builder.take()
        if graph is not None and graph.generation == self.generation:
            self.install_graph(graph)
        if self.builder.error is not None:
            self.error = self.builder.error
        pending = []
        for future in self._spawn_jobs:
            if not future.done():
                pending.append(future); continue
            try: route = future.result()
            except CancelledError: continue
            except Exception as exc:
                self.error = exc; continue
            if route is not None and self.graph is not None and route.generation == self.graph.generation:
                if self._seeding and len(self._route_pool) < self._seed_target:
                    self._route_pool.append(route)
                    if len(self._route_pool) >= self._seed_target:
                        self._seeding = False
                else:
                    self._ready_routes.append(route)
        self._spawn_jobs = pending
        f = self.fleet
        for i, (life, future, prefix) in tuple(self._reroutes.items()):
            if not future.done():
                continue
            del self._reroutes[i]
            try: suffix = future.result()
            except CancelledError: continue
            except Exception as exc:
                self.error = exc; continue
            if not f.active[i] or f.life[i] != life or f.retiring[i]:
                continue
            if suffix is not None and suffix.generation == self.graph.generation:
                route = self.graph.compile_route((*prefix, *suffix.edges))
                if route is not None:
                    f.routes[i] = route
                    f.destination[i] = self.graph.nodes[route.destination].key
                    self._prediction_state = None
        if (self.graph is not None and len(self.graph.origins) and self.target_count
                and self.clock >= self._spawn_wait_until):
            if self._seeding:
                # Seed a small route pool concurrently. Tail spawning can then
                # reuse those legal trips instead of paying A* for every car.
                while len(self._spawn_jobs) + len(self._route_pool) < self._seed_target:
                    future = self.router.random_request(int(self.rng.integers(0, 2**63)))
                    if future is None:
                        break
                    self._spawn_jobs.append(future)
            elif not (self._route_pool and f.n < self.target_count):
                desired = min(32, max(0, self.target_count - f.n) + 8)
                while len(self._spawn_jobs) + len(self._ready_routes) < desired:
                    future = self.router.random_request(int(self.rng.integers(0, 2**63)))
                    if future is None:
                        break
                    self._spawn_jobs.append(future)

    def occupancy(self):
        f = self.fleet
        active = np.flatnonzero(f.active & (f.lane >= 0))
        merging = active[f.lane_from[active] >= 0]
        extra = ()
        if len(merging):
            progress = f.progress[merging] * self.graph.table.length[f.lane_from[merging]] / self.graph.table.length[f.lane[merging]]
            extra = (merging, f.lane_from[merging], progress, f.speed[merging], f.kind[merging])
        return LaneOccupancy(active, f.lane[active], f.progress[active], f.speed[active], f.kind[active], extra)

    def spawn_route(self, route, *, lane=None, progress=None, velocity=None, kind=None,
                    color=None, fade=True, occupancy=None):
        """Worker/test entry point. Warm-start anywhere on the first routed segment."""
        graph, f = self.graph, self.fleet
        if graph is None or route is None or route.generation != graph.generation or not route.edges or f.n >= self.target_count:
            return None
        self.shared.sync_slots(f)
        self.intersections.sync_slots(f)
        kind = int(self.rng.choice(KINDS, p=KIND_PROBABILITIES)) if kind is None else int(kind)
        edge = graph.edges[route.edges[0]]
        allowed = route.allowed_lanes[0]
        if lane is None:
            # A short initial segment has no room for a 30 m indicator lead-in.
            choices = allowed if edge.length < 100. else edge.lanes
            lane = int(self.rng.choice(choices))
        if lane not in edge.lanes:
            return None
        length = graph.lanes[lane].length
        occupancy = self.occupancy() if occupancy is None else occupancy
        factor, headway_scale, accel_scale = sample_temperament(self.rng, bool(HEAVY[kind]))
        desired = edge.limit * factor
        warm_start = velocity is None
        velocity = min(desired, edge.limit * .92 if velocity is None else max(0., velocity))
        acceleration = HEAVY_ACCELERATION if HEAVY[kind] else DEFAULT_DRIVER.acceleration
        found = None
        for _ in range(8 if progress is None else 1):
            p = float(self.rng.uniform(0., max(0., length - 8.))) if progress is None else float(np.clip(progress, 0., length))
            speed = min(velocity, float(np.interp(p, graph.lanes[lane].arc, graph.lanes[lane].curve_speed)))
            if warm_start:
                speed = self._warm_speed(route, lane, p, speed, factor)
            event = self._approach_event(route, 0, lane, p, max(80., speed**2 / 4. + 8.), LENGTHS[kind])
            if event is not None:
                stop_distance, _ = event
                if stop_distance < -1e-9: continue  # never create an uncommitted body inside the stop bound
                if warm_start:
                    # Even a green warm spawn must be able to stop if the phase
                    # changes on its first tick. Preserve the ordinary 85% cap.
                    speed = min(speed, math.sqrt(BRAKE_ENVELOPE * max(0., stop_distance - 2.)))
            terminal = self._terminal_stop(route, 0, lane, p, max(80., speed**2 / 4. + 8.), LENGTHS[kind])
            if terminal is not None:
                if terminal < -1e-9: continue
                if warm_start: speed = min(speed, math.sqrt(BRAKE_ENVELOPE * max(0., terminal - 2.)))
            shared = self.shared.event(self, route, 0, lane, p, LENGTHS[kind], max(80., speed**2 / 4. + 8.))
            if shared is not None:
                if not self.shared.available(shared): continue
                if warm_start and shared.remaining >= 0.:
                    speed = min(speed, math.sqrt(BRAKE_ENVELOPE * max(0., shared.remaining - 2.)))
            leader = self._route_leader(route, 0, lane, p, occupancy, horizon=max(120., speed**2 / 2. + 30.))
            follower = self._lane_follower(lane, p, occupancy, horizon=240.)
            if leader is not None:
                gap = leader[1] - p - .5 * (LENGTHS[kind] + leader[3])
                if gap < desired_gap(speed, leader[2], acceleration=acceleration): continue
            if follower is not None:
                gap = p - follower[1] - .5 * (LENGTHS[kind] + follower[3])
                rear_acceleration = HEAVY_ACCELERATION if HEAVY[f.kind[follower[0]]] else DEFAULT_DRIVER.acceleration
                if gap < desired_gap(follower[2], speed, acceleration=rear_acceleration): continue
            found = p; break
        if found is None:
            return None
        i = f.allocate()
        if i is None:
            return None
        f.lane[i], f.progress[i], f.cursor[i] = lane, found, 0
        if shared is not None and shared.remaining + STOP_MARGIN <= 0.:
            # The candidate has no slot until allocation succeeds.
            for group, direction in shared.reservations:
                self.shared.commit(i, group, direction, shared.origin,
                                   entered=bool(self.graph.shared_memberships[lane]), life=f.life[i],
                                   admission=shared.remaining + STOP_MARGIN)
        f.factor[i], f.speed[i] = factor, speed
        f.headway[i], f.accel_scale[i] = DEFAULT_DRIVER.time_headway * headway_scale, accel_scale
        f.kind[i] = kind
        paint = PAINTS[int(self.rng.integers(len(PAINTS)))].copy() if color is None else np.asarray(color, np.float32)
        if kind == 5: paint = np.array([.98, .80, .05], np.float32)
        elif kind == 3: paint = np.array([.72, .12, .12], np.float32)
        elif kind == 6: paint = np.array([.90, .90, .88], np.float32)
        f.color[i], f.routes[i] = paint, route
        f.origin[i], f.destination[i] = graph.nodes[route.origin].key, graph.nodes[route.destination].key
        f.source[i] = edge.source[0]
        f.lane_from[i] = f.planned[i] = -1
        for array in (f.acceleration, f.merge_elapsed, f.cooldown, f.distance, f.steer,
                      f.stationary, f.junction_wait, f.retire_elapsed, f.yaw_rate, f.pitch_rate):
            array[i] = 0.
        f.age[i] = 0. if fade else 1.
        f.visibility[i] = 0. if fade else 1.
        f.retiring[i] = False
        f.phase[i] = int(self.rng.integers(256)) << 4
        f.wheel[i] = float(self.rng.uniform(0., 2 * math.pi))
        f.flags[i], f.stop_reason[i] = LAMPS | f.phase[i], ''
        # Retain the previous snapshot only to prove that these are the sole
        # edits since the last tick/publication. A known new body refreshes every
        # dependent route; any unrelated fixture/remap edit still rebuilds all.
        self._publication_spawns.add(int(i))
        xyz, heading, pitch, _, _ = graph.table.sample(np.array([lane]), np.array([found]))
        f.pose[i] = (*xyz[0], heading[0], pitch[0])
        occupancy.reserve(i, lane, found, f.speed[i], LENGTHS[kind])
        return int(i)

    def _warm_speed(self, route, lane, progress, speed, factor):
        """A late city spawn must already be able to brake for its first bend."""
        graph = self.graph
        horizon = max(80., speed**2 / 4. + 8.)
        for road, remaining, _, cid in self._route_roads(route, 0, lane, progress, horizon):
            if road != lane:
                entry = max(0., remaining - graph.table.length[road])
                limit = min(graph.limits[road] * factor, graph.splines[road].curve_speed[0])
                speed = min(speed, math.sqrt(limit**2 + BRAKE_ENVELOPE * entry))
            if cid >= 0:
                speed = min(speed, math.sqrt(graph.curve_min[cid]**2 + BRAKE_ENVELOPE * remaining))
                if graph.controlled[graph.splines[cid].node]:
                    break
        return speed

    def _spawn(self):
        f = self.fleet
        if f.n >= self.target_count:
            # Startup seeding is done; later replacements use ordinary trips.
            self._route_pool.clear()
            self._seeding = False
            return
        if (self.graph is None or (not self._ready_routes and not self._route_pool)
                or self.clock < self._spawn_wait_until):
            return
        occupancy = self.occupancy()
        spawned = 0
        for _ in range(min(32, self.target_count - f.n)):
            if self._ready_routes:
                route = self._ready_routes.popleft()
            elif self._route_pool:
                source = self._route_pool[int(self.rng.integers(len(self._route_pool)))]
                cursor = int(self.rng.integers(len(source.edges)))
                route = self.graph.compile_route(source.edges[cursor:])
                if route is None:
                    continue
            else:
                break
            spawned += self.spawn_route(route, occupancy=occupancy) is not None
            if f.n >= self.target_count:
                break
        if f.n >= self.target_count:
            self._route_pool.clear()
            self._seeding = False
        elif not spawned:
            # Try other streets before backing off; one occupied origin must not
            # leave the rest of the city empty. Paused saturated roads still sleep.
            self._spawn_wait_until = self.clock + .2

    def _route_trace(self, route, cursor, lane, virtual=False):
        """Bounded immutable topology/arc offsets, never driver progress or gaps."""
        key = (id(route), cursor, lane, virtual)
        trace = self._route_traces.get(key)
        if trace is not None:
            self._route_traces.move_to_end(key)
            return trace
        graph = self.graph
        roads, following, continuation = [], [], None
        preparation = approach = None
        distance = float(graph.table.length[lane])
        if lane >= graph.lane_count:
            lane = graph.splines[lane].outgoing
            cursor += 1
            following.append((lane, distance))
            distance += float(graph.table.length[lane])
        for _ in range(self.TRACE_STEPS):
            if cursor >= len(route.edges): break
            cid = graph.next_connection(route, cursor, lane)
            roads.append((lane, distance, cursor, cid))
            if approach is None and cid >= 0 and graph.controlled[graph.splines[cid].node]:
                approach = (distance, cid)
            edge = graph.edges[route.edges[cursor]]
            if (preparation is None and cursor + 1 < len(route.edges)
                    and ((route.turns[cursor] != STRAIGHT and graph.controlled[edge.end])
                         or len(route.allowed_lanes[cursor]) < edge.lane_count)):
                preparation = (distance, route.turns[cursor], cursor, route.allowed_lanes[cursor])
            if cid < 0:
                if not virtual or cursor + 1 >= len(route.edges): break
                allowed = route.allowed_lanes[cursor]
                alternative = min(allowed, key=lambda a: abs(graph.lanes[a].index - graph.lanes[lane].index))
                cid = graph.next_connection(route, cursor, alternative)
                if cid < 0: break
            following.append((cid, distance))
            lane = graph.splines[cid].outgoing
            following.append((lane, distance + float(graph.table.length[cid])))
            distance += float(graph.table.length[cid]) + float(graph.table.length[lane])
            cursor += 1
        else:
            if cursor < len(route.edges):
                continuation = (cursor, lane, distance - float(graph.table.length[lane]))
        trace = _RouteTrace(route, tuple(roads), tuple(following), continuation,
                            frozenset(s for s, _ in following), preparation, approach)
        self._route_traces[key] = trace
        if len(self._route_traces) > self.TRACE_CACHE_SIZE:
            self._route_traces.popitem(last=False)
        return trace

    def _route_splines(self, route, cursor, lane):
        if route is None or lane < 0:
            return
        while True:
            trace = self._route_trace(route, cursor, lane)
            for following, _ in trace.following:
                yield following
            if trace.continuation is None: return
            cursor, lane, _ = trace.continuation

    def _future_splines(self, i, lane=None):
        f = self.fleet
        yield from self._route_splines(f.routes[i], int(f.cursor[i]),
                                      int(f.lane[i]) if lane is None else int(lane))

    def _route_leader(self, route, cursor, lane, p, occupancy, exclude=-1, horizon=80.):
        """Neighbour in this route, expressed in the current lane's coordinates."""
        graph = self.graph
        leader, _ = occupancy.neighbours(lane, p, exclude)
        if leader is not None:
            return leader
        if route is None or lane < 0: return None
        offset = 0.
        heads, lengths = occupancy.heads, graph.table.length
        while True:
            trace = self._route_trace(route, cursor, lane)
            for following, start in trace.following:
                first = heads.get(following)
                if first is not None and first[0] == exclude:
                    first = occupancy.first(following, exclude)
                if first is not None:
                    return first[0], offset + start + first[1], first[2], first[3]
                # Preserve the query-before-horizon-check seam semantics.
                if offset + start + lengths[following] - p > horizon:
                    return None
            if trace.continuation is None: return None
            cursor, lane, advance = trace.continuation
            offset += advance

    def _lane_follower(self, lane, p, occupancy, exclude=-1, horizon=120.):
        """Nearest rear neighbour, including bodies traversing preceding seams."""
        graph = self.graph
        _, follower = occupancy.neighbours(lane, p, exclude)
        if follower is not None:
            return follower
        # Most shape/tile seams have one predecessor. Cache that immutable prefix
        # rather than rebuilding a heap/set for every candidate lane every 0.1 s.
        # Occupants, exclusion, progress and the rear-braking horizon stay live;
        # branches/cycles retain the original bounded nearest-first search.
        chain = self._upstream_chains.get(lane)
        if chain is None:
            prefix, seen = [], set()
            following = graph.predecessors.get(lane, ())
            while len(following) == 1 and len(prefix) < self.UPSTREAM_STEPS:
                previous = following[0]
                if previous in seen: break
                seen.add(previous); prefix.append(previous)
                following = graph.predecessors.get(previous, ())
            chain = tuple(prefix), tuple(following)
            self._upstream_chains[lane] = chain
            if len(self._upstream_chains) > self.UPSTREAM_CACHE_SIZE:
                self._upstream_chains.popitem(last=False)
        else:
            self._upstream_chains.move_to_end(lane)
        prefix, following = chain
        distance, horizon = float(p), float(horizon)
        for previous in prefix:
            if distance > horizon: return None
            length = graph.splines[previous].length
            last = occupancy.last(previous, exclude)
            if last is not None:
                rear_distance = distance + length - last[1]
                return ((last[0], p - rear_distance, last[2], last[3])
                        if rear_distance < horizon else None)
            distance += length
        if not following: return None
        pending = [(distance, previous) for previous in following]
        heapq.heapify(pending)
        seen, best, best_distance = set(prefix), None, horizon
        while pending and len(seen) < 128:
            distance, previous = heapq.heappop(pending)
            if distance > best_distance:
                break
            if previous in seen:
                continue
            seen.add(previous)
            length = graph.splines[previous].length
            last = occupancy.last(previous, exclude)
            if last is not None:
                rear_distance = distance + length - last[1]
                if rear_distance < best_distance:
                    best_distance = rear_distance
                    best = last[0], p - rear_distance, last[2], last[3]
            else:
                for predecessor in graph.predecessors.get(previous, ()):
                    heapq.heappush(pending, (distance + length, predecessor))
        return best

    def _lane_leader(self, i, occupancy, lane, p):
        f, graph = self.fleet, self.graph
        leader = self._route_leader(f.routes[i], int(f.cursor[i]), lane, p, occupancy,
                                    int(i), max(65., f.speed[i] * 4.))
        if leader is not None:
            return leader[1] - p - .5 * (LENGTHS[f.kind[i]] + leader[3]), leader[2]
        return np.inf, 0.

    def _leader(self, i, occupancy):
        f, graph = self.fleet, self.graph
        lane, p = int(f.lane[i]), float(f.progress[i])
        nearest = self._lane_leader(i, occupancy, lane, p)
        old = int(f.lane_from[i])
        if old >= 0:
            # During the full lateral blend the body still occupies both lanes.
            old_p = p * graph.table.length[old] / graph.table.length[lane]
            other = self._lane_leader(i, occupancy, old, old_p)
            if other[0] < nearest[0]: nearest = other
        return nearest

    def _leaders(self, ids, occupancy):
        """Batch same-lane/future neighbours so a denser fleet keeps up with time."""
        f, graph = self.fleet, self.graph
        leader, position, velocity, length = occupancy.leaders(f.lane[ids], f.progress[ids], ids)
        gaps = position - f.progress[ids] - .5 * (LENGTHS[f.kind[ids]] + length)
        missing = np.flatnonzero(leader < 0)
        if len(missing):
            slots = ids[missing]
            batch = self._road_batch(slots)
            following = batch.dependency_lanes[slots]
            starts = batch.dependency_distance[slots]
            horizon = np.maximum(65., f.speed[slots] * 4.)
            # The scalar probe queries a spline before checking its end against
            # the horizon. In particular, it always probes the first seam ahead.
            probes = (following >= 0) & (starts - f.progress[slots, None] <= horizon[:, None])
            probes[:, 0] = following[:, 0] >= 0
            rows, columns = np.nonzero(probes)
            found = np.zeros(len(slots), bool)
            if len(rows):
                # Query only reachable splines, not all 65 cached dependencies
                # for every car. nonzero keeps each route's probe order intact.
                next_id, next_pos, next_v, next_length = occupancy.first_batch(following[rows, columns], slots[rows])
                hits = np.flatnonzero(next_id >= 0)
                hits = hits[np.r_[True, np.diff(rows[hits]) != 0]] if len(hits) else hits
                selected = rows[hits]
                gaps[missing[selected]] = (starts[selected, columns[hits]] + next_pos[hits] - f.progress[slots[selected]]
                                           - .5 * (LENGTHS[f.kind[slots[selected]]] + next_length[hits]))
                velocity[missing[selected]] = next_v[hits]
                found[selected] = True
            continuation = batch.dependency_continuation[slots] - f.progress[slots] <= horizon
            for k in np.flatnonzero(continuation & ~found):
                i = int(slots[k])
                gaps[missing[k]], velocity[missing[k]] = self._lane_leader(i, occupancy, int(f.lane[i]), float(f.progress[i]))
        for k in np.flatnonzero(f.lane_from[ids] >= 0):
            i, old = int(ids[k]), int(f.lane_from[ids[k]])
            progress = f.progress[i] * graph.table.length[old] / graph.table.length[f.lane[i]]
            other = self._lane_leader(i, occupancy, old, progress)
            if other[0] < gaps[k]:
                gaps[k], velocity[k] = other
        return gaps, velocity

    def _walk_roads(self, i, horizon, *, virtual=False):
        """(lane, distance to its stop line, route cursor, connector) ahead.

        Distances include actual lane and connector lengths, including the
        receiving lane when the car is currently on a straight connector.
        Virtual lane selection is only for planning a future dedicated lane.
        """
        f = self.fleet
        yield from self._route_roads(f.routes[i], int(f.cursor[i]), int(f.lane[i]),
                                     float(f.progress[i]), horizon, virtual=virtual)

    def _route_roads(self, route, cursor, lane, progress, horizon, *, virtual=False):
        if route is None or lane < 0:
            return
        offset = 0.
        while True:
            trace = self._route_trace(route, cursor, lane, virtual)
            for road, distance, j, cid in trace.roads:
                remaining = offset + distance - progress
                if not remaining <= horizon: return
                yield road, remaining, j, cid
            if trace.continuation is None: return
            cursor, lane, advance = trace.continuation
            offset += advance

    def _turn_ahead(self, i, horizon=None):
        f, graph = self.fleet, self.graph
        route, cursor = f.routes[i], int(f.cursor[i])
        if route is None or f.lane[i] < 0:
            return None
        lane = graph.splines[f.lane[i]]
        if lane.incoming >= 0:
            if lane.turn != STRAIGHT and graph.controlled[lane.node]:
                return lane.turn, 0., cursor, route.allowed_lanes[cursor]
        if horizon is None:
            road = graph.lanes[lane.outgoing] if lane.incoming >= 0 else lane
            changes = max(1, graph.edges[road.edge].lane_count - 1)
            # Each adjacent change needs a fresh 30 m warning and a complete
            # lateral blend. Wide/fast roads must start before the fixed minimum.
            horizon = max(self.PREPARE_DISTANCE, changes * (self.INDICATOR_DISTANCE
                          + max(8., road.limit * self.MERGE_SECONDS + 3.)) + LENGTHS[f.kind[i]] + 10.)
        event = self._route_event(route, cursor, int(f.lane[i]), float(f.progress[i]),
                                  horizon, True, 'preparation')
        if event is None: return None
        distance, turn, j, allowed = event
        return turn, distance, j, allowed

    def _route_event(self, route, cursor, lane, progress, horizon, virtual, name):
        """First topology-only turn/control event, with live progress and horizon."""
        offset = 0.
        while True:
            trace = self._route_trace(route, cursor, lane, virtual)
            event = getattr(trace, name)
            if event is not None:
                distance = offset + event[0] - progress
                return (distance, *event[1:]) if distance <= horizon else None
            if trace.continuation is None: return None
            cursor, lane, advance = trace.continuation
            offset += advance
            if offset - progress > horizon: return None

    def _required_lane(self, i):
        f, graph = self.fleet, self.graph
        current = graph.lanes[f.lane[i]]
        ahead = self._turn_ahead(i)
        if ahead is None:
            return None
        turn, _, _, allowed = ahead
        ordinal = min(graph.lanes[a].index for a in allowed) if turn in (TURN_LEFT, UTURN) else max(graph.lanes[a].index for a in allowed) if turn == TURN_RIGHT else min((graph.lanes[a].index for a in allowed), key=lambda n: abs(n - current.index))
        lanes = graph.edges[current.edge].lanes
        target = lanes[min(ordinal, len(lanes) - 1)]
        return target if target != current.id else None

    def _required_lanes(self, ids):
        """Topology-cached preparation targets with live per-driver horizons."""
        return self._road_batch(ids).required_lanes(self, ids)

    def _mobil(self, i, target, occupancy, mandatory):
        f, graph = self.fleet, self.graph
        old = int(f.lane[i])
        p = float(f.progress[i])
        new_p = p * graph.splines[target].length / graph.splines[old].length
        route, cursor = f.routes[i], int(f.cursor[i])
        horizon = max(120., float(f.speed[i]) * 5.)
        old_leader = self._route_leader(route, cursor, old, p, occupancy, int(i), horizon)
        new_leader = self._route_leader(route, cursor, target, new_p, occupancy, int(i), horizon)
        old_follower = self._lane_follower(old, p, occupancy, int(i), horizon)
        new_follower = self._lane_follower(target, new_p, occupancy, int(i), horizon)
        # Positions in the two splines differ slightly around offset curves.
        delta = new_p - p
        old_leader = None if old_leader is None else (old_leader[0], old_leader[1] + delta, *old_leader[2:])
        old_follower = None if old_follower is None else (old_follower[0], old_follower[1] + delta, *old_follower[2:])
        desired = graph.lanes[target].limit * float(f.factor[i])
        options = dict(velocity=float(f.speed[i]), desired=desired, length=float(LENGTHS[f.kind[i]]),
                               position=new_p, old_leader=old_leader, old_follower=old_follower,
                               new_leader=new_leader, new_follower=new_follower,
                               follower_desired=lambda slot: graph.limits[f.lane[slot]] * f.factor[slot])
        accepted, incentive = mobil_decision(**options, mandatory=mandatory)
        if (not mandatory and graph.edges[graph.lanes[old].edge].priority >= 4
                and graph.lanes[target].index > graph.lanes[old].index
                and new_leader is None and incentive > -.05):
            accepted = mobil_decision(**options, mandatory=True)[0]
            incentive += .25  # use the same keep-right bias when revalidating intent
        return accepted, incentive

    def _parallel_connection(self, route, cursor, cid, alternative):
        graph = self.graph
        c = graph.splines[cid]
        if graph.controlled[c.node] or c.turn != STRAIGHT:
            return -1
        other = graph.next_connection(route, cursor, alternative)
        if (other < 0 or graph.splines[other].turn != STRAIGHT
                or graph.splines[other].outgoing == c.outgoing):
            return -1
        return other

    def _merge_room(self, i, target, needed):
        """Room through parallel shape/tile seams, but never across a box/merge."""
        f, graph = self.fleet, self.graph
        route, cursor, old = f.routes[i], int(f.cursor[i]), int(f.lane[i])
        p = f.progress[i] * graph.table.length[target] / graph.table.length[old]
        room = graph.table.length[target] - p
        while room < needed:
            cid = graph.next_connection(route, cursor, target)
            if cid < 0: break
            other = self._parallel_connection(route, cursor, cid, old)
            if other < 0: break
            room += graph.table.length[cid]
            target, old = graph.splines[cid].outgoing, graph.splines[other].outgoing
            room += graph.table.length[target]
            cursor += 1
        return room

    def _lane_decisions(self, occupancy):
        f, graph = self.fleet, self.graph
        changed = False
        ids = np.flatnonzero(f.active & ~f.retiring & (f.lane >= 0) & (f.lane < graph.lane_count) & (f.lane_from < 0))
        ids = np.asarray([i for i in ids.tolist() if graph.edges[graph.lanes[f.lane[i]].edge].lane_count >= 2
                          and not (f.planned[i] >= 0 and f.distance[i] - f.intent_distance[i] < self.INDICATOR_DISTANCE)], np.int64)
        required_lanes = self._required_lanes(ids) if len(ids) else ()
        for i, required in zip(ids.tolist(), required_lanes):
            lane = graph.lanes[f.lane[i]]
            edge = graph.edges[lane.edge]
            required = int(required) if required >= 0 else None
            # The overtaking cooldown must not strand a multi-lane turn. Every
            # mandatory step still gets its own indicator lead-in and MOBIL test.
            if f.cooldown[i] > 0 and required is None:
                continue
            if f.planned[i] >= 0:
                target = int(f.planned[i])
                if target >= graph.lane_count or graph.lanes[target].edge != edge.id:
                    f.planned[i] = -1; continue
                if f.distance[i] - f.intent_distance[i] < self.INDICATOR_DISTANCE:
                    continue
                # Ordinary shape vertices may be crossed during a blend; boxes,
                # turns and converging lanes must be reached after it completes.
                needed = max(f.speed[i], self.MIN_MERGE_SPEED) * self.MERGE_SECONDS + 3.
                if self._merge_room(int(i), target, needed) < needed:
                    continue
                accepted, _ = self._mobil(int(i), target, occupancy, required is not None)
                if not accepted:
                    continue
                old = int(f.lane[i])
                f.progress[i] *= graph.table.length[target] / graph.table.length[old]
                f.lane_from[i], f.lane[i], f.merge_elapsed[i] = old, target, 0.
                f.planned[i] = -1
                occupancy.reserve(int(i), target, f.progress[i], f.speed[i], LENGTHS[f.kind[i]])
                changed = True
                continue
            targets = []
            if required is not None:
                direction = 1 if graph.lanes[required].index > lane.index else -1
                targets.append((edge.lanes[lane.index + direction], True))
            elif edge.priority >= 4:
                if lane.index > 0: targets.append((edge.lanes[lane.index - 1], False))
                if lane.index + 1 < edge.lane_count: targets.append((edge.lanes[lane.index + 1], False))
            best = None
            for target, mandatory in targets:
                needed = self.INDICATOR_DISTANCE + max(f.speed[i], self.MIN_MERGE_SPEED) * self.MERGE_SECONDS + 3.
                if self._merge_room(int(i), target, needed) < needed:
                    continue
                accepted, incentive = self._mobil(int(i), target, occupancy, mandatory)
                if accepted and (best is None or incentive > best[0]):
                    best = (incentive, target)
            if best is not None:
                f.planned[i] = best[1]
                f.intent_distance[i] = f.distance[i]
        return changed

    def _approach_event(self, route, cursor, lane, progress, horizon, length):
        """Scalar physical stop event, including an uncommitted connector approach."""
        graph = self.graph
        if lane < 0 or route is None: return None
        if np.isfinite(graph.entry_progress[lane]):
            distance = graph.entry_progress[lane] - STOP_MARGIN - .5 * length - progress
            return (distance, lane) if distance <= horizon else None
        event = self._route_event(route, cursor, lane, progress,
                                  horizon + graph.max_stop_back + .5 * length, False, 'approach')
        if event is None: return None
        distance, cid = event
        distance += graph.approach_offsets[cid] - .5 * length
        return (distance, cid) if distance <= horizon else None

    def _terminal_stop(self, route, cursor, lane, progress, horizon, length):
        """A trip ending at a controlled short link must fade outside its box."""
        graph = self.graph
        for road, remaining, j, _ in self._route_roads(
                route, cursor, lane, progress, horizon + graph.max_stop_back + .5 * length):
            if j + 1 >= len(route.edges) and graph.controlled[graph.edges[graph.lanes[road].edge].end]:
                stop = min(remaining - .001, remaining + graph.stop_offsets[road] - .5 * length)
                return stop if stop <= horizon else None
        return None

    def _approaches(self, ids):
        f, graph = self.fleet, self.graph
        ids = ids[~f.retiring[ids] & (f.lane[ids] >= 0)]
        if not len(ids): return [], {}
        batch = self._road_batch(ids)
        progress = f.progress[ids]
        horizon = np.maximum(80., f.speed[ids] ** 2 / 4. + 8.)
        connections = batch.approach_connection[ids].copy()
        remaining = batch.approach_distance[ids] - progress
        connected = connections >= 0
        remaining[connected] += graph.approach_offsets[connections[connected]] - .5 * LENGTHS[f.kind[ids[connected]]]
        owned = np.fromiter((self.intersections._leases.get(int(i), -1) for i in ids), np.int64, count=len(ids))
        routed = np.fromiter((f.routes[int(i)] is not None for i in ids), bool, count=len(ids))
        current = np.isfinite(graph.entry_progress[f.lane[ids]]) & (owned != f.lane[ids]) & routed
        connections[current] = f.lane[ids[current]]
        remaining[current] = (graph.entry_progress[connections[current]] - STOP_MARGIN
                              - .5 * LENGTHS[f.kind[ids[current]]] - progress[current])
        selected = (connections >= 0) & (remaining <= horizon)
        result, by_slot = [], {}
        for k in np.flatnonzero(selected):
            i = int(ids[k])
            approach = Approach(i, int(connections[k]), float(remaining[k]),
                                float(f.speed[i]), float(LENGTHS[f.kind[i]]), float(f.junction_wait[i]))
            result.append(approach); by_slot[i] = approach
        fallback = batch.needs_continuation(ids, progress, horizon + graph.max_stop_back
                                            + .5 * LENGTHS[f.kind[ids]]) & ~selected & ~current
        for k in np.flatnonzero(fallback):
            i = int(ids[k])
            event = self._approach_event(f.routes[i], int(f.cursor[i]), int(f.lane[i]),
                                         float(progress[k]), float(horizon[k]), float(LENGTHS[f.kind[i]]))
            if event is not None:
                distance, cid = event
                approach = Approach(i, cid, distance, float(f.speed[i]),
                                    float(LENGTHS[f.kind[i]]), float(f.junction_wait[i]))
                result.append(approach); by_slot[i] = approach
        if fallback.any():
            result = [by_slot[i] for i in ids.tolist() if i in by_slot]
        return result, by_slot

    def _road_batch(self, ids):
        batch = self._road_lookahead
        if (batch is None or batch.graph is not self.graph or batch.steps != self.TRACE_STEPS
                or len(batch.routes) != self.fleet.capacity):
            batch = self._road_lookahead = RoadLookahead(self.fleet.capacity, self.TRACE_STEPS, self.graph)
        return batch.ensure(self, ids)

    def _road_limit_scalar(self, i, horizon, desired):
        """Reference/continuation path with the original road-walker semantics."""
        f, graph = self.fleet, self.graph
        lane, route = int(f.lane[i]), f.routes[i]
        stop, reason = np.inf, ''
        length = float(LENGTHS[f.kind[i]])
        for road_lane, remaining, j, cid in self._walk_roads(i, horizon + graph.max_stop_back + .5 * length):
            in_horizon = remaining <= horizon
            if road_lane != lane and in_horizon:
                entry = max(0., remaining - graph.table.length[road_lane])
                limit = min(graph.limits[road_lane] * f.factor[i], graph.splines[road_lane].curve_speed[0])
                desired = min(desired, math.sqrt(limit**2 + BRAKE_ENVELOPE * entry))
            stopping = min(remaining - .001, remaining + graph.stop_offsets[road_lane] - .5 * length)
            if cid >= 0 and in_horizon:
                desired = min(desired, math.sqrt(graph.curve_min[cid]**2 + BRAKE_ENVELOPE * remaining))
            if cid < 0 and route is not None and j + 1 < len(route.edges) and stopping <= horizon:
                stop, reason = stopping, 'turn lane'
            if (route is not None and j + 1 >= len(route.edges) and stopping <= horizon):
                if i in self._reroutes:
                    stop, reason = min(stop, stopping), 'routing'
                elif graph.controlled[graph.edges[graph.lanes[road_lane].edge].end]:
                    stop, reason = min(stop, stopping), 'destination'
            if cid >= 0 and graph.controlled[graph.splines[cid].node]:
                break
        return desired, stop, reason

    def _road_limits(self, ids, desired, horizon):
        """Vectorized braking envelopes with live progress, speed and driver factor."""
        f, graph = self.fleet, self.graph
        batch = self._road_batch(ids)
        progress = f.progress[ids]
        distance = batch.distance[ids] - progress[:, None]
        rows, columns = np.nonzero(batch.valid[ids] & (distance <= (horizon + graph.max_stop_back
                                                                  + .5 * LENGTHS[f.kind[ids]])[:, None]))
        slots = ids[rows]
        remaining, roads = distance[rows, columns], batch.lane[slots, columns]
        # Evaluate only roads inside each live braking horizon. A reduction at
        # the input row preserves repeated roads, input order and empty rows.
        desired = desired.copy()
        in_horizon = remaining <= horizon[rows]
        future = (roads != f.lane[slots]) & in_horizon
        entry = np.maximum(0., remaining[future] - graph.table.length[roads[future]])
        limit = np.minimum(batch.edge_limit[slots[future], columns[future]] * f.factor[slots[future]],
                           batch.entry_curve[slots[future], columns[future]])
        np.minimum.at(desired, rows[future], np.sqrt(limit**2 + BRAKE_ENVELOPE * entry))
        connected = batch.has_connection[slots, columns] & in_horizon
        curve = batch.connection_curve[slots[connected], columns[connected]]
        np.minimum.at(desired, rows[connected], np.sqrt(curve**2 + BRAKE_ENVELOPE * remaining[connected]))
        turn_stop, route_stop, arrival_stop = (np.full(len(ids), np.inf) for _ in range(3))
        stopping = np.minimum(remaining - .001, remaining + graph.stop_offsets[roads] - .5 * LENGTHS[f.kind[slots]])
        turns = batch.turn_stop[slots, columns] & (stopping <= horizon[rows])
        np.minimum.at(turn_stop, rows[turns], stopping[turns])
        rerouting = np.zeros(len(ids), bool)
        if self._reroutes:
            rerouting = np.fromiter((int(i) in self._reroutes for i in ids.tolist()), bool, count=len(ids))
            routing = batch.route_end[slots, columns] & rerouting[rows] & (stopping <= horizon[rows])
            np.minimum.at(route_stop, rows[routing], stopping[routing])
        destinations = (batch.route_end[slots, columns] & ~rerouting[rows] & (stopping <= horizon[rows])
                        & graph.controlled_ends[roads])
        np.minimum.at(arrival_stop, rows[destinations], stopping[destinations])
        stop = np.minimum(np.minimum(turn_stop, route_stop), arrival_stop)
        reasons = [''] * len(ids)
        for k in np.flatnonzero(np.isfinite(turn_stop)): reasons[k] = 'turn lane'
        for k in np.flatnonzero(np.isfinite(route_stop)): reasons[k] = 'routing'
        for k in np.flatnonzero(np.isfinite(arrival_stop)): reasons[k] = 'destination'
        for k in np.flatnonzero(batch.needs_continuation(ids, progress, horizon + graph.max_stop_back
                                                        + .5 * LENGTHS[f.kind[ids]])):
            desired[k], stop[k], reasons[k] = self._road_limit_scalar(int(ids[k]), float(horizon[k]), float(desired[k]))
        return desired, stop, reasons

    def _previous_spline(self, route, cursor, lane):
        """Reverse only the driver's legal route, not arbitrary upstream branches."""
        graph = self.graph
        if lane >= graph.lane_count: return graph.splines[lane].incoming, cursor
        if route is None or cursor <= 0 or cursor >= len(route.edges): return -1, cursor
        candidates = [cid for cid in graph.transitions.get((route.edges[cursor - 1], route.edges[cursor]), ())
                      if graph.splines[cid].outgoing == lane]
        if not candidates: return -1, cursor
        cid = min(candidates, key=lambda c: abs(graph.lanes[graph.splines[c].incoming].index - graph.lanes[lane].index))
        return cid, cursor - 1

    def _stop_before(self, i, allowance, occupancy, reason):
        """Exceptional boundary recovery, including a stop behind short shape seams.

        Ordinary braking is capped before integration. This fallback also handles
        graph/fixture edits and unusually large diagnostic timesteps safely; it
        must never pin a blocked bumper to the end of a too-short final link.
        """
        f, graph = self.fleet, self.graph
        allowance = min(0., float(allowance))
        lane, cursor, progress = int(f.lane[i]), int(f.cursor[i]), float(f.progress[i] + allowance)
        source, planned = int(f.lane_from[i]), int(f.planned[i])
        rewind = allowance
        for _ in range(128):
            if progress >= 0.: break
            previous, previous_cursor = self._previous_spline(f.routes[i], cursor, lane)
            if previous < 0:
                rewind -= progress
                progress = 0.; break
            if source >= 0: source, _ = self._previous_spline(f.routes[i], cursor, source)
            if planned >= 0: planned, _ = self._previous_spline(f.routes[i], cursor, planned)
            lane, cursor = previous, previous_cursor
            progress += graph.table.length[lane]
        f.lane[i], f.cursor[i], f.progress[i] = lane, cursor, max(0., progress)
        f.lane_from[i], f.planned[i] = source, planned
        f.speed[i], f.yaw_rate[i], f.pitch_rate[i] = 0., 0., 0.
        f.stop_reason[i] = reason
        if progress < 0.: rewind -= progress
        f.distance[i] = max(0., f.distance[i] + rewind)
        f.wheel[i] = (f.wheel[i] + rewind / WHEEL_RADII[f.kind[i]]) % (2 * math.pi)
        self.shared.rewind(int(i), f.distance[i])
        cid = self.intersections._leases.get(int(i))
        anchor = self.intersections._lease_distance.get(int(i))
        if cid is not None and anchor is not None and f.distance[i] < anchor + graph.entry_progress[cid] - .5 * LENGTHS[f.kind[i]] - 1e-9:
            self.intersections._release_lease(int(i))
        self._prediction_clearance[i] = 0.
        road = graph.splines[lane].outgoing if lane >= graph.lane_count else lane
        f.source[i] = graph.edges[graph.lanes[road].edge].source[0]
        occupancy.reserve(int(i), lane, f.progress[i], 0., LENGTHS[f.kind[i]])
        if source >= 0:
            occupancy.reserve(int(i), source, f.progress[i] * graph.table.length[source] / graph.table.length[lane],
                              0., LENGTHS[f.kind[i]])

    def _recover_entry_stops(self, approaches, shared_approaches, occupancy):
        """Restore revoked, unconsumed margins before freezing IDM neighbours.

        A green body can advance past its stop margin without reaching actual
        entry. If storage, priority or the phase then denies it, a zero movement
        cap alone leaves that old negative allowance in place indefinitely.
        Recover only such denied bodies along their legal route; consumed entry
        stays leased. The caller rebuilds occupancy before any integration, so a
        same-lane rewind cannot leave followers using stale frozen positions.
        """
        pending = {}
        for i, approach in approaches.items():
            if approach.remaining >= -1e-9 or self.intersections._leases.get(i) == approach.connection:
                continue
            reason = self.intersections.blocked(approach, occupancy)
            if reason: pending[i] = approach.remaining, reason
        for i, approach in shared_approaches.items():
            if approach.remaining >= -1e-9 or self.shared.available(approach): continue
            if i not in pending or approach.remaining < pending[i][0]:
                pending[i] = approach.remaining, 'shared'
        for i, (remaining, reason) in pending.items():
            self._stop_before(i, remaining, occupancy, reason)
        if pending:
            # Set the recovered confirmed orientation before the normal motion
            # sample; a stopped body must not retain a synthetic rewind yaw rate.
            self._poses(np.asarray(tuple(pending), np.int64))
        return bool(pending)

    def _shared_boundary(self, i, occupancy):
        """Catch every crossed corridor, even across multiple diagnostic seams."""
        f = self.fleet
        for _ in range(128):
            event = self.shared.event(self, f.routes[i], int(f.cursor[i]), int(f.lane[i]),
                                      float(f.progress[i]), float(LENGTHS[f.kind[i]]), 0., int(i))
            if event is None: return True
            if not self.shared.available(event):
                if event.remaining < 0.:
                    self._stop_before(i, event.remaining, occupancy, 'shared')
                    # A large step may stop on a later legal spline. Normalize
                    # that permitted prefix before exposing its final pose.
                    return f.progress[i] >= self.graph.table.length[f.lane[i]]
                return True
            if event.remaining + STOP_MARGIN > 1e-9: return True
            self.shared.commit_approach(event, f.distance[i] + event.origin, life=f.life[i])
        return False

    def _cross_boundaries(self, ids, occupancy):
        f, graph = self.fleet, self.graph
        for i in ids:
            for _ in range(128):
                lane = int(f.lane[i])
                if graph.shared_corridors and not self._shared_boundary(int(i), occupancy): break
                if lane < 0 or f.progress[i] < graph.table.length[lane]:
                    break
                overshoot = f.progress[i] - graph.table.length[lane]
                if lane >= graph.lane_count:
                    c = graph.splines[lane]
                    if graph.controlled[c.node] and self.intersections._leases.get(int(i)) != lane:
                        remaining = graph.entry_progress[lane] - STOP_MARGIN - .5 * LENGTHS[f.kind[i]] - f.progress[i]
                        approach = Approach(int(i), lane, float(remaining), float(f.speed[i]),
                                            float(LENGTHS[f.kind[i]]), float(f.junction_wait[i]))
                        reason = self.intersections.blocked(approach, occupancy)
                        if reason:
                            self._stop_before(i, remaining, occupancy, reason); break
                        if remaining + STOP_MARGIN <= 1e-9:
                            self.intersections.commit(int(i), lane, f.distance[i] - f.progress[i], life=f.life[i])
                    f.lane[i], f.progress[i] = c.outgoing, overshoot
                    f.cursor[i] += 1
                    f.source[i] = graph.edges[graph.lanes[c.outgoing].edge].source[0]
                    if f.lane_from[i] >= 0:
                        f.lane_from[i] = graph.splines[f.lane_from[i]].outgoing
                    if f.planned[i] >= 0:
                        f.planned[i] = graph.splines[f.planned[i]].outgoing
                    occupancy.reserve(int(i), c.outgoing, overshoot, f.speed[i], LENGTHS[f.kind[i]])
                    if f.lane_from[i] >= 0:
                        old = int(f.lane_from[i])
                        occupancy.reserve(int(i), old, overshoot * graph.table.length[old] / graph.table.length[c.outgoing],
                                          f.speed[i], LENGTHS[f.kind[i]])
                    continue
                route, cursor = f.routes[i], int(f.cursor[i])
                if i in self._reroutes and (route is None or cursor + 1 >= len(route.edges)):
                    remaining = min(graph.table.length[lane] - .001,
                                    graph.table.length[lane] + graph.stop_offsets[lane] - .5 * LENGTHS[f.kind[i]]) - f.progress[i]
                    self._stop_before(i, remaining, occupancy, 'routing')
                    break
                if route is None or cursor + 1 >= len(route.edges):
                    if graph.controlled[graph.edges[graph.lanes[lane].edge].end]:
                        remaining = min(graph.table.length[lane] - .001,
                                        graph.table.length[lane] + graph.stop_offsets[lane] - .5 * LENGTHS[f.kind[i]]) - f.progress[i]
                        self._stop_before(i, remaining, occupancy, 'destination')
                    else:
                        f.progress[i] = graph.table.length[lane]; f.speed[i] = 0.
                    self._retire(int(i), 'destination')
                    break
                cid = graph.next_connection(route, cursor, lane)
                other = self._parallel_connection(route, cursor, cid, int(f.lane_from[i])) if cid >= 0 and f.lane_from[i] >= 0 else -1
                if cid < 0 or (f.lane_from[i] >= 0 and other < 0):
                    remaining = min(graph.table.length[lane] - .001,
                                    graph.table.length[lane] + graph.stop_offsets[lane] - .5 * LENGTHS[f.kind[i]]) - f.progress[i]
                    self._stop_before(i, remaining, occupancy, 'turn lane')
                    break
                f.lane[i], f.progress[i] = cid, overshoot
                if other >= 0: f.lane_from[i] = other
                if f.planned[i] >= 0:
                    f.planned[i] = self._parallel_connection(route, cursor, cid, int(f.planned[i]))
                if graph.controlled[graph.splines[cid].node]:
                    remaining = graph.entry_progress[cid] - STOP_MARGIN - .5 * LENGTHS[f.kind[i]] - overshoot
                    approach = Approach(int(i), cid, float(remaining), float(f.speed[i]),
                                        float(LENGTHS[f.kind[i]]), float(f.junction_wait[i]))
                    reason = self.intersections.blocked(approach, occupancy)
                    if reason and remaining < 0.:
                        self._stop_before(i, remaining, occupancy, reason); break
                    if reason: f.stop_reason[i] = reason
                    elif remaining + STOP_MARGIN <= 1e-9 and self.intersections._leases.get(int(i)) != cid:
                        self.intersections.commit(int(i), cid, f.distance[i] - overshoot, life=f.life[i])
                occupancy.reserve(int(i), cid, overshoot, f.speed[i], LENGTHS[f.kind[i]])

    def _poses(self, ids, dt=0.):
        f, graph = self.fleet, self.graph
        valid = ids[f.lane[ids] >= 0]
        if not len(valid):
            return
        previous_yaw, previous_pitch = f.pose[valid, 3].copy(), f.pose[valid, 4].copy()
        xyz, yaw, pitch, curvature, _ = graph.table.sample(f.lane[valid], f.progress[valid])
        merging = f.lane_from[valid] >= 0
        if merging.any():
            slots = valid[merging]
            from_progress = f.progress[slots] * graph.table.length[f.lane_from[slots]] / graph.table.length[f.lane[slots]]
            old_xyz, old_yaw, old_pitch, old_curve, _ = graph.table.sample(f.lane_from[slots], from_progress)
            t = np.clip(f.merge_elapsed[slots] / self.MERGE_SECONDS, 0., 1.)
            blend = 10 * t**3 - 15 * t**4 + 6 * t**5
            # Lateral phase slows with forward motion below city crawling speed.
            # Its per-metre tangent stays bounded instead of pointing sideways
            # when a queue stops a driver halfway through a lane change.
            derivative = ((30 * t**2 - 60 * t**3 + 30 * t**4)
                          / (self.MERGE_SECONDS * np.maximum(f.speed[slots], self.MIN_MERGE_SPEED)))
            delta = xyz[merging] - old_xyz
            xyz[merging] = old_xyz + delta * blend[:, None]
            tangent = np.c_[np.cos(yaw[merging]), np.sin(yaw[merging])]
            direction = tangent + delta[:, :2] * derivative[:, None]
            yaw[merging] = np.where(f.speed[slots] > .01, np.arctan2(direction[:, 1], direction[:, 0]),
                                    previous_yaw[merging])
            pitch[merging] = old_pitch + (pitch[merging] - old_pitch) * blend
            curvature[merging] = old_curve + (curvature[merging] - old_curve) * blend
        if dt:
            change = np.arctan2(np.sin(yaw - previous_yaw), np.cos(yaw - previous_yaw))
            f.yaw_rate[valid] = np.clip(change / dt, -2., 2.)
            f.pitch_rate[valid] = np.clip((pitch - previous_pitch) / dt, -.5, .5)
            steering_curve = np.where(merging, f.yaw_rate[valid] / np.maximum(f.speed[valid], 1.), curvature)
            target = np.clip(np.arctan(WHEELBASES[f.kind[valid]] * steering_curve), -.62, .62)
            f.steer[valid] += np.clip(target - f.steer[valid], -.9 * dt, .9 * dt)
        f.pose[valid, :3], f.pose[valid, 3], f.pose[valid, 4] = xyz, yaw, pitch

    def _lights(self, ids):
        f, graph = self.fleet, self.graph
        if not len(ids): return
        flags = f.phase[ids] | LAMPS
        stopped = np.fromiter((bool(f.stop_reason[i]) for i in ids.tolist()), bool, count=len(ids))
        flags[(f.acceleration[ids] < -.4) | (stopped & (f.speed[ids] < .2))] |= BRAKE
        valid = f.lane[ids] >= 0
        merging = valid & (f.lane_from[ids] >= 0)
        planned = valid & ~merging & (f.planned[ids] >= 0)
        automatic = np.flatnonzero(valid & ~merging & ~planned)
        if len(automatic):
            slots = ids[automatic]
            turns, remaining = self._road_batch(slots).indicators(self, slots, self.INDICATOR_DISTANCE)
            indicating = (turns != STRAIGHT) & (remaining <= self.INDICATOR_DISTANCE)
            flags[automatic[indicating]] |= np.where(np.isin(turns[indicating], (TURN_LEFT, UTURN)), LEFT, RIGHT)
        def ordinal(spline):
            s = graph.splines[spline]
            return graph.lanes[s.incoming if s.incoming >= 0 else spline].index
        for k in np.flatnonzero(merging | planned).tolist():
            i, lane = int(ids[k]), int(f.lane[ids[k]])
            other = int(f.lane_from[i] if merging[k] else f.planned[i])
            left = ordinal(lane) < ordinal(other) if merging[k] else ordinal(other) < ordinal(lane)
            flags[k] |= LEFT if left else RIGHT
        f.flags[ids] = flags

    def tick(self, dt=STEP):
        if self.graph is None:
            self.clock += dt
            return
        start = time.perf_counter()
        f, graph = self.fleet, self.graph
        self._prediction_state = None
        self._prediction_clearance.fill(0.)
        self.clock += dt
        ids = np.flatnonzero(f.active & (f.lane >= 0) & ~f.retiring)
        self.shared.sync_slots(f)
        self.intersections.sync_slots(f)
        occupancy = self.occupancy()
        self._decisions += dt
        if self._decisions >= .1 - 1e-9:
            if self._lane_decisions(occupancy):
                occupancy = self.occupancy()
            self._decisions %= .1
        approaches, by_slot = self._approaches(ids)
        active = {int(i): (int(f.lane[i]), float(f.progress[i]), float(LENGTHS[f.kind[i]]), float(f.distance[i]), int(f.life[i]))
                   for i in np.flatnonzero(f.active & (f.lane >= 0))}
        if graph.shared_corridors: self.shared.begin_tick(active)
        shared_approaches = self.shared.approaches(self, ids)
        self.intersections.begin_tick(self.clock, active, approaches)
        if self._recover_entry_stops(by_slot, shared_approaches, occupancy):
            # Recovery can cross an ordinary seam without entering a junction.
            # Remove old occupancy reservations and refresh legal-route rows and
            # closest-approach buckets BEFORE computing any frozen neighbour cap.
            occupancy = self.occupancy()
            approaches, by_slot = self._approaches(ids)
            active = {int(i): (int(f.lane[i]), float(f.progress[i]), float(LENGTHS[f.kind[i]]), float(f.distance[i]), int(f.life[i]))
                      for i in np.flatnonzero(f.active & (f.lane >= 0))}
            if graph.shared_corridors: self.shared.begin_tick(active)
            shared_approaches = self.shared.approaches(self, ids)
            self.intersections.begin_tick(self.clock, active, approaches)
        if len(ids):
            curve_limit = graph.table.curve_limit(f.lane[ids], f.progress[ids])
            desired = np.minimum(graph.limits[f.lane[ids]] * f.factor[ids], curve_limit)
            horizon = np.maximum(80., f.speed[ids] ** 2 / 4. + 8.)
            desired, stops, stop_reasons = self._road_limits(ids, desired, horizon)
            gaps, leader_speed = self._leaders(ids, occupancy)
            caps = np.maximum(0., gaps - .05)
            slot_rows = np.empty(f.capacity, np.int64)
            slot_rows[ids] = np.arange(len(ids))
            for i, reason in zip(ids.tolist(), stop_reasons):
                f.stop_reason[i] = reason
            for i, approach in by_slot.items():
                reason = self.intersections.blocked(approach, occupancy)
                if reason:
                    k = slot_rows[i]
                    stops[k] = min(stops[k], approach.remaining)
                    f.stop_reason[i] = reason
            for i, approach in shared_approaches.items():
                if not self.shared.available(approach):
                    k = slot_rows[i]
                    stops[k] = min(stops[k], approach.remaining)
                    f.stop_reason[i] = 'shared'
            # All stop distances are centre travel allowances derived from the
            # physical front-bumper bound, including upstream short-link seams.
            stop_gaps = np.maximum(0., stops)
            leader_speed = np.where(stop_gaps < gaps, 0., leader_speed)
            gaps = np.minimum(gaps, stop_gaps)
            caps = np.minimum(caps, stop_gaps)
            acceleration = idm_acceleration(f.speed[ids], desired, gaps, leader_speed,
                                            acceleration=np.where(HEAVY[f.kind[ids]], HEAVY_ACCELERATION, DEFAULT_DRIVER.acceleration) * f.accel_scale[ids],
                                            time_headway=f.headway[ids])
            before_lanes, before_sources, before_cursors = f.lane[ids].copy(), f.lane_from[ids].copy(), f.cursor[ids].copy()
            scale = np.ones(len(ids))
            blending = before_sources >= 0
            scale[blending] = np.maximum(1., graph.table.length[before_sources[blending]]
                                        / graph.table.length[before_lanes[blending]])
            velocity, move = integrate_idm(f.speed[ids], acceleration, dt, caps / scale)
            # Arbitrate the actual body-entry crossings serially BEFORE any
            # position is published. Trimmed connector starts are not box entry.
            entering = dict.fromkeys((*by_slot, *shared_approaches))
            for i in entering:
                k = slot_rows[i]
                shared = shared_approaches.get(i)
                for _ in range(128):
                    if shared is None: break
                    if not self.shared.available(shared):
                        allowed = max(0., shared.remaining)
                        if move[k] > allowed: move[k], velocity[k] = allowed, 0.
                        caps[k] = min(caps[k], max(0., shared.remaining)); f.stop_reason[i] = 'shared'
                        break
                    if shared.remaining + STOP_MARGIN > move[k] + 1e-9: break
                    self.shared.commit_approach(shared, f.distance[i] + shared.origin, life=f.life[i])
                    shared = self.shared.event(self, f.routes[i], int(f.cursor[i]), int(f.lane[i]),
                                               float(f.progress[i]), float(LENGTHS[f.kind[i]]),
                                               max(80., f.speed[i]**2 / 4. + 8.), i)
                approach = by_slot.get(i)
                crossing = (approach is not None and self.intersections._leases.get(i) != approach.connection
                            and approach.remaining + STOP_MARGIN <= move[k] + 1e-9)
                reason = self.intersections.blocked(approach, occupancy) if crossing else None
                if reason:
                    move[k] = min(move[k], max(0., approach.remaining)); velocity[k] = 0.
                    caps[k] = min(caps[k], max(0., approach.remaining)); f.stop_reason[i] = reason
                self.shared.rewind(i, f.distance[i] + move[k])
                if i in shared_approaches:
                    next_shared = self.shared.event(self, f.routes[i], int(f.cursor[i]), int(f.lane[i]),
                                                    float(f.progress[i]), float(LENGTHS[f.kind[i]]),
                                                    max(80., f.speed[i]**2 / 4. + 8.), i)
                    if next_shared is None: del shared_approaches[i]
                    else: shared_approaches[i] = next_shared
                if crossing and reason is None:
                    cid = approach.connection
                    origin = (-f.progress[i] if f.lane[i] == cid else approach.remaining
                              - graph.approach_offsets[cid] + .5 * approach.length)
                    self.intersections.commit(i, cid, f.distance[i] + origin, life=f.life[i])
            # Freeze known obstacles at their pre-tick positions, then subtract
            # our advance in both occupied splines. Offset curves can have
            # different lengths, so source-lane progress may advance faster.
            self._prediction_clearance[ids] = np.maximum(0., caps - move * scale)
            for i, approach in by_slot.items():
                if self.intersections._leases.get(i) != approach.connection:
                    k = slot_rows[i]
                    margin = 0. if self.intersections.blocked(approach, occupancy) else STOP_MARGIN
                    entry = max(0., approach.remaining + margin - move[k] * scale[k])
                    self._prediction_clearance[i] = min(self._prediction_clearance[i], entry)
            for i, approach in shared_approaches.items():
                if not self.shared.admitted(approach):
                    k = slot_rows[i]
                    margin = STOP_MARGIN if self.shared.available(approach) else 0.
                    entry = max(0., approach.remaining + margin - move[k] * scale[k])
                    self._prediction_clearance[i] = min(self._prediction_clearance[i], entry)
            # A driver's own speed ceiling (limit x driver factor) is never exceeded, including downhill curves.
            f.speed[ids] = np.minimum(velocity, graph.limits[f.lane[ids]] * f.factor[ids])
            f.acceleration[ids] = acceleration
            f.progress[ids] += move
            f.distance[ids] += move
            f.wheel[ids] = np.mod(f.wheel[ids] + move / WHEEL_RADII[f.kind[ids]], 2 * np.pi)
            f.stationary[ids] = np.where((f.speed[ids] < .1) & (move < .004), f.stationary[ids] + dt, 0.)
            waiting = np.fromiter((bool(f.stop_reason[i]) for i in ids.tolist()), bool, count=len(ids))
            f.junction_wait[ids] = np.where(waiting & (f.speed[ids] < .3), f.junction_wait[ids] + dt, 0.)
            for k, i in enumerate(ids.tolist()):
                if stop_reasons[k] == 'destination' and stops[k] - move[k] <= 2.1 and f.speed[i] < .3:
                    self._retire(i, 'destination')
            merging = ids[f.lane_from[ids] >= 0]
            f.merge_elapsed[merging] += dt * np.minimum(1., f.speed[merging] / self.MIN_MERGE_SPEED)
            done = merging[f.merge_elapsed[merging] >= self.MERGE_SECONDS]
            f.lane_from[done] = -1; f.cooldown[done] = 5.
            ending = ids[f.progress[ids] >= graph.table.length[f.lane[ids]]]
            self._cross_boundaries(ending, occupancy)
            # A receiving road can have a lower limit than the spline used for
            # this tick's integration, especially for a late warm-start spawn.
            f.speed[ids] = np.minimum(f.speed[ids], graph.limits[f.lane[ids]] * f.factor[ids])
            for i in ids[f.stationary[ids] > self.STUCK_SECONDS]:
                self._retire(int(i), 'stuck')
            self._poses(ids, dt)
            changed = ((before_lanes != f.lane[ids]) | (before_cursors != f.cursor[ids])
                       | ((before_sources != f.lane_from[ids]) & (f.lane_from[ids] >= 0)))
            if changed.any():
                dirty = set(map(int, f.lane[ids[changed]]))
                dirty.update(map(int, f.lane_from[ids[changed]][f.lane_from[ids[changed]] >= 0]))
                self._refresh_transition_clearance(dirty)
        all_ids = np.flatnonzero(f.active)
        f.age[all_ids] += dt
        f.cooldown[all_ids] = np.maximum(0., f.cooldown[all_ids] - dt)
        retiring = all_ids[f.retiring[all_ids]]
        f.retire_elapsed[retiring] += dt
        f.visibility[all_ids] = np.minimum(1., f.age[all_ids] / .8)
        f.visibility[retiring] *= np.maximum(0., 1. - f.retire_elapsed[retiring] / self.FADE_SECONDS)
        f.speed[retiring] = 0.; f.yaw_rate[retiring] = 0.; f.pitch_rate[retiring] = 0.
        for i in retiring[f.retire_elapsed[retiring] >= self.FADE_SECONDS - 1e-9]:
            self.shared.release(int(i))
            if int(i) in self.intersections._leases: self.intersections._release_lease(int(i))
            f.release(int(i))
        self._lights(np.flatnonzero(f.active))
        self._refresh_entry_clearance(all_ids)
        self._remember_prediction()
        self.last_tick_ms = (time.perf_counter() - start) * 1000.
        self._tick_cost = self.last_tick_ms / 1000. if not self._tick_cost else .9 * self._tick_cost + .1 * self.last_tick_ms / 1000.
        self.tick_count += 1
        self._tick_history.append(self.last_tick_ms)

    def _prediction_arrays(self):
        f = self.fleet
        return f.active, f.life, f.lane, f.lane_from, f.progress, f.kind, f.cursor, f.speed, f.retiring

    def _control_signature(self):
        manager = self.intersections
        if manager is None: return None
        return (manager.clock, tuple((node, signal.index) for node, signal in manager.signals.items()),
                tuple(manager._leases.items()), tuple(self._reroutes), self.shared.signature())

    def _control_clearance(self, ids, occupancy):
        """Live front-body bounds; prediction cannot consume entry permission.

        Red/yield/blocked approaches keep their stop margin. A green but not-yet
        leased approach may be shown up to actual entry, never inside the box.
        The worker alone commits entry, including entry before a spline seam.
        """
        f, graph = self.fleet, self.graph
        safe = np.zeros(len(ids))
        active = np.flatnonzero(~f.retiring[ids] & (f.lane[ids] >= 0))
        if not len(active): return safe
        slots = ids[active]
        horizon = np.maximum(80., f.speed[slots] ** 2 / 4. + 8.)
        _, bounds, _ = self._road_limits(slots, graph.limits[f.lane[slots]] * f.factor[slots], horizon)
        safe[active] = np.maximum(0., bounds)
        _, approaches = self._approaches(slots)
        rows = {int(i): k for k, i in enumerate(ids)}
        for i, approach in approaches.items():
            if self.intersections._leases.get(i) == approach.connection: continue
            bound = approach.remaining + (0. if self.intersections.blocked(approach, occupancy) else STOP_MARGIN)
            safe[rows[i]] = min(safe[rows[i]], max(0., bound))
        for i, approach in self.shared.approaches(self, slots).items():
            bound = approach.remaining + (STOP_MARGIN if self.shared.available(approach) else 0.)
            safe[rows[i]] = min(safe[rows[i]], max(0., bound))
        return safe

    def _route_occupancy_changed(self, route, cursor, lane, progress, horizon, dirty):
        """Conservative topology dependency check, including bounded trace chunks."""
        offset = 0.
        while True:
            if lane in dirty: return True
            if route is None: return False
            trace = self._route_trace(route, cursor, lane)
            if not trace.occupied_lanes.isdisjoint(dirty): return True
            if trace.continuation is None: return False
            cursor, lane, advance = trace.continuation
            offset += advance
            if offset - progress > horizon: return False

    def _refresh_transition_clearance(self, dirty, *, spawned=()):
        """A new occupant may arrive from a connector not on a follower's route.

        Frozen pre-tick bounds remain safe for unchanged topology, but not when
        a body enters a previously empty route spline. Re-query only dependent
        drivers, retaining the stricter frozen/stop-line allowance.
        """
        f = self.fleet
        spawned = set(spawned)
        candidates = f.active & (f.lane >= 0) & (f.speed > 0.)
        if spawned:
            new = np.asarray(tuple(spawned), np.int64)
            candidates[new] = f.active[new] & (f.lane[new] >= 0)
        ids = np.flatnonzero(candidates)
        if not len(ids): return 0
        changed = self._road_batch(ids).dependencies(self, ids, dirty)
        if spawned: changed |= np.isin(ids, tuple(spawned))
        affected = ids[changed].tolist()
        if affected:
            occupancy = self.occupancy()
            if len(affected) > 32:
                selected = np.asarray(affected, np.int64)
                gaps, _ = self._leaders(selected, occupancy)
                safe = np.maximum(0., gaps - .05)
                self._prediction_clearance[selected] = np.where(np.isin(selected, tuple(spawned)), safe,
                                                                np.minimum(self._prediction_clearance[selected], safe))
            else:
                for i in affected:
                    gap, _ = self._leader(i, occupancy)
                    safe = max(0., gap - .05)
                    self._prediction_clearance[i] = safe if i in spawned else min(self._prediction_clearance[i], safe)
            selected = np.asarray(affected, np.int64)
            self._prediction_clearance[selected] = np.minimum(self._prediction_clearance[selected],
                                                              self._control_clearance(selected, occupancy))
        return len(affected)

    def _refresh_entry_clearance(self, ids):
        """Recheck revocable permission before remembering publication bounds.

        Another approach's receiving space can open during ordinary forward
        motion, giving it priority without any lane/cursor change. Frozen IDM
        gaps remain safe, but an unconsumed yield/entry allowance may not. Query
        only pending entries whose 100 ms prediction can exceed the denied bound;
        never rebuild leaders or increase a stricter frozen queue/stop allowance.
        """
        f, graph = self.fleet, self.graph
        selected = ids[f.active[ids] & (f.lane[ids] >= 0) & ~f.retiring[ids]
                       & (f.speed[ids] > 0.) & (self._prediction_clearance[ids] > 0.)]
        if not len(selected): return 0
        advance = np.zeros(f.capacity)
        endpoint = graph.table.length[f.lane[selected]] - f.progress[selected]
        advance[selected] = np.minimum(np.minimum(f.speed[selected] * .1, self._prediction_clearance[selected]),
                                       np.maximum(0., endpoint))
        selected = selected[advance[selected] > 0.]
        if not len(selected): return 0
        _, approaches = self._approaches(selected)
        occupancy, queries = None, 0
        for i, approach in approaches.items():
            if self.intersections._leases.get(i) == approach.connection: continue
            if advance[i] <= max(0., approach.remaining): continue
            if occupancy is None: occupancy = self.occupancy()
            margin = 0. if self.intersections.blocked(approach, occupancy) else STOP_MARGIN
            self._prediction_clearance[i] = min(self._prediction_clearance[i], max(0., approach.remaining + margin))
            queries += 1
        for i, approach in self.shared.approaches(self, selected).items():
            if advance[i] <= max(0., approach.remaining): continue
            margin = STOP_MARGIN if self.shared.available(approach) else 0.
            self._prediction_clearance[i] = min(self._prediction_clearance[i], max(0., approach.remaining + margin))
            queries += 1
        return queries

    def _remember_prediction(self):
        self._prediction_state = tuple(a.copy() for a in self._prediction_arrays())
        self._prediction_routes = tuple(self.fleet.routes)
        self._prediction_controls = self._control_signature()
        self._publication_spawns.clear()

    def _publication_clearance(self, ids):
        """Reuse tick caps; known spawns refresh dependencies, unknown edits rebuild."""
        f = self.fleet
        self.shared.sync_slots(f)
        self.intersections.sync_slots(f)
        unchanged = (self._prediction_state is not None
                     and all(np.array_equal(a, b) for a, b in zip(self._prediction_arrays(), self._prediction_state))
                      and all(a is b for a, b in zip(f.routes, self._prediction_routes)))
        controls_changed = self._prediction_controls != self._control_signature()
        if not unchanged:
            start = time.perf_counter()
            untouched = np.ones(f.capacity, bool)
            untouched[list(self._publication_spawns)] = False
            known_spawns = (self._prediction_state is not None and bool(self._publication_spawns)
                            and len(f.routes) == len(self._prediction_routes)
                            and all(a.shape == b.shape and np.array_equal(a[untouched], b[untouched])
                                    for a, b in zip(self._prediction_arrays(), self._prediction_state))
                            and all(f.routes[i] is self._prediction_routes[i] for i in np.flatnonzero(untouched)))
            if known_spawns:
                dirty = {int(f.lane[i]) for i in self._publication_spawns if f.active[i] and f.lane[i] >= 0}
                queries = self._refresh_transition_clearance(dirty, spawned=self._publication_spawns)
            else:
                self._prediction_clearance.fill(0.)
                occupancy = self.occupancy()
                valid = ids[f.lane[ids] >= 0]
                queries = len(valid)
                if len(valid) > 32:
                    gaps, _ = self._leaders(valid, occupancy)
                    self._prediction_clearance[valid] = np.maximum(0., gaps - .05)
                else:
                    for i in valid.tolist():
                        gap, _ = self._leader(i, occupancy)
                        self._prediction_clearance[i] = max(0., gap - .05)
                self._prediction_clearance[valid] = np.minimum(self._prediction_clearance[valid],
                                                               self._control_clearance(valid, occupancy))
            if known_spawns and controls_changed:
                valid = ids[f.lane[ids] >= 0]
                self._prediction_clearance[valid] = np.minimum(self._prediction_clearance[valid],
                                                               self._control_clearance(valid, self.occupancy()))
            self._remember_prediction()
            self._clearance_history.append(dict(mode='spawn' if known_spawns else 'full', queries=queries,
                                                milliseconds=(time.perf_counter() - start) * 1000.))
        elif controls_changed:
            valid = ids[f.lane[ids] >= 0]
            self._prediction_clearance[valid] = np.minimum(self._prediction_clearance[valid],
                                                           self._control_clearance(valid, self.occupancy()))
            self._prediction_controls = self._control_signature()
        return self._prediction_clearance

    def publish(self):
        start = time.perf_counter()
        f = self.fleet
        ids = np.flatnonzero(f.active)
        rows = self._packed
        rows.fill(0.)
        if len(ids):
            yaw, pitch = f.pose[ids, 3], f.pose[ids, 4]
            c, s, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
            rows[ids, 0], rows[ids, 1], rows[ids, 2] = c * cp, s * cp, sp
            rows[ids, 4], rows[ids, 5] = -s, c
            rows[ids, 8], rows[ids, 9], rows[ids, 10] = -c * sp, -s * sp, cp
            rows[ids, 3], rows[ids, 7], rows[ids, 11] = f.yaw_rate[ids], f.pitch_rate[ids], WHEEL_RADII[f.kind[ids]]
            rows[ids, 12:15], rows[ids, 15] = f.pose[ids, :3], 1.
            rows[ids, 16:19], rows[ids, 19] = f.color[ids], f.flags[ids]
            rows[ids, 20], rows[ids, 21], rows[ids, 23] = f.wheel[ids], f.steer[ids], f.visibility[ids]
            rows[ids, 22] = f.speed[ids]
            if self.graph is not None:
                clearance = self._publication_clearance(ids)
                valid = ids[f.lane[ids] >= 0]
                # Never predict past a spline end before the worker commits the
                # next connector, even if the route continues through that seam.
                endpoint = self.graph.table.length[f.lane[valid]] - f.progress[valid]
                safe = np.minimum(clearance[valid], np.maximum(0., endpoint))
                rows[valid, 22] = np.minimum(f.speed[valid], safe / .1)
        counts = {}
        for i in ids:
            key = f.source[i]
            counts[key] = counts.get(key, 0) + 1
        self._sequence += 1
        arrays, identities, distances = [], [], []
        for kind in range(KINDS):
            selected = ids[f.kind[ids] == kind]
            a = np.ascontiguousarray(rows[selected]) if len(selected) else None
            if a is not None: a.flags.writeable = False
            arrays.append(a)
            identity = ((f.life[selected].astype(np.uint64) << np.uint64(32))
                        | selected.astype(np.uint64)) if len(selected) else None
            distance = np.ascontiguousarray(f.distance[selected]) if len(selected) else None
            if identity is not None:
                identity.flags.writeable = distance.flags.writeable = False
            identities.append(identity); distances.append(distance)
        self.frame = VehicleFrame(arrays, clock=self.clock,
                                   generation=self.graph.generation if self.graph is not None else 0,
                                   sequence=self._sequence, counts=counts, ids=identities, distances=distances,
                                   signals=self._signal_stream.snapshot(self.clock) if self.graph is not None else None)
        self.last_publish_ms = (time.perf_counter() - start) * 1000.
        self._publish_history.append(self.last_publish_ms)
        return self.frame

    def update(self, dt):
        if self.closed:
            return self.frame
        self._collect_services()
        self._spawn()
        self.accum += max(0., min(float(dt), .25))
        ticks, step = min(8, int((self.accum + 1e-9) / self.STEP)), self.STEP
        if ticks:
            # Keep simulated time locked to real time. If the ticks of this update would
            # cost more wall time than they represent (a big city, a slow machine), take
            # fewer, longer steps instead of falling behind and playing in slow motion.
            # The car-following model integrates ballistically and every move is clearance
            # capped, so a step of up to MAX_STEP stays collision free.
            budget = .8 * self.accum - self.last_publish_ms / 1000.
            affordable = max(1, int(budget / max(self._tick_cost, 1e-4)))
            if ticks > affordable:
                ticks = affordable
                step = min(self.MAX_STEP, self.accum / ticks)
                ticks = max(1, min(ticks, int((self.accum + 1e-9) / step)))
        self.accum = max(0., self.accum - ticks * step)
        for _ in range(ticks):
            self.tick(step)
        return self.publish()

    def close(self, wait=True):
        if self.closed:
            return
        self.closed = True
        self.builder.close(wait=wait)
        self.router.close(wait=wait)
        self._spawn_jobs.clear(); self._ready_routes.clear(); self._reroutes.clear()
        self._route_pool.clear(); self._seeding = False
        self._route_traces.clear()
        self._upstream_chains.clear()
        self._road_lookahead = None
        self._prediction_state = None
        self._prediction_routes = ()
        self._prediction_controls = None
        self._publication_spawns.clear()

    def stats(self):
        graph = self.graph
        return dict(vehicles=self.fleet.n, target_count=self.target_count, nodes=len(graph.nodes) if graph else 0,
                    edges=len(graph.edges) if graph else 0, lanes=len(graph.lanes) if graph else 0,
                    connectors=len(graph.connections) if graph else 0, routes=self.router.cached,
                    route_hits=self.router.hits, routing=self.router.pending,
                    stuck_recycled=self.recycled_stuck, completed=self.completed_trips,
                    tick_ms=self.last_tick_ms, publish_ms=self.last_publish_ms,
                     ticks=self.tick_count, clock=self.clock, generation=self.generation,
                     tiles=len(self.tiles), graph_generation=graph.generation if graph else 0,
                     graph_building=self.builder.busy,
                    graph_build_ms=self.builder.last_build_ms,
                    error=str(self.error) if self.error else None)

    def diagnostics(self, *, safety=False, topology=False, reset=False):
        """Owner-thread inspection, explicitly outside the render/tick workload."""
        result = dict(stats=self.stats(), tick_ms=list(self._tick_history),
                      publish_ms=list(self._publish_history), clearance=[dict(record) for record in self._clearance_history])
        if safety and self.graph is not None:
            f, graph = self.fleet, self.graph
            ids = np.flatnonzero(f.active & (f.lane >= 0))
            occupancy = self.occupancy()
            minimum = math.inf
            for _, positions, _, lengths in occupancy.entries.values():
                if len(positions) > 1:
                    minimum = min(minimum, float(np.min(np.diff(positions) - .5 * (lengths[:-1] + lengths[1:]))))
            excess = float(np.max(f.speed[ids] - graph.limits[f.lane[ids]] * f.factor[ids])) if len(ids) else 0.
            prediction = 0.
            controls = self._control_clearance(ids, occupancy)
            for k, i in enumerate(ids):
                gap, _ = self._leader(int(i), occupancy)
                end = graph.table.length[f.lane[i]] - f.progress[i]
                safe = max(0., min(gap - .05, end, controls[k]))
                prediction = max(prediction, float(self._packed[i, 22]) * .1 - safe)
            result['safety'] = dict(
                finite=bool(np.isfinite(f.pose[ids]).all() and np.isfinite(f.speed[ids]).all()
                            and all(a is None or np.isfinite(a).all() for a in self.frame)
                            and (self.frame.signals is None or np.isfinite(self.frame.signals).all())),
                nonnegative_speed=bool((f.speed[ids] >= 0.).all()),
                minimum_bumper_gap=minimum if math.isfinite(minimum) else None,
                speed_limit_excess=excess, maximum_prediction_excess=prediction,
                moving=int((f.speed[ids] > .1).sum()), retiring=int(f.retiring[ids].sum()),
                route_trace_entries=len(self._route_traces),
                 junction_conflict_pairs=len(self.intersections._conflicts))
            result['safety']['shared_opposing_conflicts'] = self.shared.conflicts()
        if topology and self.graph is not None:
            # Expensive graph inspection is opt-in and never part of a tick or
            # ordinary frame publication. Only this bounded summary crosses IPC.
            result['topology'] = dict(
                splines_valid=all(np.all(np.diff(s.arc) > 0.) and all(
                    np.isfinite(getattr(s, name)).all() for name in
                    ('points', 'heading', 'pitch', 'curvature', 'curve_speed'))
                    for s in self.graph.splines),
                route_pairs=list(dict.fromkeys((r.origin, r.destination)
                    for r in self.fleet.routes if r is not None))[:64])
        if reset:
            self._tick_history.clear(); self._publish_history.clear()
            self._clearance_history.clear()
        return result


class TrafficSimulator:
    """Nonblocking process facade; ``threaded=False`` is for CPU fixtures.

    ``backend='thread'`` keeps the old owner thread for instrumentation/tests.
    The application default is spawn isolation, not a GIL scheduling workaround.
    """
    MAX_VIEW_DIST2 = 1500. ** 2

    def __init__(self, target_count=DEFAULT_TRAFFIC_COUNT, *, threaded=True, backend=None):
        self.target_count = max(0, int(target_count))
        self.tiles = {}
        self.backend = backend or ('process' if threaded else 'sync')
        if self.backend not in ('process', 'thread', 'sync'):
            raise ValueError('traffic backend must be process, thread or sync')
        if not threaded and self.backend != 'sync':
            raise ValueError('threaded=False requires the synchronous backend')
        threaded = self.backend != 'sync'
        self._threaded = bool(threaded)
        self._cv = threading.Condition()
        self._services_ready = False
        self.core = (TrafficCore(self.target_count, threaded_services=threaded,
                                 notify=self._notify_services if threaded else None)
                     if self.backend != 'process' else None)
        self._commands = {}
        self._pending_dt = 0.
        self._elapsed = 0.
        self._closed = False
        self._working = False
        self._frame = self.core.frame if self.core is not None else VehicleFrame([None] * KINDS)
        self._presentation = TrafficPresentation()
        self._state = {}
        self._thread = None
        self._worker = None
        if self.backend == 'process':
            from .traffic_worker import ProcessTrafficWorker
            self._worker = ProcessTrafficWorker(self.target_count, cv=self._cv)
        elif threaded:
            self._thread = threading.Thread(target=self._run, name='traffic-stream', daemon=True)
            self._thread.start()

    def add_tile(self, key, paths, origin, scale, roads=None):
        if key in self.tiles or self._closed:
            return
        paths = list(paths or ())
        self.tiles[key] = dict(paths=paths, origin=tuple(origin), scale=tuple(scale),
                               roads=roads)
        tile = GraphTile(key, tuple(paths), tuple(origin), tuple(scale))
        with self._cv:
            self._commands[key] = tile
            if self._worker is not None:
                self._submit_tiles()
            self._cv.notify_all()

    def remove_tile(self, key):
        if self._closed:
            return
        self.tiles.pop(key, None)
        with self._cv:
            self._commands[key] = None
            if self._worker is not None:
                self._submit_tiles()
            self._cv.notify_all()

    def _submit_tiles(self):
        # One latest authoritative snapshot: churn cannot accumulate tombstones
        # or retain evicted road arrays while an earlier build is still running.
        tiles = {key: GraphTile(key, tuple(t['paths']), t['origin'], t['scale'])
                 for key, t in self.tiles.items()}
        self._worker.submit(tiles=tiles)
        self._commands.clear()

    def _apply_commands(self, commands):
        if not commands:
            return
        tiles = dict(self.core.tiles)
        for key, tile in commands.items():
            if tile is None: tiles.pop(key, None)
            else: tiles[key] = tile
        self.core.set_tiles(tiles)

    def _accept(self, frame):
        self._frame = frame
        self._state = {key: SimpleNamespace(n=n) for key, n in frame.counts.items()}

    def _notify_services(self):
        with self._cv:
            self._services_ready = True
            self._cv.notify_all()

    def _run(self):
        try:
            while True:
                with self._cv:
                    while not self._closed and not self._commands and self._pending_dt <= 0 and not self._services_ready:
                        self._cv.wait()
                    if self._closed:
                        break
                    commands, self._commands = self._commands, {}
                    dt, self._pending_dt = self._pending_dt, 0.
                    self._services_ready = False
                    self._working = True
                self._apply_commands(commands)
                frame = self.core.update(dt)
                with self._cv:
                    self._accept(frame)
                    self._working = False
                    self._cv.notify_all()
        except Exception as exc:
            with self._cv:
                self.core.error = exc
                self._working = False
                self._cv.notify_all()

    def update(self, dt, cam_xy):
        dt = min(max(float(dt), 0.), .25)
        if self._closed:
            return None
        self._elapsed += dt
        with self._cv:
            if self._closed:
                return None
            if self._worker is not None:
                self._worker.submit(dt=dt)
                self._accept(self._worker.frame)
            elif self._threaded:
                self._pending_dt = min(.25, self._pending_dt + dt)
                self._cv.notify_all()
            else:
                commands, self._commands = self._commands, {}
                self._apply_commands(commands)
                self._accept(self.core.update(dt))
            frame = self._frame
        if not frame.generation:
            return [None] * KINDS
        asynchronous = self._worker is not None or self._threaded
        view = self._presentation.sample(frame, self._elapsed) if asynchronous else frame
        prediction = 0. if asynchronous else max(0., min(.1, self._elapsed - frame.clock))
        clock = view.clock + prediction
        arrays, identities, distances = [], [], []
        for kind, rows in enumerate(view):
            if rows is None:
                arrays.append(None); identities.append(None); distances.append(None); continue
            a = predict_instances(rows, prediction)
            near = (a[:, 12] - cam_xy[0])**2 + (a[:, 13] - cam_xy[1])**2 <= self.MAX_VIEW_DIST2
            arrays.append(np.ascontiguousarray(a[near]) if near.any() else None)
            identity = view.ids[kind][near].copy() if getattr(view, 'ids', None) is not None and near.any() else None
            distance = (view.distances[kind][near] + rows[near, 22] * prediction
                        if getattr(view, 'distances', None) is not None and near.any() else None)
            if identity is not None: identity.flags.writeable = False
            if distance is not None: distance.flags.writeable = False
            identities.append(identity); distances.append(distance)
        signals = view.signals
        if signals is not None:
            near = (signals[:, 12] - cam_xy[0])**2 + (signals[:, 13] - cam_xy[1])**2 <= self.MAX_VIEW_DIST2
            signals = np.ascontiguousarray(signals[near]) if near.any() else None
        return VehicleFrame(arrays, clock=clock, generation=view.generation,
                              sequence=view.sequence, counts=view.counts, signals=signals,
                              ids=identities if getattr(view, 'ids', None) is not None else None,
                              distances=distances if getattr(view, 'distances', None) is not None else None)

    def synchronize(self, timeout=10.):
        """Wait outside rendering (tests/shutdown), never called by update/draw."""
        deadline = time.monotonic() + float(timeout)
        if self._closed:
            return True
        if self._worker is not None:
            ready = self._worker.synchronize(timeout)
            with self._cv:
                self._accept(self._worker.frame)
            return ready
        if not self._threaded:
            self.update(0., (0., 0.))
            self.update(0., (0., 0.))
            return True
        with self._cv:
            self._cv.notify_all()
            while time.monotonic() < deadline:
                if self.core.error is not None:
                    raise RuntimeError('traffic worker failed') from self.core.error
                ready = (not self._commands and self._pending_dt <= 0 and not self._working
                         and not self._services_ready and not self.core._spawn_jobs and not self.core._reroutes
                         and not self.core.builder.busy and self._frame.generation == self.core.generation
                         and self.core.router.pending == 0)
                if ready:
                    return True
                self._cv.wait(max(0., deadline - time.monotonic()))
        return False

    def stats(self):
        if self._worker is not None:
            return dict(self._worker.stats(), **self._presentation.stats())
        return dict(self.core.stats(), backend=self.backend, **self._presentation.stats())

    def diagnostics(self, *, safety=False, topology=False, reset=False, timeout=10.):
        """Explicit blocking inspection; never used by update or draw."""
        if self._worker is not None:
            result = self._worker.diagnostics(safety=safety, topology=topology, reset=reset, timeout=timeout)
            with self._cv:
                self._accept(self._worker.frame)
            return result
        if not self.synchronize(timeout):
            raise TimeoutError('traffic diagnostics did not settle')
        return self.core.diagnostics(safety=safety, topology=topology, reset=reset)

    def close(self, wait=True):
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._commands.clear(); self._pending_dt = 0.
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join()
        if self._worker is not None:
            self._worker.close(wait=wait)
        else:
            self.core.close(wait=wait)
        self.tiles.clear(); self._state.clear()
        self._presentation.reset()

    shutdown = close

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
