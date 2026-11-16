"""Fixed-capacity, topology-only road lookahead rows for NumPy batch probes.

The route LRU remains bounded separately; this holds at most TRACE_STEPS scalar
road records and 2 * TRACE_STEPS + 1 dependency IDs per fleet slot, not complete
routes or live gaps/progress/horizons.
Longer lookahead falls back to the existing chunked route walker. Route identity,
lane and cursor changes refresh only their row, and graph installation clears it.
"""
from __future__ import annotations

import numpy as np


class RoadLookahead:
    def __init__(self, capacity, steps, graph):
        self.graph, self.steps = graph, int(steps)
        shape = (int(capacity), self.steps)
        self.routes = [None] * int(capacity)
        self.current_lane = np.full(capacity, -1, np.int64)
        self.cursor = np.full(capacity, -1, np.int64)
        self.valid = np.zeros(shape, bool)
        self.lane = np.zeros(shape, np.int64)
        self.distance = np.zeros(shape)
        self.edge_limit = np.zeros(shape)
        self.entry_curve = np.zeros(shape)
        self.connection_curve = np.zeros(shape)
        self.has_connection = np.zeros(shape, bool)
        self.turn_stop = np.zeros(shape, bool)
        self.route_end = np.zeros(shape, bool)
        self.approach_distance = np.full(capacity, np.inf)
        self.approach_connection = np.full(capacity, -1, np.int64)
        self.continuation = np.zeros(capacity, bool)
        self.last_distance = np.zeros(capacity)
        self.indicator_ready = np.zeros(capacity, bool)
        self.indicator_turn = np.zeros(capacity, np.int8)
        self.indicator_distance = np.full(capacity, np.inf)
        self.indicator_committed = np.zeros(capacity, bool)
        self.indicator_continuation = np.full(capacity, np.inf)
        self.preparation_target = np.full(capacity, -1, np.int64)
        # Full route-trace dependencies, not the road-limit row's shorter prefix
        # ending at its first control. A foreign connector can enter that future
        # route; retaining these bounded IDs avoids rebuilding every driver's LRU
        # trace on each sparse spawn/transition refresh.
        self.dependency_lanes = np.full((int(capacity), 2 * self.steps + 1), -1, np.int64)
        self.dependency_distance = np.zeros(self.dependency_lanes.shape)
        self.dependency_continuation = np.full(capacity, np.inf)

    def ensure(self, core, ids):
        f, graph = core.fleet, self.graph
        unused = np.flatnonzero((self.current_lane >= 0) & (~f.active | (f.lane < 0)))
        for i in unused:
            self.routes[i] = None
        self.current_lane[unused] = -1
        self.indicator_ready[unused] = False
        route_changed = np.fromiter((self.routes[i] is not f.routes[i] for i in ids.tolist()),
                                    bool, count=len(ids))
        changed = ((self.current_lane[ids] != f.lane[ids]) | (self.cursor[ids] != f.cursor[ids])
                   | route_changed)
        for i in ids[changed].tolist():
            route, lane, cursor = f.routes[i], int(f.lane[i]), int(f.cursor[i])
            self.routes[i], self.current_lane[i], self.cursor[i] = route, lane, cursor
            self.valid[i].fill(False)
            self.approach_distance[i], self.approach_connection[i] = np.inf, -1
            self.continuation[i] = False
            self.indicator_ready[i] = False
            self.dependency_lanes[i].fill(-1)
            self.dependency_continuation[i] = np.inf
            if route is None or lane < 0:
                continue
            trace = core._route_trace(route, cursor, lane)
            self.dependency_lanes[i, :len(trace.following)] = [s for s, _ in trace.following]
            self.dependency_distance[i, :len(trace.following)] = [d for _, d in trace.following]
            if trace.continuation is not None:
                self.dependency_continuation[i] = trace.continuation[2]
            if trace.approach is not None:
                self.approach_distance[i], self.approach_connection[i] = trace.approach
            controlled = False
            for j, (road, distance, road_cursor, cid) in enumerate(trace.roads):
                self.valid[i, j] = True
                self.lane[i, j], self.distance[i, j] = road, distance
                self.edge_limit[i, j] = graph.limits[road]
                self.entry_curve[i, j] = graph.splines[road].curve_speed[0]
                self.has_connection[i, j] = cid >= 0
                self.connection_curve[i, j] = graph.curve_min[cid] if cid >= 0 else 0.
                self.turn_stop[i, j] = cid < 0 and road_cursor + 1 < len(route.edges)
                self.route_end[i, j] = road_cursor + 1 >= len(route.edges)
                self.last_distance[i] = distance
                if cid >= 0 and graph.controlled[graph.splines[cid].node]:
                    controlled = True
                    break
            self.continuation[i] = not controlled and trace.continuation is not None
        return self

    def _prepare_indicators(self, core, ids):
        """Cache only the virtual turn's topology, shared by planning and lamps."""
        from .road_graph import STRAIGHT, TURN_LEFT, TURN_RIGHT, UTURN

        f, graph = core.fleet, self.graph
        for i in ids[~self.indicator_ready[ids]].tolist():
            self.indicator_ready[i] = True
            self.indicator_turn[i], self.indicator_distance[i] = STRAIGHT, np.inf
            self.indicator_committed[i], self.indicator_continuation[i] = False, np.inf
            self.preparation_target[i] = -1
            route, lane, cursor = self.routes[i], int(f.lane[i]), int(f.cursor[i])
            if route is None or lane < 0: continue
            spline = graph.splines[lane]
            if spline.incoming >= 0 and spline.turn != STRAIGHT and graph.controlled[spline.node]:
                self.indicator_turn[i], self.indicator_distance[i] = spline.turn, 0.
                self.indicator_committed[i] = True
                continue
            trace = core._route_trace(route, cursor, lane, True)
            if trace.preparation is not None:
                self.indicator_distance[i], self.indicator_turn[i] = trace.preparation[:2]
                if lane < graph.lane_count:
                    _, turn, _, allowed = trace.preparation
                    current = graph.lanes[lane]
                    indices = [graph.lanes[a].index for a in allowed]
                    ordinal = (min(indices) if turn in (TURN_LEFT, UTURN) else max(indices) if turn == TURN_RIGHT
                               else min(indices, key=lambda n: abs(n - current.index)))
                    lanes = graph.edges[current.edge].lanes
                    target = lanes[min(ordinal, len(lanes) - 1)]
                    if target != lane: self.preparation_target[i] = target
            elif trace.continuation is not None:
                self.indicator_continuation[i] = trace.continuation[2]

    def indicators(self, core, ids, horizon):
        """First virtual preparation event; lamp phases and progress stay live."""
        f = core.fleet
        self._prepare_indicators(core, ids)
        turns = self.indicator_turn[ids].copy()
        remaining = np.where(self.indicator_committed[ids], 0., self.indicator_distance[ids] - f.progress[ids])
        fallback = (np.isfinite(self.indicator_continuation[ids])
                    & (self.indicator_continuation[ids] - f.progress[ids] <= horizon))
        for k in np.flatnonzero(fallback):
            event = core._turn_ahead(int(ids[k]), horizon)
            if event is not None: turns[k], remaining[k] = event[:2]
        return turns, remaining

    def required_lanes(self, core, ids):
        """Same turn-lane targets as the scalar probe, without repeat route walks."""
        from ..vehicles.vehicle_instances import LENGTHS

        f, graph = core.fleet, self.graph
        self._prepare_indicators(core, ids)
        changes = np.fromiter((max(1, graph.edges[graph.lanes[int(lane)].edge].lane_count - 1)
                               for lane in f.lane[ids]), np.int64, count=len(ids))
        horizon = np.maximum(core.PREPARE_DISTANCE, changes * (core.INDICATOR_DISTANCE
                             + np.maximum(8., graph.limits[f.lane[ids]] * core.MERGE_SECONDS + 3.))
                             + LENGTHS[f.kind[ids]] + 10.)
        targets = np.where(self.indicator_distance[ids] - f.progress[ids] <= horizon,
                           self.preparation_target[ids], -1)
        fallback = (np.isfinite(self.indicator_continuation[ids])
                    & (self.indicator_continuation[ids] - f.progress[ids] <= horizon))
        for k in np.flatnonzero(fallback):
            target = core._required_lane(int(ids[k]))
            targets[k] = -1 if target is None else target
        return targets

    def dependencies(self, core, ids, dirty):
        """Scalar-equivalent conservative occupancy dependencies with live horizons.

        The first complete trace is queried even beyond the horizon, matching the
        original seam semantics. Longer continuations and blending source lanes
        retain the scalar walker; cached IDs never contain gaps or progress.
        """
        f, graph = core.fleet, self.graph
        changed = np.asarray(tuple(dirty), np.int64)
        dirty_lanes = np.zeros(len(graph.splines) + 1, bool)
        valid = (changed >= 0) & (changed < len(graph.splines))
        dirty_lanes[changed[valid] + 1] = True
        # Index zero represents padding, never an occupied spline. Dense boolean
        # membership avoids repeatedly sorting 65 dependency IDs per driver.
        affected = dirty_lanes[f.lane[ids] + 1]
        affected |= np.any(dirty_lanes[self.dependency_lanes[ids] + 1], axis=1)
        horizon = np.maximum(65., f.speed[ids] * 4.)
        continuation = self.dependency_continuation[ids]
        fallback = np.isfinite(continuation) & (continuation - f.progress[ids] <= horizon)
        for k in np.flatnonzero(fallback & ~affected).tolist():
            i = int(ids[k])
            affected[k] = core._route_occupancy_changed(f.routes[i], int(f.cursor[i]), int(f.lane[i]),
                                                       float(f.progress[i]), float(horizon[k]), dirty)
        for k in np.flatnonzero((f.lane_from[ids] >= 0) & ~affected).tolist():
            i, source = int(ids[k]), int(f.lane_from[ids[k]])
            progress = float(f.progress[i] * graph.table.length[source] / graph.table.length[f.lane[i]])
            affected[k] = core._route_occupancy_changed(f.routes[i], int(f.cursor[i]), source,
                                                       progress, float(horizon[k]), dirty)
        return affected

    def needs_continuation(self, ids, progress, horizon):
        # Conservative fallback before any unrecorded road can enter the horizon.
        return self.continuation[ids] & (self.last_distance[ids] - progress <= horizon)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in vars(self).values() if isinstance(a, np.ndarray))
