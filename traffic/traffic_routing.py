"""Bounded background graph construction and cached A*/Dijkstra requests."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import queue
import threading
import time

import numpy as np

from .road_graph import RoadGraph


def _resolved(value):
    future = Future()
    future.set_result(value)
    return future


class GraphBuilder:
    """At most one build and one coalesced replacement, never a growing job queue."""
    def __init__(self, threaded=True, notify=None):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='traffic-graph') if threaded else None
        self._lock = threading.Lock()
        self._busy = False
        self._pending = None
        self._closed = False
        self._ready = queue.SimpleQueue()
        self.error = None
        self.last_build_ms = 0.
        self.notify = notify

    def request(self, tiles, generation):
        snapshot = (dict(tiles), int(generation))
        with self._lock:
            if self._closed:
                return
            if self._busy:
                self._pending = snapshot
                return
            self._busy = True
        self._submit(snapshot)

    def _submit(self, snapshot):
        if self._pool is None:
            try:
                self._finish(_resolved(self._build(*snapshot)))
            except Exception as exc:
                future = Future(); future.set_exception(exc); self._finish(future)
        else:
            try:
                self._pool.submit(self._build, *snapshot).add_done_callback(self._finish)
            except RuntimeError:
                with self._lock: self._busy = False

    def _build(self, tiles, generation):
        start = time.perf_counter()
        try:
            return RoadGraph.build(tiles, generation)
        finally:
            self.last_build_ms = (time.perf_counter() - start) * 1000.

    def _finish(self, future):
        try:
            result = future.result()
        except Exception as exc:
            result = None
            self.error = exc
        with self._lock:
            next_build, self._pending = self._pending, None
            if self._closed:
                self._busy = False
                return
            if next_build is None:
                self._busy = False
                if result is not None:
                    self._ready.put(result)
        if next_build is not None:
            self._submit(next_build)
        elif self.notify is not None:
            self.notify()

    def take(self):
        newest = None
        while True:
            try: newest = self._ready.get_nowait()
            except queue.Empty: return newest

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def close(self, wait=True):
        with self._lock:
            self._closed = True
            self._pending = None
        if self._pool is not None:
            self._pool.shutdown(wait=wait, cancel_futures=True)
        self.take()


class RoutingService:
    """Versioned LRU routes, in-flight deduplication and explicit backpressure.

    The simulation only submits and takes completed Futures. Search and random
    reachable destination selection execute on routing threads. Old snapshots
    stay valid for running searches, but old-generation results cannot be used
    by a newly published city graph. ``threaded=False`` is a deterministic test
    harness; the application always uses the background implementation.
    """
    def __init__(self, workers=2, cache_size=2048, max_pending=64, threaded=True, notify=None):
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix='traffic-route') if threaded else None
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._destinations = OrderedDict()
        self._inflight = {}
        self._pending = set()
        self._closed = False
        self.graph = None
        self.cache_size = max(1, int(cache_size))
        self.max_pending = max(1, int(max_pending))
        self.hits = self.searches = self.stale_results = 0
        self.notify = notify

    def set_graph(self, graph):
        with self._lock:
            self.graph = graph
            # Cache entries carry integer indices into exactly one snapshot.
            self._cache.clear()
            self._destinations.clear()
            pending = tuple(self._pending)
        # Future.cancel invokes callbacks synchronously. Notifications must not
        # hold the routing lock while the traffic facade holds its own condition.
        for future in pending:
            future.cancel()

    def _search(self, graph, origin, destination, incoming, algorithm):
        key = (graph.generation, origin, destination, incoming, algorithm)
        with self._lock:
            if key in self._cache:
                result = self._cache.pop(key); self._cache[key] = result
                self.hits += 1
                return result
            self.searches += 1
        # NumPy pose/physics probes briefly release the GIL. A continuous queue of
        # Python searches must let their owner regain it, not impose a full thread
        # switching interval on every tiny array operation.
        cooperate = (lambda: time.sleep(.0001)) if self._pool is not None else None
        result = graph.find_route(origin, destination, incoming=incoming, algorithm=algorithm,
                                  _cooperate=cooperate)
        with self._lock:
            if not self._closed and self.graph is graph:
                self._cache[key] = result
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            else:
                self.stale_results += 1
        return result

    def _submit(self, function, *args):
        with self._lock:
            if self._closed or len(self._pending) >= self.max_pending:
                return None
            if self._pool is None:
                try: return _resolved(function(*args))
                except Exception as exc:
                    future = Future(); future.set_exception(exc); return future
            future = self._pool.submit(function, *args)
            self._pending.add(future)
            future.add_done_callback(self._finished)
            return future

    def _finished(self, future):
        with self._lock:
            self._pending.discard(future)
        if self.notify is not None:
            self.notify()

    def request(self, origin, destination, *, incoming=-1, algorithm='astar'):
        with self._lock:
            graph = self.graph
            if graph is None or self._closed:
                return None
            key = (graph.generation, int(origin), int(destination), int(incoming), algorithm)
            if key in self._cache:
                self.hits += 1
                result = self._cache.pop(key); self._cache[key] = result
                return _resolved(result)
            existing = self._inflight.get(key)
            if existing is not None:
                return existing
            future = self._submit(self._search, graph, *key[1:])
            if future is not None:
                self._inflight[key] = future
                # Callback removal is identity checked: a stale completion must
                # not remove another request for the same origin/destination.
                def remove(done):
                    with self._lock:
                        if self._inflight.get(key) is done:
                            self._inflight.pop(key, None)
                future.add_done_callback(remove)
            return future

    def _random(self, graph, seed, origin, incoming, algorithm):
        if not len(graph.origins):
            return None
        rng = np.random.default_rng(seed)
        source = int(rng.choice(graph.origins, p=graph.origin_weights)) if origin is None else int(origin)
        if not (0 <= source < len(graph.nodes)):
            return None
        components = graph.destination_components(source, incoming)
        with self._lock:
            destinations = self._destinations.get(components) if self.graph is graph else None
            if destinations is not None:
                self._destinations.move_to_end(components)
        if destinations is None:
            destinations = graph.reachable_destinations(components)
            with self._lock:
                if not self._closed and self.graph is graph:
                    self._destinations[components] = destinations
                    while len(self._destinations) > 128:
                        self._destinations.popitem(last=False)
        if not destinations:
            return None
        for _ in range(16):
            destination = int(rng.choice(destinations))
            if source == destination:
                continue
            result = self._search(graph, source, destination, incoming, algorithm)
            if result is not None and result.edges:
                return result
        for eid in graph.nodes[source].outgoing:
            if incoming < 0 or (incoming, eid) in graph.transitions:
                return graph.compile_route((eid,))
        return None

    def random_request(self, seed, *, origin=None, incoming=-1, algorithm='astar'):
        with self._lock:
            if self.graph is None:
                return None
            return self._submit(self._random, self.graph, int(seed), origin, int(incoming), algorithm)

    @property
    def pending(self):
        with self._lock: return len(self._pending)

    @property
    def cached(self):
        with self._lock: return len(self._cache)

    def close(self, wait=True):
        with self._lock:
            self._closed = True
            pending = tuple(self._pending)
            self._cache.clear()
            self._destinations.clear()
        for future in pending: future.cancel()
        if self._pool is not None:
            self._pool.shutdown(wait=wait, cancel_futures=True)
        with self._lock:
            self._inflight.clear(); self._pending.clear(); self.graph = None
