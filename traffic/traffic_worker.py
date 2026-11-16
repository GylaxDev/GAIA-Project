"""Spawn-isolated traffic ownership with bounded, coalesced IPC.

Only immutable packed frames and small status records cross back to rendering.
Road snapshots are serialized by an IO thread, never by update/add_tile. There is
one command in flight and one latest pending snapshot/time allowance. The reply
pipe has OS backpressure and is continuously drained into a one-frame mailbox.
No graphs, drivers, locks, Futures or GL resources cross the process boundary.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
import traceback

from ..vehicles.vehicle_instances import VehicleFrame

_UNCHANGED = object()


def _settled(core):
    return (not core._spawn_jobs and not core._reroutes and not core.builder.busy
            and core.frame.generation == core.generation and core.router.pending == 0)


def _traffic_process(requests, replies, stop, target_count, factory):
    """Importable spawn entry point; all traffic services live in this process."""
    from .core import TrafficCore

    cv = threading.Condition()
    pending = None
    services = False
    core = None

    def notify():
        nonlocal services
        with cv:
            services = True
            cv.notify_all()

    def receive():
        nonlocal pending
        try:
            while not stop.is_set():
                message = requests.recv()
                with cv:
                    if message is None:
                        stop.set()
                    else:
                        # The parent waits for publication/ack before sending
                        # another update, so this slot cannot grow into a queue.
                        if pending is not None:
                            raise RuntimeError('traffic command mailbox overflow')
                        pending = message
                    cv.notify_all()
        except (EOFError, OSError):
            stop.set()
            with cv: cv.notify_all()
        except Exception:
            stop.set()
            with cv: cv.notify_all()

    try:
        core = (factory or TrafficCore)(target_count, notify=notify)
        serial = 0
        replies.send(('frame', serial, core.frame, dict(core.stats(), worker_pid=os.getpid()), True, None))
        receiver = threading.Thread(target=receive, name='traffic-commands', daemon=True)
        receiver.start()
        while not stop.is_set():
            with cv:
                while pending is None and not services and not stop.is_set():
                    # Also observe a parent's bounded shutdown if its sender is
                    # interrupted while serializing a large topology snapshot.
                    cv.wait(.25)
                if stop.is_set(): break
                message, pending = pending, None
                service_wake = services
                services = False
            diagnostic = None
            tiles = None
            dt = 0.
            if message is not None:
                serial, tiles, dt, diagnostic = message
                if tiles is not None:
                    core.set_tiles(tiles)
            # A diagnostic-only RPC observes the published state without an
            # extra spawn/pack operation or a synthetic frame/timing sample.
            frame = (core.update(dt) if message is None or service_wake or tiles is not None or dt > 0.
                     else core.frame)
            details = None
            if diagnostic is not None:
                tag, safety, topology, reset = diagnostic
                details = (tag, core.diagnostics(safety=safety, topology=topology, reset=reset))
            with cv:
                ready = pending is None and not services and _settled(core)
            replies.send(('frame', serial, frame, dict(core.stats(), worker_pid=os.getpid()), ready, details))
    except Exception:
        try: replies.send(('error', traceback.format_exc()))
        except (OSError, EOFError): pass
    finally:
        if core is not None:
            core.close(wait=not stop.is_set())
        requests.close(); replies.close()


class ProcessTrafficWorker:
    """Render-side IO bridge; synchronize/diagnostics/close may block, submit may not."""
    MAX_PENDING_DT = .25
    SHUTDOWN_TIMEOUT = 5.

    def __init__(self, target_count, *, factory=None, cv=None):
        context = mp.get_context('spawn')
        child_requests, self._requests = context.Pipe(duplex=False)
        self._replies, child_replies = context.Pipe(duplex=False)
        self._stop = context.Event()
        self._cv = cv if cv is not None else threading.Condition()
        self._diagnostic_lock = threading.Lock()
        self._tiles = _UNCHANGED
        self._dt = 0.
        self._diagnostic = None
        self._diagnostic_reply = None
        self._revision = self._acked = self._sent = 0
        self._inflight = self._ready = self._closing = self.closed = False
        self._error = None
        self._dropped_dt = 0.
        self.frame = VehicleFrame()
        self._stats = dict(vehicles=0, nodes=0, lanes=0, routes=0, route_hits=0, routing=0,
                           stuck_recycled=0, tick_ms=0., publish_ms=0., ticks=0, clock=0., error=None,
                           tiles=0, generation=0, graph_generation=0, graph_building=False)
        self.process = context.Process(target=_traffic_process,
                                       args=(child_requests, child_replies, self._stop, target_count, factory),
                                       name='traffic-simulation', daemon=True)
        try:
            self.process.start()
        except BaseException:
            child_requests.close(); child_replies.close()
            self._requests.close(); self._replies.close()
            self.process.close()
            raise
        self.pid = self.process.pid
        child_requests.close(); child_replies.close()
        self._sender = threading.Thread(target=self._send, name='traffic-send', daemon=True)
        self._reader = threading.Thread(target=self._read, name='traffic-frames', daemon=True)
        self._sender.start(); self._reader.start()

    def submit(self, *, tiles=_UNCHANGED, dt=0.):
        with self._cv:
            if self._closing or self.closed or self._error is not None: return
            dt = max(0., float(dt))
            if tiles is _UNCHANGED and dt <= 0.: return
            if tiles is not _UNCHANGED: self._tiles = dict(tiles)
            self._dropped_dt += max(0., self._dt + dt - self.MAX_PENDING_DT)
            self._dt = min(self.MAX_PENDING_DT, self._dt + dt)
            self._revision += 1
            self._cv.notify_all()

    def _fail(self, message):
        with self._cv:
            if not self._closing and self._error is None:
                self._error = RuntimeError(message)
                self._stats = dict(self._stats, error=message)
            self._inflight = False
            self._cv.notify_all()

    def _send(self):
        try:
            while True:
                with self._cv:
                    while (not self._closing and self._error is None
                           and (self._inflight or self._sent >= self._revision)):
                        self._cv.wait()
                    if self._error is not None: return
                    if self._closing:
                        message = None
                    else:
                        self._sent = self._revision
                        tiles = None if self._tiles is _UNCHANGED else self._tiles
                        message = (self._sent, tiles, self._dt, self._diagnostic)
                        self._tiles, self._dt, self._diagnostic = _UNCHANGED, 0., None
                        self._inflight = True
                        self._cv.notify_all()
                # Pickling and a potentially full OS pipe are off the caller.
                self._requests.send(message)
                if message is None: return
        except (OSError, EOFError, BrokenPipeError) as exc:
            self._fail(f'traffic command channel failed: {exc}')
        except Exception:
            self._fail(traceback.format_exc())

    def _read(self):
        try:
            while True:
                message = self._replies.recv()
                if message[0] == 'error':
                    self._fail(message[1]); return
                _, serial, frame, stats, ready, details = message
                # Decoding/copy ownership is complete before acquiring the very
                # short mailbox lock. The render thread never holds an IPC lock.
                for rows in frame:
                    if rows is not None: rows.flags.writeable = False
                for table in (getattr(frame, 'ids', None), getattr(frame, 'distances', None)):
                    if table is not None:
                        for rows in table:
                            if rows is not None: rows.flags.writeable = False
                if frame.signals is not None: frame.signals.flags.writeable = False
                with self._cv:
                    self.frame, self._stats, self._ready = frame, stats, ready
                    self._acked = max(self._acked, serial)
                    if serial >= self._sent: self._inflight = False
                    if details is not None: self._diagnostic_reply = details
                    self._cv.notify_all()
        except (OSError, EOFError) as exc:
            self._fail(f'traffic process exited before shutdown: {exc}')
        except Exception:
            self._fail(traceback.format_exc())

    def _check_error(self):
        if self.process is not None and self.process.exitcode is not None and not self._closing:
            self._fail(f'traffic process exited with code {self.process.exitcode}')
        if self._error is not None:
            raise RuntimeError('traffic worker failed') from self._error
        if self._stats.get('error'):
            raise RuntimeError(f"traffic worker failed: {self._stats['error']}")

    def synchronize(self, timeout=10.):
        deadline = time.monotonic() + max(0., float(timeout))
        with self._cv:
            while not self.closed and not self._closing:
                self._check_error()
                if self._acked >= self._revision and not self._inflight and self._ready:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0.: return False
                self._cv.wait(remaining)
        return True

    def diagnostics(self, *, safety=False, topology=False, reset=False, timeout=10.):
        # One explicitly blocking inspector; concurrent inspectors cannot create
        # an unbounded RPC queue or overwrite an earlier caller's reply slot.
        deadline = time.monotonic() + max(0., float(timeout))
        with self._diagnostic_lock:
            with self._cv:
                self._check_error()
                if self._closing or self.closed: raise RuntimeError('traffic worker is closed')
                self._revision += 1
                tag = self._revision
                self._diagnostic = (tag, bool(safety), bool(topology), bool(reset))
                self._cv.notify_all()
                while self._diagnostic_reply is None or self._diagnostic_reply[0] != tag:
                    self._check_error()
                    if self._closing or self.closed: raise RuntimeError('traffic worker is closed')
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.: raise TimeoutError('traffic diagnostics timed out')
                    self._cv.wait(remaining)
                return self._diagnostic_reply[1]

    def stats(self):
        with self._cv:
            return dict(self._stats, backend='process', worker_pid=self.pid,
                        pending_dt=self._dt, dropped_dt=self._dropped_dt,
                        commands_in_flight=int(self._inflight))

    @property
    def alive(self):
        return self.process is not None and self.process.is_alive()

    def close(self, wait=True):
        with self._cv:
            if self.closed or self._closing: return
            self._closing = True
            self._tiles, self._dt, self._diagnostic = _UNCHANGED, 0., None
            self._cv.notify_all()
        self._stop.set()
        process = self.process
        if wait: process.join(self.SHUTDOWN_TIMEOUT)
        if process.is_alive():
            process.terminate(); process.join(2.)
        if process.is_alive():
            process.kill(); process.join()
        self._requests.close(); self._replies.close()
        self._sender.join(2.); self._reader.join(2.)
        process.close()
        self.process = None
        with self._cv:
            self.closed = True
            self._inflight = False
            self._cv.notify_all()
