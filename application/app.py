"""App, chunk streaming, chunk manager, and world orchestration."""
from __future__ import annotations

import gc
import math
import os
import queue
import sys
import tempfile
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np

from ..core import telemetry as tm
from ..vehicles.skidmarks import SkidMarks
from ..core.camera import Camera
from ..vehicles.driving import DrivingController, DriveControls, PlayerCar, vehicle_obstacles
from types import SimpleNamespace
from ..peds.walker import Walker, WalkControls
from ..peds.pedestrians import PedestrianCrowd
from ..vehicles.driving_surface import DrivingScene, DrivingTile
from ..vehicles.vehicle_instances import VEHICLE_NAMES, HEIGHTS, LENGTHS, WIDTHS, LEFT, RIGHT
from ..core.config import ACCENT, CITIES, SkyClock, StreamConfig
from ..world.data_fetcher import OSMDataFetcher, TileCancelled
from ..world.mesh_extruder import (
    ChunkResult, MeshCache, mesh_tile_job, shared_mesh_pool, shutdown_mesh_pool,
    MeshData, TrafficSimulator,
)
from ..ui.mapscreen import MapScreen
from ..ui.maps_ui import keycaps
from ..media.session import create_media_session
from ..ui.media_ui import NowPlayingPanel
from ..ui.minimap import MinimapPanel, build_massing
from ..navigation.core import Destination, NavigationController
from ..ui.navigation_ui import NavigationPanel
from ..world.places import parse_places
from ..ui.settings_ui import SettingsPanel
from ..ui import theme as T
from ..ui import weather_ui
from ..render.overlay import Overlay2D, TileCache
from ..vehicles import pick
from ..core.projection import ChunkGrid, LocalProjection, tile_bounds, tile_d2
from ..render.renderer import Renderer, SkyState, compute_sky
from ..render.lod import WATER_VERTEX
from ..world.weather_system import WeatherSystem
from ..world.live_weather import LiveWeather, local_hours
from .staging import StagedResult, stage_result, discard_future, shutdown_staging_cleanup
from .app_menu import MenuHost
from dataclasses import replace as _replace


QUEUED, LOADING, DONE, FAILED = "queued", "loading", "done", "failed"

_DPI_AWARE = False


APP_NAME = "GAIA"
ICON_FILE = Path(__file__).resolve().parents[1] / "assets" / "icons" / "gaia_64.png"


def _set_app_identity():
    """Give the process its own taskbar identity (Windows would otherwise group it under python.exe
    and show the Python icon) and set the window icon to the GAIA badge."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GAIA.Viewer")
        except Exception:
            pass


def _window_icon(pygame):
    try:
        pygame.display.set_icon(pygame.image.load(str(ICON_FILE)))
    except Exception:
        pass            # a missing icon file must never stop the app


def _enable_dpi_awareness():
    """Opt out of Windows' bitmap scaling.  Without this Windows renders the window at
    96 dpi and stretches the result, which blurs every pixel of the UI."""
    global _DPI_AWARE
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)      # per-monitor
        _DPI_AWARE = True
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            _DPI_AWARE = True
        except Exception:
            pass


def _system_ui_scale():
    """Logical -> device pixel factor for the UI (OSM3D_UI_SCALE overrides)."""
    env = os.environ.get("OSM3D_UI_SCALE")
    if env:
        try:
            return max(0.75, min(4.0, float(env)))
        except ValueError:
            pass
    if sys.platform != "win32" or not _DPI_AWARE:
        return 1.0
    import ctypes
    try:
        dpi = ctypes.windll.user32.GetDpiForSystem()
    except Exception:
        try:
            hdc = ctypes.windll.user32.GetDC(0)
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
            ctypes.windll.user32.ReleaseDC(0, hdc)
        except Exception:
            return 1.0
    return max(1.0, min(4.0, round(dpi / 96.0 * 4) / 4))


def _traffic_status_text(status):
    """A bounded set of startup/error labels; never wait for the traffic worker."""
    if status is None:
        return "Traffic: starting worker"
    if status.get('error'):
        return "Traffic: worker failed (F2 for details)"
    if not status.get('nodes'):
        return "Traffic: building roads" if status.get('tiles') else "Traffic: waiting for roads"
    if not status.get('vehicles'):
        return "Traffic: preparing routes"
    line = f"Traffic: {status['vehicles']} cars"
    if status.get('presentation_interval_ms', 0.) > 100.:
        line += "  \u00b7  worker behind"
    if status.get('generation', 0) > status.get('graph_generation', 0):
        line += "  \u00b7  updating roads"
    return line


def _cached_chunk_job(cache, z, key):
    """CPU-heavy cached geometry/road preparation runs outside the render GIL."""
    result = cache.load(z, key)
    if result is not None:
        # Older mesh caches remain usable. Backfill small labels/POIs from raw
        # map data in this worker, without re-extruding or overwriting the mesh.
        if result.places is None or any(not getattr(path, 'name', '') for path in result.paths or ()):
            raw = OSMDataFetcher._read_cache(
                cache.dir.parent / f'overpass_v4_{z}_{key[0]}_{key[1]}.json.gz')
            if raw is not None:
                s, w, n, e = tile_bounds(z, *key)
                if result.places is None:
                    result.places = parse_places(raw, LocalProjection((s + n) * .5, (w + e) * .5), (z, *key))
                names = {el.get('id'): (el.get('tags') or {}).get('name', '')
                         for el in raw.get('elements', ()) if el.get('type') == 'way'}
                for path in result.paths or ():
                    if not getattr(path, 'name', ''): path.name = names.get(path.uid, '')
            if result.places is None: result.places = ()
        return _stage_chunk(result, cache.dir.parent / 'staging')
    return result


def _stage_chunk(result, directory):
    from ..traffic.traffic_vec import TileRoads
    result.roads = TileRoads(result.paths or [])
    result.driving = DrivingTile.from_meshes(result.b_mesh, result.g_mesh, result.w_mesh)
    result.minimap = build_massing(result.b_mesh, result.driving)
    # Rendering uses packed buffers, precomputed bounds and neon source samples.
    for mesh in (result.b_mesh, result.g_mesh, result.w_mesh, result.s_mesh, result.p_mesh):
        if mesh is None: continue
        mesh.vertices = np.empty((len(mesh.vertices), 0), np.float32)
        if mesh.packed_edges is not None:
            mesh.edges = np.empty((len(mesh.edges), 0), np.float32)
    return stage_result(result, directory)


def _built_chunk_job(payload, cache):
    tile = mesh_tile_job(payload)
    b, g, w, s, p = [MeshData(*m) if m else None for m in
                    (tile.b_mesh, tile.g_mesh, tile.w_mesh, tile.s_mesh, tile.p_mesh)]
    if cache is not None:
        cache.store(payload[3], tile.key, b, g, w, tile.trees, lights=tile.lights,
                    parked=tile.parked, paths=tile.paths, s_mesh=s, p_mesh=p, places=tile.places)
    from ..render.lod import prepare_mesh, align_neon
    align_neon(b, s)
    for mesh, kind in ((b, 'building'), (g, 'terrain'), (w, 'water'), (s, 'neon'), (p, 'props')):
        prepare_mesh(mesh, kind)
    result = ChunkResult(tile.key, b, g, w, tile.trees, lights=tile.lights,
                          parked=tile.parked, paths=tile.paths, s_mesh=s, p_mesh=p, places=tile.places)
    directory = cache.dir.parent / 'staging' if cache is not None else Path(tempfile.gettempdir()) / 'gaia-staging'
    return _stage_chunk(result, directory)


class ChunkStreamer:
    def __init__(self, fetcher, grid, load_r2, workers=3, mesh_cache=None,
                 process_pool=True, prefetch_tiles=0.0):
        self.fetcher, self.grid = fetcher, grid
        self.cache = mesh_cache
        self.default_height = fetcher.default_height
        self.level_height = fetcher.level_height
        self.pool = shared_mesh_pool(workers, process_pool)
        self.results = queue.Queue(maxsize=2)
        self._cv = threading.Condition()
        self._state = {}; self._fails = {}
        self._ahead = set(); self._focus = None; self._stop = False
        self._threads = [threading.Thread(target=self._worker, name=f"chunk-{i}", daemon=True)
                         for i in range(max(1, workers))]
        for t in self._threads: t.start()

    def set_focus(self, key): self._focus = key

    def request(self, key, ahead=False):
        with self._cv:
            st = self._state.get(key)
            if st is not None:
                if not ahead: self._ahead.discard(key)
                if (st != FAILED
                        or time.monotonic() < self._fails.get(key, (0, 0.0, ""))[1]):
                    return False
            self._state[key] = QUEUED
            tm.BOARD.set(key, "queued", "waiting for a free worker")
            if ahead: self._ahead.add(key)
            else: self._ahead.discard(key)
            self._cv.notify(); return True

    def state(self, key): return self._state.get(key)
    def error(self, key): return self._fails.get(key, (0, 0.0, "unknown error"))[2]

    def forget(self, key):
        with self._cv:
            if self._state.get(key) in (QUEUED, DONE, FAILED):
                del self._state[key]
                self._fails.pop(key, None); self._ahead.discard(key)

    def prune(self, keep):
        with self._cv:
            for key in [k for k, s in self._state.items()
                        if s in (DONE, FAILED) and not keep(k)]:
                del self._state[key]
                self._fails.pop(key, None); self._ahead.discard(key)

    def counts(self):
        with self._cv:
            c = {QUEUED: 0, LOADING: 0, DONE: 0, FAILED: 0}
            for s in self._state.values(): c[s] += 1
            c["ahead"] = sum(1 for k in self._ahead
                             if self._state.get(k) in (QUEUED, LOADING))
        return c

    def shutdown(self):
        with self._cv:
            self._stop = True
            # Enqueue and shutdown share this lock: no worker can publish a
            # late mapped result after the manager has drained the queue.
            while True:
                try: self.results.get_nowait()
                except queue.Empty: break
            self._cv.notify_all()

    def _dist2(self, key):
        return 0 if self._focus is None else tile_d2(key, self._focus)

    def _cancelled(self, key):
        return self._stop or self._dist2(key) > 9.0

    def _take(self):
        with self._cv:
            while not self._stop:
                best, best_key = None, None
                for key, st in list(self._state.items()):
                    if st != QUEUED: continue
                    d = self._dist2(key)
                    if d > 9.0:
                        del self._state[key]; self._ahead.discard(key); continue
                    cand = (1 if key in self._ahead else 0, d)
                    if best is None or cand < best: best, best_key = cand, key
                if best_key is not None:
                    self._state[best_key] = LOADING; return best_key
                self._cv.wait(0.4)
        return None

    def _worker(self):
        while True:
            key = self._take()
            if key is None: return
            result = None
            try:
                for result in self._build(key):
                    while True:
                        with self._cv:
                            if self._cancelled(key): raise TileCancelled()
                            try:
                                self.results.put_nowait(result)
                                break
                            except queue.Full:
                                self._cv.wait(.05)
                    if self._stop: return
                    result = None
            except TileCancelled:
                tm.BOARD.drop(key)
                with self._cv:
                    self._state.pop(key, None); self._ahead.discard(key)
                continue
            except Exception as exc:
                with self._cv:
                    self._ahead.discard(key)
                    n = self._fails.get(key, (0, 0.0, ""))[0] + 1
                    backoff = min(60.0, 3.0 * 2 ** (n - 1))
                    self._fails[key] = (n, time.monotonic() + backoff, str(exc)[:160])
                    self._state[key] = FAILED
                tm.BOARD.set(key, "failed", str(exc)[:80])
                tm.LOG.error(f"tile {self.grid.zoom}/{key[0]}/{key[1]} failed: {str(exc)[:140]}")
                continue
            finally:
                # A cancelled full-queue producer otherwise pins its mapping
                # while it waits for the next tile, even after world teardown.
                result = None
            with self._cv:
                self._state[key] = DONE
                self._fails.pop(key, None); self._ahead.discard(key)

    def _run_job(self, payload, key):
        try:
            future = self.pool.submit_task(_built_chunk_job, payload, self.cache)
        except Exception:
            result = _built_chunk_job(payload, self.cache)
            return result.open(defer_cleanup=True) if isinstance(result, StagedResult) else result

        return self._wait_job(future, key)

    def _wait_job(self, future, key):
        while True:
            try:
                result = future.result(timeout=0.25)
                if self._cancelled(key):
                    if isinstance(result, StagedResult): result.discard()
                    raise TileCancelled()
                return result.open(defer_cleanup=True) if isinstance(result, StagedResult) else result
            except FutureTimeout:
                if self._cancelled(key):
                    if not future.cancel(): future.add_done_callback(discard_future)
                    raise TileCancelled()
            except TileCancelled:
                raise
            except Exception as e:
                tm.LOG.error(f"mesh job failed for tile {key}: {e}")
                raise e

    def _build(self, key):
        z = self.grid.zoom; tag = f"{z}/{key[0]}/{key[1]}"
        if self.cache is not None:
            tm.BOARD.set(key, "cache", "looking for a cached 3D mesh")
            hit = self._wait_job(self.pool.submit_task(_cached_chunk_job, self.cache, z, key), key)
            if hit is not None:
                tm.LOG.info(f"tile {tag}: 3D mesh loaded from cache")
                tm.BOARD.set(key, "upload", "sending to GPU")
                yield hit; return
        raw = self.fetcher.fetch_tile_raw(z, key[0], key[1],
                                          cancel=lambda: self._cancelled(key))
        lat0, lon0 = self.grid.center(key)
        n_el = len(raw.get("elements", ())) if isinstance(raw, dict) else 0
        tm.BOARD.set(key, "mesh", f"extruding {n_el:,} map elements")
        tm.LOG.info(f"tile {tag}: building 3D geometry from {n_el:,} elements")
        t0 = time.monotonic()
        heights, sea_z = self.fetcher.tile_terrain(z, key[0], key[1])
        if heights is not None:
            tm.LOG.info(f"tile {tag}: terrain elevation loaded")
        payload = (raw, lat0, lon0, z, key[0], key[1],
                   self.default_height, self.level_height, heights, sea_z)
        result = self._run_job(payload, key)
        tris = sum(len(m.indices) // 3 for m in (result.b_mesh, result.g_mesh, result.w_mesh) if m is not None)
        tm.LOG.info(f"tile {tag}: geometry ready, {tris:,} triangles in {time.monotonic() - t0:.1f}s")
        tm.BOARD.set(key, "upload", "sending to GPU")
        yield result


class ChunkManager:
    MIN_PREFETCH_SPEED = 12.0

    def __init__(self, renderer, grid, streamer, cfg, traffic_sim=None):
        self.renderer, self.grid, self.streamer, self.cfg = renderer, grid, streamer, cfg
        self.traffic_sim = (traffic_sim if traffic_sim is not None
                            else TrafficSimulator() if streamer is not None else None)
        self.chunks = {}
        self.driving_scene = DrivingScene()
        self.vram = 0
        self.focus = None
        self.ahead = ()
        self._last_update = 0.0
        self.uploaded = self.evicted = 0
        self._deferred = {}
        self._upload_job = None

    def update(self, target_xy, now, velocity=None):
        if self.streamer is None: return
        x, y = float(target_xy[0]), float(target_xy[1])
        focus = self.grid.key_at(x, y)
        ahead = self._prefetch_keys(x, y, focus, velocity)
        if (focus == self.focus and ahead == self.ahead
                and now - self._last_update < 0.5):
            return
        if focus != self.focus:
            for key in self._deferred: self.streamer.forget(key)
            self._deferred.clear()
        self.focus, self.ahead, self._last_update = focus, ahead, now
        self.streamer.set_focus(focus)
        for key in self.grid.keys_within(focus, self.cfg.load_r2):
            chunk = self.chunks.get(key)
            if chunk is not None: chunk.last_used = now
            elif key not in self._deferred: self.streamer.request(key)
        for key in ahead:
            chunk = self.chunks.get(key)
            if chunk is not None: chunk.last_used = now
            elif key not in self._deferred: self.streamer.request(key, ahead=True)
        self._evict(focus)
        self.streamer.prune(lambda k: tile_d2(k, focus) <= self.cfg.retain_r2)

    def _prefetch_keys(self, x, y, focus, velocity):
        if not self.cfg.prefetch or velocity is None: return ()
        vx, vy = float(velocity[0]), float(velocity[1])
        speed = math.hypot(vx, vy)
        if speed < self.MIN_PREFETCH_SPEED: return ()
        span = max(1, min(3, int(round(self.cfg.prefetch_tiles))))
        tw, th = self.grid.size_m(focus)
        reach = span * max(min(tw, th), 1.0)
        dx, dy = vx * self.cfg.prefetch_lead, vy * self.cfg.prefetch_lead
        dist = math.hypot(dx, dy)
        if dist > reach: dx, dy = dx * (reach / dist), dy * (reach / dist)
        step_x = 0 if abs(vx) < 0.35 * speed else (1 if vx > 0 else -1)
        step_y = 0 if abs(vy) < 0.35 * speed else (-1 if vy > 0 else 1)
        if not (step_x or step_y): return ()
        key = self.grid.key_at(x + dx, y + dy)
        if step_x and (key[0] - focus[0]) * step_x < span:
            key = (focus[0] + step_x * span, key[1])
        if step_y and (key[1] - focus[1]) * step_y < span:
            key = (key[0], focus[1] + step_y * span)
        if step_x and not step_y:
            cands = [key, (key[0], key[1] - 1), (key[0], key[1] + 1)]
        elif step_y and not step_x:
            cands = [key, (key[0] - 1, key[1]), (key[0] + 1, key[1])]
        else:
            cands = [key]
        edge = 1 << self.grid.zoom
        ring = set(self.grid.keys_within(focus, self.cfg.load_r2))
        return tuple(k for k in cands
                     if 0 <= k[0] < edge and 0 <= k[1] < edge
                     and k not in ring and k != focus)

    def _evict(self, focus):
        self.vram = sum(c.nbytes for c in self.chunks.values())
        far = sorted((c for c in self.chunks.values()
                      if tile_d2(c.key, focus) > self.cfg.retain_r2),
                     key=lambda c: c.last_used)
        for c in far: self._drop(c)
        budget = int(self.cfg.vram_mb * 1024 * 1024)
        if self.vram > budget:
            lru = sorted((c for c in self.chunks.values()
                          if c.key != focus),
                         key=lambda c: (-tile_d2(c.key, focus), c.last_used))
            for c in lru:
                if self.vram <= budget: break
                self._deferred[c.key] = c.nbytes
                self._drop(c)

    def _admit(self, res):
        size = sum(len(m.vertices)*stride + m.indices.nbytes +
                   (m.lod_indices.nbytes if m.lod_indices is not None else 0)
                   for m, stride in ((res.b_mesh, 48), (res.g_mesh, 52),
                                     (res.w_mesh, WATER_VERTEX.itemsize),
                                     (res.s_mesh, 48), (res.p_mesh, 48)) if m is not None)
        budget = max(0, int(self.cfg.vram_mb*1048576))
        old = self.chunks.get(res.key)
        required = size - (old.nbytes if old is not None else 0)
        if self.vram + required > budget and self.focus is not None:
            candidates = sorted((c for c in self.chunks.values() if c.key != res.key and
                                 tile_d2(c.key, self.focus) > tile_d2(res.key, self.focus)),
                                key=lambda c: -tile_d2(c.key, self.focus))
            for c in candidates:
                self._deferred[c.key] = c.nbytes
                self._drop(c)
                if self.vram + required <= budget: break
        if self.vram + required <= budget: return True
        self._deferred[res.key] = size
        tm.BOARD.set(res.key, 'done', 'cached; outside GPU residency budget')
        return False

    def _drop(self, c):
        self.driving_scene.remove_tile(c.key)
        self.renderer.free_chunk(c)
        if self.traffic_sim is not None:
            self.traffic_sim.remove_tile(c.key)
        del self.chunks[c.key]
        self.vram -= c.nbytes
        self.evicted += 1
        self.streamer.forget(c.key)
        tm.BOARD.drop(c.key)

    _upload_job = None      # (ChunkResult, upload generator) carried over between frames

    def pump_uploads(self, max_ms=4.0):
        """Move finished tiles to the GPU within a per-frame time budget.

        A big tile used to be uploaded in one go (tens of ms, a visible hitch).  The upload is
        now a generator that yields after each buffer set, so it is spread over several
        frames and the budget is honoured between steps."""
        if self.streamer is None: return
        t0 = time.perf_counter()
        while True:
            job = self._upload_job
            if job is not None and self.focus is not None and tile_d2(job[0].key, self.focus) > self.cfg.retain_r2:
                job[1].close()
                self.streamer.forget(job[0].key)
                self._upload_job = job = None
            if job is None:
                try: res = self.streamer.results.get_nowait()
                except queue.Empty: return
                if (self.focus is not None
                        and tile_d2(res.key, self.focus) > self.cfg.retain_r2):
                    self.streamer.forget(res.key); tm.BOARD.drop(res.key); continue
                if not (res.b_mesh or res.g_mesh or res.w_mesh
                        or (res.trees is not None and len(res.trees) > 0)):
                    c = None
                else:
                    if not self._admit(res):
                        if (time.perf_counter()-t0)*1000 >= max_ms: return
                        continue
                    old = self.chunks.pop(res.key, None)
                    if old is not None:
                        self.driving_scene.remove_tile(old.key)
                        self.renderer.free_chunk(old); self.vram -= old.nbytes
                        if self.traffic_sim is not None:
                            self.traffic_sim.remove_tile(old.key)
                    gen = self.renderer.upload_chunk_steps(
                        res.key, res.b_mesh, res.g_mesh, res.w_mesh, res.trees,
                        self.grid.origin(res.key),
                        scale=(self.grid.x_scale(res.key), 1.0),
                        lights=getattr(res, "lights", None),
                        parked=getattr(res, "parked", None),
                        s_mesh=getattr(res, "s_mesh", None),
                        p_mesh=getattr(res, "p_mesh", None))
                    job = (res, gen)
            else:
                res = job[0]
            c = None
            if job is not None:
                res, gen = job
                done = False
                while True:
                    try:
                        next(gen)
                    except StopIteration as stop:
                        c = stop.value; done = True; break
                    if (time.perf_counter() - t0) * 1000.0 >= max_ms:
                        break
                if not done:
                    self._upload_job = job          # resume next frame
                    return
                self._upload_job = None
            if c is not None:
                self.chunks[res.key] = c
                driving = getattr(res, 'driving', None)
                if driving is not None:
                    self.driving_scene.add_tile(res.key, driving, getattr(res, 'paths', None),
                                                self.grid.origin(res.key), (self.grid.x_scale(res.key), 1.0),
                                                getattr(res, 'minimap', None), places=getattr(res, 'places', ()))
                if self.driving_scene.filter_parked(c):
                    self.renderer._parked_key = None
                self.vram += c.nbytes; self.uploaded += 1
                if self.traffic_sim is not None:
                    paths = getattr(res, "paths", None)
                    if paths:
                        self.traffic_sim.add_tile(res.key, paths,
                                                  origin=self.grid.origin(res.key),
                                                  scale=(self.grid.x_scale(res.key), 1.0),
                                                  roads=res.roads)
            st = tm.BOARD.get(res.key)
            took = f" in {time.time() - st.t0:.1f}s" if st is not None else ""
            tm.BOARD.set(res.key, "done")
            tm.LOG.ok(f"tile {self.grid.zoom}/{res.key[0]}/{res.key[1]}: ready{took}")
            if (time.perf_counter() - t0) * 1000.0 >= max_ms: return

    def stats(self):
        if self.streamer is None: return None
        c = self.streamer.counts()
        return {"resident": len(self.chunks), "queued": c[QUEUED],
                "loading": c[LOADING], "failed": c[FAILED],
                "ahead": c["ahead"], "vram_mb": self.vram / 1048576.0}

    def release_all(self):
        if self._upload_job is not None: self._upload_job[1].close()
        self._upload_job = None
        if self.streamer is not None: self.streamer.shutdown()
        if self.streamer is not None:
            while True:
                try: self.streamer.results.get_nowait()
                except queue.Empty: break
        for c in list(self.chunks.values()): self.renderer.free_chunk(c)
        self.chunks.clear(); self.vram = 0
        self.driving_scene.clear()
        if self.traffic_sim is not None:
            self.traffic_sim.close()


class World:
    def __init__(self, renderer, label, cfg, proj=None, fetcher=None, radius=350.0,
                  static_mesh=None, mesh_cache=None, static_paths=None, static_places=None, traffic_count=None):
        self.proj, self.label = proj, label
        if static_mesh is not None:
            self.grid = self.streamer = self.center_key = None
            self.traffic_sim = None
            self.manager = ChunkManager(renderer, None, None, cfg, traffic_sim=None)
            # The modular extruder returns the complete scene, not one mesh.
            # Keep accepting a single building mesh for older static callers.
            parts = static_mesh if isinstance(static_mesh, tuple) else (static_mesh, None, None, None, None, None)
            b, g, w, trees, lights, parked = parts
            mesh_radius = max((m.radius for m in (b, g, w) if m is not None), default=radius)
            self.stream_radius = max(mesh_radius * 1.2, 500.0)
            self.home_distance = mesh_radius * 1.5
            chunk = renderer.upload_chunk((0, 0), b, g, w, trees, (0., 0.), lights=lights, parked=parked)
            self.manager.chunks[chunk.key] = chunk
            driving = DrivingTile.from_meshes(b, g, w)
            self.manager.driving_scene.add_tile(chunk.key, driving, static_paths,
                                                 minimap=build_massing(b, driving), places=static_places)
            self.manager.vram = chunk.nbytes
            self.manager.uploaded = 1
            return
        self.grid = ChunkGrid(proj, cfg.zoom)
        self.traffic_sim = TrafficSimulator(traffic_count) if traffic_count else TrafficSimulator()
        self.streamer = ChunkStreamer(fetcher, self.grid, cfg.load_r2, cfg.workers,
                                      mesh_cache, cfg.process_pool,
                                      cfg.prefetch_tiles if cfg.prefetch else 0.0)
        self.manager = ChunkManager(renderer, self.grid, self.streamer, cfg,
                                    traffic_sim=self.traffic_sim)
        self.center_key = self.grid.key_at(0.0, 0.0)
        tw, th = self.grid.size_m(self.center_key)
        self.stream_radius = (cfg.load_radius + 0.5) * max(tw, th)
        self.home_distance = radius * 1.5
        self.cfg = cfg

    def prime(self):
        if self.streamer is None: return
        self.streamer.set_focus(self.center_key)
        self.streamer.request(self.center_key)

    def close(self): self.manager.release_all()


class App(MenuHost):
    UPLOAD_BUDGET_MS = 4.0

    def __init__(self, fetcher, radius, cfg, demo_mesh=None, start=None, size=(1280, 800),
                 tile_url=None, tile_invert=True, cache_dir=Path(".osm_cache"),
                 start_map=False, start_hours=13.0, day_speed=0.05,
                 shadow_size=4096, shadows=True, fullscreen=False, vehicle_shadow_size=4096,
                 near_shadow_size=2048, sky_date=None, utc_offset=None, light_pollution=.35,
                  constellations=False, sky_location=(40.758, -73.9855), demo_paths=None, demo_places=None,
                  skip_menu=False):
        self.sky_clock = SkyClock(start_hours, day_speed, date=sky_date)
        self.utc_offset = None if utc_offset is None else float(utc_offset)
        self.light_pollution = float(light_pollution)
        self.sky_location = tuple(map(float, sky_location))
        if (self.utc_offset is not None and not math.isfinite(self.utc_offset)) or not math.isfinite(self.light_pollution):
            raise ValueError('UTC offset and light pollution must be finite')
        _enable_dpi_awareness()
        _set_app_identity()
        self.ui_scale = _system_ui_scale()
        size = (int(round(size[0] * self.ui_scale)), int(round(size[1] * self.ui_scale)))
        import pygame
        self.pg = pygame
        pygame.init()
        _window_icon(pygame)                # before the window exists, so the taskbar picks it up
        self._init_menu_state(cache_dir, skip_menu)
        tm.LOG.attach_file(Path(cache_dir) / "osm3d.log")
        tm.LOG.info(f"starting {APP_NAME}")
        self._joystick = None
        if pygame.joystick.get_count() > 0:
            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            tm.LOG.info(f"Gamepad detected: {self._joystick.get_name()}")
        self._create_window(*size)
        if fullscreen:
            try:
                pygame.display.toggle_fullscreen()
            except pygame.error:
                pass
        pygame.display.set_caption(APP_NAME)
        self.fetcher, self.radius, self.cfg = fetcher, radius, cfg
        w, h = pygame.display.get_window_size()
        self.ui = Overlay2D()               # first, so the boot screen can draw during the slow steps
        self.ui.scale = self.ui_scale
        self._boot(0.08, "Starting up")
        self.skidmarks = SkidMarks()
        self._boot(0.22, "Preparing the renderer")
        self.renderer = Renderer(w, h, shadow_size, shadows, vehicle_shadow_size=vehicle_shadow_size,
                                 near_shadow_size=near_shadow_size)
        self.renderer.skid_marks = self.skidmarks
        self.renderer.constellations_enabled = bool(constellations)
        self.renderer.set_aa(4)             # 4x MSAA on the scene target; F8 toggles
        self._boot(0.62, "Loading the sky and weather")
        self.weather = WeatherSystem("clear")
        self.live_weather = LiveWeather()
        self._live_applied = None
        self._live_offset_on = False       # live weather borrowed the clock's UTC offset
        self._utc_offset_saved = None      # ...and this is the offset to give back
        self.mesh_cache = MeshCache(
            cache_dir, cfg.mesh_cache,
            tag=f"d{float(getattr(fetcher, 'terrain_datum', 0.0)):.1f}")
        if tile_url is None:
            from ..core.config import DEFAULT_TILE_URL
            tile_url = DEFAULT_TILE_URL
        self.tiles = TileCache(tile_url, Path(cache_dir))
        self.map = MapScreen(self.tiles, CITIES, tile_invert)
        self.chrome = self.map.chrome
        self.settings = None               # the old F1 panel; the menu's Settings pages replaced it
        self._settings_capture = False
        self.media = NowPlayingPanel(create_media_session())
        self.show_media = False           # F7: now-playing card is opt-in
        self.show_stats = False           # F3: streaming / draw / fps diagnostics
        self._hint_key = None             # hints fade out a few seconds after the mode changes
        self._hint_t0 = 0.0
        self.minimap = MinimapPanel()     # bottom-left flat map inset
        self.navigation = NavigationController()
        self.destinations = NavigationPanel()
        self._navigation_capture = False
        self._sync_ui_size(w, h)
        self.camera, self.proj, self.label = None, None, ""
        self.world, self.next_world = None, None
        self.wireframe, self.show_edges = False, False   # outlines off: they read as glowing roof trim
        self.mode = "map"
        self.loading, self.load_label, self.load_start = False, "", 0.0
        self.load_key, self.load_attempts = None, 0
        self.show_log = False
        self.show_hud = True              # F9: hide every overlay element for clean shots
        self.geo_result = None
        self.message = ("", 0.0)
        self.typing, self.query = False, ""
        self.fade = 0.0
        self.press_dist = 0.0
        self.last_click = (0.0, (0, 0))
        self.vel = np.zeros(2, np.float64)
        self._vel_xy = self._vel_t = None
        self._traffic_instances = None
        self._traffic_status, self._traffic_status_at = None, 0.
        self._traffic_error_logged = None
        self.driving = DrivingController()
        self._init_walking()
        self._frame_sky = None            # sky state computed once per frame by _render

        # Per-frame key/mouse press state
        self._keys_pressed = set()
        self._mouse_left_pressed = False
        self._mouse_right_pressed = False

        self._boot(0.82, "Building the menus")
        self._init_menu()
        self._title_pending = bool(self.start_menu)
        self._boot(0.95, "Opening the world")
        if demo_mesh is not None:
            self._activate(World(self.renderer, "Demo city", cfg, static_mesh=demo_mesh,
                                   mesh_cache=self.mesh_cache, static_paths=demo_paths,
                                   static_places=demo_places), arrival=False)
            self.mode = "map" if start_map else "scene"
        elif start is not None and not start_map:
            self.mode = "scene"
            self._travel(*start)
        if self.world is None and not self.loading and self._title_pending:
            self._title_pending = False
            self.menu.open("main", title=True)       # started on the world map: greet with the menu

    def _create_window(self, w, h):
        pg = self.pg
        flags = pg.OPENGL | pg.DOUBLEBUF | pg.RESIZABLE
        # The HDR scene is resolved by post-processing; default-FBO MSAA only
        # replicated the fullscreen composite and HUD four times.
        for samples in (0,):
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
            pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
            pg.display.gl_set_attribute(pg.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
            pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 1 if samples else 0)
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, samples)
            # Frame pacing: without vsync the loop was throttled by a millisecond-granular
            # sleep (clock.tick(120)), which alternates 8 / 9 ms frames and tears.  Ask the
            # driver to pace the swap instead; fall back to the old cap if it refuses.
            try:
                pg.display.gl_set_attribute(pg.GL_SWAP_CONTROL, 1)
            except (AttributeError, pg.error):
                pass
            try:
                try:
                    pg.display.set_mode((w, h), flags, vsync=1)
                except TypeError:                       # pygame < 2: no vsync kwarg
                    pg.display.set_mode((w, h), flags)
                return
            except pg.error:
                pg.display.quit()
                pg.display.init()
        raise RuntimeError("Could not create an OpenGL 3.3 core context")

    def _activate(self, world, arrival):
        if getattr(self, 'navigation', None) is not None: self.navigation.clear()
        if getattr(self, 'destinations', None) is not None: self.destinations.clear()
        self._navigation_capture = False
        old, self.world = self.world, world
        if old is not None: old.close()
        gc.collect(); gc.freeze()          # keep the old world's long-lived objects out of later scans
        self.renderer.stream_radius = world.stream_radius
        self.camera = Camera(world.stream_radius, world.home_distance)
        self.camera.collide = self._camera_collide
        self.apply_camera_prefs()
        self.garage_cars = []
        self.vel[:] = 0.0; self._vel_xy = self._vel_t = None
        self._traffic_instances = None
        self._traffic_status, self._traffic_status_at = None, 0.
        self._traffic_error_logged = None
        self.driving = DrivingController()
        self._init_walking()
        self.proj, self.label = world.proj, world.label
        world.manager.pump_uploads(max_ms=60.0)
        if arrival:
            self.camera.begin_arrival(); self.fade = 1.0
        if world.proj:
            self.map.marker = (world.proj.lat0, world.proj.lon0, world.label)
        self.mode = "scene"
        if self._title_pending:
            self._title_pending = False
            self.menu.open("main", title=True)

    def _open_map(self, local=False):
        if self.mode == "map": return
        driving = getattr(self, 'driving', None)
        if driving is not None: driving.pause()
        if getattr(self, 'destinations', None) is not None: self.destinations.open = False
        self.mode = "map"
        if self.proj:
            anchor = driving.car.position if driving is not None and driving.active else self.camera.target
            lat, lon = self.proj.to_latlon(*anchor[:2])
            local = local or (driving is not None and driving.active)
            self.map.jump_to(lat, lon, 16.0 if local else 8.0)
            if not local: self.map.fly_to(lat, lon, 10.2, dur=1.1)

    def _navigation_anchor(self):
        if self.camera is None: return None
        driving = getattr(self, 'driving', None)
        if driving is not None and driving.active:
            return driving.car.position, driving.car.heading, driving.car.speed
        if self.walker.active:
            return self.walker.position, self.walker.heading, self.walker.speed
        anchor = self.camera.position if self.camera.mode == self.camera.FLY else self.camera.target
        return np.asarray(anchor[:2], np.float64), None, 0.

    def _update_navigation(self):
        nav = getattr(self, 'navigation', None)
        anchor = self._navigation_anchor()
        if nav is not None and self.world is not None and anchor is not None:
            nav.update(self.world.manager.driving_scene, *anchor)

    def _set_destination(self, destination):
        if self.loading:
            self._say("Wait for travel to finish before choosing directions", 3.)
            return
        self.navigation.set_destination(destination)
        self.destinations.open, self.typing = False, False
        if self.proj is not None:
            lat, lon = self.proj.to_latlon(*destination.position)
            self.map.marker = (lat, lon, destination.name)
        if self.driving.active: self.driving.pause()
        self.mode = "scene"
        self.show_hud = True
        self._say(f"Routing to {destination.name}", 3.)

    def _picker_open(self):
        return bool(getattr(getattr(self, 'destinations', None), 'open', False))

    def _toggle_destinations(self, place=None):
        panel = getattr(self, 'destinations', None)
        anchor = self._navigation_anchor()
        if panel is None or anchor is None or self.world is None:
            self._say("Travel to a place first to use driving directions", 3.)
            return
        if self.loading:
            self._say("Wait for travel to finish before choosing directions", 3.)
            return
        if panel.open and place is None:
            panel.open = False
        else:
            panel.show(self.world.manager.driving_scene, anchor[0], place)
            self.typing = False
            self.mode, self.show_hud, self.show_help = "scene", True, False
            self.map.end_drag()
            if self.driving.active: self.driving.pause()

    def _navigation_action(self, action):
        panel = self.destinations
        if action.startswith('place:'):
            index = int(action.split(':')[1])
            if 0 <= index < len(panel.items): self._set_destination(Destination.from_place(panel.items[index]))
        elif action == 'open': self._toggle_destinations()
        elif action == 'close': panel.open = False
        elif action == 'cancel':
            self.navigation.clear()
            self._say("Route ended", 2.)
        elif action == 'choose_map':
            panel.open = False
            if self.proj is not None: self._open_map(local=True)
            else: self._say("Right-click the minimap to drop a destination pin", 4.)

    def _navigate_marker(self):
        if self.world is None:
            self._say("Use Travel to first, then choose driving directions", 4.)
        elif self.proj is None or self.map.marker is None:
            self._toggle_destinations()
        else:
            lat, lon, label = self.map.marker
            x, y = self.proj.to_local(lat, lon)
            self._set_destination(Destination(label, float(x), float(y)))

    def _say(self, msg, secs=5.0):
        self.message = (msg, time.time() + secs)

    def _travel(self, lat, lon, label):
        if self.loading:
            self._say("Still travelling - press Esc to cancel"); return
        self.map.marker = (lat, lon, label)
        if self.mode == "map": self.map.fly_to(lat, lon, 9.0, dur=1.0)
        self.loading, self.load_label, self.load_start = True, label, time.time()
        self.load_attempts = 0
        tm.LOG.info(f"travel to {label} ({lat:.4f}, {lon:.4f})")
        self.next_world = World(self.renderer, label, self.cfg,
                                LocalProjection(lat, lon), self.fetcher, self.radius,
                                mesh_cache=self.mesh_cache, traffic_count=self.traffic_count)
        self.load_key = self.next_world.center_key
        self.next_world.prime()

    def _cancel_travel(self, reason="cancelled"):
        nw = self.next_world
        if nw is not None: nw.close()
        self.next_world, self.loading, self.load_key = None, False, None
        tm.LOG.warn(f"travel to {self.load_label} {reason}")
        if self.world is None:
            self.mode = "map"
            self.menu.open("main", title=True)      # nothing to go back to: return to the title menu
        self._say(f"Travel {reason}", 3.0)

    def _travel_marker(self):
        if self.map.marker is None:
            self._say("Click the map to pick a destination first")
        else:
            self._travel(*self.map.marker)

    def _ui_action(self, act):
        """A click that landed on the Apple-Maps chrome (sidebar, buttons, compass, zoom)."""
        ch, pg = self.chrome, self.pg
        w, h = pg.display.get_window_size()
        if act.startswith("city:"):
            i = int(act[5:])
            if 0 <= i < len(CITIES):
                self._travel(CITIES[i][1], CITIES[i][2], CITIES[i][0])
            return
        if act == "search":
            self.typing, self.query = True, ""
        elif act == "guides":
            ch.guides_open = not ch.guides_open
        elif act == "directions":
            self._navigate_marker()
        elif act == "route_cancel":
            self.navigation.clear()
            ch.navigation_active = False
            self._say("Route ended", 2.)
        elif act == "go":
            if self.map.marker: self._travel_marker()
            else: self._say("Pick a place on the map first", 2.5)
        elif act == "sidebar":
            ch.sidebar_open = not ch.sidebar_open
        elif act == "weather":
            ch.weather_open = not ch.weather_open
        elif act == "weather_close":
            ch.weather_open = False
        elif act == "weather_units":
            ch.weather_units.imperial = not ch.weather_units.imperial
        elif act == "weather_live":
            self.set_live_weather(not self.weather.live)
        elif act == "weather_panel":
            pass                            # a click on the card itself must not reach the 3D view
        elif act == "fullscreen":
            self._toggle_fullscreen()
        elif act == "map":
            if self.mode == "scene":
                self._open_map()
            else:
                self.map.toggle_style()
                self._say("Dark map" if self.map.invert > 0.5 else "Standard map", 1.8)
        elif act == "locate":
            if self.mode == "scene":
                if self.camera: self._reset_view()
            elif self.camera:
                self.mode = "scene"
            elif self.map.marker:
                self._travel_marker()
        elif act == "compass":
            if self.mode == "scene" and self.camera:
                if self.driving.active: self.camera.recenter_drive()
                else: self.camera.azimuth = math.radians(-90)
        elif act in ("zoom_in", "zoom_out"):
            d = 1.0 if act == "zoom_in" else -1.0
            if self.mode == "map": self.map.zoom_step(d)
            elif self.camera: self.camera.zoom(d * 2.0)

    def _toggle_fullscreen(self):
        pg = self.pg
        try:
            pg.display.toggle_fullscreen()
        except pg.error as exc:
            self._say(f"Fullscreen unavailable: {exc}", 3.0)
            return
        w, h = pg.display.get_window_size()
        self.renderer.resize(w, h)
        self._sync_ui_size(w, h)
        self._say("Fullscreen on (F11 to exit)" if pg.display.is_fullscreen()
                  else "Fullscreen off", 1.6)

    def _poll_background(self):
        nw = self.next_world
        if nw is not None:
            st = nw.streamer.state(nw.center_key)
            if st == FAILED:
                self.load_attempts += 1
                err = nw.streamer.error(nw.center_key)
                if self.load_attempts < 3:
                    tm.LOG.warn(f"travel: attempt {self.load_attempts}/3 failed ({err}) - retrying")
                    nw.streamer.forget(nw.center_key)
                    nw.streamer.request(nw.center_key)
                else:
                    tm.LOG.error(f"travel failed after 3 attempts: {err}")
                    self._say(f"Travel failed: {err}", 8)
                    self.show_log = True
                    self._cancel_travel("failed")
            elif st == DONE:
                waiting = (self.mode == "map" and self.map.flying
                           and time.time() - self.load_start < 7.0)
                if not waiting:
                    tm.LOG.ok(f"arrived at {self.load_label} "
                              f"({time.time() - self.load_start:.1f}s)")
                    self.next_world, self.loading, self.load_key = None, False, None
                    self._activate(nw, arrival=True)
        if self.geo_result is not None:
            kind, *rest = self.geo_result
            self.geo_result = None
            if kind == "err":
                tm.LOG.error(f"search failed: {rest[0]}")
                self._say(f"Search failed: {rest[0]}", 6)
            else:
                lat, lon, name = rest
                tm.LOG.ok(f"search found {name} ({lat:.4f}, {lon:.4f})")
                self.map.marker = (lat, lon, name)
                self._open_map()
                self.map.fly_to(lat, lon, 9.5, dur=1.2)
                self._say("Found it - press Enter (or double-click) to travel there", 7)

    def _update_world(self, dt):
        if self.mode == "scene" and self.world is not None and self.camera is not None:
            now = time.time()
            # Turning the view moves the fly target by hundreds of metres even
            # when the eye stands still. Residency/prefetch must track the eye.
            driving = getattr(self, 'driving', None)
            is_driving = driving is not None and driving.active
            anchor = (driving.car.position if is_driving else self.walker.position if self.walker.active
                      else self.camera.position if self.camera.mode == self.camera.FLY else self.camera.target)
            xy = np.asarray(anchor[:2], np.float64)
            if is_driving:
                self.vel = driving.car.velocity[:2].copy()
            elif self._vel_xy is not None:
                delta = max(now - self._vel_t, 1e-3)
                self.vel = self.vel * 0.75 + ((xy - self._vel_xy) / delta) * 0.25
                if float(np.hypot(*self.vel)) < 8.0: self.vel *= 0.5
            self._vel_xy, self._vel_t = xy, now
            self.world.manager.update(xy, now, self.vel)
            self.world.manager.pump_uploads(self.UPLOAD_BUDGET_MS)
            sim = getattr(self.world, "traffic_sim", None)
            if sim is not None:
                self._traffic_instances = sim.update(dt, xy)
                if now >= self._traffic_status_at:
                    self._traffic_status = sim.stats()
                    self._traffic_status_at = now + 1.
                    error = self._traffic_status.get('error')
                    if error and error != self._traffic_error_logged:
                        tm.LOG.error(f"traffic worker failed: {error}")
                        self._traffic_error_logged = error
            else:
                self._traffic_instances = None
                self._traffic_status = None

    def _sky(self):
        eye = self.camera.position if self.camera is not None else np.array([0., 0., 100.])
        lat, lon = self.proj.to_latlon(*eye[:2]) if self.proj is not None else self.sky_location
        height = float(eye[2]) + float(getattr(self.fetcher, 'terrain_datum', 0.))
        return compute_sky(self.sky_clock.hours, lat, date=self.sky_clock.date, lon_deg=lon,
                           utc_offset=self.utc_offset, altitude_m=height, light_pollution=self.light_pollution)

    def _search(self, q):
        self._say(f"Searching for '{q}'...", 8)
        tm.LOG.info(f"searching for '{q}'")
        def work():
            try: self.geo_result = ("ok", *self.fetcher.geocode_full(q))
            except Exception as exc: self.geo_result = ("err", str(exc))
        threading.Thread(target=work, daemon=True).start()

    def _key_typing(self, ev):
        pg = self.pg
        if ev.key == pg.K_ESCAPE: self.typing = False
        elif ev.key in (pg.K_RETURN, pg.K_KP_ENTER):
            self.typing = False
            if self.query.strip(): self._search(self.query.strip())
        elif ev.key == pg.K_BACKSPACE: self.query = self.query[:-1]
        elif ev.unicode and ev.unicode.isprintable() and len(self.query) < 60:
            self.query += ev.unicode

    def _click(self, pos):
        if self.show_help:
            return  # the settings popup is modal; never pick through it
        now = time.time()
        double = (now - self.last_click[0] < 0.4
                  and math.hypot(pos[0] - self.last_click[1][0],
                                 pos[1] - self.last_click[1][1]) < 8)
        self.last_click = (0.0, pos) if double else (now, pos)
        w, h = self.pg.display.get_window_size()
        lpos, lw, lh = self._lm(pos), w / self.ui_scale, h / self.ui_scale
        destinations = getattr(self, 'destinations', None)
        if self.mode == "scene" and self.show_hud and destinations is not None:
            action = destinations.hit(lpos)
            if destinations.open or action is not None:
                self.last_click = (0.0, pos)
                if action is not None: self._navigation_action(action)
                else: destinations.open = False
                return
        panel = getattr(self, 'media', None)
        if (self.mode == "scene" and self.show_hud and getattr(self, 'show_media', True)
                and panel is not None and panel.press(lpos, lw, lh)):
            return
        if (self.mode == "scene" and self.show_hud
                and getattr(self, 'minimap', None) is not None
                and self.minimap.hit(lpos)):
            self.last_click = (0.0, pos)
            place = self.minimap.place_at(lpos)
            if place is not None: self._toggle_destinations(place)
            elif self.proj is not None: self._open_map(local=True)
            else: self._toggle_destinations()
            return
        act = self.chrome.hit(lpos, lw, lh, self.mode, len(CITIES)) if self.show_hud else None
        if act is not None:
            self.last_click = (0.0, pos)
            self._ui_action(act); return
        if self.mode == "map":
            if self.typing:
                self.typing = False; return
            picked = self.map.pick(*lpos)
            if picked is None: return
            place = self.map.place_near(lpos) if self.show_hud else None
            near = self.map.city_near(*lpos, CITIES)
            if place is not None and self.proj is not None:
                lat, lon = self.proj.to_latlon(*place.position)
                label = place.name
            elif near:
                lat, lon, label = near[1], near[2], near[0]
            else:
                lat, lon = picked
                label = f"{lat:.4f}, {lon:.4f}"
            self.map.marker = (lat, lon, label)
            if double: self._travel(lat, lon, label)
        elif double and self.camera and not self.driving.active:
            w, h = self.pg.display.get_window_size()
            hit = self.camera.ground_hit(pos[0], pos[1], w, h)
            if (hit is None
                    or math.hypot(hit[0] - self.camera.target[0],
                                  hit[1] - self.camera.target[1]) > 20000):
                return
            self.camera.fly_to(hit)

    def _right_click(self, pos):
        """A stationary right-click pins a destination; right-drag still pans."""
        if self.show_help or self._picker_open() or self.typing:
            return
        lpos = self._lm(pos)
        if self.mode == "scene" and self.show_hud and self.minimap.hit(lpos):
            place = self.minimap.place_at(lpos)
            xy = place.position if place is not None else self.minimap.pick(lpos)
            if xy is not None:
                destination = Destination.from_place(place) if place is not None else Destination("Dropped pin", *xy)
                self._set_destination(destination)
        elif self.mode == "map":
            if self.show_hud and self.map.blocks(lpos): return
            picked = self.map.pick(*lpos)
            if picked is None: return
            place = self.map.place_near(lpos) if self.show_hud else None
            if place is not None and self.proj is not None:
                lat, lon = self.proj.to_latlon(*place.position)
                label = place.name
            else:
                lat, lon = picked
                label = "Dropped pin"
            self.map.marker = (lat, lon, label)
            self._navigate_marker()

    def _update_camera(self):
        """Driving consumes movement keys; otherwise feed the existing fly rig."""
        if self.menu.is_open:
            self._neutral_controls()
            return
        if self._picker_open():
            return
        pg = self.pg
        keys = pg.key.get_pressed()
        kmods = pg.key.get_mods()
        cam = self.camera
        fwd = (keys[pg.K_w] or keys[pg.K_UP]) - (keys[pg.K_s] or keys[pg.K_DOWN])
        side = (keys[pg.K_d] or keys[pg.K_RIGHT]) - (keys[pg.K_a] or keys[pg.K_LEFT])
        if self.walker.active:
            self._walk_controls = WalkControls(float(fwd), float(side), bool(kmods & pg.KMOD_SHIFT),
                                               bool(kmods & pg.KMOD_CTRL), bool(keys[pg.K_SPACE]))
            return
        driving = getattr(self, 'driving', None)
        if driving is not None and driving.active:
            driving.controls = DriveControls(float(fwd), float(-side), bool(keys[pg.K_SPACE]))
            return
        climb = keys[pg.K_e] - keys[pg.K_q]
        boost = (cam.sprint if kmods & pg.KMOD_SHIFT
                 else cam.crawl if kmods & pg.KMOD_CTRL else 1.0)
        # once flying, keep feeding the rig so the momentum dampers settle
        if cam.mode == cam.FLY or fwd or side or climb:
            cam.fly(fwd, side, climb, boost)

    def _driving_obstacles(self, *, exclude=None, excluded_parked=None):
        return vehicle_obstacles(self.world.manager.chunks.values(), self._traffic_instances,
                                 self.driving.cars, exclude, excluded_parked)

    def _toggle_driving(self):
        if self.camera is None or self.world is None:
            return
        driver, cam = self.driving, self.camera
        if driver.active:
            parked = driver.car
            driver.leave()
            if self._step_out_of(parked):
                return
            cam.release()
            self._say("Car parked - free flight (Enter to drive again)", 3.)
            return
        on_foot = self.walker.active
        if self.loading:
            self._say("Wait for the destination to load before driving", 3.)
            return
        scene = self.world.manager.driving_scene
        chunks = list(self.world.manager.chunks.values())
        w, h = self.pg.display.get_window_size()
        if on_foot:
            # on foot: only a car within arm's reach (a few strides) can be entered
            anchor = np.asarray(self.walker.position, np.float64)
            hit = pick.nearest_drive_vehicle(chunks, driver.cars, anchor, max_distance=6.5)
            aimed = True
            if hit is None:
                self._say("No car within reach - walk up to one and press Enter", 3.)
                return
        else:
            hit = pick.pick_drive_vehicle(cam, chunks, driver.cars, *self.pg.mouse.get_pos(), w, h)
            aimed = hit is not None
            anchor = cam.position if cam.mode == cam.FLY else cam.target
            if hit is None:
                hit = pick.nearest_drive_vehicle(chunks, driver.cars, anchor)
        car = None
        if hit is not None:
            candidate = hit.car or PlayerCar.from_instance(hit.row, hit.kind)
            excluded = (hit.chunk, hit.kind, hit.index) if hit.chunk is not None else None
            obstacles = self._driving_obstacles(exclude=hit.car, excluded_parked=excluded)
            if driver.place(candidate, scene, obstacles):
                car = candidate
            elif aimed:
                self._say("That car has no clear driving space - aim at another parked car", 4.)
                return
        if car is None:
            car = driver.spawn(scene, np.asarray(anchor), cam.forward(), self._driving_obstacles())
            if car is None:
                self._say("No clear road loaded nearby - move toward a street and try Enter", 4.)
                return
            hit = None
        if hit is not None and hit.chunk is not None:
            scene.claimed.add(scene.parked_token(hit.chunk, hit.kind, hit.row))
            scene.filter_parked(hit.chunk)
            self.renderer._parked_key = None
        if on_foot:
            self._end_walking()                       # the chase camera takes over from the walk camera
        driver.enter(car)
        cam.follow_vehicle(car.position, car.heading, snap=True, height=HEIGHTS[car.kind],
                           length=LENGTHS[car.kind])
        self.vel[:] = 0.; self._vel_xy = self._vel_t = None
        self.show_help = self.show_log = False
        self._say(f"Driving {VEHICLE_NAMES[car.kind]} - WASD/arrows, Space brake, Enter exit", 4.)

    # ------------------------------------------------------------------ on foot
    WALK_HINTS = (("W A S D", "Walk"), ("Shift", "Run"), ("Ctrl", "Stroll"), ("Space", "Jump"),
                  ("Enter", "Enter car"), ("Tab", "Garage"), ("Esc", "Free flight"))
    CROWD_LEVELS = ((0.0, "off"), (0.35, "quiet"), (1.0, "normal"), (1.5, "packed"))

    def _init_walking(self):
        self._set_mouse_capture(False)
        self.walker = Walker()
        self.crowd = PedestrianCrowd()
        self.crowd_level = 2
        self._walk_controls = WalkControls()
        self._frame_no = 0
        self._veh_cache = (-99, None, None, None)

    def _set_mouse_capture(self, on):
        """Hide and confine the cursor while walking so the mouse looks around."""
        on = bool(on)
        if on == getattr(self, '_mouse_captured', False):
            return
        pg = self.pg
        try:
            pg.event.set_grab(on)
            pg.mouse.set_visible(not on)
            if on: pg.mouse.get_rel()
        except pg.error:
            return
        self._mouse_captured = on

    def _wants_capture(self):
        return bool(self.walker.active and self.mode == "scene" and not self.typing and not self.menu.is_open
                    and not self.show_log and not self.loading and not self._picker_open()
                    and self.pg.key.get_focused())

    def _camera_collide(self, origin, direction, max_distance):
        scene = self.world.manager.driving_scene if self.world is not None else None
        if scene is None or not self.walker.active:
            return max_distance
        return self.walker.query.ray_distance(scene, origin, direction, max_distance)

    def _vehicle_table(self, xy, every=1):
        """Cars within 100 m of xy as (rows, kinds); cached for `every` frames."""
        frame, cached_xy, rows, kinds = self._veh_cache
        if rows is not None and self._frame_no - frame < every:
            return rows, kinds
        owner = SimpleNamespace(position=np.asarray(xy, np.float64))
        rows, kinds = vehicle_obstacles(self.world.manager.chunks.values(), self._traffic_instances,
                                        self.driving.cars, owner)
        self._veh_cache = (self._frame_no, owner.position, rows, kinds)
        return rows, kinds

    def _begin_walking(self, say=True):
        walker, cam = self.walker, self.camera
        cam.follow_walker(walker.position, walker.heading, 0., snap=True, height=walker.height)
        self.vel[:] = 0.; self._vel_xy = self._vel_t = None
        self.show_help = self.show_log = False
        self._veh_cache = (-99, None, None, None)
        if say:
            self._say("On foot - WASD walk, Shift run, Space jump, Enter car, Esc free flight", 5.)

    def _end_walking(self):
        self.walker.active = False
        self._set_mouse_capture(False)
        self.renderer.motion_car = None

    def _stop_walking(self):
        self._end_walking()
        if self.camera is not None:
            self.camera.release()
        self._say("Free flight (U to walk again)", 2.5)

    def _toggle_walking(self):
        if self.camera is None or self.world is None or self.mode != "scene":
            return
        if self.walker.active:
            self._stop_walking()
            return
        if self.driving.active:
            self._say("Leave the car first (Enter) - you step out onto the pavement", 3.)
            return
        if self.loading:
            self._say("Wait for the destination to load before walking", 3.)
            return
        cam, scene = self.camera, self.world.manager.driving_scene
        anchor = cam.position if cam.mode == cam.FLY else cam.target
        fwd = cam.forward()
        heading = math.atan2(float(fwd[1]), float(fwd[0]))
        if not self.walker.place_near(scene, float(anchor[0]), float(anchor[1]), heading, None, radius=60.):
            self._say("No walkable ground loaded here yet - move over a street and press U", 4.)
            return
        self._begin_walking()

    def _step_out_of(self, car):
        """After parking: put the player beside the car and hand over to the walk camera."""
        if self.world is None or self.camera is None:
            return False
        scene = self.world.manager.driving_scene
        owner = SimpleNamespace(position=np.asarray(car.position, np.float64))
        rows, kinds = vehicle_obstacles(self.world.manager.chunks.values(), self._traffic_instances,
                                        self.driving.cars, owner)
        if not self.walker.exit_car(scene, car, (rows, kinds)):
            return False
        self._begin_walking(say=False)
        self._say("On foot - walk up to a car and press Enter to drive again", 4.)
        return True

    def _update_walking(self, raw, active):
        walker, cam = self.walker, self.camera
        controls = self._walk_controls
        self._walk_controls = WalkControls()   # keys are re-read every frame
        if not walker.active:
            return
        if self.mode != "scene" or self.world is None or cam is None:
            return
        paused = (not active or self.typing or self.menu.is_open or self.show_log or self.loading
                  or self._picker_open())
        if not paused:
            fwd = cam.forward()
            yaw = math.atan2(float(fwd[1]), float(fwd[0]))
            rows, kinds = self._vehicle_table(walker.position[:2])
            near = np.hypot(rows[:, 12] - walker.position[0], rows[:, 13] - walker.position[1]) < 16.
            walker.update(raw, self.world.manager.driving_scene, controls, yaw, (rows[near], kinds[near]))
        cam.follow_walker(walker.position, walker.heading, walker.speed, height=walker.height)
        # The character moves with the camera: keep it out of the motion blur.
        p = walker.position
        self.renderer.motion_car = (float(p[0]), float(p[1]), float(p[2]) - .05, float(walker.heading),
                                    .55, .55, float(walker.height) + .15)

    def _update_crowd(self, dt):
        crowd, cam = self.crowd, self.camera
        if self.mode != "scene" or self.world is None or cam is None or self.loading:
            if crowd.peds: crowd.clear()
            return
        scene = self.world.manager.driving_scene
        driving, walker = self.driving, self.walker
        if walker.active:
            anchor, player = walker.position, (walker.position[:2], walker.speed)
        elif driving.active:
            anchor, player = driving.car.position, None
        else:
            anchor, player = (cam.position if cam.mode == cam.FLY else cam.target), None
        eye = cam.position
        ground = float(np.median([p.z for p in crowd.peds])) if crowd.peds else float(anchor[2]) if \
            (walker.active or driving.active) else 0.
        sky = self._frame_sky
        night = float(np.clip((2. - math.degrees(sky.alt)) / 12., 0., 1.)) if sky is not None else 0.
        crowd.density = self.CROWD_LEVELS[self.crowd_level][0]
        crowd.enabled = crowd.density > 0.
        vehicles = self._vehicle_table(anchor[:2], every=1 if walker.active else 4) if crowd.peds else None
        crowd.update(dt, scene, np.asarray(anchor[:2], np.float64), float(eye[2]) - ground,
                     player=player, vehicles=vehicles, rain=float(self.weather.rain), night=night)

    def _ped_rows(self):
        rows = self.crowd.rows if self.crowd.peds else None
        if self.walker.active:
            mine = self.walker.instance_row()[None]
            rows = mine if rows is None else np.vstack((rows, mine))
        return rows

    def _cycle_crowd(self):
        self.crowd_level = (self.crowd_level + 1) % len(self.CROWD_LEVELS)
        self._say(f"Pedestrians: {self.CROWD_LEVELS[self.crowd_level][1]}", 2.0)

    def _reset_view(self):
        driving = getattr(self, 'driving', None)
        if driving is not None and driving.active:
            if driving.reset(self.world.manager.driving_scene, self._driving_obstacles(exclude=driving.car)):
                car = driving.car
                self.camera.follow_vehicle(car.position, car.heading, snap=True, height=HEIGHTS[car.kind],
                                           length=LENGTHS[car.kind])
                self._say("Car recovered to a clear road", 2.)
            else:
                self._say("No clear road loaded for recovery yet", 3.)
        else:
            self.camera.reset()

    def _update_driving(self, dt, active=True):
        if not self.driving.active:
            self.driving.controls = DriveControls()
            self.renderer.motion_car = None
            if self.skidmarks is not None:
                self.skidmarks.release()
                self.skidmarks.clock += float(dt)      # marks keep ageing while you walk away
            return
        if self.mode != "scene" or self.world is None:
            self.driving.pause()
            return
        driver, car = self.driving, self.driving.car
        if not active or self.typing or self.menu.is_open or self.show_log or self.loading or self._picker_open():
            driver.pause()
        else:
            driver.update(dt, self.world.manager.driving_scene, self._driving_obstacles(exclude=car),
                          self.weather.wetness)
        # Chase the interpolated pose, not the fixed-step pose: whole 1/120 s steps
        # land 0-3 at a time per frame and the un-interpolated aim juddered.
        pose = driver.render_pose() or (car.body_position, car.heading)
        self.camera.fov_effect = bool(getattr(self.renderer, 'speed_fov', True))
        self.camera.follow_vehicle(pose[0], pose[1], car.speed, height=HEIGHTS[car.kind],
                                   steer=car.steering, length=LENGTHS[car.kind], slip=car.slip_angle)
        # The chassis moves with the camera: tell the motion blur so it is not smeared
        # as if it were static scenery (x, y, z, heading, half length/width, height).
        self.renderer.motion_car = (float(pose[0][0]), float(pose[0][1]), float(pose[0][2]),
                                    float(pose[1]), float(LENGTHS[car.kind]) * .5 + .15,
                                    float(WIDTHS[car.kind]) * .5 + .15, float(HEIGHTS[car.kind]) + .1)
        # Tyre marks from the slide state, laid under the interpolated wheels.
        if self.skidmarks is not None and not driver.blocked:
            self.skidmarks.update(car, pose[0], pose[1], dt)
        elif self.skidmarks is not None:
            self.skidmarks.release()

    def _focus(self):
        """F: swoop onto the building or vehicle under the cursor, or release."""
        cam = self.camera
        if cam is None or self.world is None:
            return
        if self.driving.active:
            cam.recenter_drive()
            self._say("Chase camera recentered", 1.6)
            return
        if cam.focused:
            cam.release()
            self._say("Free flight", 1.6)
            return
        w, h = self.pg.display.get_window_size()
        chunks = list(self.world.manager.chunks.values())
        hit = pick.pick_focus(cam, chunks, self.driving.frame(self._traffic_instances),
                              *self.pg.mouse.get_pos(), w, h)
        if hit is None:
            # nothing under the cursor: settle into an orbit of the aimed point
            anchor = np.asarray(cam.target, np.float64).copy()
            anchor[2] = 0.0
            cam.focus_on(anchor, radius=max(30.0, cam.distance * 0.35))
            self._say("Focused on the view centre  (F or Esc to release)", 2.6)
            return
        anchor, radius, kind = hit
        cam.focus_on(anchor, radius)
        self._say(f"Focused on a {kind}  (F or Esc to release)", 3.0)

    def _handle(self, ev):
        pg = self.pg
        if ev.type == pg.QUIT: return False
        if self.menu.is_open and ev.type not in (pg.VIDEORESIZE, getattr(pg, "WINDOWRESIZED", -1)):
            return self._menu_event(ev)

        # Track key/mouse presses
        if ev.type == pg.KEYDOWN and not getattr(ev, 'repeat', False):
            self._keys_pressed.add(ev.key)
        elif ev.type == pg.MOUSEBUTTONDOWN:
            if ev.button == 1:
                self._mouse_left_pressed = True
            elif ev.button == 3:
                self._mouse_right_pressed = True
        elif ev.type == pg.KEYUP:
            self._keys_pressed.discard(ev.key)
        elif ev.type == pg.MOUSEBUTTONUP:
            if ev.button == 1:
                self._mouse_left_pressed = False
            elif ev.button == 3:
                self._mouse_right_pressed = False

        if ev.type == pg.KEYDOWN:
            mods = pg.key.get_mods()
            if ev.key == pg.K_F11 or (ev.key in (pg.K_RETURN, pg.K_KP_ENTER) and mods & pg.KMOD_ALT) or (ev.key == pg.K_f
                                      and mods & (pg.KMOD_CTRL | getattr(pg, "KMOD_LGUI", 0))):
                self._toggle_fullscreen(); return True
            if ev.key == pg.K_F6:
                if not getattr(ev, 'repeat', False): self._toggle_destinations()
                return True
            if self._picker_open():
                action, place = self.destinations.key(ev, pg)
                if action == 'destination': self._set_destination(Destination.from_place(place))
                return True
            if self.typing:
                self._key_typing(ev); return True
            k = ev.key
            if k == pg.K_F1:
                self.open_menu("settings"); return True
            if k in (pg.K_TAB, pg.K_F4):
                if not getattr(ev, 'repeat', False): self.open_garage()
                return True
            if k == pg.K_ESCAPE:
                if self.loading: self._cancel_travel()
                elif self.show_log: self.show_log = False
                elif self.mode == "scene" and self.walker.active:
                    self._stop_walking()
                elif self.mode == "scene" and self.driving.active:
                    self._toggle_driving()
                elif self.mode == "scene" and self.camera and self.camera.focused:
                    self.camera.release()
                    self._say("Free flight", 1.6)
                elif self.mode == "map" and self.camera: self.mode = "scene"
                else: self.open_menu("main")
            elif k in (pg.K_F2, pg.K_BACKQUOTE):
                self.show_log = not self.show_log
            elif k == pg.K_F9:
                self.show_hud = not self.show_hud
                self._say(f"HUD {'hidden' if not self.show_hud else 'shown'}", 2.0)
            elif k == pg.K_F3:
                self.show_stats = not self.show_stats
                self._say(f"Diagnostics {'on' if self.show_stats else 'off'}", 1.6)
            elif k == pg.K_F8:
                r = self.renderer
                r.set_aa(0 if r.aa_samples else 4)
                self._say("Anti-aliasing " + (f"{r.aa_samples}x MSAA" if r.aa_samples else "off"), 2.2)
            elif k == pg.K_F7:
                panel = getattr(self, 'media', None)
                if panel is None or panel.session is None:
                    self._say("Now playing needs a Windows media session", 2.4)
                else:
                    self.show_media = not getattr(self, 'show_media', True)
                    self._say(f"Now playing {'shown' if self.show_media else 'hidden'}", 2.0)
            elif k == pg.K_m:
                if self.mode == "scene": self._open_map()
                elif self.camera: self.mode = "scene"
            elif k == pg.K_g and self.mode == "map":
                self._ui_action("map")
            elif k == pg.K_SLASH or ev.unicode == "/":
                self.typing, self.query = True, ""
            elif k in (pg.K_RETURN, pg.K_KP_ENTER) and self.mode == "map":
                self._travel_marker()
            elif k == pg.K_k: self.sky_clock.shift(-0.5)
            elif k == pg.K_l: self.sky_clock.shift(0.5)
            elif k == pg.K_p:
                self.sky_clock.toggle()
                self._say(f"Time {'paused' if not self.sky_clock.running else 'running'}", 2.0)
            elif k == pg.K_t:
                if mods & pg.KMOD_SHIFT:
                    self.set_live_weather(not self.weather.live)
                else:
                    mode = self.weather.cycle()
                    self._say(f"Weather: {self.weather.label}", 3.0)
            elif k == pg.K_i:
                r = self.renderer
                r.constellations_enabled = not r.constellations_enabled
                self._say(f"Constellation guides {'on' if r.constellations_enabled else 'off'}", 2.0)
            elif pg.K_1 <= k <= pg.K_9 or k == pg.K_0:
                i = 9 if k == pg.K_0 else k - pg.K_1
                self._travel(CITIES[i][1], CITIES[i][2], CITIES[i][0])
            elif self.mode == "scene":
                if k in (pg.K_RETURN, pg.K_KP_ENTER):
                    if not getattr(ev, 'repeat', False): self._toggle_driving()
                elif k == pg.K_u:
                    if not getattr(ev, 'repeat', False): self._toggle_walking()
                elif k == pg.K_j:
                    if not getattr(ev, 'repeat', False): self._cycle_crowd()
                elif self.driving.active and k in (pg.K_q, pg.K_e):
                    if not getattr(ev, 'repeat', False): self.driving.toggle_indicator(LEFT if k == pg.K_q else RIGHT)
                elif self.driving.active and k == pg.K_o:
                    if not getattr(ev, 'repeat', False): self.driving.car.lights = not self.driving.car.lights
                elif k == pg.K_f: self._focus()
                elif k == pg.K_x: self.wireframe = not self.wireframe
                elif k == pg.K_c: self.show_edges = not self.show_edges
                elif k == pg.K_v:
                    r = self.renderer
                    # Cycle Neutral -> Colorful -> Matrix -> Neutral.  The grade stays enabled:
                    # look 0 is already the neutral (ungraded) path in the shader, and turning
                    # grade_enabled off here made the next press a no-op (stuck on Neutral).
                    r.grade_enabled = True
                    r.look = (r.look + 1) % len(r.LOOK_NAMES)
                    self._say(f"Look: {r.LOOK_NAMES[r.look]}", 2.0)
                elif k == pg.K_h:
                    r = self.renderer
                    r.ae_enabled = not r.ae_enabled; r.reset_adaptation()
                    self._say(f"Eye adaptation {'on' if r.ae_enabled else 'off'}", 2.0)
                elif k == pg.K_n:
                    r = self.renderer
                    if mods & pg.KMOD_SHIFT:
                        r.flare_fstop = 14.0 if r.flare_fstop > 18.0 else 22.0
                        r.flare_enabled = True
                        self._say(f"Lens flare f/{int(r.flare_fstop)}", 2.0)
                    else:
                        r.flare_enabled = not r.flare_enabled
                        self._say(f"Lens flare {'on (f/%d)' % int(r.flare_fstop) if r.flare_enabled else 'off'}", 2.0)
                elif k == pg.K_b:
                    r = self.renderer
                    r.vignette = 0.0 if r.vignette > 0.0 else 0.42
                    self._say(f"Vignette {'on' if r.vignette > 0.0 else 'off'}", 2.0)
                elif k == pg.K_g:
                    self.renderer.ssr_enabled = not self.renderer.ssr_enabled
                    self._say(f"Reflections {'on' if self.renderer.ssr_enabled else 'off'}", 2.0)
                elif k == pg.K_r and self.camera: self._reset_view()
        elif ev.type in (pg.VIDEORESIZE, getattr(pg, "WINDOWRESIZED", -1)):
            w, h = pg.display.get_window_size()
            self.renderer.resize(w, h); self._sync_ui_size(w, h)
        elif ev.type == pg.MOUSEWHEEL:
            if self.show_help:
                w, h = pg.display.get_window_size()
                self.settings.scroll_by(ev.y, self.ui, w / self.ui_scale, h / self.ui_scale)
            elif self._picker_open():
                self.destinations.scroll_by(ev.y)
            elif (self.show_hud and self.mode == "scene"
                  and self.destinations.hit(self._lm(pg.mouse.get_pos())) is not None):
                return True
            elif self.mode == "map":
                pos = self._lm(pg.mouse.get_pos())
                if not self.map.blocks(pos):
                    self.map.zoom_by(ev.y, *pos)
            elif self.camera: self.camera.zoom(ev.y)
        elif ev.type == pg.MOUSEBUTTONDOWN:
            self.press_dist = 0.0
            w, h = pg.display.get_window_size()
            if self.show_help:
                self._settings_capture = self.map.press_ui = True
                if ev.button == 1:
                    action = self.settings.press(self.ui, self._lm(ev.pos), w / self.ui_scale,
                                                 h / self.ui_scale, self.renderer)
                    if action == "close":
                        self.show_help = False
                return True
            if getattr(self, '_mouse_captured', False):
                self.map.press_ui = False          # the hidden cursor must not "press" UI under it
                return True
            self._navigation_capture = self._picker_open()
            pos, lw, lh = self._lm(ev.pos), w / self.ui_scale, h / self.ui_scale
            self.map.press_ui = self._navigation_capture or (self.show_hud and (
                self.chrome.over_ui(pos, lw, lh, self.mode, len(CITIES))
                or (self.mode == "scene" and (self.destinations.hit(pos) is not None
                    or self.minimap.hit(pos) or (getattr(self, 'show_media', True)
                    and getattr(self, 'media', None) is not None and self.media.hit(pos, lw, lh))))))
        elif ev.type == pg.MOUSEBUTTONUP:
            if self._swallow_up:                    # the release that closed a menu
                self._swallow_up = False
                self.map.press_ui = False
                return True
            if self.show_help or getattr(self, '_settings_capture', False):
                self.settings.release()
                self._settings_capture = self.map.press_ui = False
                return True
            if getattr(self, '_navigation_capture', False) and not self._picker_open():
                self._navigation_capture = self.map.press_ui = False
                return True
            if self.mode == "map": self.map.end_drag()
            if getattr(self, '_mouse_captured', False):
                self._navigation_capture = self.map.press_ui = False
                return True
            if ev.button == 1 and self.press_dist < 5 and not self.typing:
                self._click(ev.pos)
            elif ev.button == 3 and self.press_dist < 5:
                self._right_click(ev.pos)
            self._navigation_capture = self.map.press_ui = False
        elif ev.type == pg.MOUSEMOTION:
            if self.show_help or getattr(self, '_settings_capture', False):
                if self.show_help and ev.buttons[0]:
                    w, h = pg.display.get_window_size()
                    self.settings.drag(self.ui, self._lm(ev.pos), w / self.ui_scale,
                                       h / self.ui_scale, self.renderer)
                return True
            dx, dy = ev.rel
            self.press_dist += abs(dx) + abs(dy)
            if self._picker_open() or getattr(self, '_navigation_capture', False):
                return True
            left, mid, right = ev.buttons
            if self.mode == "map":
                if (left or mid or right) and not self.map.press_ui:
                    self.map.pan(dx / self.ui_scale, dy / self.ui_scale)
            elif self.camera and (not self.map.press_ui or getattr(self, '_mouse_captured', False)):
                if self.walker.active:
                    # captured: the mouse always looks; otherwise any held button drags the view
                    if getattr(self, '_mouse_captured', False) or left or mid or right:
                        self.camera.orbit(dx, dy)
                elif self.driving.active:
                    if left or mid or right: self.camera.orbit(dx, dy)
                elif right or mid or (left and pg.key.get_mods() & pg.KMOD_SHIFT):
                    self.camera.pan(dx, dy)
                elif left: self.camera.orbit(dx, dy)
        return True

    # ------------------------------------------------------------------ HUD
    # All HUD code works in logical pixels (``ui.scale`` maps them to the framebuffer)
    # and draws through themed containers, so nothing floats raw over the scene.
    MUTED = T.TEXT_3
    GREEN = T.OK
    LEVEL_COLOR = T.LEVEL
    STEPS = (("cache", "Check local cache"), ("download", "Download map data"),
             ("mesh", "Build 3D geometry"), ("upload", "Upload to GPU"))

    SCENE_HINTS = (("W A S D", "Fly"), ("F6", "Directions"), ("Shift", "Sprint"), ("F", "Focus"), ("R", "Reset"),
                   ("Enter", "Drive"), ("Tab", "Garage"), ("U", "Walk"), ("M", "Map"), ("Esc", "Menu"))
    DRIVE_HINTS = (("W A S D", "Drive"), ("F6", "Directions"), ("Space", "Brake"), ("Q / E", "Signals"), ("O", "Lights"),
                   ("F", "Chase view"), ("R", "Recover"), ("Enter", "Exit"), ("Esc", "Menu"))

    # Only the shortcuts that matter for the current mode, shown briefly then faded out (F1 lists all).
    SCENE_HINTS_MIN = (("W A S D", "Fly"), ("Enter", "Drive"), ("U", "Walk"), ("F6", "Directions"), ("F1", "All shortcuts"))
    DRIVE_HINTS_MIN = (("W A S D", "Drive"), ("Space", "Brake"), ("Enter", "Exit"), ("F1", "All shortcuts"))
    HINT_SECONDS = 9.0

    def _hint_alpha(self, key):
        now = time.time()
        if key != self._hint_key:
            self._hint_key, self._hint_t0 = key, now
        age = now - self._hint_t0
        return max(0.0, min(1.0, (self.HINT_SECONDS - age) / 1.5))

    def _hud_scene(self, w, h):
        ui, M = self.ui, T.MARGIN
        sky = self._frame_sky or self.weather.effective_sky(self._sky())
        mouse = self._lm(self.pg.mouse.get_pos())
        cam = self.camera
        coords = ""
        if self.proj:
            lat, lon = self.proj.to_latlon(cam.target[0], cam.target[1])
            coords = f"{lat:.4f}, {lon:.4f}"
        az = cam.azimuth if cam is not None else 0.0
        # one quiet line under the clock: conditions, plus a dot when they are live
        sub = self.weather.label + ("  \u00b7  live" if self.weather.live else "")
        if not self.sky_clock.running:
            sub += "  \u00b7  paused"
        wx = weather_ui.build_info(self.weather, self.live_weather, sky.stamp)
        self.chrome.draw_scene_ui(ui, w, h, mouse, self.label or "Scene", coords, sky.stamp, sub,
                                  self.typing, self.query, self.pg.display.is_fullscreen(),
                                  (math.cos(az), math.sin(az)), weather=wx)
        # bottom-left: streaming progress only while something is loading, full diagnostics on F3
        rows = []
        st = self.world.manager.stats() if self.world else None
        busy = (st["queued"] + st["loading"]) if st else 0
        if st and (self.show_stats or busy):
            head = f"{st['resident']} chunks"
            if busy: head += f"  \u00b7  {busy} streaming"
            if st["failed"]: head += f"  \u00b7  {st['failed']} retrying"
            rows.append((head, T.TEXT, T.T_BODY, "semibold"))
            if self.show_stats:
                lit = int(round(self.renderer.lights.count / 100.0)) * 100
                rows.append((f"{st['vram_mb']:.0f} MB  \u00b7  {self.renderer.last_drawn} drawn"
                             + (f"  \u00b7  {lit} lights" if lit else "") + f"  \u00b7  {self.fps} fps",
                             T.TEXT_3, T.T_SMALL, "medium"))
                if self.world and getattr(self.world, 'traffic_sim', None) is not None:
                    rows.append((_traffic_status_text(self._traffic_status), T.TEXT_3, T.T_SMALL, "regular"))
        cw = 0
        status_h = 0.0
        if rows:
            prog = (st["resident"] / max(1, st["resident"] + busy)) if (st and busy) else None
            cw, status_h = self.chrome.status_card(ui, M, h - M, rows, progress=prog,
                                                   shimmer=time.time(), min_w=180)
        # bottom centre: a faint hint line that disappears on its own
        if self.show_hints:
            drive = self.driving.active
            walk = self.walker.active
            fade = self._hint_alpha("drive" if drive else "walk" if walk else "fly")
            if fade > 0.01:
                items = (self.DRIVE_HINTS_MIN if drive else self.WALK_HINTS if walk else self.SCENE_HINTS_MIN)
                self.chrome.hint_bar(ui, 0, h - M - 36, w, items, fade=fade, center=True)
        if self.show_minimap:
            self._hud_minimap(w, h, mouse, status_h)
        else:
            self.minimap.clear()
        if self.driving.active:
            self._hud_driving(w, h)
        anchor = self._navigation_anchor()
        if anchor is not None:
            self.destinations.draw(ui, w, h, mouse, self.navigation,
                                   self.world.manager.driving_scene, anchor[0], status_h)

    def _hud_minimap(self, w, h, mouse, status_h):
        if self.world is None:
            self.minimap.clear()
            return
        scene = self.world.manager.driving_scene
        if not scene.tiles:
            self.minimap.clear()
            return
        car = self.driving.car if self.driving.active else None
        if car is not None:
            pose = self.driving.render_pose()
            focus, heading = pose if pose is not None else (car.position, car.heading)
        else:
            cam = self.camera
            if cam is None:
                self.minimap.clear()
                return
            focus = cam.position if cam.mode == cam.FLY else cam.target
            heading = cam.azimuth + math.pi
        self.minimap.draw(self.ui, w, h, mouse, scene, focus, heading, status_h, self.navigation)

    def _hud_media(self, w, h):
        panel = getattr(self, 'media', None)
        if panel is None or not getattr(self, 'show_media', True):
            return
        panel.draw(self.ui, w, h, self._lm(self.pg.mouse.get_pos()))

    def _hud_driving(self, w, h):
        car, ui, M = self.driving.car, self.ui, T.MARGIN
        kmh = int(round(abs(car.speed) * 3.6))
        mph = self.speed_units == "mph"
        shown = int(round(kmh * 0.621371)) if mph else kmh
        state = ("Blocked \u00b7 R to recover" if self.driving.blocked else
                 "Braking" if car.braking else
                 "Left signal" if car.indicator == LEFT else
                 "Right signal" if car.indicator == RIGHT else "Enter to park / exit")
        sig = -1 if car.indicator == LEFT else 1 if car.indicator == RIGHT else 0
        x = w - M - 236
        y = h - M - 124
        self.chrome.speed_card(ui, x, y, VEHICLE_NAMES[car.kind], shown, car.gear, state, sig,
                               bool(car.braking), kmh / 180.0, unit="mph" if mph else "km/h")

    # F1 panel contents: every bound key, grouped.  F8 / F9 show live state, so the
    # rows name the action and the panel header reports the current mode.
    SHORTCUTS = (
        ("Menus", (("Esc", "menu / back / release"), ("Tab / F4", "garage: call a car"),
                   ("F1", "settings"), ("F2", "streaming log"), ("M", "world map"))),
        ("Camera", (("W A S D", "fly"), ("Q / E", "down / up"), ("Shift", "sprint"),
                    ("Ctrl", "crawl"), ("Wheel", "speed"), ("F", "focus"), ("R", "reset"),
                    ("Esc", "release focus"))),
        ("Driving", (("Enter", "enter / park / exit"), ("WASD / arrows", "accelerate / reverse / steer"),
                     ("Space", "brake"), ("Q / E", "left / right signal"), ("O", "headlights"),
                     ("F / Wheel", "chase view / distance"), ("R / Esc", "recover / exit"))),
        ("On foot", (("U", "walk / return to free flight"), ("WASD", "walk (camera-relative)"),
                     ("Shift / Ctrl", "run / stroll"), ("Space", "jump"), ("Mouse", "look around"),
                     ("Enter", "enter a nearby car"),
                     ("J", "crowd: off / quiet / normal / packed"), ("Esc", "free flight"))),
        ("Navigation", (("F6", "nearby destinations / directions"), ("M", "map / scene"),
                        ("/", "world search"), ("1 - 0", "cities"),
                        ("Right-click map", "navigate to a pin"), ("Click minimap place", "choose destination"),
                        ("Double-click map", "travel here"), ("Enter on map", "travel to marker"))),
        ("Display", (("X", "wireframe"), ("C", "edges"), ("V", "colour grade"),
                     ("B", "vignette"), ("F8", "anti-aliasing"), ("F9", "hide HUD"),
                     ("F11", "full screen"))),
        ("Media", (("F7", "now-playing card"), ("Click", "play / pause / skip"))),
        ("World", (("K / L", "time - / +"), ("P", "pause time"), ("T", "weather"),
                   ("Shift + T", "live weather"), ("I", "constellations"), ("N", "lens flare"),
                   ("H", "eye adaptation"), ("G", "reflections"))),
        ("Debug", (("F2", "streaming log"), ("Esc", "close panels"))),
    )

    def _hud_settings(self, w, h):
        self.settings.draw(self.ui, w, h, self._lm(self.pg.mouse.get_pos()),
                           self.renderer, self.show_hud)

    def _hud_loading(self, w, h):
        ui, now, M = self.ui, time.time(), T.MARGIN
        elapsed = now - self.load_start
        st = tm.BOARD.get(self.load_key) if self.load_key is not None else None
        stage = st.stage if st is not None else "queued"
        cur = max(1, tm.STAGE_INDEX.get(stage, 1))
        frac = st.fraction(now) if st is not None else 0.0
        lines = tm.LOG.tail(4)
        hint = None
        if elapsed > 12.0 and stage in ("queued", "cache", "download"):
            hint = ("Overpass is slow right now \u00b7 still trying other servers"
                    if elapsed < 40.0 else
                    "Still working. Downloads are cached, so this city loads instantly next time")
        elif elapsed > 12.0 and stage == "mesh":
            hint = "Dense city \u00b7 building the 3D geometry takes a moment"
        pad, srow = 20, 28
        cw = int(min(560, w - 2 * M)); x = (w - cw) // 2; y = M
        inner = cw - 2 * pad
        ch = (pad + 50 + 30 + len(self.STEPS) * srow + (26 if hint else 0) + 10
              + (len(lines) * 18 + 10 if lines else 0) + 40)
        ui.container(x, y, cw, ch, "popup", r=T.R_PANEL + 2)
        # header: spinner, title, coordinates, elapsed time
        ui.spinner(x + pad + 10, y + pad + 20, 9, now)
        title = ui.fit(f"Travelling to {self.load_label}", T.T_TITLE + 1, inner - 34 - 56, weight="semibold")
        ui.text(title, x + pad + 34, y + pad + 2, T.T_TITLE + 1, T.TEXT, weight="semibold")
        ui.text(f"{int(elapsed)} s", x + cw - pad, y + pad + 5, T.T_BODY, T.TEXT_3,
                anchor="right", weight="medium")
        if self.next_world is not None and self.next_world.proj is not None:
            p = self.next_world.proj
            ui.text(f"{p.lat0:.4f}, {p.lon0:.4f}", x + pad + 34, y + pad + 26, T.T_SMALL, T.TEXT_3,
                    weight="medium")
        by = y + pad + 56
        ui.bar(x + pad, by, inner - 48, 6, frac, shimmer=now if stage in ("download", "mesh") else 0.0)
        ui.text_c(f"{int(frac * 100)}%", x + cw - pad, by + 3, T.T_BASE, T.TEXT, anchor="right",
                  weight="semibold")
        sy = by + 22
        for i, (key, label) in enumerate(self.STEPS):
            ry = sy + i * srow; idx = tm.STAGE_INDEX[key]; cy = ry + srow / 2
            if cur > idx:
                ui.tick(x + pad + 10, cy, 9, T.OK); col, wt = T.TEXT_2, "regular"
            elif cur == idx:
                ui.spinner(x + pad + 10, cy, 6, now, dots=8); col, wt = T.TEXT, "semibold"
            else:
                ui.circle(x + pad + 10, cy, 3, (1, 1, 1, 0.25)); col, wt = T.TEXT_4, "regular"
            ui.text_c(label, x + pad + 34, cy, T.T_BASE, col, weight=wt)
            if cur == idx and st is not None and st.detail:
                ui.text_c(ui.fit(st.detail, T.T_SMALL, inner * 0.5), x + cw - pad, cy, T.T_SMALL,
                          T.TEXT_3, anchor="right", weight="medium")
        ly = sy + len(self.STEPS) * srow + 6
        if hint:
            ui.text_c(ui.fit(hint, T.T_SMALL, inner), x + pad, ly + 10, T.T_SMALL, T.WARN, weight="medium")
            ly += 26
        if lines:
            ui.divider(x + pad, ly, inner); ly += 10
            for t, level, msg in lines:
                ui.text(ui.fit(f"{time.strftime('%H:%M:%S', time.localtime(t))}  {msg}", T.T_SMALL, inner),
                        x + pad, ly, T.T_SMALL, self.LEVEL_COLOR.get(level, T.TEXT_3), weight="regular")
                ly += 18
        # footer: key hints
        fy = y + ch - 36
        kw = keycaps(ui, x + pad, fy, "Esc", 11)
        ui.text_c("Cancel", x + pad + kw + 8, fy + 10.5, T.T_SMALL, T.TEXT_3, weight="medium")
        k2 = x + pad + kw + 8 + ui.text_w("Cancel", T.T_SMALL, weight="medium") + 18
        kw2 = keycaps(ui, k2, fy, "F2", 11)
        ui.text_c("Full log", k2 + kw2 + 8, fy + 10.5, T.T_SMALL, T.TEXT_3, weight="medium")
        return ch

    def _hud_log(self, w, h):
        ui, M = self.ui, T.MARGIN
        n = 16 if h >= 600 else 8
        lines = tm.LOG.tail(n)
        cw = int(min(w - 2 * M, 880)); ch = 52 + 18 * n
        x, y = M, h - M - 36 - 12 - 40 - 12 - ch      # clears the toast row
        ui.container(x, y, cw, ch, "popup", r=T.R_PANEL)
        ui.text("Log", x + 16, y + 14, T.T_LABEL, T.WARN, weight="semibold")
        hx = x + cw - 16
        tw = ui.text_w("close", T.T_SMALL, weight="medium")
        ui.text_c("close", hx, y + 24, T.T_SMALL, T.TEXT_3, anchor="right", weight="medium")
        kw = keycaps(ui, hx - tw - 8 - 30, y + 14, "F2", 11)
        ui.text_c("saved to osm3d.log", hx - tw - 8 - 30 - 14, y + 24, T.T_SMALL, T.TEXT_4,
                  anchor="right", weight="regular")
        ui.divider(x + 16, y + 42, cw - 32)
        for i, (t, level, msg) in enumerate(lines):
            ui.text(ui.fit(f"{time.strftime('%H:%M:%S', time.localtime(t))}  {msg}", T.T_SMALL, cw - 32),
                    x + 16, y + 50 + i * 18, T.T_SMALL, self.LEVEL_COLOR.get(level, T.TEXT_3), weight="regular")

    def _hud_common(self, w, h, dt):
        ui = self.ui
        if not self.show_hud:
            self.fade = 0.0
            return
        if self.fade > 0:
            ui.rect(0, 0, w, h, (0.03, 0.034, 0.048, min(1.0, self.fade)))
            self.fade -= dt / 1.2
        if self.loading and self.world is not None:
            self._hud_loading(w, h)
        if self.show_fps and self.fps and not self.show_stats:
            ui.text(f"{self.fps} fps", w - T.MARGIN, h - T.MARGIN - 16, T.T_SMALL, T.TEXT_2,
                    anchor="right", weight="semibold")
        if self.show_log: self._hud_log(w, h)
        msg, until = self.message
        if msg and time.time() < until:
            tw = ui.text_w(msg, T.T_BASE, weight="semibold")
            bw = round(tw + 36)
            ty = h - T.MARGIN - 36 - 12 - 40
            ui.container(round(w / 2 - bw / 2), ty, bw, 40, "popup", r=T.R_PANEL - 2)
            ui.text_c(msg, w / 2, ty + 20, T.T_BASE, T.TEXT, anchor="center", weight="semibold")

    def _update_cursor(self, w, h):
        """Hand cursor over clickable chrome, arrow elsewhere."""
        pg = self.pg
        want = pg.SYSTEM_CURSOR_ARROW
        pos = self._lm(pg.mouse.get_pos())
        panel = getattr(self, 'media', None)
        if self.menu.is_open:
            if self.menu.cursor(pos):
                want = pg.SYSTEM_CURSOR_HAND
        elif self.show_hud and self.mode == "scene" and self._picker_open():
            action = self.destinations.hit(pos)
            if action == 'field': want = pg.SYSTEM_CURSOR_IBEAM
            elif action is not None and action != 'picker_bg': want = pg.SYSTEM_CURSOR_HAND
        elif (self.show_hud and self.mode == "scene"
              and self.destinations.hit(pos) not in (None, 'guidance_bg')):
            want = pg.SYSTEM_CURSOR_HAND
        elif (self.show_hud and self.mode == "scene" and getattr(self, 'show_media', True)
              and panel is not None and panel.hit(pos, w, h)):
            want = pg.SYSTEM_CURSOR_HAND
        elif (self.show_hud and self.mode == "scene"
              and getattr(self, 'minimap', None) is not None and self.minimap.hit(pos)):
            want = pg.SYSTEM_CURSOR_HAND
        elif self.show_hud and self.chrome.clickable(pos, w, h, self.mode, len(CITIES)):
            want = pg.SYSTEM_CURSOR_HAND
        if want != getattr(self, "_cursor", None):
            try: pg.mouse.set_cursor(want)
            except (AttributeError, TypeError, pg.error): pass
            self._cursor = want

    def _lm(self, pos):
        """Window (physical) pixels -> logical UI pixels."""
        return (pos[0] / self.ui_scale, pos[1] / self.ui_scale)

    def _sync_ui_size(self, w, h):
        S = self.ui_scale
        self.map.set_scale(S)
        self.map.set_size(w / S, h / S)

    # ------------------------------------------------------------ live weather
    def _here_latlon(self):
        """Where the camera is looking, in degrees, or ``None`` before a city loads."""
        if self.proj is None or self.camera is None:
            return None
        try:
            target = self.camera.target if self.camera.mode != self.camera.FLY else self.camera.position
            return self.proj.to_latlon(float(target[0]), float(target[1]))
        except Exception:
            return None

    def _sync_live_clock(self, state):
        """Point the sky clock at the observed place's real local time.

        Live weather means today's sky, not just today's clouds: the clock is
        held to the location's wall-clock time (shortest-way round, so a jump to
        another time zone does not spin the sun the long way round).  Turning
        live weather off leaves the clock running from wherever it ended up.
        """
        offset = (state or {}).get("utc_offset")
        hours = local_hours(offset)
        if hours is None:
            return
        # Civil local time needs the place's real offset, otherwise the sky is
        # computed for local mean solar time (up to an hour out at zone edges).
        if not self._live_offset_on:
            self._utc_offset_saved = self.utc_offset
            self._live_offset_on = True
        self.utc_offset = float(offset) / 3600.0
        delta = (hours - self.sky_clock.hours + 12.0) % 24.0 - 12.0
        if abs(delta) >= 1.0 / 3600.0:              # ignore sub-second churn
            self.sky_clock.shift(delta)

    def _live_weather_tick(self):
        """Keep the live feed pointed at the camera and apply each new observation once."""
        if not self.weather.live:
            return
        here = self._here_latlon()
        if here is None:
            return
        self.live_weather.request(*here)
        state = self.live_weather.state
        if state is None:
            return
        self._sync_live_clock(state)
        if state.get("stamp") != self._live_applied:
            self.weather.apply_live(state)
            self._live_applied = state.get("stamp")

    def set_live_weather(self, on):
        """Toggle real reported conditions for the place on screen."""
        on = bool(on)
        if not on:
            self._live_applied = None
            self.weather.clear_live()
            if self._live_offset_on:
                self.utc_offset = self._utc_offset_saved
                self._live_offset_on = False
            self._say("Manual weather", 2.4)
            return
        here = self._here_latlon()
        if here is None:
            self._say("Load a city before switching to live weather", 3.0)
            return
        self.live_weather.request(*here, force=True)
        state = self.live_weather.state
        if state is not None:
            self.weather.apply_live(state)
            self._live_applied = state.get("stamp")
            self._sync_live_clock(state)
            self._say(f"Live weather \u00b7 {self.weather.label} \u00b7 {self.sky_clock.stamp} local", 3.0)
        else:
            self.weather.live = True
            self._say("Live weather \u00b7 fetching current conditions\u2026", 3.0)

    def _render(self, dt):
        w, h = self.pg.display.get_window_size()
        lw, lh = w / self.ui_scale, h / self.ui_scale
        if self.world is None and self.loading:
            # first city (or a cancelled one): a full-screen loading screen instead of a blank view
            self.renderer.clear()
            self.ui.begin(w, h)
            self._draw_city_loading(lw, lh)
        elif self.mode == "scene":
            if self.world is not None:
                self._live_weather_tick()
                self.weather.update(dt)
                sky = self._frame_sky = self.weather.effective_sky(self._sky())
                if not self.city_shadows and sky.shadows:
                    sky = self._frame_sky = _replace(sky, shadows=False)
                self.renderer.draw(self.camera, self.world.manager.chunks.values(),
                                   self.wireframe, self.show_edges, sky,
                                   traffic=self.driving.frame(self._traffic_instances),
                                   weather=self.weather, pedestrians=self._ped_rows())
            else:
                self.renderer.clear()
            self.ui.begin(w, h)
            if self.world is not None and self.show_hud: self._hud_scene(lw, lh)
        else:
            self.renderer.clear()
            self.map.render(self.sky_clock.hours)
            self.ui.begin(w, h)
            if self.show_hud:
                anchor = self._navigation_anchor()
                self.map.draw(self.ui, self._lm(self.pg.mouse.get_pos()), self.loading,
                              self.typing, self.query, self.pg.display.is_fullscreen(),
                              self.camera is not None, navigation=self.navigation,
                              scene=self.world.manager.driving_scene if self.world is not None else None,
                              projection=self.proj, position=anchor[0] if anchor is not None else None)
        if self.menu.is_open:
            self.menu.draw(self.ui, lw, lh, self._lm(self.pg.mouse.get_pos()), dt)

        self._hud_common(lw, lh, dt)
        self._update_cursor(lw, lh)
        self.ui.end()

    def run(self):
        pg = self.pg
        clock = pg.time.Clock()
        running, frames, t_fps = True, 0, time.time()
        # Garbage collection: generation-2 sweeps over millions of mesh/numpy objects stalled
        # single frames by tens of ms.  Collect rarely and never mid-frame in bulk.
        gc.collect(); gc.freeze()
        gc.set_threshold(50_000, 20, 100)
        try:
            vsync_on = bool(pg.display.gl_get_attribute(pg.GL_SWAP_CONTROL))
        except (AttributeError, pg.error):
            vsync_on = False
        t_prev = time.perf_counter()
        dt_s = 1.0 / 60.0                      # smoothed frame time driving all animation
        while running:
            # Bound GPU power even on high-refresh displays; yield when unfocused.
            active = pg.display.get_active() and pg.key.get_focused()
            clock.tick(self.cfg.max_fps if active else 15)
            t_now = time.perf_counter()
            raw = min(max(t_now - t_prev, 1e-4), 0.25)
            t_prev = t_now
            # Smooth the step: a single slow frame (a chunk upload, a GC) used to become one
            # big camera / sky / traffic jump.  Limit how far one frame can move the average,
            # then relax toward it.  Over a second the total elapsed time is still correct.
            dt_s += (min(raw, dt_s * 2.0) - dt_s) * 0.25
            dt = dt_s
            # Spend less of the frame on mesh uploads when frames are running long.
            self.UPLOAD_BUDGET_MS = float(np.clip(4.0 - (dt_s * 1000.0 - 16.7) * 0.5, 1.0, 4.0))
            for ev in pg.event.get():
                running = self._handle(ev) and running
            self._frame_no += 1
            if self._quit: running = False
            self._set_mouse_capture(self._wants_capture())
            if active and self.mode == "scene" and self.camera and not self.typing:
                self._update_camera()
            self.sky_clock.advance(dt)
            self._poll_background()
            self._update_driving(raw, active)
            self._update_walking(raw, active)
            self._update_world(dt)
            self._update_crowd(dt)
            if self.mode == "map":
                self.map.update(dt)
            elif self.camera:
                # Chase: the car pose is interpolated with the raw frame time, so the rig
                # must integrate with the same clock; the smoothed dt made it jitter.
                self.camera.update(raw if (self.driving.active or self.walker.active) else dt)
            self._update_navigation()

            mouse_pos = self._lm(self.pg.mouse.get_pos())
            keys = self.pg.key.get_pressed()
            # pg.key.get_pressed() returns a ScancodeWrapper (not iterable in pygame-ce 2.5+)
            # Iterate over valid key codes directly
            key_dict = {}
            for k in range(512):  # SDL_SCANCODE_MAX is 512
                if keys[k]:
                    key_dict[k] = True

            # Add per-frame press flags (KEYDOWN/MOUSEBUTTONDOWN events)
            for k in self._keys_pressed:
                key_dict[k] = True
            key_dict["mouse_left_pressed"] = self._mouse_left_pressed
            key_dict["mouse_right_pressed"] = self._mouse_right_pressed

            # Reset press flags for next frame (only the "pressed this frame" ones)
            self._keys_pressed.clear()
            self._mouse_left_pressed = False
            self._mouse_right_pressed = False

            # Gamepad state
            gamepad_state = {}
            if hasattr(self, '_joystick') and self._joystick:
                j = self._joystick
                lx = j.get_axis(0)
                ly = j.get_axis(1)
                gamepad_state["lx"] = lx if abs(lx) > 0.15 else 0
                gamepad_state["ly"] = ly if abs(ly) > 0.15 else 0
                gamepad_state["a_pressed"] = j.get_button(0)
                gamepad_state["b_pressed"] = j.get_button(1)
                gamepad_state["lb_pressed"] = j.get_button(4) if j.get_numbuttons() > 4 else False
                gamepad_state["rb_pressed"] = j.get_button(5) if j.get_numbuttons() > 5 else False

            self._render(dt)
            pg.display.flip()
            frames += 1
            if time.time() - t_fps >= 1.0:
                where = self.label if self.mode == "scene" else "Map"
                pg.display.set_caption(
                    f"{APP_NAME} - {where}  |  {frames} fps  |  {self.weather.label}"
                    f"{'  |  wireframe' if self.wireframe else ''}")
                self.fps = frames
                frames, t_fps = 0, time.time()
        try:
            self.store.save()
        except Exception as exc:
            tm.LOG.warn(f"settings: could not save on exit ({exc})")
        self.navigation.close()
        for wld in (self.world, self.next_world):
            if wld is not None: wld.close()
        if getattr(self, 'media', None) is not None:
            self.media.close()
        self.renderer.spill.close()
        self.renderer.release_vegetation()
        self.renderer.release_billboards()
        self.renderer.release_shadows()
        self.renderer.release_weather()
        self.renderer.release_water_texture()
        self.renderer.release_atmosphere()
        shutdown_mesh_pool()
        shutdown_staging_cleanup()
        self.fetcher.close()
        tm.LOG.close()
        pg.quit()
