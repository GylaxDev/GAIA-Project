"""App, chunk streaming, chunk manager, and world orchestration."""
from __future__ import annotations

import math
import queue
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np

from . import telemetry as tm
from .camera import Camera
from .config import ACCENT, CITIES, SkyClock, StreamConfig
from .data_fetcher import OSMDataFetcher, TileCancelled
from .mesh_extruder import (
    ChunkResult, MeshCache, mesh_tile_job, shared_mesh_pool, shutdown_mesh_pool,
    MeshData,
)
from .overlay import Overlay2D, TileCache, WorldMap
from .projection import ChunkGrid, LocalProjection, tile_d2
from .renderer import Renderer, SkyState, compute_sky


QUEUED, LOADING, DONE, FAILED = "queued", "loading", "done", "failed"


class ChunkStreamer:
    def __init__(self, fetcher, grid, load_r2, workers=3, mesh_cache=None,
                 process_pool=True, prefetch_tiles=0.0):
        self.fetcher, self.grid = fetcher, grid
        self.cache = mesh_cache
        self.default_height = fetcher.default_height
        self.level_height = fetcher.level_height
        self.pool = shared_mesh_pool(workers, process_pool)
        self.results = queue.Queue()
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
            self._stop = True; self._cv.notify_all()

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
            try:
                for result in self._build(key):
                    self.results.put(result)
                    if self._stop and result.coarse: return
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
            with self._cv:
                self._state[key] = DONE
                self._fails.pop(key, None); self._ahead.discard(key)

    def _run_job(self, payload, key):
        try: 
            future = self.pool.submit(payload)
        except Exception:
            return mesh_tile_job(payload)
        
        while True:
            try:
                return future.result(timeout=0.25)
            except FutureTimeout:
                if self._cancelled(key):
                    future.cancel()
                    raise TileCancelled()
            except Exception as e:
                tm.LOG.error(f"mesh job failed for tile {key}: {e}")
                raise e

    def _build(self, key):
        z = self.grid.zoom; tag = f"{z}/{key[0]}/{key[1]}"
        if self.cache is not None:
            tm.BOARD.set(key, "cache", "looking for a cached 3D mesh")
            hit = self.cache.load(z, key)
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
        payload = (raw, lat0, lon0, z, key[0], key[1],
                   self.default_height, self.level_height)
        b_arrays, g_arrays, w_arrays, t_arr = self._run_job(payload, key)
        b_mesh = MeshData(*b_arrays) if b_arrays else None
        g_mesh = MeshData(*g_arrays) if g_arrays else None
        w_mesh = MeshData(*w_arrays) if w_arrays else None
        tris = sum(len(m.indices) // 3 for m in (b_mesh, g_mesh, w_mesh) if m is not None)
        tm.LOG.info(f"tile {tag}: geometry ready, {tris:,} triangles in {time.monotonic() - t0:.1f}s")
        tm.BOARD.set(key, "upload", "sending to GPU")
        yield ChunkResult(key, b_mesh, g_mesh, w_mesh, t_arr)
        if self.cache is not None:
            self.cache.store(z, key, b_mesh, g_mesh, w_mesh, t_arr)


class ChunkManager:
    MIN_PREFETCH_SPEED = 12.0

    def __init__(self, renderer, grid, streamer, cfg):
        self.renderer, self.grid, self.streamer, self.cfg = renderer, grid, streamer, cfg
        self.chunks = {}
        self.vram = 0
        self.focus = None
        self.ahead = ()
        self._last_update = 0.0
        self.uploaded = self.evicted = 0

    def update(self, target_xy, now, velocity=None):
        if self.streamer is None: return
        x, y = float(target_xy[0]), float(target_xy[1])
        focus = self.grid.key_at(x, y)
        ahead = self._prefetch_keys(x, y, focus, velocity)
        if (focus == self.focus and ahead == self.ahead
                and now - self._last_update < 0.5):
            return
        self.focus, self.ahead, self._last_update = focus, ahead, now
        self.streamer.set_focus(focus)
        for key in self.grid.keys_within(focus, self.cfg.load_r2):
            chunk = self.chunks.get(key)
            if chunk is not None: chunk.last_used = now
            else: self.streamer.request(key)
        for key in ahead:
            chunk = self.chunks.get(key)
            if chunk is not None: chunk.last_used = now
            else: self.streamer.request(key, ahead=True)
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
        far = sorted((c for c in self.chunks.values()
                      if tile_d2(c.key, focus) > self.cfg.retain_r2),
                     key=lambda c: c.last_used)
        for c in far: self._drop(c)
        budget = int(self.cfg.vram_mb * 1024 * 1024)
        if self.vram > budget:
            lru = sorted((c for c in self.chunks.values()
                          if tile_d2(c.key, focus) > self.cfg.load_r2),
                         key=lambda c: c.last_used)
            for c in lru:
                if self.vram <= budget: break
                self._drop(c)

    def _drop(self, c):
        self.renderer.free_chunk(c)
        del self.chunks[c.key]
        self.vram -= c.nbytes
        self.evicted += 1
        self.streamer.forget(c.key)
        tm.BOARD.drop(c.key)

    def pump_uploads(self, max_ms=4.0):
        if self.streamer is None: return
        t0 = time.perf_counter()
        while True:
            try: res = self.streamer.results.get_nowait()
            except queue.Empty: return
            if (self.focus is not None
                    and tile_d2(res.key, self.focus) > self.cfg.retain_r2):
                self.streamer.forget(res.key); tm.BOARD.drop(res.key); continue
            if (res.b_mesh or res.g_mesh or res.w_mesh
                    or (res.trees is not None and len(res.trees) > 0)):
                old = self.chunks.pop(res.key, None)
                if old is not None:
                    self.renderer.free_chunk(old); self.vram -= old.nbytes
                c = self.renderer.upload_chunk(
                    res.key, res.b_mesh, res.g_mesh, res.w_mesh, res.trees,
                    self.grid.origin(res.key),
                    scale=(self.grid.x_scale(res.key), 1.0))
                self.chunks[res.key] = c
                self.vram += c.nbytes; self.uploaded += 1
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
        if self.streamer is not None: self.streamer.shutdown()
        for c in list(self.chunks.values()): self.renderer.free_chunk(c)
        self.chunks.clear(); self.vram = 0


class World:
    def __init__(self, renderer, label, cfg, proj=None, fetcher=None, radius=350.0,
                 static_mesh=None, mesh_cache=None):
        self.proj, self.label = proj, label
        if static_mesh is not None:
            self.grid = self.streamer = self.center_key = None
            self.manager = ChunkManager(renderer, None, None, cfg)
            self.stream_radius = max(static_mesh.radius * 1.2, 500.0)
            self.home_distance = static_mesh.radius * 1.5
            return
        self.grid = ChunkGrid(proj, cfg.zoom)
        self.streamer = ChunkStreamer(fetcher, self.grid, cfg.load_r2, cfg.workers,
                                      mesh_cache, cfg.process_pool,
                                      cfg.prefetch_tiles if cfg.prefetch else 0.0)
        self.manager = ChunkManager(renderer, self.grid, self.streamer, cfg)
        self.center_key = self.grid.key_at(0.0, 0.0)
        tw, th = self.grid.size_m(self.center_key)
        self.stream_radius = (cfg.load_radius + 0.5) * max(tw, th)
        self.home_distance = radius * 1.5
        self.cfg = cfg

    def prime(self):
        if self.streamer is None: return
        # Only the centre tile gates arrival. Requesting its 8 neighbours here made them compete
        # for the same few request slots and slowed the one tile we are actually waiting on;
        # the ChunkManager requests the surrounding ring right after arrival.
        self.streamer.set_focus(self.center_key)
        self.streamer.request(self.center_key)

    def close(self): self.manager.release_all()


class App:
    UPLOAD_BUDGET_MS = 4.0

    def __init__(self, fetcher, radius, cfg, demo_mesh=None, start=None, size=(1280, 800),
                 tile_url=None, tile_invert=True, cache_dir=Path(".osm_cache"),
                 start_map=False, start_hours=13.0, day_speed=0.05,
                 shadow_size=2048, shadows=True):
        import pygame
        self.pg = pygame
        pygame.init()
        import sys
        sys.setswitchinterval(0.002)
        tm.LOG.attach_file(Path(cache_dir) / "osm3d.log")
        tm.LOG.info("starting OSM 3D")
        self._create_window(*size)
        pygame.display.set_caption("OSM 3D")
        self.fetcher, self.radius, self.cfg = fetcher, radius, cfg
        w, h = pygame.display.get_window_size()
        self.renderer = Renderer(w, h, shadow_size, shadows)
        self.sky_clock = SkyClock(start_hours, day_speed)
        self.ui = Overlay2D()
        self.mesh_cache = MeshCache(cache_dir, cfg.mesh_cache)
        if tile_url is None:
            from .config import DEFAULT_TILE_URL
            tile_url = DEFAULT_TILE_URL
        self.tiles = TileCache(tile_url, Path(cache_dir))
        self.map = WorldMap(self.tiles, tile_invert)
        self.map.set_size(w, h)
        self.camera, self.proj, self.label = None, None, ""
        self.world, self.next_world = None, None
        self.wireframe, self.show_edges = False, True
        self.mode = "map"
        self.loading, self.load_label, self.load_start = False, "", 0.0
        self.load_key, self.load_attempts = None, 0
        self.show_log = False
        self.geo_result = None
        self.message = ("", 0.0)
        self.typing, self.query = False, ""
        self.fade = 0.0
        self.press_dist = 0.0
        self.last_click = (0.0, (0, 0))
        self.vel = np.zeros(2, np.float64)
        self._vel_xy = self._vel_t = None
        self.tiles.request((0, 0, 0))
        if demo_mesh is not None:
            self._activate(World(self.renderer, "Demo city", cfg, static_mesh=demo_mesh,
                                 mesh_cache=self.mesh_cache), arrival=False)
            self.mode = "map" if start_map else "scene"
        elif start is not None and not start_map:
            self.mode = "scene"
            self._travel(*start)

    def _create_window(self, w, h):
        pg = self.pg
        flags = pg.OPENGL | pg.DOUBLEBUF | pg.RESIZABLE
        for samples in (4, 0):
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
            pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
            pg.display.gl_set_attribute(pg.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
            # Explicitly request a 24-bit depth buffer
            pg.display.gl_set_attribute(pg.GL_DEPTH_SIZE, 24)
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 1 if samples else 0)
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, samples)
            try:
                pg.display.set_mode((w, h), flags)
                return
            except pg.error:
                pg.display.quit()
                pg.display.init()
        raise RuntimeError("Could not create an OpenGL 3.3 core context")

    def _activate(self, world, arrival):
        old, self.world = self.world, world
        if old is not None: old.close()
        self.renderer.stream_radius = world.stream_radius
        self.camera = Camera(world.stream_radius, world.home_distance)
        self.vel[:] = 0.0; self._vel_xy = self._vel_t = None
        self.proj, self.label = world.proj, world.label
        world.manager.pump_uploads(max_ms=60.0)
        if arrival:
            self.camera.begin_arrival(); self.fade = 1.0
        if world.proj:
            self.map.marker = (world.proj.lat0, world.proj.lon0, world.label)
        self.mode = "scene"

    def _open_map(self):
        if self.mode == "map": return
        self.mode = "map"
        if self.proj:
            lat, lon = self.proj.to_latlon(self.camera.target[0], self.camera.target[1])
            self.map.jump_to(lat, lon, 13.0); self.map.fly_to(lat, lon, 4.0)

    def _say(self, msg, secs=5.0):
        self.message = (msg, time.time() + secs)

    def _travel(self, lat, lon, label):
        if self.loading:
            self._say("Still travelling - press Esc to cancel"); return
        self.map.marker = (lat, lon, label)
        if self.mode == "map": self.map.fly_to(lat, lon, 15.0)
        self.loading, self.load_label, self.load_start = True, label, time.time()
        self.load_attempts = 0
        tm.LOG.info(f"travel to {label} ({lat:.4f}, {lon:.4f})")
        self.next_world = World(self.renderer, label, self.cfg,
                                LocalProjection(lat, lon), self.fetcher, self.radius,
                                mesh_cache=self.mesh_cache)
        self.load_key = self.next_world.center_key
        self.next_world.prime()

    def _cancel_travel(self, reason="cancelled"):
        nw = self.next_world
        if nw is not None: nw.close()
        self.next_world, self.loading, self.load_key = None, False, None
        tm.LOG.warn(f"travel to {self.load_label} {reason}")
        if self.world is None and self.mode == "scene": self.mode = "map"
        self._say(f"Travel {reason}", 3.0)

    def _travel_marker(self):
        if self.map.marker is None:
            self._say("Click the map to pick a destination first")
        else:
            self._travel(*self.map.marker)

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
                self.map.fly_to(lat, lon, 11.0)
                self._say("Found it - press Enter (or double-click) to travel there", 7)

    def _update_world(self):
        if self.mode == "scene" and self.world is not None and self.camera is not None:
            now = time.time()
            xy = np.asarray(self.camera.target[:2], np.float64)
            if self._vel_xy is not None:
                dt = max(now - self._vel_t, 1e-3)
                self.vel = self.vel * 0.75 + ((xy - self._vel_xy) / dt) * 0.25
                if float(np.hypot(*self.vel)) < 8.0: self.vel *= 0.5
            self._vel_xy, self._vel_t = xy, now
            self.world.manager.update(xy, now, self.vel)
            self.world.manager.pump_uploads(self.UPLOAD_BUDGET_MS)

    def _sky(self):
        lat = self.proj.lat0 if self.proj is not None else 40.758
        return compute_sky(self.sky_clock.hours, lat)

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
        now = time.time()
        double = (now - self.last_click[0] < 0.4
                  and math.hypot(pos[0] - self.last_click[1][0],
                                 pos[1] - self.last_click[1][1]) < 8)
        self.last_click = (0.0, pos) if double else (now, pos)
        if self.mode == "map":
            row = self.map.city_row_at(*pos, len(CITIES))
            if row is not None:
                self._travel(CITIES[row][1], CITIES[row][2], CITIES[row][0]); return
            near = self.map.city_near(*pos, CITIES)
            if near:
                lat, lon, label = near[1], near[2], near[0]
            else:
                lat, lon = self.map.screen_to_latlon(*pos)
                label = f"{lat:.4f}, {lon:.4f}"
            self.map.marker = (lat, lon, label)
            if double: self._travel(lat, lon, label)
        elif double and self.camera:
            w, h = self.pg.display.get_window_size()
            hit = self.camera.ground_hit(pos[0], pos[1], w, h)
            if (hit is None
                    or math.hypot(hit[0] - self.camera.target[0],
                                  hit[1] - self.camera.target[1]) > 20000):
                return
            self.camera.fly_to(hit)

    def _handle(self, ev):
        pg = self.pg
        if ev.type == pg.QUIT: return False
        if ev.type == pg.KEYDOWN:
            if self.typing:
                self._key_typing(ev); return True
            k = ev.key
            if k == pg.K_ESCAPE:
                if self.loading: self._cancel_travel()
                elif self.show_log: self.show_log = False
                elif self.mode == "map" and self.camera: self.mode = "scene"
                else: return False
            elif k in (pg.K_F2, pg.K_BACKQUOTE):
                self.show_log = not self.show_log
            elif k == pg.K_m:
                if self.mode == "scene": self._open_map()
                elif self.camera: self.mode = "scene"
            elif k == pg.K_SLASH or ev.unicode == "/":
                self.typing, self.query = True, ""
            elif k in (pg.K_RETURN, pg.K_KP_ENTER) and self.mode == "map":
                self._travel_marker()
            elif k == pg.K_k: self.sky_clock.shift(-0.5)
            elif k == pg.K_l: self.sky_clock.shift(0.5)
            elif k == pg.K_p:
                self.sky_clock.toggle()
                self._say(f"Time {'paused' if not self.sky_clock.running else 'running'}", 2.0)
            elif pg.K_1 <= k <= pg.K_9 or k == pg.K_0:
                i = 9 if k == pg.K_0 else k - pg.K_1
                self._travel(CITIES[i][1], CITIES[i][2], CITIES[i][0])
            elif self.mode == "scene":
                if k == pg.K_f: self.wireframe = not self.wireframe
                elif k == pg.K_e: self.show_edges = not self.show_edges
                elif k == pg.K_r and self.camera: self.camera.reset()
        elif ev.type in (pg.VIDEORESIZE, getattr(pg, "WINDOWRESIZED", -1)):
            w, h = pg.display.get_window_size()
            self.renderer.resize(w, h); self.map.set_size(w, h)
        elif ev.type == pg.MOUSEWHEEL:
            if self.mode == "map": self.map.zoom_by(ev.y, *pg.mouse.get_pos())
            elif self.camera: self.camera.zoom(ev.y)
        elif ev.type == pg.MOUSEBUTTONDOWN:
            self.press_dist = 0.0
        elif ev.type == pg.MOUSEBUTTONUP:
            if ev.button == 1 and self.press_dist < 5 and not self.typing:
                self._click(ev.pos)
        elif ev.type == pg.MOUSEMOTION:
            dx, dy = ev.rel
            self.press_dist += abs(dx) + abs(dy)
            left, mid, right = ev.buttons
            if self.mode == "map":
                if left or mid or right: self.map.pan(dx, dy)
            elif self.camera:
                if right or mid or (left and pg.key.get_mods() & pg.KMOD_SHIFT):
                    self.camera.pan(dx, dy)
                elif left: self.camera.orbit(dx, dy)
        return True

    def _hud_scene(self, w, h):
        ui = self.ui; sky = self._sky()
        ui.text(self.label or "Scene", 16, 14, 30, (1, 1, 1, 0.95))
        ui.text(sky.stamp, w - 16, 10, 30, (1, 1, 1, 0.92), anchor="right")
        body = f"sun {math.degrees(sky.alt):+.0f}\u00b0" if sky.is_day else "night"
        body += "  shadows" if sky.shadows else "  no shadows"
        if sky.moon_intensity > 0.05: body += f"  moon {int(sky.moon_phase * 100)}%"
        if not self.sky_clock.running: body += "  paused"
        ui.text(body, w - 16, 44, 16, (0.60, 0.70, 0.84, 0.85), anchor="right")
        if self.proj:
            lat, lon = self.proj.to_latlon(self.camera.target[0], self.camera.target[1])
            ui.text(f"{lat:.5f}, {lon:.5f}", 16, 44, 19, (0.6, 0.72, 0.85, 0.9))
        st = self.world.manager.stats() if self.world else None
        if st:
            busy = st["queued"] + st["loading"]
            line = (f"chunks {st['resident']} resident | {busy} streaming"
                    f"{' | ' + str(st['ahead']) + ' ahead' if st.get('ahead') else ''}"
                    f"{' | ' + str(st['failed']) + ' retrying' if st['failed'] else ''}"
                    f" | {st['vram_mb']:.0f} MB | {self.renderer.last_drawn} drawn")
            ui.text(line, 16, 70, 16, (0.5, 0.65, 0.8, 0.85))
            if busy:
                ui.bar(16, 92, 240, 4, st["resident"] / max(1, st["resident"] + busy),
                       shimmer=time.time())
        ui.text("M: map   /: search   1-0: cities   Double-click: fly   "
                "F: wire   E: edges   R: reset   K/L: time   P: pause   F2: log",
                16, h - 24, 16, (0.62, 0.7, 0.8, 0.85))

    MUTED = (0.62, 0.72, 0.86, 0.92)
    GREEN = (0.45, 0.92, 0.60, 1.0)
    LEVEL_COLOR = {"info": (0.74, 0.82, 0.92, 0.95), "ok": (0.55, 0.92, 0.68, 1.0),
                   "warn": (1.0, 0.80, 0.45, 1.0), "error": (1.0, 0.52, 0.52, 1.0)}
    STEPS = (("cache", "Check local cache"), ("download", "Download map data"),
             ("mesh", "Build 3D geometry"), ("upload", "Upload to GPU"))

    def _hud_loading(self, w, h):
        """Travel card: what is happening, how far along, and the latest log lines."""
        ui, now = self.ui, time.time()
        elapsed = now - self.load_start
        st = tm.BOARD.get(self.load_key) if self.load_key is not None else None
        stage = st.stage if st is not None else "queued"
        cur = max(1, tm.STAGE_INDEX.get(stage, 1))          # "queued" shows as step 1
        frac = st.fraction(now) if st is not None else 0.0
        lines = tm.LOG.tail(4)
        hint = None
        if elapsed > 12.0 and stage in ("queued", "cache", "download"):
            hint = ("Overpass is slow right now - still trying other servers"
                    if elapsed < 40.0 else
                    "Still working. Downloads are cached, so this city loads instantly next time")
        elif elapsed > 12.0 and stage == "mesh":
            hint = "Dense city - building the 3D geometry takes a moment"
        cw = int(min(560, w - 40)); x = (w - cw) // 2; y = 14
        ch = 98 + len(self.STEPS) * 26 + (26 if hint else 0) + 14 + 17 * len(lines) + 30
        ui.panel(x, y, cw, ch)
        ui.spinner(x + 30, y + 34, 11, now)
        ui.text(f"Travelling to {self.load_label[:26]}", x + 56, y + 14, 28, (1, 1, 1, 1))
        ui.text(f"{int(elapsed)} s", x + cw - 16, y + 18, 22, self.MUTED, anchor="right")
        if self.next_world is not None and self.next_world.proj is not None:
            p = self.next_world.proj
            ui.text(f"{p.lat0:.4f}, {p.lon0:.4f}", x + 56, y + 42, 17, self.MUTED)
        by = y + 68
        ui.bar(x + 16, by, cw - 32 - 54, 8, frac,
               shimmer=now if stage in ("download", "mesh") else 0.0)
        ui.text(f"{int(frac * 100)}%", x + cw - 16, by - 6, 20, ACCENT, anchor="right")
        sy = by + 22
        for i, (key, label) in enumerate(self.STEPS):
            ry = sy + i * 26; idx = tm.STAGE_INDEX[key]
            if cur > idx:
                ui.tick(x + 30, ry + 9, 9, self.GREEN); col = (0.66, 0.90, 0.74, 1)
            elif cur == idx:
                ui.spinner(x + 30, ry + 9, 6, now, dots=8); col = (1, 1, 1, 1)
            else:
                ui.circle(x + 30, ry + 9, 3, (1, 1, 1, 0.25)); col = (0.55, 0.62, 0.72, 1)
            ui.text(label, x + 52, ry + 2, 21, col)
            if cur == idx and st is not None and st.detail:
                ui.text(st.detail[:48], x + cw - 16, ry + 4, 16, self.MUTED, anchor="right")
        ly = sy + len(self.STEPS) * 26 + 4
        if hint:
            ui.text(hint, x + 16, ly + 2, 18, self.LEVEL_COLOR["warn"]); ly += 26
        ui.rect(x + 16, ly, cw - 32, 1, (1, 1, 1, 0.10)); ly += 8
        for t, level, msg in lines:
            ui.text(f"{time.strftime('%H:%M:%S', time.localtime(t))}  {msg}"[:78],
                    x + 16, ly, 15, self.LEVEL_COLOR.get(level, self.MUTED), shadow=False)
            ly += 17
        ui.text("Esc  cancel        F2  full log", x + cw / 2, y + ch - 22, 15,
                (0.55, 0.65, 0.78, 0.9), anchor="center", shadow=False)
        return ch

    def _hud_log(self, w, h):
        ui = self.ui
        n = 16 if h >= 600 else 8
        lines = tm.LOG.tail(n)
        cw = int(min(w - 24, 860)); ch = 40 + 18 * n
        x, y = 12, h - 38 - ch
        ui.panel(x, y, cw, ch, accent=(1.0, 0.80, 0.45, 1.0), bg=(0.02, 0.025, 0.04, 0.93))
        ui.text("LOG", x + 14, y + 11, 20, (1.0, 0.80, 0.45, 1.0))
        ui.text("F2 or Esc to close   -   also saved to osm3d.log", x + cw - 14, y + 14, 15,
                (0.55, 0.65, 0.78, 0.9), anchor="right", shadow=False)
        for i, (t, level, msg) in enumerate(lines):
            ui.text(f"{time.strftime('%H:%M:%S', time.localtime(t))}  {msg}"[:int(cw / 6.6)],
                    x + 14, y + 36 + i * 18, 15, self.LEVEL_COLOR.get(level, self.MUTED),
                    shadow=False)

    def _hud_common(self, w, h, dt):
        ui = self.ui
        if self.fade > 0:
            ui.rect(0, 0, w, h, (0.03, 0.034, 0.048, min(1.0, self.fade)))
            self.fade -= dt / 1.2
        y = 14
        if self.loading:
            y += self._hud_loading(w, h) + 10
        if self.typing:
            ui.panel(w / 2 - 250, y, 500, 44)
            cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
            ui.text("Search a place: " + self.query + cursor,
                    w / 2 - 236, y + 12, 24, (1, 1, 1, 1))
        if self.show_log: self._hud_log(w, h)
        msg, until = self.message
        if msg and time.time() < until:
            ui.text(msg, w / 2, h - 70, 22, (1, 0.93, 0.7, 1), anchor="center")

    def _render(self, dt):
        w, h = self.pg.display.get_window_size()
        if self.mode == "scene":
            if self.world is not None:
                self.renderer.draw(self.camera, self.world.manager.chunks.values(),
                                   self.wireframe, self.show_edges, self._sky())
            else:
                self.renderer.clear()
            self.ui.begin(w, h)
            if self.world is not None: self._hud_scene(w, h)
        else:
            self.renderer.clear()
            self.ui.begin(w, h)
            self.map.draw(self.ui, self.pg.mouse.get_pos(), CITIES, self.loading)
            if self.map.marker and not self.loading and not self.typing:
                cw = int(min(600, w - 40))
                self.ui.panel(w / 2 - cw / 2, h - 118, cw, 38)
                self.ui.text(f"{self.map.marker[2][:34]}   -   Enter or double-click to travel",
                             w / 2, h - 108, 21, (1, 1, 1, 1), anchor="center")
        self._hud_common(w, h, dt)
        self.ui.end()

    def run(self):
        pg = self.pg
        clock = pg.time.Clock()
        running, frames, t_fps = True, 0, time.time()
        while running:
            dt = clock.tick(120) / 1000.0
            for ev in pg.event.get():
                running = self._handle(ev) and running
            if self.mode == "scene" and self.camera and not self.typing:
                keys = pg.key.get_pressed()
                kx = (keys[pg.K_d] or keys[pg.K_RIGHT]) - (keys[pg.K_a] or keys[pg.K_LEFT])
                ky = (keys[pg.K_s] or keys[pg.K_DOWN]) - (keys[pg.K_w] or keys[pg.K_UP])
                if kx or ky:
                    self.camera.pan(-kx * 400 * dt, -ky * 400 * dt)
            self.sky_clock.advance(dt)
            self._poll_background()
            self._update_world()
            if self.mode == "map":
                self.map.update(dt); self.tiles.pump()
            elif self.camera:
                self.camera.update(dt)
            self._render(dt)
            pg.display.flip()
            frames += 1
            if time.time() - t_fps >= 1.0:
                where = self.label if self.mode == "scene" else "World map"
                pg.display.set_caption(
                    f"OSM 3D - {where}  |  {frames} fps"
                    f"{'  |  wireframe' if self.wireframe else ''}")
                frames, t_fps = 0, time.time()
        for wld in (self.world, self.next_world):
            if wld is not None: wld.close()
        shutdown_mesh_pool()
        self.fetcher.close()
        tm.LOG.close()
        pg.quit()