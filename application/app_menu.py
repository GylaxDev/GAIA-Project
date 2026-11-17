"""Menu, settings, garage and loading-screen plumbing for :class:`App`.

``App`` inherits :class:`MenuHost`.  The menu pages are duck-typed against this API
(``menu_*``, ``garage_*``, ``diagnostics`` and the plain attributes the settings schema
reads and writes), so none of them import the application.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .. import __version__
from ..core import telemetry as tm
from ..core.config import CITIES, DEFAULT_TRAFFIC_COUNT
from ..vehicles.driving import DriveControls
from ..ui.loading_ui import LoadingScreen
from ..ui.menu_ui import MenuSystem
from ..ui.settings_pages import build as build_settings
from ..settings.store import SettingsStore
from ..vehicles.vehicle_instances import HEIGHTS, LENGTHS, VEHICLE_NAMES, WIDTHS  # noqa: F401  (re-exported for pages)
from ..peds.walker import WalkControls


class MenuHost:
    version = __version__
    garage_enter = True                       # "Drive right away" toggle in the Garage

    # ------------------------------------------------------------------ setup
    def _init_menu_state(self, cache_dir, skip_menu):
        """Plain attributes the settings schema reads; called before the schema is built."""
        self.cache_dir = Path(cache_dir)
        self.cam_prefs = {"sensitivity": 1.0, "invert_y": False, "fly_speed": 60.0}
        self.city_shadows = True
        self.show_minimap = True
        self.show_hints = True
        self.show_fps = False
        self.speed_units = "kmh"
        self.start_menu = not skip_menu
        self.traffic_count = DEFAULT_TRAFFIC_COUNT
        self.garage_cars = []
        self.fps = 0
        self._quit = False
        self._swallow_up = False
        self._title_pending = False
        self._gpu_name = None
        self.show_help = False                # legacy F1 panel flag; the menu replaced it
        self.loader = LoadingScreen()
        self._t_start = time.perf_counter()

    def _init_menu(self):
        """Build the settings store and the menu once every setting's owner exists."""
        self.store = SettingsStore(self.cache_dir / "settings.json")
        self.store.register(build_settings(self))
        self.store.capture_defaults()
        applied = self.store.apply_stored()
        if applied:
            tm.LOG.info(f"settings: restored {applied} saved values")
        self.menu = MenuSystem(self, self.store)

    # ------------------------------------------------------------ boot splash
    def _boot(self, frac, label):
        """Draw one start-up frame between blocking phases; never fatal."""
        if getattr(self, "_boot_off", False):
            return
        try:
            pg = self.pg
            pg.event.pump()
            w, h = pg.display.get_window_size()
            S = self.ui_scale
            self.ui.begin(w, h)
            self.loader.draw_boot(self.ui, w / S, h / S, frac, label, self.version,
                                  time.perf_counter() - self._t_start)
            self.ui.end()
            pg.display.flip()
        except Exception as exc:              # a cosmetic screen must never stop the app
            self._boot_off = True
            tm.LOG.warn(f"boot screen disabled ({exc})")

    # ------------------------------------------------------- settings targets
    def apply_camera_prefs(self):
        cam = getattr(self, "camera", None)
        if cam is None:
            return
        cam.sensitivity = float(self.cam_prefs["sensitivity"])
        cam.invert_y = bool(self.cam_prefs["invert_y"])
        cam.speed = float(self.cam_prefs["fly_speed"])

    def toggle_fullscreen(self):
        self._toggle_fullscreen()

    def set_ui_scale(self, v):
        v = float(min(2.0, max(0.75, v)))
        self.ui_scale = v
        self.ui.scale = v
        w, h = self.pg.display.get_window_size()
        self._sync_ui_size(w, h)

    def reset_all_settings(self):
        self.store.reset_all()
        self.store.save(force=True)
        self._say("All settings restored to their defaults", 3.0)

    # --------------------------------------------------------------- menu host
    def menu_info(self):
        sky = getattr(self, "_frame_sky", None)
        mode = ("Driving" if self.driving.active else "On foot" if self.walker.active
                else "Map" if self.mode == "map" else "Free flight" if self.world is not None else "–")
        coords = None
        if self.proj is not None and self.camera is not None:
            lat, lon = self.proj.to_latlon(self.camera.target[0], self.camera.target[1])
            coords = f"{lat:.4f}, {lon:.4f}"
        return {"world": self.world is not None, "loading": bool(self.loading),
                "location": self.label or None, "time": getattr(sky, "stamp", None),
                "weather": self.weather.label, "mode": mode, "coords": coords,
                "vehicle": VEHICLE_NAMES[self.driving.car.kind] if self.driving.active else None}

    def menu_resume(self):
        if self.camera is not None and self.world is not None:
            self.mode = "scene"

    def menu_quit(self):
        self._quit = True

    def menu_open_map(self):
        if self.camera is not None:
            self._open_map()
        else:
            self.mode = "map"

    def menu_search(self):
        self.menu_open_map()
        self.typing, self.query = True, ""

    def menu_travel(self, i):
        if not 0 <= i < len(CITIES):
            return
        if self.world is None:
            self.mode = "scene"               # no map flight: go straight to the loading screen
        self._travel(CITIES[i][1], CITIES[i][2], CITIES[i][0])

    def menu_show_log(self):
        self.show_log = True

    def open_menu(self, page="main"):
        self.menu.open(page, title=False)
        self.typing = False

    def open_garage(self):
        if self.world is None or self.camera is None:
            self._say("Load a city first, then open the Garage", 3.0)
            self.ui_sound.error()
            return
        if self.loading:
            self._say("Wait for the destination to load", 3.0)
            return
        self.menu.open("garage", title=False)
        self.typing = False

    def diagnostics(self):
        st = self.world.manager.stats() if self.world is not None else None
        if self._gpu_name is None:
            try:
                import OpenGL.GL as gl
                raw = gl.glGetString(gl.GL_RENDERER)
                self._gpu_name = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            except Exception:
                self._gpu_name = "–"
        traffic = "–"
        if self.world is not None and getattr(self.world, "traffic_sim", None) is not None:
            from .app import _traffic_status_text
            traffic = _traffic_status_text(self._traffic_status).replace("Traffic: ", "")
        return {"gpu": self._gpu_name, "fps": f"{self.fps} fps" if self.fps else "–",
                "chunks": f"{st['resident']} resident, {st['queued'] + st['loading']} streaming" if st else "–",
                "vram": f"{st['vram_mb']:.0f} MB" if st else "–", "traffic": traffic,
                "cache": str(self.cache_dir)}

    # ------------------------------------------------------------------ garage
    def garage_count(self):
        return len(self.garage_cars)

    def _leave_driving(self):
        driver = self.driving
        if driver.active:
            driver.leave()
            audio = getattr(self, "audio", None)
            if audio is not None:
                audio.leave()

    def garage_spawn(self, kind, rgb, enter=True):
        """Place a new car on a clear road near the camera.  Returns ``(ok, message)``."""
        if self.world is None or self.camera is None:
            return False, "Load a city first"
        if self.loading:
            return False, "Wait for the destination to load"
        driver, cam = self.driving, self.camera
        scene = self.world.manager.driving_scene
        if not scene.tiles:
            return False, "The streets are still loading"
        self._leave_driving()
        if self.walker.active:
            anchor = self.walker.position
        else:
            anchor = cam.position if cam.mode == cam.FLY else cam.target
        car = driver.spawn(scene, np.asarray(anchor), cam.forward(), self._driving_obstacles(),
                           int(kind), color=np.asarray(rgb, np.float32))
        if car is None:
            return False, "No clear road nearby. Move closer to a street and try again"
        self.garage_cars.append(car)
        name = VEHICLE_NAMES[car.kind]
        if enter:
            if self.walker.active:
                self._end_walking()
            driver.enter(car)
            audio = getattr(self, "audio", None)
            if audio is not None:
                audio.enter()
            cam.follow_vehicle(car.position, car.heading, snap=True, height=HEIGHTS[car.kind],
                               length=LENGTHS[car.kind])
            self.vel[:] = 0.0
            self._vel_xy = self._vel_t = None
            self.show_log = False
            self._say(f"Driving {name} - WASD/arrows, Space brake, Enter exit", 4.0)
        else:
            driver.cars.append(car)
            car.lights = False
            if not self.walker.active:
                try:
                    cam.fly_to(np.asarray(car.position, np.float64))
                except Exception:
                    pass
            self._say(f"{name} is waiting nearby - walk up and press Enter, or open the Garage", 4.0)
        self.mode = "scene"
        return True, f"{name} ready"

    def garage_clear(self):
        driver, n = self.driving, len(self.garage_cars)
        for car in list(self.garage_cars):
            if car is driver.car and driver.active:
                self._leave_driving()
                if self.camera is not None:
                    self.camera.release()
            if car in driver.cars:
                driver.cars.remove(car)
            if car is driver.car:
                driver.car = None
        self.garage_cars.clear()
        if n:
            self._say(f"Removed {n} spawned vehicle{'s' if n != 1 else ''}", 3.0)
        return n

    # ------------------------------------------------------------ input helpers
    def _menu_event(self, ev):
        pg = self.pg
        if ev.type == pg.KEYDOWN:
            mods = pg.key.get_mods()
            if ev.key == pg.K_F11 or (ev.key in (pg.K_RETURN, pg.K_KP_ENTER) and mods & pg.KMOD_ALT):
                self._toggle_fullscreen()
                return True
            if ev.key == pg.K_F2 or (ev.key == pg.K_ESCAPE and self.show_log):
                self.show_log = not self.show_log if ev.key == pg.K_F2 else False
                return True
        pos = self._lm(ev.pos) if hasattr(ev, "pos") else None
        if ev.type == pg.MOUSEBUTTONDOWN:
            self._swallow_up = True           # the matching release must not click the scene
        elif ev.type == pg.MOUSEBUTTONUP:
            self._swallow_up = False
        self.menu.handle(ev, pg, pos)
        return not self._quit

    def _neutral_controls(self):
        """Stop a held key from driving the car or walker while a menu is open."""
        if self.driving.active:
            self.driving.controls = DriveControls()
        if self.walker.active:
            self._walk_controls = WalkControls(0.0, 0.0, False, False, False)
            self._profile_held = False

    # ----------------------------------------------------------- loading screen
    def _draw_city_loading(self, w, h):
        now = time.time()
        elapsed = now - self.load_start
        st = tm.BOARD.get(self.load_key) if self.load_key is not None else None
        stage = st.stage if st is not None else "queued"
        cur = max(1, tm.STAGE_INDEX.get(stage, 1))
        frac = st.fraction(now) if st is not None else 0.0
        steps = []
        for key, label in self.STEPS:
            idx = tm.STAGE_INDEX[key]
            steps.append((label, "done" if cur > idx else "active" if cur == idx else "todo"))
        hint = None
        if elapsed > 12.0 and stage in ("queued", "cache", "download"):
            hint = ("Overpass is slow right now · still trying other servers" if elapsed < 40.0 else
                    "Still working. Downloads are cached, so this city loads instantly next time")
        elif elapsed > 12.0 and stage == "mesh":
            hint = "Dense city · building the 3D geometry takes a moment"
        lines = [(time.strftime("%H:%M:%S", time.localtime(t)), level, msg) for t, level, msg in tm.LOG.tail(5)]
        coords = ""
        if self.next_world is not None and self.next_world.proj is not None:
            p = self.next_world.proj
            coords = f"{p.lat0:.4f}, {p.lon0:.4f}"
        self.loader.draw_travel(self.ui, w, h, time.perf_counter() - self._t_start,
                                title=self.load_label, coords=coords, elapsed=elapsed, frac=frac, steps=steps,
                                detail=st.detail if st is not None and st.detail else "", hint=hint,
                                lines=lines, level_colors=self.LEVEL_COLOR, cancel=True)
