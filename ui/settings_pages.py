"""The settings schema: eight pages of live controls, each bound to a real value.

``build(app)`` returns the categories shown by the Settings screen.  Every setting
reads its current value from the object that owns it (renderer, camera, sky clock ...)
and writes back through the same object, so a keyboard shortcut such as ``V`` or ``T``
and the menu always agree.  Nothing here draws; ``menu_ui.SettingsPage`` does.
"""
from __future__ import annotations

import math

from ..core.camera import Camera
from ..core.config import DEFAULT_TRAFFIC_COUNT
from ..settings.store import Category, Section, Setting
from ..world.weather_system import WeatherSystem


# ------------------------------------------------------------------ builders
def toggle(id, label, get, set, desc="", **kw):
    return Setting(id, label, "toggle", get, set, desc, **kw)


def slider(id, label, get, set, lo, hi, step, fmt, desc="", **kw):
    return Setting(id, label, "slider", get, set, desc, lo=lo, hi=hi, step=step, fmt=fmt, **kw)


def choice(id, label, get, set, choices, desc="", **kw):
    return Setting(id, label, "choice", get, set, desc, choices=tuple(choices), **kw)


def action(id, label, fn, desc="", button="Run", **kw):
    return Setting(id, label, "action", action=fn, desc=desc, button=button, persist=False, **kw)


def info(id, label, get, desc=""):
    return Setting(id, label, "info", get=get, desc=desc, persist=False)


# ---------------------------------------------------------------- formatters
def pct(v):
    return f"{int(round(float(v) * 100))}%"


def times(v):
    return "Off" if float(v) <= 0.001 else f"{float(v):.2f}×"


def stops(v):
    return f"{float(v):+.2f} EV"


def clock_text(v):
    h = float(v) % 24.0
    hh, mm = int(h), int(round((h - int(h)) * 60.0)) % 60
    if mm == 0 and round((h - int(h)) * 60.0) >= 60:
        hh = (hh + 1) % 24
    return f"{hh:02d}:{mm:02d}"


def degrees(v):
    return f"{float(v):.0f}°"


def meters_per_s(v):
    return f"{float(v):.0f} m/s"


# time-flow presets: (key, label, hours of sky time per real second)
FLOW = (("paused", "Paused", 0.0), ("slow", "Slow", 0.02), ("normal", "Normal", 0.05),
        ("fast", "Fast", 0.25), ("rush", "Rush", 1.0))
WEATHER_SHORT = {"clear": "Clear", "overcast": "Cloudy", "rain": "Rain", "storm": "Storm", "snow": "Snow"}
TRAFFIC_LEVELS = ((1200, "Light"), (DEFAULT_TRAFFIC_COUNT, "Normal"), (6000, "Heavy"))
CROWD_NAMES = ("Off", "Quiet", "Normal", "Packed")


def build(app):
    """Create the categories for ``app`` (the host that owns all the live objects)."""
    r = lambda: app.renderer                                    # noqa: E731  (late-bound: resized/rebuilt)
    pg = app.pg

    # ---------------------------------------------------------------- helpers
    def set_attr(obj, name):
        return lambda v: setattr(obj(), name, v)

    def flow_get():
        clock = app.sky_clock
        if not clock.running:
            return "paused"
        live = [(k, s) for k, _, s in FLOW if k != "paused"]
        return min(live, key=lambda ks: abs(ks[1] - clock.speed))[0]

    def flow_set(key):
        clock = app.sky_clock
        for k, _, speed in FLOW:
            if k == key:
                clock.running = key != "paused"
                if speed > 0:
                    clock.speed = speed
                return

    def set_time(v):
        app.sky_clock.shift(float(v) - app.sky_clock.hours)

    def set_look(v):
        rr = r()
        rr.grade_enabled = True
        rr.look = int(v)

    def set_ae(v):
        rr = r()
        rr.ae_enabled = bool(v)
        rr.reset_adaptation()

    def set_aa(v):
        r().set_aa(int(v))

    def set_flare_aperture(v):
        rr = r()
        rr.flare_fstop = float(v)
        rr.flare_enabled = True

    def set_fov(v):
        Camera.BASE_FOV = float(v)

    def cam_pref(name):
        def _set(v):
            app.cam_prefs[name] = v
            app.apply_camera_prefs()
        return _set

    def set_fullscreen(v):
        if bool(v) != bool(pg.display.is_fullscreen()):
            app.toggle_fullscreen()

    def hdr():
        return bool(getattr(r(), "hdr_scene", False))

    def ae_on():
        return hdr() and bool(r().ae_enabled) and bool(getattr(r(), "ae_ok", True))

    def flare_on():
        return bool(r().flare_enabled)

    def media_ok():
        return getattr(app.media, "session", None) is not None

    # ------------------------------------------------------------- 1 graphics
    graphics = Category("graphics", "Graphics", "monitor", "Window, frame rate and render quality", [
        Section("Display", [
            toggle("fullscreen", "Full screen", lambda: bool(pg.display.is_fullscreen()), set_fullscreen,
                   "Switch between a window and full screen (F11).", persist=False),
            choice("fps_cap", "Frame-rate limit", lambda: int(app.cfg.max_fps), set_attr(lambda: app.cfg, "max_fps"),
                   ((30, "30"), (60, "60"), (90, "90"), (120, "120"), (144, "144"), (0, "Max")),
                   "Caps the render loop to save power. V-sync stays on."),
        ]),
        Section("Quality", [
            choice("aa", "Anti-aliasing", lambda: int(r().aa_samples), set_aa,
                   ((0, "Off"), (4, "4× MSAA")), "Smooths jagged edges on buildings and cars (F8)."),
            toggle("reflections", "Screen-space reflections", lambda: bool(r().ssr_enabled),
                   set_attr(r, "ssr_enabled"), "Wet roads and glass reflect the street around them (G)."),
        ]),
        Section("Shadows", [
            toggle("city_shadows", "City shadows", lambda: bool(app.city_shadows), set_attr(lambda: app, "city_shadows"),
                   "Sun shadows cast by buildings and terrain across the district."),
            toggle("near_shadows", "Close-up shadows", lambda: bool(r().near_shadows_enabled),
                   set_attr(r, "near_shadows_enabled"), "Sharp, detailed shadows around the camera.",
                   enabled=lambda: getattr(r(), "near_shadow", None) is not None),
            toggle("vehicle_shadows", "Vehicle shadows", lambda: bool(r().vehicle_shadows_enabled),
                   set_attr(r, "vehicle_shadows_enabled"), "Cars and pedestrians shade the road beneath them.",
                   enabled=lambda: getattr(r(), "vehicle_shadow", None) is not None),
        ]),
        Section("Water", [
            toggle("water_waves", "Waves", lambda: bool(r().water_waves_enabled), set_attr(r, "water_waves_enabled"),
                   "Animated swell on rivers, lakes and the sea."),
            toggle("water_ripples", "Ripples", lambda: bool(r().water_ripples_enabled),
                   set_attr(r, "water_ripples_enabled"), "Fine surface ripples that catch the light."),
            toggle("water_sss", "Subsurface glow", lambda: bool(r().water_sss_enabled), set_attr(r, "water_sss_enabled"),
                   "Light scattering through shallow water."),
        ]),
    ])

    # -------------------------------------------------------- 2 post-process
    post = Category("post", "Post-processing", "sparkle", "Colour, lens and exposure effects", [
        Section("Image", [
            choice("look", "Colour look", lambda: int(r().look), set_look,
                   tuple((i, n) for i, n in enumerate(app.renderer.LOOK_NAMES)),
                   "Colour grade applied to the final image (V)."),
            slider("bloom", "Bloom", lambda: float(r().bloom_strength), set_attr(r, "bloom_strength"),
                   0.0, 0.30, 0.01, lambda v: pct(float(v) / 0.30), "Glow around bright lights and the sun.",
                   enabled=hdr),
            slider("vignette", "Vignette", lambda: float(r().vignette), set_attr(r, "vignette"),
                   0.0, 0.80, 0.02, lambda v: "Off" if v <= 0.001 else pct(float(v) / 0.80),
                   "Darkens the corners of the frame (B)."),
        ]),
        Section("Lens", [
            toggle("flare", "Lens flare", lambda: bool(r().flare_enabled), set_attr(r, "flare_enabled"),
                   "Starbursts and ghosts around the brightest lights (N)."),
            choice("flare_f", "Aperture", lambda: 14.0 if r().flare_fstop < 18.0 else 22.0, set_flare_aperture,
                   ((14.0, "f/14"), (22.0, "f/22")), "A narrower aperture draws longer, crisper spikes.",
                   enabled=flare_on),
            slider("motion_blur", "Motion blur", lambda: float(r().motion_blur), set_attr(r, "motion_blur"),
                   0.0, 1.5, 0.05, times, "Smears fast camera and car movement. 0 turns it off."),
        ]),
        Section("Exposure", [
            toggle("ae", "Eye adaptation", lambda: bool(r().ae_enabled), set_ae,
                   "Exposure follows the scene like a real eye (H).", enabled=lambda: hdr()),
            slider("ae_brightness", "Brightness", lambda: float(r().ae_brightness), set_attr(r, "ae_brightness"),
                   -1.0, 2.0, 0.05, stops, "Raise to lift shaded streets.", enabled=ae_on),
            slider("ae_strength", "Adaptation strength", lambda: float(r().ae_strength), set_attr(r, "ae_strength"),
                   0.0, 1.0, 0.05, pct, "How strongly exposure corrects toward the target.", enabled=ae_on),
        ]),
    ])

    # ----------------------------------------------------------------- camera
    camera = Category("camera", "Camera", "camera", "Field of view and mouse control", [
        Section("View", [
            slider("fov", "Field of view", lambda: float(Camera.BASE_FOV), set_fov, 40.0, 90.0, 1.0, degrees,
                   "Vertical field of view of the main camera."),
            toggle("speed_fov", "Speed FOV effect", lambda: bool(r().speed_fov), set_attr(r, "speed_fov"),
                   "The chase camera widens as the car speeds up."),
        ]),
        Section("Controls", [
            slider("look_sens", "Mouse sensitivity", lambda: float(app.cam_prefs["sensitivity"]),
                   cam_pref("sensitivity"), 0.25, 3.0, 0.05, times, "Orbit, look and steering-view speed."),
            toggle("invert_y", "Invert vertical look", lambda: bool(app.cam_prefs["invert_y"]),
                   cam_pref("invert_y"), "Move the mouse up to look down."),
            slider("fly_speed", "Free-fly speed", lambda: float(app.cam_prefs["fly_speed"]),
                   cam_pref("fly_speed"), 10.0, 300.0, 5.0, meters_per_s,
                   "Base speed of free flight; the mouse wheel still adjusts it."),
        ]),
    ])

    # ------------------------------------------------------------------ world
    world = Category("world", "World", "clock", "Time of day, weather and sky", [
        Section("Time", [
            slider("time_of_day", "Time of day", lambda: float(app.sky_clock.hours), set_time, 0.0, 24.0, 0.25,
                   clock_text, "Local mean solar time (K / L step by half an hour).", persist=False),
            choice("time_flow", "Time flow", flow_get, flow_set, tuple((k, n) for k, n, _ in FLOW),
                   "How fast the day passes (P pauses)."),
        ]),
        Section("Atmosphere", [
            choice("weather", "Weather", lambda: app.weather.mode, lambda v: app.weather.set_mode(v),
                   tuple((m, WEATHER_SHORT[m]) for m in WeatherSystem.MODES), "Changes the sky, fog and roads (T).",
                   persist=False),
            slider("light_pollution", "Light pollution", lambda: float(app.light_pollution),
                   set_attr(lambda: app, "light_pollution"), 0.0, 1.0, 0.05, pct,
                   "Night-sky glow: 0% shows the faintest stars, 100% is a bright city."),
            toggle("constellations", "Constellation guides", lambda: bool(r().constellations_enabled),
                   set_attr(r, "constellations_enabled"), "Draw catalogue constellation lines at night (I)."),
        ]),
    ])

    # --------------------------------------------------------- traffic/people
    traffic = Category("traffic", "Traffic & People", "crowd", "Who and what shares the streets", [
        Section("Pedestrians", [
            choice("crowd", "Crowd density", lambda: int(app.crowd_level), lambda v: setattr(app, "crowd_level", int(v)),
                   tuple((i, n) for i, n in enumerate(CROWD_NAMES)), "How many people walk the pavements (J)."),
        ]),
        Section("Vehicles", [
            choice("traffic", "Traffic density", lambda: int(app.traffic_count),
                   lambda v: setattr(app, "traffic_count", int(v)), TRAFFIC_LEVELS,
                   "Number of cars driving around.", note="next travel"),
            action("clear_cars", "Spawned vehicles", lambda: app.garage_clear(),
                   "Remove every car you spawned from the Garage.", button="Clear all"),
        ]),
    ])

    # -------------------------------------------------------------- interface
    interface = Category("interface", "Interface", "layout", "What the HUD shows and how big it is", [
        Section("Heads-up display", [
            toggle("hud", "Show HUD", lambda: bool(app.show_hud), set_attr(lambda: app, "show_hud"),
                   "Hide every overlay for clean screenshots (F9).", persist=False),
            toggle("minimap", "Mini-map", lambda: bool(app.show_minimap), set_attr(lambda: app, "show_minimap"),
                   "The round street map in the lower corner."),
            toggle("hints", "Key hints", lambda: bool(app.show_hints), set_attr(lambda: app, "show_hints"),
                   "The shortcut bar along the bottom of the screen."),
            toggle("media", "Now-playing card", lambda: bool(app.show_media), set_attr(lambda: app, "show_media"),
                   "Show the system media player (F7).", enabled=media_ok),
            toggle("fps", "Frame-rate counter", lambda: bool(app.show_fps), set_attr(lambda: app, "show_fps"),
                   "A small FPS readout in the corner."),
        ]),
        Section("Units & size", [
            choice("units", "Speed units", lambda: app.speed_units, set_attr(lambda: app, "speed_units"),
                   (("kmh", "km/h"), ("mph", "mph")), "Used by the speedometer."),
            slider("ui_scale", "Interface scale", lambda: float(app.ui_scale), lambda v: app.set_ui_scale(v),
                   0.75, 2.0, 0.05, pct, "Size of text and panels."),
        ]),
        Section("Start-up", [
            toggle("start_menu", "Show main menu on launch", lambda: bool(app.start_menu),
                   set_attr(lambda: app, "start_menu"), "Open the main menu once the city has loaded."),
        ]),
    ])

    # --------------------------------------------------------------- controls
    controls = Category("controls", "Controls", "keyboard", "Every key and mouse binding", [], custom="controls")

    # --------------------------------------------------------------- advanced
    def diag(name):
        return lambda: app.diagnostics().get(name, "–")

    advanced = Category("advanced", "Advanced", "terminal", "Debug views and diagnostics", [
        Section("Debug views", [
            toggle("wireframe", "Wireframe", lambda: bool(app.wireframe), set_attr(lambda: app, "wireframe"),
                   "Draw geometry as lines (X).", persist=False),
            toggle("edges", "Building edges", lambda: bool(app.show_edges), set_attr(lambda: app, "show_edges"),
                   "Outline building silhouettes (C).", persist=False),
            toggle("log", "Streaming log", lambda: bool(app.show_log), set_attr(lambda: app, "show_log"),
                   "Show the live load and error log (F2).", persist=False),
        ]),
        Section("Diagnostics", [
            info("d_fps", "Frame rate", diag("fps")),
            info("d_chunks", "City chunks", diag("chunks")),
            info("d_vram", "Geometry memory", diag("vram")),
            info("d_traffic", "Traffic", diag("traffic")),
            info("d_gpu", "Graphics card", diag("gpu")),
            info("d_cache", "Cache folder", diag("cache")),
        ]),
        Section("Maintenance", [
            action("reset_all", "Reset all settings", lambda: app.reset_all_settings(),
                   "Restore every setting on every page to its default.", button="Reset"),
        ]),
    ])

    return [graphics, post, camera, world, traffic, interface, controls, advanced]
