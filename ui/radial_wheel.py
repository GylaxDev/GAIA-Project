"""Watch Dogs styled radial wheel UI component.

A contextual pie-menu with:
- Central hub (primary action / close)
- 4-8 radial segments for abilities/options
- Smooth open/close animation with stagger
- Keyboard/gamepad navigation (WASD/arrows or left stick)
- Watch Dogs aesthetic: cyan accents, geometric brackets, scanlines
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import theme as T
from . import icons

# Optional pygame for key constants (only needed for key mapping)
try:
    import pygame as _pygame
except ImportError:
    _pygame = None
from .ui_kit import ease_out, lerp, mix


@dataclass
class RadialAction:
    """A single wedge action in the radial wheel."""
    id: str
    label: str
    icon: str                    # icon name from icons.py
    color: tuple = T.SIGNAL      # accent color for this wedge
    enabled: bool = True
    callback: Optional[Callable] = None
    # Optional: sub-wheel for nested menus
    sub_actions: list = field(default_factory=list)


@dataclass
class RadialWheelState:
    """Runtime state for the radial wheel."""
    open: bool = False
    anim: float = 0.0            # 0..1 open progress
    hover_index: int = -1        # currently highlighted wedge (-1 = center hub)
    center_hover: float = 0.0    # center hub hover animation
    stagger_delay: float = 0.04  # per-wedge entrance delay
    last_input_time: float = 0.0
    # Navigation
    using_gamepad: bool = False
    nav_angle: float = 0.0       # current navigation angle for stick input


class RadialWheel:
    """Watch Dogs styled radial ability wheel."""

    # Geometry constants
    HUB_RADIUS = 56.0
    INNER_RADIUS = 72.0
    OUTER_RADIUS = 160.0
    MIN_WEDGES = 4
    MAX_WEDGES = 8
    OPEN_TIME = 0.18
    CLOSE_TIME = 0.12

    def __init__(self, actions: list[RadialAction], center_label: str = "EXIT", center_icon: str = "close"):
        """Create a radial wheel with the given actions.

        Args:
            actions: List of RadialAction (4-8 items)
            center_label: Label for center hub
            center_icon: Icon for center hub
        """
        self.actions = actions[:self.MAX_WEDGES]
        self.center_label = center_label
        self.center_icon = center_icon
        self.state = RadialWheelState()
        self._callback_result = None
        self._pending_action: Optional[RadialAction] = None

        # Validate
        if len(self.actions) < self.MIN_WEDGES:
            raise ValueError(f"Radial wheel needs at least {self.MIN_WEDGES} actions, got {len(self.actions)}")

    def open(self, mouse_pos: tuple = None):
        """Open the wheel at the current mouse position or screen center."""
        self.state.open = True
        self.state.anim = 0.0
        self.state.hover_index = -1
        self.state.center_hover = 0.0
        self.state.last_input_time = time.time()
        self._callback_result = None
        self._pending_action = None

    def close(self):
        """Close the wheel."""
        self.state.open = False
        self.state.anim = 1.0  # will animate to 0
        self._callback_result = None

    def toggle(self, mouse_pos: tuple = None):
        if self.state.open or self.state.anim > 0:
            self.close()
        else:
            self.open(mouse_pos)

    def update(self, dt: float, mouse_pos: tuple, keys: dict, gamepad: dict = None):
        """Update animation and input state. Returns action callback result if any."""
        prev_anim = self.state.anim

        # Animate open/close
        if self.state.open:
            self.state.anim = min(1.0, self.state.anim + dt / self.OPEN_TIME)
        else:
            self.state.anim = max(0.0, self.state.anim - dt / self.CLOSE_TIME)

        # If fully closed, stop processing
        if self.state.anim <= 0.0 and not self.state.open:
            self.state.hover_index = -1
            self.state.center_hover = 0.0
            return self._callback_result

        # Process input
        self._process_input(mouse_pos, keys, gamepad, dt)

        # Execute pending action on release
        if self._pending_action and self._pending_action.callback:
            self._callback_result = self._pending_action.callback()
            self._pending_action = None
            self.close()
            return self._callback_result

        return None

    def _process_input(self, mouse_pos: tuple, keys: dict, gamepad: dict, dt: float):
        """Handle mouse, keyboard, and gamepad navigation.

        `keys` can be either:
        - dict with string names (from demo): {"w": True, "up": True, "mouse_left_pressed": True}
        - dict with pygame key constants (from app): {pygame.K_w: True, pygame.K_UP: True}
        """
        import pygame
        cx, cy = self._get_center()
        mx, my = mouse_pos

        # Normalize keys to string names for internal use
        if _pygame:
            key_map = {
                _pygame.K_w: "w", _pygame.K_UP: "up",
                _pygame.K_s: "s", _pygame.K_DOWN: "down",
                _pygame.K_a: "a", _pygame.K_LEFT: "left",
                _pygame.K_d: "d", _pygame.K_RIGHT: "right",
                _pygame.K_RETURN: "enter_pressed", _pygame.K_KP_ENTER: "enter_pressed",
                _pygame.K_SPACE: "space_pressed",
                _pygame.K_ESCAPE: "escape_pressed",
            }
        else:
            key_map = {}
        # Build normalized key dict
        norm_keys = {}
        for k, v in keys.items():
            if isinstance(k, int):
                name = key_map.get(k)
                if name:
                    norm_keys[name] = v
            else:
                norm_keys[k] = v

        # Gamepad left stick navigation
        if gamepad and (abs(gamepad.get("lx", 0)) > 0.25 or abs(gamepad.get("ly", 0)) > 0.25):
            self.state.using_gamepad = True
            lx, ly = gamepad["lx"], -gamepad["ly"]  # invert Y
            self.state.nav_angle = math.atan2(ly, lx)
            # Convert angle to wedge index
            self._update_hover_from_angle(cx, cy, self.state.nav_angle)
            self.state.last_input_time = time.time()

            # A button confirms
            if gamepad.get("a_pressed"):
                self._confirm_selection()
        else:
            self.state.using_gamepad = False

            # Mouse hover
            dx, dy = mx - cx, my - cy
            dist = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)

            if dist < self.HUB_RADIUS * 1.1:
                self.state.hover_index = -1
                self.state.center_hover = min(1.0, self.state.center_hover + dt * 15)
            elif dist < self.OUTER_RADIUS * 1.1:
                self._update_hover_from_angle(cx, cy, angle)
                self.state.center_hover = max(0.0, self.state.center_hover - dt * 15)
            else:
                self.state.hover_index = -1
                self.state.center_hover = max(0.0, self.state.center_hover - dt * 10)

            # Keyboard navigation (WASD / arrows)
            nav_dir = None
            if norm_keys.get("w") or norm_keys.get("up"):
                nav_dir = (0, -1)
            elif norm_keys.get("s") or norm_keys.get("down"):
                nav_dir = (0, 1)
            elif norm_keys.get("a") or norm_keys.get("left"):
                nav_dir = (-1, 0)
            elif norm_keys.get("d") or norm_keys.get("right"):
                nav_dir = (1, 0)

            if nav_dir:
                self.state.using_gamepad = True
                nav_angle = math.atan2(-nav_dir[1], nav_dir[0])
                self._update_hover_from_angle(cx, cy, nav_angle)
                self.state.last_input_time = time.time()

            # Click / Enter / Space confirms
            if norm_keys.get("mouse_left_pressed") or norm_keys.get("enter_pressed") or norm_keys.get("space_pressed"):
                self._confirm_selection()

            # Right click / Escape cancels
            if norm_keys.get("mouse_right_pressed") or norm_keys.get("escape_pressed"):
                self.close()

    def _update_hover_from_angle(self, cx: float, cy: float, angle: float):
        """Convert angle to wedge index."""
        # Normalize angle: 0 = right, CCW positive
        # Wedges are distributed around the circle
        n = len(self.actions)
        # Shift so first wedge is at top (-90 deg)
        slice_angle = 2 * math.pi / n
        relative = angle + math.pi / 2  # shift so 0 is at top
        if relative < 0:
            relative += 2 * math.pi
        index = int(relative / slice_angle) % n
        self.state.hover_index = index

    def _confirm_selection(self):
        """Confirm the currently highlighted action."""
        if self.state.hover_index == -1:
            # Center hub - just close
            self.close()
        elif 0 <= self.state.hover_index < len(self.actions):
            action = self.actions[self.state.hover_index]
            if action.enabled:
                self._pending_action = action
            else:
                # Disabled - shake feedback?
                pass

    def _get_center(self) -> tuple:
        """Get wheel center position. Override for custom positioning."""
        # Default to screen center - will be overridden by draw() with actual center
        return (0, 0)

    def draw(self, ui, w: int, h: int, mouse_pos: tuple, now: float = None):
        """Draw the radial wheel centered on screen."""
        cx, cy = w / 2, h / 2
        self._get_center = lambda: (cx, cy)

        if self.state.anim <= 0.0 and not self.state.open:
            return

        # Eased animation
        t = ease_out(self.state.anim)

        # Scrim backdrop
        scrim_alpha = 0.55 * t
        ui.rect(0, 0, w, h, (0.01, 0.015, 0.025, scrim_alpha))

        # Subtle scanline overlay (Watch Dogs aesthetic)
        if t > 0.1:
            self._draw_scanlines(ui, w, h, t * 0.08)

        n = len(self.actions)
        slice_angle = 2 * math.pi / n
        start_angle = -math.pi / 2  # start at top

        # Stagger entrance for each wedge
        for i, action in enumerate(self.actions):
            wedge_t = max(0.0, min(1.0, (t - i * self.state.stagger_delay) / (1.0 - (n - 1) * self.state.stagger_delay)))
            wedge_ease = ease_out(wedge_t)

            if wedge_ease <= 0.01:
                continue

            self._draw_wedge(ui, cx, cy, action, i, start_angle, slice_angle, wedge_ease, now)

        # Center hub (draw last so it's on top)
        self._draw_center_hub(ui, cx, cy, t, now)

        # Radial brackets at outer edge (decorative)
        if t > 0.3:
            self._draw_outer_brackets(ui, cx, cy, n, start_angle, slice_angle, t)

    def _draw_wedge(self, ui, cx, cy, action: RadialAction, index: int,
                    start_angle: float, slice_angle: float, t: float, now: float):
        """Draw a single wedge segment."""
        is_hover = (self.state.hover_index == index and not self.state.using_gamepad) or \
                   (self.state.using_gamepad and index == self.state.hover_index)
        enabled = action.enabled

        # Colors
        base_color = action.color if enabled else T.TEXT_4
        fill_color = mix((0.08, 0.09, 0.11, 0.0), base_color + (0.18,), t * (0.85 if is_hover else 0.45))
        border_color = mix(T.BORDER_SOFT, base_color + (1.0,), t * (1.0 if is_hover else 0.6))
        text_color = mix(T.TEXT_3, base_color + (1.0,), t) if enabled else T.TEXT_4

        # Wedge geometry - build as triangle fan
        inner_r = lerp(self.HUB_RADIUS + 8, self.INNER_RADIUS, t)
        outer_r = lerp(self.HUB_RADIUS + 8, self.OUTER_RADIUS, t)

        a0 = start_angle + index * slice_angle
        a1 = a0 + slice_angle

        # Slight gap between wedges
        gap = 0.015 * (1.0 - t * 0.5)
        a0 += gap
        a1 -= gap

        # Build wedge vertices for filled polygon
        # Use the overlay's triangle batching
        segments = max(8, int(slice_angle * outer_r / 8))
        verts = []
        colors = []

        # Center point (actually inner arc center for ring segment)
        # For a ring segment, we need inner and outer arc points
        for j in range(segments + 1):
            aa = a0 + (a1 - a0) * j / segments
            # Outer vertex
            verts.append([cx + outer_r * math.cos(aa), cy + outer_r * math.sin(aa)])
            colors.append(fill_color)
            # Inner vertex
            verts.append([cx + inner_r * math.cos(aa), cy + inner_r * math.sin(aa)])
            colors.append(fill_color)

        # Submit as triangle strip
        if len(verts) >= 6:
            ui.triangles(np.array(verts, np.float32), np.array(colors, np.float32))

        # Border arcs (thin lines)
        if t > 0.3:
            bw = lerp(0.5, 1.8, t) * (2.0 if is_hover else 1.0)
            # Outer arc
            self._draw_arc_line(ui, cx, cy, outer_r, a0, a1, border_color, bw, segments)
            # Inner arc
            self._draw_arc_line(ui, cx, cy, inner_r, a1, a0, border_color, bw * 0.7, segments)
            # Radial edges
            for aa in (a0, a1):
                x0, y0 = cx + inner_r * math.cos(aa), cy + inner_r * math.sin(aa)
                x1, y1 = cx + outer_r * math.cos(aa), cy + outer_r * math.sin(aa)
                ui.line(x0, y0, x1, y1, bw, border_color)

        # Icon and label
        if t > 0.4:
            mid_angle = (a0 + a1) / 2
            icon_r = lerp(inner_r, (inner_r + outer_r) * 0.5, t)
            label_r = lerp((inner_r + outer_r) * 0.55, outer_r - 18, t)

            ix = cx + icon_r * math.cos(mid_angle)
            iy = cy + icon_r * math.sin(mid_angle)
            lx = cx + label_r * math.cos(mid_angle)
            ly = cy + label_r * math.sin(mid_angle)

            # Icon
            icon_size = lerp(0, 24, t) * (1.15 if is_hover else 1.0)
            icons.draw(ui, action.icon, ix, iy, icon_size, text_color if enabled else T.TEXT_4, 1.8)

            # Label (radial orientation)
            label_alpha = t * (1.0 if is_hover else 0.8)
            ui.text_c(action.label, lx, ly, T.T_SMALL,
                      (text_color[0], text_color[1], text_color[2], label_alpha),
                      weight="semibold", anchor="center")

            # Hover highlight ring
            if is_hover and enabled:
                pulse = 0.7 + 0.3 * math.sin(now * 4.5)
                ui.ring(ix, iy, icon_size * 0.7 + 6 * pulse, 2.0 * pulse,
                        (base_color[0], base_color[1], base_color[2], 0.5 * pulse))

    def _draw_arc_line(self, ui, cx, cy, r, a0, a1, color, width, segments):
        """Draw an arc as a series of line segments."""
        pts = []
        for i in range(segments + 1):
            aa = a0 + (a1 - a0) * i / segments
            pts.append((cx + r * math.cos(aa), cy + r * math.sin(aa)))
        ui.outline_poly(pts, width, color, closed=False)

    def _draw_center_hub(self, ui, cx, cy, t: float, now: float):
        """Draw the central hub button."""
        is_hover = self.state.hover_index == -1
        r = lerp(0, self.HUB_RADIUS, t)
        hover_t = self.state.center_hover

        # Background ring
        bg_alpha = 0.92 * t
        ui.circle(cx, cy, r, (0.07, 0.08, 0.10, bg_alpha))

        # Accent ring
        ring_w = lerp(0, 2.5, t) * (1.5 if is_hover else 1.0)
        ring_color = mix(T.BORDER_SOFT, T.SIGNAL, hover_t)
        ui.ring(cx, cy, r - 1, ring_w, ring_color)

        # Inner fill on hover
        if hover_t > 0.05:
            ui.circle(cx, cy, r - 4, (T.SIGNAL[0], T.SIGNAL[1], T.SIGNAL[2], 0.12 * hover_t))

        # Bracket corners (Watch Dogs style)
        if t > 0.5:
            bracket_t = ease_out((t - 0.5) / 0.5)
            arm = lerp(0, 14, bracket_t) * (1.2 if is_hover else 1.0)
            bracket_color = mix((T.SIGNAL[0], T.SIGNAL[1], T.SIGNAL[2], 0.0), T.SIGNAL, hover_t * bracket_t)
            bracket_w = lerp(0, 2.0, bracket_t)
            self._draw_brackets(ui, cx - r, cy - r, 2 * r, 2 * r, arm, bracket_color, bracket_w)

        # Icon
        if t > 0.3:
            icon_size = lerp(0, 26, t) * (1.1 if is_hover else 1.0)
            icon_color = mix(T.TEXT_3, T.SIGNAL, hover_t)
            icons.draw(ui, self.center_icon, cx, cy, icon_size, icon_color, 1.8)

        # Label below hub
        if t > 0.6:
            label_t = ease_out((t - 0.6) / 0.4)
            label_color = mix(T.TEXT_3, T.SIGNAL, hover_t * label_t)
            ui.text_c(self.center_label, cx, cy + r + 18, T.T_CAPTION, label_color, weight="semibold")

        # Scanline pulse on hover
        if is_hover and hover_t > 0.3:
            pulse = 0.5 + 0.5 * math.sin(now * 6)
            for i in range(-2, 3):
                y = cy + i * 8
                ui.line(cx - r * 0.7, y, cx + r * 0.7, y, 1.0,
                        (T.SIGNAL[0], T.SIGNAL[1], T.SIGNAL[2], 0.08 * pulse * hover_t))

    def _draw_outer_brackets(self, ui, cx, cy, n: int, start_angle: float,
                             slice_angle: float, t: float):
        """Draw decorative brackets at outer radius."""
        bracket_t = ease_out(max(0.0, (t - 0.3) / 0.7))
        if bracket_t < 0.05:
            return

        outer_r = self.OUTER_RADIUS
        for i in range(n):
            a0 = start_angle + i * slice_angle
            a1 = a0 + slice_angle
            mid = (a0 + a1) / 2

            # Only draw at wedge centers
            x = cx + outer_r * math.cos(mid)
            y = cy + outer_r * math.sin(mid)

            # Small bracket marking the wedge boundary
            arm = 8 * bracket_t
            bw = 1.5 * bracket_t
            # Perpendicular to radius
            px, py = -math.sin(mid), math.cos(mid)
            bx, by = math.cos(mid), math.sin(mid)

            ui.line(x - px * arm, y - py * arm, x + px * arm, y + py * arm, bw, (T.SIGNAL[0], T.SIGNAL[1], T.SIGNAL[2], 0.4 * bracket_t))

    def _draw_brackets(self, ui, x, y, w, h, arm, color, width):
        """Draw four L-shaped corner brackets."""
        half = width / 2
        for px, py, sx, sy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            ox, oy = px + sx * half, py + sy * half
            ui.line(ox - sx * half, oy, ox + sx * arm, oy, width, color)
            ui.line(ox, oy - sy * half, ox, oy + sy * arm, width, color)

    def _draw_scanlines(self, ui, w, h, alpha):
        """Subtle horizontal scanlines for Watch Dogs aesthetic."""
        spacing = 4
        line_alpha = alpha * 0.5
        color = (0.0, 0.0, 0.0, line_alpha)
        for y in range(0, int(h), spacing):
            ui.rect(0, y, w, 1, color)


# Predefined wheel configurations for common game actions

def create_hacking_wheel(on_select: Callable[[str], None]) -> RadialWheel:
    """Create the main hacking/profiling wheel."""
    actions = [
        RadialAction("profile", "PROFILE", "info", T.SIGNAL, True,
                     lambda: on_select("profile")),
        RadialAction("hack", "HACK", "bolt", (1.0, 0.55, 0.15, 1.0), True,
                     lambda: on_select("hack")),
        RadialAction("camera", "CAMERA", "monitor", (0.36, 0.86, 1.0, 1.0), True,
                     lambda: on_select("camera")),
        RadialAction("distract", "DISTRACT", "speaker", (0.36, 1.0, 0.55, 1.0), True,
                     lambda: on_select("distract")),
        RadialAction("block", "BLOCK", "lock", (0.8, 0.4, 1.0, 1.0), True,
                     lambda: on_select("block")),
        RadialAction("trigger", "TRIGGER", "sparkle", (1.0, 0.85, 0.2, 1.0), True,
                     lambda: on_select("trigger")),
    ]
    return RadialWheel(actions, "EXIT", "close")


def create_vehicle_wheel(on_select: Callable[[str], None]) -> RadialWheel:
    """Create vehicle-specific wheel."""
    actions = [
        RadialAction("enter", "ENTER", "car", T.SIGNAL, True,
                     lambda: on_select("enter")),
        RadialAction("hack_vehicle", "HACK", "bolt", (1.0, 0.55, 0.15, 1.0), True,
                     lambda: on_select("hack_vehicle")),
        RadialAction("camera", "CAMERA", "monitor", (0.36, 0.86, 1.0, 1.0), True,
                     lambda: on_select("camera")),
        RadialAction("track", "TRACK", "route", (0.36, 1.0, 0.55, 1.0), True,
                     lambda: on_select("track")),
    ]
    return RadialWheel(actions, "BACK", "chevron_l")


def create_combat_wheel(on_select: Callable[[str], None]) -> RadialWheel:
    """Create combat/weapon wheel."""
    actions = [
        RadialAction("pistol", "PISTOL", "bolt", T.SIGNAL, True,
                     lambda: on_select("pistol")),
        RadialAction("rifle", "RIFLE", "monitor", (1.0, 0.55, 0.15, 1.0), True,
                     lambda: on_select("rifle")),
        RadialAction("grenade", "GRENADE", "sparkle", (1.0, 0.85, 0.2, 1.0), True,
                     lambda: on_select("grenade")),
        RadialAction("lure", "LURE", "speaker", (0.36, 1.0, 0.55, 1.0), True,
                     lambda: on_select("lure")),
        RadialAction("drone", "DRONE", "globe", (0.8, 0.4, 1.0, 1.0), True,
                     lambda: on_select("drone")),
        RadialAction("rc", "RC CAR", "car", (1.0, 0.4, 0.6, 1.0), True,
                     lambda: on_select("rc")),
    ]
    return RadialWheel(actions, "HOLSTER", "chevron_l")


# Integration helper for the main app

class RadialWheelManager:
    """Manages multiple radial wheels and context switching."""

    def __init__(self):
        self.wheels: dict[str, RadialWheel] = {}
        self.active: Optional[RadialWheel] = None
        self.context: str = "none"

    def register(self, name: str, wheel: RadialWheel):
        self.wheels[name] = wheel

    def open(self, name: str):
        """Open a wheel by name, closing any active wheel."""
        if self.active:
            self.active.close()
        wheel = self.wheels.get(name)
        if wheel:
            self.active = wheel
            self.context = name
            wheel.open()

    def close(self):
        if self.active:
            self.active.close()
            self.active = None
            self.context = "none"

    def toggle(self, name: str):
        if self.active and self.context == name:
            self.close()
        else:
            self.open(name)

    def update(self, dt: float, mouse_pos: tuple, keys: dict, gamepad: dict = None):
        if self.active:
            return self.active.update(dt, mouse_pos, keys, gamepad)
        return None

    def draw(self, ui, w: int, h: int, mouse_pos: tuple, now: float):
        if self.active:
            self.active.draw(ui, w, h, mouse_pos, now)

    def is_open(self) -> bool:
        return self.active is not None and (self.active.state.open or self.active.state.anim > 0)


# Example usage in the main app:
#
# wheel_manager = RadialWheelManager()
# wheel_manager.register("hacking", create_hacking_wheel(handle_hack_action))
# wheel_manager.register("vehicle", create_vehicle_wheel(handle_vehicle_action))
# wheel_manager.register("combat", create_combat_wheel(handle_combat_action))
#
# # In main loop:
# result = wheel_manager.update(dt, mouse_pos, keys, gamepad_state)
# wheel_manager.draw(ui, width, height, mouse_pos, time.time())
#
# # Open with Tab key or gamepad LB:
# if keys.get("tab_pressed") or gamepad.get("lb_pressed"):
#     wheel_manager.toggle("hacking")