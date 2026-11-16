"""Watch Dogs themed HUD components - minimap, speed card, status cards, etc."""
from __future__ import annotations

import math
import time

import numpy as np

from . import theme as T
from .watchdogs_theme import (
    lerp, lerp_color, ease_out,
    WD_CYAN, WD_CYAN_DIM, WD_CYAN_BRIGHT, WD_CYAN_SOFT, WD_CYAN_FAINT,
    WD_ORANGE, WD_GREEN, WD_PURPLE,
    WD_BG_DEEP, WD_BG_DARK, WD_BG_MEDIUM, WD_BG_LIGHT,
    WD_PANEL_GRADIENT, WD_CARD_GRADIENT, WD_BUTTON_GRADIENT,
    WD_BORDER_PRIMARY, WD_BORDER_SECONDARY, WD_BORDER_SUBTLE, WD_BORDER_FOCUS,
    WD_TEXT_PRIMARY, WD_TEXT_SECONDARY, WD_TEXT_MUTED, WD_TEXT_DISABLED,
    WD_OK, WD_WARN, WD_ERROR,
    WD_R_SMALL, WD_R_MEDIUM, WD_R_LARGE, WD_R_PILL, WD_R_CARD,
    WD_T_MICRO, WD_T_CAPTION, WD_T_SMALL, WD_T_BODY, WD_T_BASE, WD_T_LABEL, WD_T_HEAD, WD_DISPLAY,
    WD_ANIM_FAST, WD_ANIM_NORMAL, WD_ANIM_SLOW,
    WD_SHADOW_MEDIUM, WD_SHADOW_STRONG,
)
from .watchdogs_icons import draw_wd
from .watchdogs_ui import wd_brackets, wd_scanlines, wd_gradient_panel


class WatchDogsMinimap:
    """Watch Dogs style minimap with holographic aesthetic."""

    def __init__(self):
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.rotation = 0.0
        self.markers = []

    def draw(self, ui, w, h, scene, focus, heading, status_h, navigation=None):
        """Draw the minimap in bottom-left corner."""
        # Map dimensions
        map_size = min(280, w * 0.3)
        margin = 20
        x = margin
        y = h - map_size - margin - status_h

        # Background panel with scanlines
        wd_gradient_panel(ui, x - 4, y - 4, map_size + 8, map_size + 8,
                          WD_PANEL_GRADIENT, border=True,
                          border_color=WD_BORDER_PRIMARY, radius=WD_R_LARGE)

        # Scanline overlay
        wd_scanlines(ui, x, y, map_size, map_size, alpha=0.02)

        ui.push_clip(x, y, map_size, map_size)

        # Draw scene tiles
        self._draw_tiles(ui, x, y, map_size, scene, focus, heading)

        # Draw navigation route if active
        if navigation and navigation.active:
            self._draw_route(ui, x, y, map_size, navigation, focus, heading)

        # Draw player marker
        self._draw_player_marker(ui, x, y, map_size, focus, heading)

        # Draw markers
        self._draw_markers(ui, x, y, map_size, focus, heading)

        ui.pop_clip()

        # Corner brackets
        wd_brackets(ui, x - 4, y - 4, map_size + 8, map_size + 8,
                    arm=14.0, color=WD_CYAN, width=2.0)

        # Zoom label
        ui.text_c(f"ZOOM {self.zoom:.1f}X", x + map_size / 2, y + map_size + 18,
                  WD_T_CAPTION, WD_TEXT_MUTED, weight="semibold")

        return (x - 4, y - 4, map_size + 8, map_size + 8)

    def _draw_tiles(self, ui, x, y, size, scene, focus, heading):
        """Draw visible road/building tiles."""
        if not scene.tiles:
            return

        half = size / 2
        center_x = x + half
        center_y = y + half

        # Transform: rotate around center, then scale
        cos_h = math.cos(-heading)
        sin_h = math.sin(-heading)
        scale = self.zoom * 0.008  # meters per pixel

        for tile in scene.tiles.values():
            if not tile.driving:
                continue
            for path in tile.driving.paths:
                if not path.points:
                    continue
                pts = []
                for px, py in path.points:
                    # World to map space
                    dx = px - focus[0]
                    dy = py - focus[1]
                    rx = dx * cos_h - dy * sin_h
                    ry = dx * sin_h + dy * cos_h
                    mx = center_x + rx * scale
                    my = center_y + ry * scale
                    pts.append((mx, my))

                if len(pts) >= 2:
                    # Road color based on type
                    if path.kind == "highway":
                        color = lerp_color(WD_CYAN, WD_CYAN_BRIGHT, 0.5)
                        width = 2.5 / ui.scale
                    elif path.kind == "primary":
                        color = WD_CYAN
                        width = 2.0 / ui.scale
                    else:
                        color = lerp_color(WD_CYAN_DIM, WD_CYAN, 0.3)
                        width = 1.5 / ui.scale

                    ui.outline_poly(pts, width, color, closed=False)

    def _draw_route(self, ui, x, y, size, navigation, focus, heading):
        """Draw navigation route with pulsing effect."""
        if not navigation.route:
            return

        half = size / 2
        center_x = x + half
        center_y = y + half
        cos_h = math.cos(-heading)
        sin_h = math.sin(-heading)
        scale = self.zoom * 0.008

        pulse = 0.5 + 0.5 * math.sin(time.time() * 4)

        pts = []
        for wp in navigation.route:
            dx = wp[0] - focus[0]
            dy = wp[1] - focus[1]
            rx = dx * cos_h - dy * sin_h
            ry = dx * sin_h + dy * cos_h
            mx = center_x + rx * scale
            my = center_y + ry * scale
            pts.append((mx, my))

        if len(pts) >= 2:
            # Pulsing route line
            color = lerp_color(WD_ORANGE, (1.0, 0.8, 0.2, 1.0), pulse)
            ui.outline_poly(pts, 3.0 / ui.scale, color)

            # Destination marker
            dest = pts[-1]
            ui.circle(dest[0], dest[1], 8, lerp_color(WD_ORANGE, (1.0, 0.6, 0.0, 1.0), pulse))
            wd_brackets(ui, dest[0] - 10, dest[1] - 10, 20, 20, arm=6, color=WD_ORANGE)

    def _draw_player_marker(self, ui, x, y, size, focus, heading):
        """Draw player position marker with heading arrow."""
        half = size / 2
        cx = x + half
        cy = y + half

        # Outer pulse ring
        pulse = 0.5 + 0.5 * math.sin(time.time() * 3)
        ui.ring(cx, cy, 12 + 4 * pulse, 2.0 / ui.scale,
                lerp_color(WD_CYAN, WD_CYAN_BRIGHT, 0.5 * pulse))

        # Heading arrow (triangle)
        arrow_len = 14
        arrow_w = 10
        hx = cx + arrow_len * math.sin(heading)
        hy = cy - arrow_len * math.cos(heading)
        bx1 = cx + arrow_w * math.sin(heading + 2.5)
        by1 = cy - arrow_w * math.cos(heading + 2.5)
        bx2 = cx + arrow_w * math.sin(heading - 2.5)
        by2 = cy - arrow_w * math.cos(heading - 2.5)

        ui.poly([(hx, hy), (bx1, by1), (bx2, by2)],
                lerp_color(WD_CYAN, WD_CYAN_BRIGHT, 0.5))

        # Center dot
        ui.circle(cx, cy, 3, WD_CYAN_BRIGHT)

        # Corner brackets on player marker
        wd_brackets(ui, cx - 16, cy - 16, 32, 32, arm=8, color=WD_CYAN, animated=True, phase=1.0)

    def _draw_markers(self, ui, x, y, size, focus, heading):
        """Draw POI markers."""
        half = size / 2
        center_x = x + half
        center_y = y + half
        cos_h = math.cos(-heading)
        sin_h = math.sin(-heading)
        scale = self.zoom * 0.008

        for marker in self.markers:
            dx = marker.x - focus[0]
            dy = marker.y - focus[1]
            rx = dx * cos_h - dy * sin_h
            ry = dx * sin_h + dy * cos_h
            mx = center_x + rx * scale
            my = center_y + ry * scale

            # Check if on screen
            if x - 10 <= mx <= x + size + 10 and y - 10 <= my <= y + size + 10:
                color = marker.color or WD_ORANGE
                ui.circle(mx, my, 5, color)
                # Small bracket
                wd_brackets(ui, mx - 7, my - 7, 14, 14, arm=4, color=color)


class WatchDogsSpeedCard:
    """Watch Dogs style speedometer card."""

    def draw(self, ui, x, y, vehicle_name, speed, gear, state, signal, braking, rpm_frac, unit="km/h"):
        w, h = 236, 124
        accent = WD_CYAN

        # Panel
        wd_gradient_panel(ui, x, y, w, h, WD_CARD_GRADIENT,
                          border=True, border_color=WD_BORDER_PRIMARY, radius=WD_R_LARGE)

        # Vehicle name
        ui.text(vehicle_name.upper(), x + 16, y + 12, WD_T_CAPTION, WD_TEXT_MUTED, weight="semibold")

        # Speed number
        speed_str = f"{int(speed):d}"
        ui.text(speed_str, x + 16, y + 36, WD_DISPLAY - 4, WD_TEXT_PRIMARY, weight="bold")

        # Unit
        ui.text(unit.upper(), x + 16 + ui.text_w(speed_str, WD_DISPLAY - 4, weight="bold") + 4,
                y + 60, WD_T_CAPTION, WD_TEXT_MUTED, weight="semibold")

        # Gear indicator
        ui.text_c(f"GEAR {gear}", x + w - 16, y + 20, WD_T_SMALL, WD_TEXT_MUTED,
                  anchor="right", weight="semibold")

        # State
        state_color = WD_GREEN if "braking" not in state.lower() else WD_ORANGE
        if "blocked" in state.lower():
            state_color = WD_ERROR
        ui.text_c(state.upper(), x + w - 16, y + 40, WD_T_CAPTION, state_color,
                  anchor="right", weight="semibold")

        # Signal indicator
        if signal != 0:
            sig_color = WD_ORANGE
            sig_text = "LEFT" if signal < 0 else "RIGHT"
            ui.text_c(f"⬤ {sig_text}", x + w - 16, y + 60, WD_T_CAPTION, sig_color,
                      anchor="right", weight="semibold")

        # RPM bar (bottom)
        bar_y = y + h - 14
        ui.box(16, bar_y, w - 32, 4, 2, WD_BG_DARK, WD_BORDER_SUBTLE)
        fill_w = (w - 32) * rpm_frac
        if fill_w > 2:
            rpm_color = lerp_color(WD_CYAN, WD_ORANGE, max(0, rpm_frac - 0.7) / 0.3)
            ui.box(16, bar_y, fill_w, 4, 2, rpm_color)

        # Corner brackets
        wd_brackets(ui, x, y, w, h, arm=10, color=WD_CYAN, animated=True, phase=1.0)


class WatchDogsStatusCard:
    """Watch Dogs style status/info card for streaming, traffic, etc."""

    def draw(self, ui, x, y, w, rows, progress=None, shimmer=0.0, min_w=220):
        """Draw a status card with rows of info."""
        # Calculate height
        row_h = 22
        h = len(rows) * row_h + 24

        # Panel
        wd_gradient_panel(ui, x, y, max(w, min_w), h, WD_CARD_GRADIENT,
                          border=True, border_color=WD_BORDER_PRIMARY, radius=WD_R_MEDIUM)

        # Rows
        for i, (text, color, size, weight) in enumerate(rows):
            ry = y + 10 + i * row_h
            if isinstance(color, tuple):
                text_color = color
            else:
                text_color = WD_TEXT_PRIMARY
            ui.text(text, x + 12, ry, size, text_color, weight=weight)

        # Progress bar
        if progress is not None:
            p = max(0.0, min(1.0, float(progress)))
            bar_y = y + h - 8
            ui.box(x + 8, bar_y, w - 16, 3, 1.5, WD_BG_DARK, WD_BORDER_SUBTLE)
            if p > 0:
                fill_w = (w - 16) * p
                if fill_w > 2:
                    # Shimmer effect
                    phase = shimmer * 2 % 1.0
                    sweep_x = x + 8 + phase * (w - 16)
                    sweep_w = min(60, w * 0.3)
                    sweep_x = max(x + 8, min(sweep_x, x + w - 8 - sweep_w))
                    ui.rect(sweep_x, bar_y, sweep_w, 3, lerp_color(WD_CYAN, WD_CYAN_BRIGHT, 0.5))
                ui.box(x + 8, bar_y, max(fill_w, 3), 3, 1.5, WD_CYAN)

        # Corner brackets
        wd_brackets(ui, x, y, max(w, min_w), h, arm=8, color=WD_CYAN, animated=True, phase=1.0)


class WatchDogsCompass:
    """Watch Dogs style compass with cardinal directions."""

    def draw(self, ui, cx, cy, r, heading):
        """Draw compass rose."""
        # Outer ring
        ui.ring(cx, cy, r, 1.5 / ui.scale, WD_BORDER_PRIMARY)

        # Cardinal directions
        for label, angle in [("N", 0), ("E", math.pi/2), ("S", math.pi), ("W", -math.pi/2)]:
            a = angle - heading
            x = cx + (r + 14) * math.sin(a)
            y = cy - (r + 14) * math.cos(a)
            ui.text_c(label, x, y, WD_T_CAPTION, WD_TEXT_PRIMARY, weight="bold")

        # Tick marks
        for i in range(16):
            a = i * math.pi / 8 - heading
            x1 = cx + (r - 4) * math.sin(a)
            y1 = cy - (r - 4) * math.cos(a)
            x2 = cx + r * math.sin(a)
            y2 = cy - r * math.cos(a)
            ui.line(x1, y1, x2, y2, 1.0 / ui.scale, WD_BORDER_SECONDARY)


class WatchDogsTimeWeatherCard:
    """Watch Dogs style time and weather display."""

    def draw(self, ui, x, y, w, time_str, weather_label, temperature=None):
        h = 56
        wd_gradient_panel(ui, x, y, w, h, WD_CARD_GRADIENT,
                          border=True, border_color=WD_BORDER_PRIMARY, radius=WD_R_MEDIUM)

        # Time
        ui.text(time_str, x + 14, y + 8, WD_T_HEAD, WD_TEXT_PRIMARY, weight="bold")

        # Weather
        weather_color = WD_CYAN
        if "rain" in weather_label.lower():
            weather_color = WD_CYAN_BRIGHT
        elif "clear" in weather_label.lower() or "sun" in weather_label.lower():
            weather_color = WD_ORANGE
        elif "storm" in weather_label.lower():
            weather_color = WD_PURPLE

        ui.text(weather_label.upper(), x + 14, y + 32, WD_T_CAPTION, weather_color, weight="semibold")

        if temperature is not None:
            ui.text_c(f"{temperature}°C", x + w - 14, y + 20, WD_T_BASE, WD_TEXT_SECONDARY, anchor="right")

        # Corner brackets
        wd_brackets(ui, x, y, w, h, arm=8, color=WD_CYAN)


class WatchDogsNotification:
    """Watch Dogs style notification toast."""

    @staticmethod
    def draw(ui, w, h, text, color=WD_CYAN, progress=0.0):
        """Draw notification at top-center with slide-down animation."""
        pass  # Would be managed by a notification queue