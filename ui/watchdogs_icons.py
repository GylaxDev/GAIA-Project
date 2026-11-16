"""Watch Dogs themed icons - geometric, technical, hacker aesthetic.

All icons designed on a 24x24 grid with:
- Sharp angles and geometric shapes
- Bracket/corner motifs
- Circuit board patterns
- Terminal/cursor elements
"""
from __future__ import annotations

import math

from . import icons  # for base drawing helpers


def _p(cx, cy, k, x, y):
    """Transform 24x24 grid point to scaled position."""
    return cx + (x - 12.0) * k, cy + (y - 12.0) * k


def _poly(ui, cx, cy, k, pts, col, w, closed=False):
    P = [_p(cx, cy, k, *q) for q in pts]
    for a, b in zip(P, P[1:] + (P[:1] if closed else [])):
        ui.line(a[0], a[1], b[0], b[1], w, col)


def _arc(ui, cx, cy, k, ox, oy, r, a0, a1, col, w, n=14):
    pts = [(ox + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
            oy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
    _poly(ui, cx, cy, k, pts, col, w)


def _ring(ui, cx, cy, k, ox, oy, r, col, w):
    x, y = _p(cx, cy, k, ox, oy)
    ui.ring(x, y, r * k, w, col)


def _dot(ui, cx, cy, k, ox, oy, r, col):
    x, y = _p(cx, cy, k, ox, oy)
    ui.circle(x, y, r * k, col)


# ========================================================================
# WATCH DOGS SPECIFIC ICONS
# ========================================================================

def wd_hack(ui, cx, cy, k, c, w):
    """Hack/breach icon - geometric shield with corner brackets."""
    # Outer shield shape
    _poly(ui, cx, cy, k, [
        (12, 3), (18, 5), (21, 12), (18, 19), (12, 21), (6, 19), (3, 12), (6, 5)
    ], c, w, closed=True)

    # Corner brackets (signature WD)
    bracket_pts = [
        (8, 8), (11, 8),       # top-left horizontal
        (8, 8), (8, 11),       # top-left vertical
        (16, 8), (13, 8),      # top-right
        (16, 8), (16, 11),     # top-right vertical
        (8, 16), (11, 16),     # bottom-left
        (8, 16), (8, 13),      # bottom-left vertical
        (16, 16), (13, 16),    # bottom-right
        (16, 16), (16, 13),    # bottom-right vertical
    ]
    _poly(ui, cx, cy, k, bracket_pts, c, w * 1.2)

    # Center data diamond
    _poly(ui, cx, cy, k, [
        (12, 10.5), (13.5, 12), (12, 13.5), (10.5, 12)
    ], c, w, closed=True)


def wd_profile(ui, cx, cy, k, c, w):
    """Profile/targeting icon - crosshair with brackets."""
    # Center crosshair
    _poly(ui, cx, cy, k, [(12, 5), (12, 19)], c, w)
    _poly(ui, cx, cy, k, [(5, 12), (19, 12)], c, w)

    # Corner targeting brackets
    size = 5.5
    for sx, sy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        bx = 12 + sx * 6.5
        by = 12 + sy * 6.5
        _poly(ui, cx, cy, k, [
            (bx - sx * size, by), (bx - sx * 1.5, by),
            (bx, by - sy * 1.5), (bx, by - sy * size)
        ], c, w)


def wd_camera(ui, cx, cy, k, c, w):
    """Camera/surveillance icon - lens with recording indicator."""
    # Camera body
    _poly(ui, cx, cy, k, [(5, 8), (19, 8), (19, 18), (5, 18)], c, w, closed=True)
    _poly(ui, cx, cy, k, [(7, 8), (10, 5), (14, 5), (17, 8)], c, w, closed=True)

    # Lens
    _ring(ui, cx, cy, k, 12, 13, 4.5, c, w)

    # Recording dot (top-right)
    _dot(ui, cx, cy, k, 16, 8, 1.8, (1.0, 0.2, 0.2, 1.0))

    # Corner brackets on lens
    _poly(ui, cx, cy, k, [
        (9, 10), (10.5, 10), (9, 10), (9, 11.5),
        (15, 10), (13.5, 10), (15, 10), (15, 11.5),
        (9, 15), (10.5, 15), (9, 15), (9, 13.5),
        (15, 15), (13.5, 15), (15, 15), (15, 13.5),
    ], c, w * 0.8)


def wd_distract(ui, cx, cy, k, c, w):
    """Distract/noise icon - speaker with signal waves."""
    # Speaker cone
    _poly(ui, cx, cy, k, [
        (5, 9), (9, 9), (14, 5), (14, 19), (9, 19)
    ], c, w, closed=True)

    # Signal waves (three arcs)
    for i, (r, alpha) in enumerate([(4, 1.0), (7, 0.6), (10, 0.3)]):
        _arc(ui, cx, cy, k, 15, 14, r, -50, 50,
             (c[0], c[1], c[2], c[3] * alpha), w * 0.8)


def wd_block(ui, cx, cy, k, c, w):
    """Block/firewall icon - hexagonal shield with slash."""
    # Hexagon
    hex_pts = []
    for i in range(6):
        angle = math.radians(-90 + i * 60)
        hex_pts.append((12 + 8.5 * math.cos(angle), 12 + 8.5 * math.sin(angle)))
    _poly(ui, cx, cy, k, hex_pts, c, w, closed=True)

    # Inner hex
    inner_pts = []
    for i in range(6):
        angle = math.radians(-90 + i * 60)
        inner_pts.append((12 + 5.5 * math.cos(angle), 12 + 5.5 * math.sin(angle)))
    _poly(ui, cx, cy, k, inner_pts, c, w, closed=True)

    # Diagonal slash
    _poly(ui, cx, cy, k, [(6, 18), (18, 6)], c, w * 1.5)

    # Corner brackets
    _poly(ui, cx, cy, k, [
        (7.5, 7.5), (10, 7.5), (7.5, 7.5), (7.5, 10),
        (16.5, 7.5), (14, 7.5), (16.5, 7.5), (16.5, 10),
        (7.5, 16.5), (10, 16.5), (7.5, 16.5), (7.5, 14),
        (16.5, 16.5), (14, 16.5), (16.5, 16.5), (16.5, 14),
    ], c, w * 0.7)


def wd_trigger(ui, cx, cy, k, c, w):
    """Trigger/trap icon - pressure plate with corners."""
    # Base plate
    _poly(ui, cx, cy, k, [(5, 7), (19, 7), (19, 17), (5, 17)], c, w, closed=True)

    # Pressure sensor pattern
    for y in [9, 11, 13, 15]:
        _poly(ui, cx, cy, k, [(6, y), (18, y)], c, w * 0.5)

    # Corner activation brackets
    _poly(ui, cx, cy, k, [
        (6, 6), (9, 6), (6, 6), (6, 9),
        (18, 6), (15, 6), (18, 6), (18, 9),
        (6, 18), (9, 18), (6, 18), (6, 15),
        (18, 18), (15, 18), (18, 18), (18, 15),
    ], c, w)

    # Center trigger diamond
    _poly(ui, cx, cy, k, [
        (12, 10), (14, 12), (12, 14), (10, 12)
    ], c, w, closed=True)


def wd_vehicle(ui, cx, cy, k, c, w):
    """Vehicle/car icon - side profile with brackets."""
    # Car body
    _poly(ui, cx, cy, k, [
        (4, 15), (4, 11), (7, 10), (11, 7), (17, 7), (20, 11), (21, 12), (21, 15)
    ], c, w, closed=True)

    # Window
    _poly(ui, cx, cy, k, [
        (9, 11), (10, 8.5), (14.5, 8.5), (16, 11)
    ], c, w * 0.8, closed=True)

    # Wheels
    _dot(ui, cx, cy, k, 8, 16, 1.8, c)
    _dot(ui, cx, cy, k, 17, 16, 1.8, c)

    # Corner brackets
    _poly(ui, cx, cy, k, [
        (5.5, 7.5), (8, 7.5), (5.5, 7.5), (5.5, 10),
        (18.5, 7.5), (16, 7.5), (18.5, 7.5), (18.5, 10),
    ], c, w * 0.8)


def wd_track(ui, cx, cy, k, c, w):
    """Track/trace icon - location pin with path."""
    # Path line
    _poly(ui, cx, cy, k, [(12, 18), (12, 8), (9, 5), (12, 2)], c, w)

    # Location pin
    pin_pts = [
        (12, 10), (10.5, 7), (9, 6), (9.5, 7), (12, 4),
        (14.5, 7), (15, 6), (13.5, 7)
    ]
    _poly(ui, cx, cy, k, pin_pts, c, w, closed=True)

    # Pin center ring
    _ring(ui, cx, cy, k, 12, 6.5, 1.8, c, w)

    # Corner brackets on pin
    _poly(ui, cx, cy, k, [
        (9.5, 7.5), (10.5, 7.5), (9.5, 7.5), (9.5, 8.5),
        (14.5, 7.5), (13.5, 7.5), (14.5, 7.5), (14.5, 8.5),
    ], c, w * 0.7)


def wd_pistol(ui, cx, cy, k, c, w):
    """Pistol/handgun icon - minimalist side profile."""
    _poly(ui, cx, cy, k, [
        (5, 13), (5, 11), (8, 10), (8, 14)
    ], c, w, closed=True)

    _poly(ui, cx, cy, k, [
        (8, 10), (17, 10), (17, 11), (15, 11), (14, 8.5), (15, 7), (17, 7),
        (17, 4), (15, 4), (15, 7), (14, 8.5), (15, 11), (13, 11), (13, 14), (8, 14)
    ], c, w, closed=True)

    _poly(ui, cx, cy, k, [
        (6, 11.5), (9, 11.5), (9, 10.5), (6, 10.5)
    ], c, w * 0.8, closed=True)


def wd_rifle(ui, cx, cy, k, c, w):
    """Rifle/assault weapon icon."""
    _poly(ui, cx, cy, k, [
        (4, 13), (4, 11), (6, 11), (6, 10), (12, 10), (12, 9), (20, 9),
        (20, 7), (22, 7), (22, 15), (20, 15), (20, 13), (12, 13), (12, 14),
        (6, 14)
    ], c, w, closed=True)

    # Bipod
    _poly(ui, cx, cy, k, [(12, 14), (12, 15), (14, 15)], c, w)
    _poly(ui, cx, cy, k, [(6, 14), (6, 15), (4, 15)], c, w)


def wd_grenade(ui, cx, cy, k, c, w):
    """Grenade/explosive icon."""
    # Grenade body
    _poly(ui, cx, cy, k, [
        (12, 4), (11, 4.5), (8, 4.5), (7, 6), (7, 16), (8, 17.5),
        (16, 17.5), (17, 16), (17, 6), (16, 4.5), (13, 4.5)
    ], c, w, closed=True)

    # Pin
    _poly(ui, cx, cy, k, [
        (10, 5.5), (11, 4.5), (12, 5.5)
    ], c, w)

    # Lever
    _poly(ui, cx, cy, k, [
        (12, 5.5), (14, 4), (15, 5.5), (14, 7)
    ], c, w)


def wd_lure(ui, cx, cy, k, c, w):
    """Lure/decoy icon - transmitter with signal."""
    # Base unit
    _poly(ui, cx, cy, k, [
        (8, 10), (16, 10), (16, 16), (8, 16)
    ], c, w, closed=True)

    # Signal emitter
    _poly(ui, cx, cy, k, [
        (12, 10), (11, 7), (12, 5), (13, 7)
    ], c, w, closed=True)

    # Signal waves
    for r, a in [(2, 0.8), (4, 0.5), (6, 0.2)]:
        _arc(ui, cx, cy, k, 12, 5, r, -60, 60,
             (c[0], c[1], c[2], c[3] * a), w * 0.8)


def wd_drone(ui, cx, cy, k, c, w):
    """Drone/UAV icon - quadcopter top view."""
    # Center body
    _poly(ui, cx, cy, k, [
        (11, 12), (12, 8), (13, 12)
    ], c, w, closed=True)
    _poly(ui, cx, cy, k, [
        (11, 12), (12, 16), (13, 12)
    ], c, w, closed=True)

    # Four rotors
    for cx_off, cy_off in [(-5, -5), (5, -5), (-5, 5), (5, 5)]:
        _poly(ui, cx, cy, k, [
            (12 + cx_off - 2, 12 + cy_off),
            (12 + cx_off, 12 + cy_off - 2),
            (12 + cx_off + 2, 12 + cy_off),
            (12 + cx_off, 12 + cy_off + 2),
        ], c, w, closed=True)

    # Rotor blades
    for cx_off, cy_off in [(-5, -5), (5, -5), (-5, 5), (5, 5)]:
        _poly(ui, cx, cy, k, [
            (12 + cx_off - 3, 12 + cy_off), (12 + cx_off + 3, 12 + cy_off),
        ], c, w)
        _poly(ui, cx, cy, k, [
            (12 + cx_off, 12 + cy_off - 3), (12 + cx_off, 12 + cy_off + 3),
        ], c, w)


def wd_rc(ui, cx, cy, k, c, w):
    """RC car/controller icon."""
    # Car body
    _poly(ui, cx, cy, k, [
        (5, 14), (5, 10), (7, 9), (11, 7), (15, 7), (18, 9.5), (20, 10), (20, 14)
    ], c, w, closed=True)

    # RC antenna
    _poly(ui, cx, cy, k, [
        (12, 10), (12.5, 6), (12, 5)
    ], c, w)

    # Signal wave on antenna
    _arc(ui, cx, cy, k, 12.5, 5.5, 1.5, -90, 90, c, w)

    # Wheels
    _dot(ui, cx, cy, k, 7.5, 15, 1.5, c)
    _dot(ui, cx, cy, k, 17, 15, 1.5, c)

    # Corner brackets
    _poly(ui, cx, cy, k, [
        (6, 7), (8, 7), (6, 7), (6, 9),
        (18, 7), (16, 7), (18, 7), (18, 9),
    ], c, w * 0.7)


def wd_exit(ui, cx, cy, k, c, w):
    """Exit/back icon - door with arrow."""
    # Door frame
    _poly(ui, cx, cy, k, [
        (5, 5), (19, 5), (19, 19), (5, 19)
    ], c, w, closed=True)

    # Exit arrow
    _poly(ui, cx, cy, k, [
        (15, 8), (10, 12), (15, 16)
    ], c, w)

    _poly(ui, cx, cy, k, [
        (10, 12), (13, 12)
    ], c, w)


def wd_settings(ui, cx, cy, k, c, w):
    """Settings/gear icon - technical gear."""
    # Outer gear teeth
    teeth = 8
    outer_r = 10
    inner_r = 7
    for i in range(teeth):
        angle = i * 2 * math.pi / teeth
        angle_next = (i + 1) * 2 * math.pi / teeth
        # Two points per tooth
        x1 = 12 + outer_r * math.cos(angle - 0.1)
        y1 = 12 + outer_r * math.sin(angle - 0.1)
        x2 = 12 + outer_r * math.cos(angle + 0.1)
        y2 = 12 + outer_r * math.sin(angle + 0.1)
        _poly(ui, cx, cy, k, [(x1, y1), (x2, y2)], c, w)

    # Inner ring
    _ring(ui, cx, cy, k, 12, 12, inner_r, c, w)

    # Center hole
    _ring(ui, cx, cy, k, 12, 12, 3, c, w)

    # Corner brackets
    _poly(ui, cx, cy, k, [
        (6, 6), (8.5, 6), (6, 6), (6, 8.5),
        (18, 6), (15.5, 6), (18, 6), (18, 8.5),
        (6, 18), (8.5, 18), (6, 18), (6, 15.5),
        (18, 18), (15.5, 18), (18, 18), (18, 15.5),
    ], c, w * 0.6)


def wd_map(ui, cx, cy, k, c, w):
    """Map/location icon - folded map with pin."""
    # Map base
    _poly(ui, cx, cy, k, [
        (4, 6), (20, 6), (20, 19), (4, 19)
    ], c, w, closed=True)

    # Fold lines
    _poly(ui, cx, cy, k, [(10, 6), (10, 19)], c, w * 0.5)
    _poly(ui, cx, cy, k, [(4, 12), (20, 12)], c, w * 0.5)

    # Location pin
    _poly(ui, cx, cy, k, [
        (12, 9), (10.5, 7), (12, 4.5), (13.5, 7)
    ], c, w, closed=True)


def wd_garage(ui, cx, cy, k, c, w):
    """Garage/vehicle bay icon."""
    # Garage door
    _poly(ui, cx, cy, k, [
        (4, 5), (20, 5), (20, 18), (4, 18)
    ], c, w, closed=True)

    # Door segments
    for y in [8, 11, 14]:
        _poly(ui, cx, cy, k, [(5, y), (19, y)], c, w * 0.5)

    # Vehicle inside
    _poly(ui, cx, cy, k, [
        (8, 16), (8, 12), (10, 11), (14, 9), (17, 9), (19, 11.5), (19.5, 12), (19.5, 16)
    ], c, w, closed=True)

    _poly(ui, cx, cy, k, [
        (10.5, 12), (11.5, 9.5), (15.5, 9.5), (16.5, 12)
    ], c, w * 0.8, closed=True)

    _dot(ui, cx, cy, k, 9.5, 17, 1.2, c)
    _dot(ui, cx, cy, k, 17.5, 17, 1.2, c)


def wd_radio(ui, cx, cy, k, c, w):
    """Radio/audio icon - frequency waves."""
    # Base
    _poly(ui, cx, cy, k, [
        (7, 10), (17, 10), (17, 16), (7, 16)
    ], c, w, closed=True)

    # Waveform
    points = []
    for i in range(12):
        x = 7 + i
        y = 13 + 2.5 * math.sin(i * 0.8)
        points.append((x, y))
    _poly(ui, cx, cy, k, points, c, w)

    # Frequency text
    _poly(ui, cx, cy, k, [
        (8.5, 11), (9.5, 10), (10.5, 11), (9.5, 12)
    ], c, w, closed=True)


def wd_phone(ui, cx, cy, k, c, w):
    """Phone/communication icon."""
    _poly(ui, cx, cy, k, [
        (6, 4), (18, 4), (18, 20), (6, 20)
    ], c, w, closed=True)

    # Screen
    _poly(ui, cx, cy, k, [
        (7.5, 5.5), (16.5, 5.5), (16.5, 18.5), (7.5, 18.5)
    ], c, w, closed=True)

    # Signal bars
    for i, h in enumerate([4, 6, 8]):
        _poly(ui, cx, cy, k, [
            (15 + i * 1.2, 19 - h), (15.8 + i * 1.2, 19 - h),
            (15.8 + i * 1.2, 19), (15 + i * 1.2, 19)
        ], c, w)


def wd_key(ui, cx, cy, k, c, w):
    """Key/access icon."""
    # Key head
    _poly(ui, cx, cy, k, [
        (11, 7), (13, 7), (13, 5), (14.5, 5), (14.5, 7), (16.5, 7),
        (15.5, 8), (12.5, 8)
    ], c, w, closed=True)

    # Key shaft
    _poly(ui, cx, cy, k, [
        (12.5, 8), (12.5, 18)
    ], c, w * 1.2)

    # Key teeth
    _poly(ui, cx, cy, k, [
        (12.5, 12), (14.5, 12), (14.5, 13), (12.5, 13)
    ], c, w)
    _poly(ui, cx, cy, k, [
        (12.5, 14), (13.5, 14), (13.5, 15), (12.5, 15)
    ], c, w)
    _poly(ui, cx, cy, k, [
        (12.5, 16), (14.5, 16), (14.5, 17), (12.5, 17)
    ], c, w)

    # Corner brackets on head
    _poly(ui, cx, cy, k, [
        (11, 5), (12, 5), (11, 5), (11, 6),
        (15, 5), (14, 5), (15, 5), (15, 6),
    ], c, w * 0.7)


# ========================================================================
# ICON REGISTRY
# ========================================================================

WD_ICONS = {
    "wd_hack": wd_hack,
    "wd_profile": wd_profile,
    "wd_camera": wd_camera,
    "wd_distract": wd_distract,
    "wd_block": wd_block,
    "wd_trigger": wd_trigger,
    "wd_vehicle": wd_vehicle,
    "wd_track": wd_track,
    "wd_pistol": wd_pistol,
    "wd_rifle": wd_rifle,
    "wd_grenade": wd_grenade,
    "wd_lure": wd_lure,
    "wd_drone": wd_drone,
    "wd_rc": wd_rc,
    "wd_exit": wd_exit,
    "wd_settings": wd_settings,
    "wd_map": wd_map,
    "wd_garage": wd_garage,
    "wd_radio": wd_radio,
    "wd_phone": wd_phone,
    "wd_key": wd_key,
}


def draw_wd(ui, name: str, cx: float, cy: float, s: float, color, width=None):
    """Draw a Watch Dogs themed icon."""
    fn = WD_ICONS.get(name)
    if fn is None:
        # Fallback to regular icons
        from .icons import draw
        draw(ui, name, cx, cy, s, color, width)
        return
    k = s / 24.0
    fn(ui, cx, cy, k, color, width if width is not None else max(1.2, 1.7 * k))