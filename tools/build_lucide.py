"""Regenerate ``ui/lucide_data.py`` from the Lucide icon set.

Lucide (https://lucide.dev, ISC licence) draws every icon as round-capped,
round-joined strokes on a 24 x 24 grid.  The overlay has no SVG renderer, so
this script flattens each icon's paths, circles, rects, lines and polylines
into plain polylines.  ``Overlay2D.icon`` strokes those polylines with an
analytic distance field at the exact device-pixel size, which reproduces the
Lucide look at any UI scale.

    npm install lucide-static
    pip install svgelements
    python tools/build_lucide.py path/to/node_modules/lucide-static

Add a name to ``WANTED`` to bundle another icon.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from svgelements import (Arc, Circle, CubicBezier, Ellipse, Line, Move, Path as SvgPath, Polygon,
                         Polyline, QuadraticBezier, Rect, SimpleLine, SVG, Close)

WANTED = """
search map map-pin map-pinned locate-fixed navigation navigation-2 compass route signpost layers globe
panel-left panel-left-close panel-left-open maximize minimize plus minus x check chevron-left chevron-right
chevron-up chevron-down arrow-up-right arrow-left arrow-right arrow-up arrow-down corner-up-right settings
sliders-horizontal keyboard info power monitor sparkles camera clock users volume-2 volume-x
layout-dashboard terminal lock lock-open rotate-ccw rotate-cw droplet trash-2 zap cloud play pause car
gamepad-2 music eye eye-off circle-help bookmark star house menu ellipsis refresh-cw download external-link
copy list layout-grid image palette moon sun cpu gauge activity wifi wifi-off bike footprints
person-standing user building-2 trees mountain waves cloud-sun cloud-moon cloudy cloud-drizzle cloud-rain
cloud-rain-wind cloud-snow cloud-lightning cloud-fog cloud-hail cloud-sun-rain cloud-moon-rain snowflake
wind thermometer thermometer-sun thermometer-snowflake droplets sunrise sunset umbrella haze tornado
sun-dim leaf triangle-alert circle-check circle-x circle-alert sun-medium orbit satellite timer calendar
mouse-pointer-2 hand joystick fullscreen sliders-vertical square-terminal hourglass sun-snow bus truck
tree-pine mic radio skip-forward skip-back film sunset book-open flag cloud-off wind-arrow-down
arrow-up-down move-3d box scan-line zoom-in zoom-out crosshair history shield-check gauge-circle
""".split()

TOL = 0.025      # flattening tolerance, in 24-grid units


def _flatten_segment(seg, out):
    """Append points of one path segment (after its start point) to ``out``."""
    if isinstance(seg, Line):
        out.append((seg.end.x, seg.end.y))
        return
    if isinstance(seg, (CubicBezier, QuadraticBezier, Arc)):
        length = max(seg.length(error=1e-3), 1e-6)
        n = max(2, min(48, int(length / 0.55) + 2))
        for i in range(1, n + 1):
            p = seg.point(i / n)
            out.append((p.x, p.y))
        return
    out.append((seg.end.x, seg.end.y))


def flatten(element):
    """Yield ``(closed, [(x, y), ...])`` strokes for one SVG element."""
    path = element if isinstance(element, SvgPath) else SvgPath(element)
    pts, start, closed = [], None, False
    for seg in path:
        if isinstance(seg, Move):
            if len(pts) > 1 or (len(pts) == 1 and closed):
                yield closed, pts
            pts, closed = [(seg.end.x, seg.end.y)], False
            start = (seg.end.x, seg.end.y)
        elif isinstance(seg, Close):
            if pts and start is not None and (abs(pts[-1][0] - start[0]) > 1e-6 or abs(pts[-1][1] - start[1]) > 1e-6):
                pts.append(start)
            closed = True
        else:
            if not pts:
                pts = [(seg.start.x, seg.start.y)]
                start = pts[0]
            _flatten_segment(seg, pts)
    if len(pts) > 1:
        yield closed, pts
    elif len(pts) == 1:
        yield False, pts * 2                       # a dot: zero-length stroke with round caps


def simplify(pts, tol=TOL):
    """Ramer-Douglas-Peucker; keeps the silhouette within ``tol`` grid units."""
    if len(pts) < 3:
        return pts
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    dx, dy = x1 - x0, y1 - y0
    norm = (dx * dx + dy * dy) ** 0.5
    best, idx = -1.0, 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        d = (abs(dy * (px - x0) - dx * (py - y0)) / norm) if norm > 1e-9 else ((px - x0) ** 2 + (py - y0) ** 2) ** 0.5
        if d > best:
            best, idx = d, i
    if best <= tol:
        return [pts[0], pts[-1]]
    return simplify(pts[:idx + 1], tol)[:-1] + simplify(pts[idx:], tol)


def icon_strokes(svg_file: Path):
    svg = SVG.parse(str(svg_file), reify=True)
    strokes = []
    for el in svg.elements():
        if isinstance(el, (SvgPath, Rect, Circle, Ellipse, SimpleLine, Polyline, Polygon)):
            p = el if isinstance(el, SvgPath) else SvgPath(el)
            for closed, pts in flatten(p):
                if isinstance(el, (Rect, Circle, Ellipse, Polygon)):
                    closed = True
                pts = simplify(pts)
                strokes.append((closed, [(round(x, 2), round(y, 2)) for x, y in pts]))
    return strokes


def main(root: Path):
    icons = root / "icons"
    out = {}
    for name in sorted(set(WANTED)):
        f = icons / f"{name}.svg"
        if not f.exists():
            print("missing", name, file=sys.stderr)
            continue
        out[name] = icon_strokes(f)
    target = Path(__file__).resolve().parents[1] / "ui" / "lucide_data.py"
    lines = ['"""Lucide icon outlines (generated by tools/build_lucide.py - do not edit by hand).',
             "",
             "Lucide is (c) Lucide Icons and Contributors, ISC licence, https://lucide.dev .",
             "Each icon is a list of strokes ``(closed, ((x, y), ...))`` on a 24 x 24 grid; they are drawn",
             'round-capped and round-joined, exactly like the original SVGs.',
             '"""',
             "from __future__ import annotations", "", "STROKES = {"]
    for name, strokes in out.items():
        body = ",".join("(%d,(%s))" % (1 if c else 0, ",".join("(%g,%g)" % p for p in pts)) for c, pts in strokes)
        lines.append(f"    {name!r}: ({body},),")
    lines.append("}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {target} ({len(out)} icons, {target.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
