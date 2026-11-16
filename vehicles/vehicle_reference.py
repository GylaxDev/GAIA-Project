"""Unbranded 2020 Camry, 2020 F-150 SuperCrew and Chevrolet Express templates.

Metres, +x forward. These are procedural interpretations of the supplied
photographs/orthographics. Each distance mesh rebuilds its own curved shell
before fitting glass and trim, preserving open arches and the pickup bed.
"""
from dataclasses import dataclass

import numpy as np

from .vehicle_models import (
    _Mesh, SEDAN, TAXI, PICKUP, VAN, COMPACT, BUS, TRUCK, X1, ONE, GLASS, TYRE, RIM, TRIM, PLATE,
    P_PAINT, P_GLASS, P_TYRE, P_HEAD, P_TAIL, P_CHROME, P_TRIM,
    P_PLATE, P_FIXED, P_MATTE, P_LED, P_PIANO, P_SUNROOF, P_HEAD_LED,
    REAR_SPIN, FRONT_SPIN, FRONT_STEER, P_LEFT, P_RIGHT,
)
from .vehicle_coachwork import _unique, _emit_grid, _const_hint, PIANO, MATTE, SLAT, CHROME, LAMP, LENS, SUNG
from .vehicle_explorer import _Curve, _Surface, _ExplorerCoach, _annulus

WHITE, AMBER = (.93, .95, .97), (.94, .38, .035)


@dataclass(frozen=True)
class ReferenceDimensions:
    name: str
    length: float
    width: float
    height: float
    wheelbase: float
    front_axle: float
    track: float
    tyre_width: float
    tyre_radius: float
    rim_radius: float
    clearance: float
    mirror_width: float
    arch_margin: float = .053       # wheel-arch gap around the tyre
    floor_half: float = .69         # half-width of the flat underfloor
    open_axles: tuple = ()          # axles with no wheel arch (rear axle under a box body)
    rear_track: float = 0.          # 0 = same as front track

    def track_at(self, ax):
        return self.rear_track if (self.rear_track and ax != self.front_axle) else self.track

    @property
    def axles(self):
        return self.front_axle, self.front_axle - self.wheelbase


CAMRY = ReferenceDimensions('2020 Camry', 4.885, 1.840, 1.445, 2.825, 1.455,
                            1.590, .215, .33415, .21590, .145, 2.100, arch_margin=.038)
# Supplied 2012 Prius (XW30), fitted to GAIA's existing envelope and 2700 mm
# wheelbase. The reference's larger alloy-wheel style uses 215/45R17 tyres.
PRIUS = ReferenceDimensions('2012 Prius', 4.460, 1.745, 1.490, 2.700, 1.300,
                            1.525, .215, .31265, .2159, .140, 1.960, arch_margin=.040, floor_half=.62)
F150 = ReferenceDimensions('2020 F-150 SuperCrew / 5.5 ft bed', 5.890, 2.029, 1.961,
                           3.683, 1.985, 1.717, .275, .40735, .22860, .235, 2.459)
EXPRESS = ReferenceDimensions('Chevrolet Express / regular cargo', 5.691, 2.012, 2.159,
                              3.429, 1.925, 1.721, .245, .38695, .20320, .195, 2.440)
# Low-floor 11 m city bus (reference: two-door urban bus) and an Isuzu N-series style
# cab-over box truck (reference: 4.5 m wheelbase, 2.2 m box). Rear box-truck axle is
# dual-tyred and sits below the box floor, so it has no wheel arch.
BUS_SPEC = ReferenceDimensions('city bus', 11.0, 2.55, 3.15, 5.65, 3.05,
                               2.10, .30, .50, .2858, .30, 2.95, arch_margin=.075, floor_half=.95)
TRUCK_SPEC = ReferenceDimensions('cab-over box truck', 7.5, 2.20, 3.05, 4.50, 2.62,
                                 1.68, .215, .40, .2032, .30, 2.55, arch_margin=.065, floor_half=.60,
                                 open_axles=(2.62 - 4.50,))
# Supplied 2020 BMW X1 xDrive18d (F48) FBX, scaled to the 2670 mm wheelbase and baked by
# tools/bake_x1_model.py. Values are measured from the baked mesh (mirror-to-mirror 2051 mm,
# 235 mm tyres on a 683 mm rolling diameter). Origin at mid-length.
X1_SPEC = ReferenceDimensions('2020 BMW X1 xDrive18d', 4.465, 1.833, 1.610, 2.670, 1.3718,
                              1.5595, .235, .34126, .2529, .209, 2.051, arch_margin=.050,
                              floor_half=.66)
DIMENSIONS = {X1: X1_SPEC, SEDAN: CAMRY, TAXI: CAMRY, PICKUP: F150, VAN: EXPRESS, COMPACT: PRIUS,
              BUS: BUS_SPEC, TRUCK: TRUCK_SPEC}
# Area-weighted centres of the fitted forward/rear-facing lenses. Express and the
# box truck use warm sealed beams; the other reference models use cool LED lenses.
LAMP_ANCHORS = {
    X1: ((1.894, .666, .772), (-2.000, .629, .960), True),
    SEDAN: ((2.346, .669, .873), (-2.360, .679, .986), True),
    TAXI: ((2.346, .669, .873), (-2.360, .679, .986), True),
    PICKUP: ((2.847, .831, 1.046), (-2.824, .890, 1.043), True),
    VAN: ((2.715, .768, 1.091), (-2.664, .837, 1.780), False),
    COMPACT: ((1.80462, .61448, .82012), (-2.04324, .50523, .90945), True),
    BUS: ((5.230, .930, .780), (-5.250, .950, 1.050), True),
    TRUCK: ((3.680, .740, .890), (-3.770, .900, .830), False),
}


def _coach(m, spec, sections, lod, front, rear, *, crown=.035, open_top=None, sweep=None):
    # Sections: x, maximum half-width, underbody z, belt z, roof-edge y, crown z.
    a = np.asarray(sections, np.float64)
    curves = [_Curve(a[:, 0], a[:, k]) for k in range(1, 6)]
    width, bottom, belt, roof_width, roof = curves
    T = dict(zb=bottom, zm=lambda x: bottom(x) + .44 * (belt(x) - bottom(x)),
             Wm=width, Wsh=lambda x: width(x) - .003, Wb=lambda x: width(x) - .018,
             zbe=belt, shdrop=.070, Wr=roof_width, zc=roof,
             crown=lambda x: crown, ridge=lambda x: 0., sculpt=None)
    max_gap = (.19, .46, 1.1)[lod]
    stations = _unique(a[:, 0], *[np.linspace(x, y, int(np.ceil((y - x) / max_gap)) + 1)
                                  for x, y in zip(a[:-1, 0], a[1:, 0])])
    arch = spec.tyre_radius + spec.arch_margin
    angles = np.linspace(-np.pi / 2, np.pi / 2, (17, 9, 5)[lod])
    xs = _unique(stations, *[ax + arch * np.sin(angles) for ax in spec.axles],
                 *[[ax - arch - .002, ax + arch + .002] for ax in spec.axles])
    xs = xs[(xs >= a[0, 0]) & (xs <= a[-1, 0])]
    C = _ExplorerCoach(m, xs, T, lod, axles=spec.axles, radius=spec.tyre_radius,
                       arch=arch, faces={1: front, -1: rear},
                       sweep=sweep or (a[-2, 0], -a[1, 0]), open_top=open_top)
    C.spec = spec
    C.emit()
    C.faces[1].emit()
    C.faces[-1].emit()
    rear_floor, front_floor = spec.axles[1] - .45, spec.axles[0] + .40
    fh = spec.floor_half
    m.quad((rear_floor, -fh, spec.clearance), (front_floor, -fh, spec.clearance),
           (front_floor, fh, spec.clearance), (rear_floor, fh, spec.clearance),
           TRIM, P_TRIM, hint=(0, 0, -1))
    return C


def _wheels(C, style):
    m, spec = C.m, C.spec
    wr, rr = spec.tyre_radius, spec.rim_radius
    n = (32, 16, 12)[C.lod]
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for ax in spec.axles:
        for s in (1, -1):
            if not any(abs(ax - o) < 1e-6 for o in spec.open_axles):
                arc = np.linspace(-np.pi / 2, np.pi / 2, (25, 9, 5)[C.lod])
                def contour(radius, offset):
                    p = [(ax - radius, spec.clearance)]
                    p += [(ax + radius * np.sin(t), wr + radius * np.cos(t)) for t in arc]
                    p += [(ax + radius, spec.clearance)]
                    return np.array([(x, s * (C.side_y(x, z) + offset), z) for x, z in p])
                inside = contour(C.arch, .008)
                outside = contour(C.arch + (.020 if style in ('sedan', 'prius') else .032), .003)
                _emit_grid(m, np.stack((inside, outside)), P_PAINT, ONE, _const_hint((0, s, .1)))
                back = inside.copy()
                back[:, 1] = s * (spec.track_at(ax) / 2 - spec.tyre_width / 2 - .018)
                _emit_grid(m, np.stack((back, inside)), P_MATTE, TRIM, _const_hint((0, s, -.3)))
                center = np.tile((ax, back[0, 1], wr), (len(back), 1))
                _emit_grid(m, np.stack((center, back)), P_MATTE, TRIM, _const_hint((0, s, 0)))
            # Rounded radial tyre: maximum radius is on the tread, not on a
            # flat polygon over the coachwork. The shell opening is real.
            hw, yc = spec.tyre_width / 2, s * spec.track_at(ax) / 2
            start = len(m.V)
            caliper = None
            bead = rr * .965 / wr
            profile = ((-.88, bead), (-1., .89), (-.72, 1.), (.72, 1.), (1., .89), (.88, bead))
            rings = [np.column_stack((ax + wr * r * np.sin(angle), np.full(n, yc + s * u * hw),
                                       wr + wr * r * np.cos(angle))) for u, r in profile]
            m.loft(np.stack(rings), TYRE, P_TYRE, cap_first=False, cap_last=False)
            yo = yc + s * (hw - .006)
            m.disc((ax, yo - s * .046, wr), rr, (0, s, 0), TRIM, P_TRIM, n)
            if not C.lod:
                m.disc((ax, yo - s * .038, wr), rr * .76, (0, s, 0), (.28, .29, .30), P_FIXED, n)
                caliper = len(m.V)
                m.box(ax + rr * .53, yo - s * .030, wr + .018, .055, .030, .105, SLAT, P_FIXED)
            _annulus(m, ax, yo, wr, rr * .93, rr * 1.036, s, RIM, P_CHROME, n)
            if style == 'steel':
                m.disc((ax, yo + s * .001, wr), rr * .94, (0, s, 0), RIM, P_CHROME, n)
                for k in range(8):
                    t = 2 * np.pi * k / 8
                    m.disc((ax + rr * .69 * np.sin(t), yo + s * .003, wr + rr * .69 * np.cos(t)),
                           rr * .145, (0, s, 0), TRIM, P_TRIM, 8 if not C.lod else 5)
            else:
                pairs = 10 if style == 'sedan' else 5 if style == 'prius' else 6
                for k in range(pairs):
                    t = 2 * np.pi * k / pairs
                    for branch in ((-1, 1) if style == 'truck' else (0,)):
                        tip = t + branch * .14
                        pts = [(.21 * rr, t - .18), (.96 * rr, tip - .055),
                               (.96 * rr, tip + .055), (.21 * rr, t + .18)]
                        m.quad(*[(ax + r * np.sin(q), yo + s * .005, wr + r * np.cos(q)) for r, q in pts],
                               RIM, P_CHROME, hint=(0, s, 0))
            m.disc((ax, yo + s * .009, wr), rr * .245, (0, s, 0), RIM, P_CHROME, 16 if not C.lod else 8)
            if not C.lod:
                lugs = 5 if style in ('sedan', 'prius') else 6 if style == 'truck' else 8
                for k in range(lugs):
                    t = k * 2 * np.pi / lugs
                    m.disc((ax + rr * .18 * np.sin(t), yo + s * .011, wr + rr * .18 * np.cos(t)),
                            .005, (0, s, 0), TRIM, P_TRIM, 6)
            front = ax == spec.front_axle
            m.mark_animation(start, (ax, yc, wr), FRONT_SPIN if front else REAR_SPIN)
            if caliper is not None:
                m.mark_animation(caliper, (ax, yc, wr), FRONT_STEER if front else 0, caliper + 24)


def _window(C, frame, pane, top=False, side=None):
    if top:
        C.top(frame, P_PIANO, PIANO, .005)
        C.top(pane, P_GLASS, GLASS, .011)
    else:
        surfaces = C.sides.values() if side is None else (C.sides[side],)
        for surface in surfaces:
            surface.patch(frame, P_PIANO, PIANO, .005)
            surface.patch(pane, P_GLASS, GLASS, .011)


def _handle(C, x, z, *, vertical=False, side=None):
    if C.lod:
        return
    for surface in (C.sides.values() if side is None else (C.sides[side],)):
        rx, rz = (.032, .080) if vertical else (.092, .024)
        surface.ellipse((x, z), rx, rz, P_PIANO, PIANO, .005, 12)
        surface.line([(x, z - .055), (x, z + .055)] if vertical else [(x - .067, z), (x + .067, z)],
                     .016, P_CHROME if C.spec is CAMRY else P_MATTE,
                     CHROME if C.spec is CAMRY else MATTE, .014)


def _seams(C, paths, side=None):
    if C.lod:
        return
    for surface in (C.sides.values() if side is None else (C.sides[side],)):
        for path in paths:
            surface.line(path, .0032, P_TRIM, TRIM, .003)


def _mirrors(C, x, z, truck=False):
    m = C.m
    hw, hh, depth = (.107, .093, .116) if truck else (.087, .047, .120)
    cy = C.spec.mirror_width / 2 - hw
    shape = np.array([(-1, -.55), (-.83, -.94), (.72, -.94), (1, -.54),
                      (1, .51), (.70, .94), (-.76, .94), (-1, .55)])
    if C.lod == 2:
        shape = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    for s in (1, -1):
        rings = [np.column_stack((np.full(len(shape), x + dx), s * (cy + shape[:, 0] * hw * scale),
                                   z + shape[:, 1] * hh * scale))
                 for dx, scale in ((-depth, .85), (-depth * .7, 1.), (depth * .5, 1.), (depth, .70))]
        m.loft(np.stack(rings), PIANO, P_PIANO)
        m.box(x + .02, s * (cy - hw * .80), z - hh * .62, .080, .150, .052, MATTE, P_MATTE)
        if C.lod < 2:
            m.quad((x - depth - .002, s * (cy - hw * .72), z - hh * .66),
                   (x - depth - .002, s * (cy + hw * .70), z - hh * .66),
                   (x - depth - .002, s * (cy + hw * .69), z + hh * .68),
                   (x - depth - .002, s * (cy - hw * .72), z + hh * .68), GLASS, P_GLASS, hint=(-1, 0, 0))


def _plate(F, z, width=.305, height=.152):
    for w, h, part, col, off in ((width + .035, height + .030, P_TRIM, TRIM, .006),
                                 (width, height, P_PLATE, PLATE, .024)):
        F.poly([(-w / 2, z - h / 2), (w / 2, z - h / 2),
                (w / 2, z + h / 2), (-w / 2, z + h / 2)], part, col, off, mirror=False)


def _lamp_wrap(C, F, side, slope=.6, reach=.55):
    """One continuous projection chart for a lamp spanning fascia and fender."""
    def chart(p):
        return np.stack((side * p[..., 1] - slope * F.s * (p[..., 0] - F.base), p[..., 2]), axis=-1)
    def select(tri):
        points = tri[:, :, :3]
        p = points.mean(axis=1)
        normal = tri[:, :, 3:6].mean(axis=1)
        facing = F.s * normal[:, 0] + slope * side * normal[:, 1]
        # Exclude nearly horizontal bonnet/boot panels that can share the same
        # chart coordinates as a lamp; retain the rounded shoulder between them.
        return ((p[:, 1] * side > .1) & (F.s * (p[:, 0] - F.base) > -reach)
                & (facing > .15))
    return _Surface(C.m, [*C.indices, *F.indices], chart, (F.s, side, 0), select)


def _band(surface, us, lower, upper, part, col, off):
    for a, b, la, lb, ha, hb in zip(us[:-1], us[1:], lower[:-1], lower[1:], upper[:-1], upper[1:]):
        surface.patch([(a, la), (b, lb), (b, hb), (a, ha)], part, col, off, normal_offset=True)


def _wipers(C, x, y, dx=.10):
    if C.lod:
        return
    for s in (1, -1):
        C.roof.line([(x, s * .06), (x - dx, s * y)], .010, P_TRIM, TRIM, .025)


def _tailpipe(m, x, y, z, lod, radius=.033):
    n = (16, 8, 6)[lod]
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rings = [np.column_stack((np.full(n, xx), y + radius * np.sin(a), z + radius * np.cos(a)))
             for xx in (x + .09, x)]
    m.loft(np.stack(rings), CHROME, P_CHROME, cap_first=False, cap_last=False)
    m.disc((x - .001, y, z), radius * .80, (-1, 0, 0), TRIM, P_TRIM, n, axis='x')


def _camry(m, lod, taxi=False):
    # LE/XLE-style nose in the supplied red-car reference: wide horizontal
    # lower grille, slim upper V and swept lamps, rather than the sport XSE mesh.
    sections = [(-2.285, .881, .275, .966, .822, 1.066), (-2.08, .905, .191, .988, .808, 1.076),
                (-1.82, .918, .153, 1.008, .726, 1.080), (-1.63, .920, .145, 1.021, .708, 1.106),
                (-1.42, .915, .145, 1.032, .685, 1.202), (-1.15, .905, .145, 1.039, .651, 1.365),
                (-.92, .904, .145, 1.032, .636, 1.424), (-.53, .905, .145, 1.012, .644, 1.445),
                (-.12, .908, .145, .984, .660, 1.444), (.23, .910, .145, .964, .672, 1.423),
                (.46, .912, .145, .948, .694, 1.366), (.70, .912, .145, .939, .742, 1.205),
                (.94, .913, .145, .925, .808, 1.045), (1.13, .916, .145, .922, .834, 1.048),
                (1.46, .920, .145, .904, .840, 1.034), (1.81, .911, .175, .884, .834, 1.005),
                (2.11, .902, .208, .853, .827, .971), (2.29, .881, .251, .821, .814, .948)]
    C = _coach(m, CAMRY, sections, lod,
               dict(base=2.190, exponent=3.6, peak=.45,
                    bulge=[(.25, .210), (.45, .2525), (.62, .237), (.77, .219), (.89, .188), (.948, .105)]),
               dict(base=-2.285, exponent=3.6, peak=.50,
                    bulge=[(.275, .085), (.50, .1575), (.72, .140), (.96, .110), (1.066, .0)]),
               sweep=(2.030, 1.970), crown=.032)
    F, R = C.faces[1], C.faces[-1]
    lower = [(-.806, .306), (.806, .306), (.766, .485), (.528, .714), (-.528, .714), (-.766, .485)]
    F.poly(lower, P_PIANO, PIANO, .005, mirror=False)
    if lod < 2:
        F.surface.line([(-.807, .311), (-.762, .486), (-.535, .715)], .016, P_CHROME, CHROME, .010)
        F.surface.line([(.807, .311), (.762, .486), (.535, .715)], .016, P_CHROME, CHROME, .010)
        for z in np.linspace(.337, .673, 9 if not lod else 4):
            F.surface.line([(-.80, z), (.80, z)], .012, P_MATTE, (.24, .25, .27), .012, clip_to=np.asarray(lower))
    F.poly([(-.631, .913), (0, .774), (.631, .913), (0, .843)], P_PIANO, PIANO, .006, mirror=False)
    F.surface.line([(-.610, .901), (0, .822), (.610, .901)], .009, P_CHROME, CHROME, .012)
    for s in (1, -1):
        wrap = _lamp_wrap(C, F, s)
        _band(wrap, (.251, .400, .720, .865, 1.072), (.823, .827, .837, .858, .953),
              (.856, .915, .942, .954, .983), P_PIANO, PIANO, .006)
        _band(wrap, (.298, .414, .720, .865, 1.040), (.841, .845, .854, .875, .954),
              (.850, .896, .922, .934, .972), P_FIXED, (.52, .57, .62), .011)
        for y in (.580, .670, .748):
            F.surface.ellipse((s * y, .873), .032, .016, P_HEAD_LED, LAMP, .017, (12, 8, 6)[lod])
        wrap.line([(.306, .838), (.721, .850), (.865, .871), (1.038, .951)],
                  .008, P_LED, WHITE, .018, normal_offset=True)
        F.surface.line([(s * .786, .902), (s * .846, .914)], .018,
                       P_LEFT if s > 0 else P_RIGHT, AMBER, .024)
        F.surface.line([(s * .672, .382), (s * .770, .383)], .019, P_HEAD_LED, LAMP, .017)
    _plate(F, .606, width=.370, height=.102)
    F.poly([(-.790, .267), (.790, .267), (.809, .287), (-.809, .287)], P_PIANO, PIANO, .006, mirror=False)

    # Distinct notchback: curved roof, sloping rear glass, separate raised boot.
    _window(C, [(1.039, -.786), (1.039, .786), (.317, .644), (.317, -.644)],
            [(1.001, -.749), (1.001, .749), (.346, .610), (.346, -.610)], top=True)
    _window(C, [(-1.649, -.695), (-1.649, .695), (-.991, .601), (-.991, -.601)],
            [(-1.609, -.657), (-1.609, .657), (-1.021, .570), (-1.021, -.570)], top=True)
    _window(C, [(.971, .954), (-.182, .993), (-.143, 1.408), (.288, 1.389)],
            [(.887, .978), (-.140, 1.018), (-.107, 1.375), (.295, 1.355)])
    _window(C, [(-.216, .997), (-1.495, 1.061), (-.960, 1.393), (-.186, 1.410)],
            [(-.258, 1.024), (-1.363, 1.080), (-.945, 1.359), (-.226, 1.377)])
    if lod < 2:
        for surface in C.sides.values():
            surface.line([(-1.116, 1.073), (-.826, 1.365)], .018, P_PIANO, PIANO, .015)
            surface.line([(.957, .951), (-.190, .991), (-1.493, 1.059)], .009, P_CHROME, CHROME, .013)
    _wipers(C, .994, .650, .040)
    _seams(C, [[(.991, .935), (.943, .645), (.901, .234)],
               [(-.188, .978), (-.227, .640), (-.219, .228)],
               [(-1.475, 1.038), (-1.552, .904), (-1.109, .633), (-1.047, .230)]])
    _handle(C, .083, .914)
    _handle(C, -.966, .964)
    _mirrors(C, .862, 1.055)
    C.side([(.875, .172), (-1.02, .172), (-1.044, .239), (.900, .239)], P_MATTE, MATTE, .005)
    if lod < 2:
        for surface in C.sides.values():
            surface.line([(.872, .311), (.330, .343), (-.560, .350), (-1.040, .310)], .008, P_MATTE, MATTE, .004)
    if not lod:
        for s in (1, -1):
            C.roof.line([(1.087, s * .665), (1.61, s * .588), (2.015, s * .510), (2.228, s * .487)],
                        .0028, P_TRIM, TRIM, .003)
        C.sides[1].line([(-1.700, .849), (-1.700, .970), (-1.890, .970), (-1.890, .849), (-1.700, .849)],
                        .003, P_TRIM, TRIM, .004)
        C.roof.line([(-1.708, -.670), (-1.730, 0), (-1.708, .670)], .003, P_TRIM, TRIM, .004)
    for s in (1, -1):
        wrap = _lamp_wrap(C, R, s, reach=.60)
        _band(wrap, (.360, .470, .720, .860, 1.183), (.901, .911, .925, .937, 1.036),
              (.918, 1.039, 1.043, 1.044, 1.057), P_PIANO, PIANO, .006)
        _band(wrap, (.402, .485, .720, .860, 1.141), (.918, .928, .942, .954, 1.036),
              (.920, 1.022, 1.025, 1.027, 1.047), P_TAIL, LENS, .013)
        R.surface.line([(s * .526, .932), (s * .788, .954)], .012, P_FIXED, LAMP, .017)
        R.surface.line([(s * .635, 1.008), (s * .752, 1.014)], .020,
                       P_LEFT if s > 0 else P_RIGHT, AMBER, .024)
        R.surface.line([(s * .650, .412), (s * .813, .425)], .014, P_TAIL, LENS, .013)
    R.poly([(-.342, .612), (.342, .612), (.406, .820), (-.406, .820)], P_MATTE, (.24, .22, .23), .005, mirror=False)
    _plate(R, .729)
    if not lod:
        R.surface.line([(-.525, .945), (-.481, .619), (-.395, .592), (.395, .592), (.481, .619), (.525, .945)],
                       .003, P_TRIM, TRIM, .004)
    R.poly([(-.795, .270), (.795, .270), (.703, .370), (-.703, .370)], P_MATTE, MATTE, .007, mirror=False)
    _tailpipe(m, -2.418, -.627, .283, lod)
    if taxi:
        m.box(-.18, 0, 1.505, .57, .235, .130, (.98, .87, .44), P_FIXED)
        m.box(-.18, 0, 1.448, .48, .195, .035, TRIM, P_TRIM)
    _wheels(C, 'sedan')
    return C


def _f150(m, lod):
    sections = [(-2.795, .947, .397, 1.245, .900, 1.315), (-2.710, 1.000, .317, 1.267, .945, 1.322),
                (-2.36, 1.006, .268, 1.277, .951, 1.322), (-1.698, 1.0145, .235, 1.282, .956, 1.322),
                (-1.07, 1.005, .235, 1.279, .940, 1.322), (-1.025, 1.005, .235, 1.279, .938, 1.322),
                (-.965, 1.004, .235, 1.283, .853, 1.860), (-.88, 1.002, .235, 1.284, .828, 1.939),
                (-.65, 1.000, .235, 1.282, .821, 1.956), (-.10, .999, .235, 1.282, .825, 1.961),
                (.40, 1.000, .235, 1.280, .827, 1.948), (.64, 1.001, .235, 1.260, .832, 1.916),
                (.84, 1.002, .235, 1.240, .859, 1.748), (1.12, 1.004, .235, 1.204, .907, 1.501),
                (1.31, 1.006, .235, 1.180, .934, 1.366), (1.55, 1.010, .235, 1.176, .916, 1.383),
                (1.985, 1.0145, .252, 1.170, .901, 1.365), (2.40, 1.001, .334, 1.130, .897, 1.330),
                (2.72, .978, .390, 1.095, .882, 1.280)]
    C = _coach(m, F150, sections, lod,
               dict(base=2.690, exponent=5., peak=.570,
                    bulge=[(.390, .170), (.570, .255), (.770, .197), (1.10, .202), (1.220, .183), (1.280, .115)]),
               dict(base=-2.795, exponent=5., peak=.430,
                    bulge=[(.397, .135), (.430, .150), (.58, .117), (.72, .052), (1.20, .024), (1.315, .010)]),
               sweep=(2.515, 2.360), open_top=(-2.710, -1.025), crown=.035)
    F, R = C.faces[1], C.faces[-1]
    # 2018-2020 facelift: twin horizontal bars and squared C-shaped lamps.
    grille = [(-.697, .839), (.697, .839), (.702, 1.224), (-.702, 1.224)]
    F.poly(grille, P_CHROME, CHROME, .005, mirror=False)
    F.poly([(-.632, .879), (.632, .879), (.650, 1.188), (-.650, 1.188)], P_PIANO, PIANO, .010, mirror=False)
    for z in (1.005, 1.121):
        F.poly([(-.685, z - .030), (.685, z - .030), (.685, z + .030), (-.685, z + .030)],
               P_CHROME, CHROME, .016, mirror=False)
    if not lod:
        for z in (.911, .951, 1.053, 1.162):
            F.surface.line([(-.625, z), (.625, z)], .008, P_MATTE, SLAT, .014)
    for s in (1, -1):
        wrap = _lamp_wrap(C, F, s)
        _band(wrap, (.591, .770, .945, 1.102), (.832, .836, .852, .901),
              (1.232, 1.241, 1.249, 1.251), P_PIANO, PIANO, .006)
        _band(wrap, (.621, .770, .922, 1.070), (.864, .868, .885, .919),
              (1.207, 1.216, 1.224, 1.227), P_FIXED, (.32, .36, .40), .011)
        for z in (.947, 1.137):
            F.surface.ellipse((s * .829, z), .063, .043, P_HEAD_LED, LAMP, .018, (16, 8, 6)[lod])
        wrap.line([(.636, 1.206), (.778, 1.213), (.902, 1.220), (.907, .885), (.638, .869)],
                  .013, P_LED, WHITE, .020, normal_offset=True)
        F.surface.line([(s * .754, 1.042), (s * .927, 1.046)], .022, P_LEFT if s > 0 else P_RIGHT, AMBER, .018)
        wrap.line([(1.053, .964), (1.053, 1.189)], .022, P_FIXED, AMBER, .016, normal_offset=True)
    F.poly([(-.974, .479), (.974, .479), (.982, .730), (-.982, .730)], P_CHROME, CHROME, .009, mirror=False)
    F.poly([(-.967, .397), (.967, .397), (.963, .471), (-.963, .471)], P_MATTE, MATTE, .011, mirror=False)
    for s in (1, -1):
        F.poly([(s * .534, .538), (s * .837, .540), (s * .836, .685), (s * .537, .684)], P_PIANO, PIANO, .017, mirror=False)
        F.poly([(s * .616, .568), (s * .781, .570), (s * .780, .650), (s * .618, .650)], P_HEAD_LED, LAMP, .020, mirror=False)
    _plate(F, .590)
    _window(C, [(1.249, -.886), (1.249, .886), (.620, .793), (.620, -.793)],
            [(1.211, -.842), (1.211, .842), (.644, .750), (.644, -.750)], top=True)
    _window(C, [(-1.009, -.796), (-1.009, .796), (-.920, .785), (-.920, -.785)],
            [(-.999, -.746), (-.999, .746), (-.932, .739), (-.932, -.739)], top=True)
    # Stepped front-door belt is a key F-150 cue. Use two convex panes.
    _window(C, [(1.203, 1.228), (.817, 1.264), (.253, 1.343), (.257, 1.899), (.603, 1.891)],
            [(1.114, 1.276), (.813, 1.301), (.300, 1.378), (.301, 1.856), (.621, 1.844)])
    _window(C, [(.218, 1.346), (-.884, 1.330), (-.862, 1.899), (.217, 1.901)],
            [(.170, 1.384), (-.832, 1.372), (-.817, 1.854), (.169, 1.857)])
    _wipers(C, 1.202, .733, .061)
    _seams(C, [[(1.256, 1.210), (1.190, .741), (1.154, .340)],
               [(.239, 1.341), (.231, .764), (.225, .332)],
               [(-.934, 1.864), (-.956, 1.314), (-.950, .330)],
               [(-1.019, 1.290), (-1.019, .348)]])
    _handle(C, .429, 1.223)
    _handle(C, -.690, 1.234)
    _mirrors(C, 1.123, 1.377, truck=True)
    for s in (1, -1):
        m.box(.129, s * 1.047, .282, 2.254, .164, .080, PIANO, P_PIANO)
        if not lod:
            for x in (-.608, .800):
                m.box(x, s * 1.050, .326, .570, .147, .014, MATTE, P_MATTE)
        # Open cargo box: floor, inner side panels, rounded wheel-house tubs.
        m.quad((-2.717, s * .913, .742), (-1.027, s * .913, .742),
               (-1.027, s * .913, 1.302), (-2.717, s * .913, 1.302), ONE, P_PAINT, hint=(0, -s, 0))
        m.box(-1.872, s * .952, 1.313, 1.692, .095, .035, MATTE, P_MATTE)
        arc = np.linspace(0, np.pi, (17, 9, 5)[lod])
        tub = [np.column_stack((F150.axles[1] + .493 * np.cos(arc), np.full(len(arc), s * y),
                                 .741 + .384 * np.sin(arc))) for y in (.651, .916)]
        _emit_grid(m, np.stack(tub), P_MATTE, MATTE, _const_hint((0, -s, .5)))
        m.quad((F150.axles[1] - .493, s * .651, .741), (F150.axles[1] + .493, s * .651, .741),
               (F150.axles[1] + .493, s * .651, .752), (F150.axles[1] - .493, s * .651, .752),
               MATTE, P_MATTE, hint=(0, -s, 0))
        # Tub's inboard arched side closes it, rather than revealing the road.
        center = np.tile((F150.axles[1], s * .651, .741), (len(arc), 1))
        _emit_grid(m, np.stack((center, tub[0])), P_MATTE, MATTE, _const_hint((0, -s, 0)))
    m.quad((-2.717, -.913, .742), (-1.027, -.913, .742), (-1.027, .913, .742), (-2.717, .913, .742),
           MATTE, P_MATTE, hint=(0, 0, 1))
    for x in (-2.717, -1.027):
        m.quad((x, -.913, .742), (x, .913, .742), (x, .913, 1.303), (x, -.913, 1.303),
               ONE, P_PAINT, hint=(1 if x < -2 else -1, 0, 0))
    if not lod:
        for y in np.linspace(-.58, .58, 11):
            m.box(-1.872, y, .750, 1.650, .020, .016, MATTE, P_MATTE)
        for s in (1, -1):
            C.roof.line([(1.361, s * .751), (1.880, s * .627), (2.453, s * .610)], .003, P_TRIM, TRIM, .003)
        C.sides[1].line([(-1.211, 1.102), (-1.211, 1.225), (-1.415, 1.225), (-1.415, 1.102), (-1.211, 1.102)],
                        .003, P_TRIM, TRIM, .004)
    R.poly([(-.772, .725), (.772, .725), (.772, 1.266), (-.772, 1.266)], P_PAINT, ONE, .003, mirror=False)
    if not lod:
        R.surface.line([(-.801, 1.297), (-.797, .705), (.797, .705), (.801, 1.297)], .004, P_TRIM, TRIM, .006)
        R.poly([(-.163, 1.167), (.163, 1.167), (.163, 1.234), (-.163, 1.234)], P_PIANO, PIANO, .009, mirror=False)
        R.surface.line([(-.688, .834), (.688, .834)], .008, P_MATTE, SLAT, .005)
    for s in (1, -1):
        R.poly([(s * .811, .785), (s * .979, .807), (s * .982, 1.297), (s * .812, 1.297)], P_PIANO, PIANO, .006, mirror=False)
        for z0, z1 in ((.842, 1.005), (1.116, 1.265)):
            R.poly([(s * .834, z0), (s * .957, z0), (s * .957, z1), (s * .834, z1)], P_TAIL, LENS, .014, mirror=False)
        R.surface.line([(s * .834, 1.065), (s * .956, 1.065)], .047, P_FIXED, LAMP, .014)
        R.surface.line([(s * .840, 1.216), (s * .945, 1.216)], .027,
                       P_LEFT if s > 0 else P_RIGHT, AMBER, .024)
        C.sides[s].patch([(-2.795, .808), (-2.609, .837), (-2.605, 1.288), (-2.795, 1.290)], P_TAIL, LENS, .009)
    R.poly([(-.974, .394), (.974, .394), (.972, .596), (-.972, .596)], P_CHROME, CHROME, .009, mirror=False)
    R.poly([(-.971, .591), (.971, .591), (.957, .657), (-.957, .657)], P_MATTE, MATTE, .012, mirror=False)
    _plate(R, .535)
    m.box(-2.869, 0, .348, .140, .100, .088, TRIM, P_TRIM)
    _tailpipe(m, -2.900, -.862, .400, lod, .042)
    _wheels(C, 'truck')
    return C


def _express(m, lod):
    sections = [(-2.645, .926, .350, 1.219, .829, 2.030), (-2.53, .981, .242, 1.239, .859, 2.128),
                (-2.28, .994, .211, 1.244, .863, 2.151), (-1.504, 1.006, .195, 1.245, .867, 2.159),
                (-.70, 1.001, .195, 1.244, .867, 2.159), (.05, .998, .195, 1.237, .866, 2.155),
                (.69, .995, .195, 1.228, .857, 2.118), (.84, .995, .195, 1.218, .866, 2.047),
                (1.07, .999, .195, 1.212, .892, 1.803), (1.36, 1.000, .195, 1.201, .922, 1.524),
                (1.54, 1.001, .195, 1.190, .932, 1.341), (1.70, 1.003, .195, 1.166, .933, 1.322),
                (1.925, 1.006, .195, 1.130, .934, 1.299), (2.31, .990, .280, 1.092, .918, 1.253),
                (2.61, .952, .373, 1.046, .869, 1.218)]
    C = _coach(m, EXPRESS, sections, lod,
               dict(base=2.605, exponent=4.6, peak=.510,
                    bulge=[(.373, .185), (.510, .2405), (.70, .191), (.84, .104), (1.05, .139), (1.16, .119), (1.218, .075)]),
               dict(base=-2.645, exponent=4.5, peak=.475,
                    bulge=[(.350, .138), (.475, .2005), (.650, .175), (.770, .060), (1.95, .014), (2.030, .005)]),
               crown=.048, sweep=(2.432, 2.20))
    F, R = C.faces[1], C.faces[-1]
    # Short rounded bonnet, split grille and rectangular sealed-beam clusters.
    grille = [(-.540, .814), (.540, .814), (.584, 1.170), (-.584, 1.170)]
    F.poly(grille, P_CHROME, CHROME, .005, mirror=False)
    F.poly([(-.493, .842), (.493, .842), (.537, 1.141), (-.537, 1.141)], P_PIANO, PIANO, .010, mirror=False)
    F.poly([(-.574, .971), (.574, .971), (.574, 1.023), (-.574, 1.023)], P_CHROME, CHROME, .017, mirror=False)
    if not lod:
        for y in np.linspace(-.495, .495, 15):
            F.surface.line([(y, .846), (y, 1.132)], .006, P_MATTE, SLAT, .014)
        for z in (.870, .901, .932, 1.059, 1.092, 1.126):
            F.surface.line([(-.510, z), (.510, z)], .005, P_MATTE, SLAT, .014)
    for s in (1, -1):
        wrap = _lamp_wrap(C, F, s)
        _band(wrap, (.527, .780, .924, 1.069), (.801, .809, .820, .851),
              (1.183, 1.191, 1.194, 1.200), P_PIANO, PIANO, .006)
        F.poly([(s * .617, 1.018), (s * .918, 1.023), (s * .918, 1.162), (s * .622, 1.155)], P_CHROME, CHROME, .011, mirror=False)
        F.poly([(s * .653, 1.041), (s * .874, 1.044), (s * .874, 1.142), (s * .655, 1.136)], P_HEAD, LAMP, .016, mirror=False)
        F.poly([(s * .617, .849), (s * .918, .853), (s * .918, .937), (s * .619, .932)], P_LEFT if s > 0 else P_RIGHT, AMBER, .015, mirror=False)
        wrap.line([(.925, .886), (1.049, .899)], .050, P_FIXED, AMBER, .015, normal_offset=True)
    F.poly([(-.969, .443), (.969, .443), (.969, .673), (-.969, .673)], P_CHROME, CHROME, .008, mirror=False)
    F.poly([(-.970, .591), (.970, .591), (.970, .639), (-.970, .639)], P_MATTE, MATTE, .014, mirror=False)
    F.poly([(-.894, .372), (.894, .372), (.956, .429), (-.956, .429)], P_MATTE, MATTE, .008, mirror=False)
    _plate(F, .530)
    _window(C, [(1.563, -.906), (1.563, .906), (.752, .815), (.752, -.815)],
            [(1.521, -.860), (1.521, .860), (.786, .770), (.786, -.770)], top=True)
    _window(C, [(1.470, 1.264), (.002, 1.294), (.013, 2.065), (.668, 2.057)],
            [(1.380, 1.302), (.049, 1.333), (.056, 2.012), (.684, 2.007)])
    _wipers(C, 1.510, .751, .076)
    _mirrors(C, 1.164, 1.389, truck=True)
    _seams(C, [[(1.559, 1.237), (1.469, .871), (1.178, .665), (1.092, .294),
               (-.066, .294), (-.065, 1.302), (-.055, 2.089)]])
    _handle(C, .037, 1.162, vertical=True)
    # Cargo variant from the photo: stamped panels instead of passenger glass.
    # Right-side barn doors and rear split doors remain individually readable.
    _seams(C, [[(-.149, 2.080), (-.158, .285)], [(-.857, 2.087), (-.857, .289)],
               [(-1.480, 2.087), (-1.483, .985)],
               [(-.158, .284), (-.850, .284), (-1.243, .288)]], side=-1)
    _handle(C, -.733, 1.160, vertical=True, side=-1)
    _handle(C, -1.355, 1.160, vertical=True, side=-1)
    if lod < 2:
        for s, surface in C.sides.items():
            surface.line([(1.237, .784), (.400, .780), (-.850, .781), (-2.498, .825)], .024, P_MATTE, MATTE, .007)
            if not lod:
                panels = [(-2.48, -.19)] if s > 0 else [(-2.47, -1.56), (-1.402, -.925), (-.793, -.216)]
                for lo, hi in panels:
                    surface.line([(lo, 1.320), (hi, 1.320), (hi, 2.008), (lo + .06, 2.008),
                                  (lo, 1.941), (lo, 1.320)], .003, P_TRIM, TRIM, .003)
        for s in (1, -1):
            C.roof.line([(-2.42, s * .762), (-1.80, s * .777), (-.40, s * .777), (.61, s * .765)],
                        .010, P_MATTE, MATTE, .011)
    if not lod:
        C.sides[1].ellipse((-1.103, 1.027), .080, .080, P_TRIM, TRIM, .004, 16)
        C.sides[1].ellipse((-1.103, 1.027), .074, .074, P_PAINT, ONE, .006, 16)
        for s in (1, -1):
            C.roof.line([(1.656, s * .748), (1.985, s * .712), (2.444, s * .667)], .003, P_TRIM, TRIM, .004)
        R.surface.line([(0, .718), (0, 2.008)], .004, P_TRIM, TRIM, .006)
        for s in (1, -1):
            R.surface.line([(s * .047, 1.314), (s * .771, 1.314), (s * .755, 1.951), (s * .059, 1.972)],
                           .003, P_TRIM, TRIM, .005)
            for z in (.911, 1.421):
                R.poly([(s * .773, z), (s * .811, z), (s * .811, z + .077), (s * .773, z + .077)],
                       P_MATTE, MATTE, .010, mirror=False)
    for s in (1, -1):
        wrap = _lamp_wrap(C, R, s)
        _band(wrap, (.779, .827, .884, .973), (1.567, 1.570, 1.581, 1.611),
              (2.008, 2.014, 2.027, 2.043), P_PIANO, PIANO, .006)
        _band(wrap, (.797, .827, .884, .951), (1.596, 1.599, 1.610, 1.630),
              (1.983, 1.989, 2.002, 2.018), P_TAIL, LENS, .014)
        wrap.line([(.799, 1.744), (.948, 1.752)], .036, P_FIXED, LAMP, .019, normal_offset=True)
        R.surface.line([(s * .800, 1.820), (s * .873, 1.820)], .030,
                       P_LEFT if s > 0 else P_RIGHT, AMBER, .024)
    R.poly([(-.953, .378), (.953, .378), (.963, .628), (-.963, .628)], P_CHROME, CHROME, .008, mirror=False)
    R.poly([(-.951, .615), (.951, .615), (.941, .664), (-.941, .664)], P_MATTE, MATTE, .013, mirror=False)
    _plate(R, .976)
    R.poly([(-.272, .906), (-.220, .906), (-.220, 1.063), (-.272, 1.063)], P_PIANO, PIANO, .009, mirror=False)
    _tailpipe(m, -2.776, -.886, .325, lod)
    _wheels(C, 'steel')
    return C


def build_reference(m, kind, lod=0):
    if lod not in (0, 1, 2):
        raise ValueError('Reference vehicle LOD must be 0, 1 or 2')
    start = len(m.I)
    if kind in (SEDAN, TAXI):
        _camry(m, lod, taxi=kind == TAXI)
    elif kind == PICKUP:
        _f150(m, lod)
    elif kind == VAN:
        _express(m, lod)
    elif kind == COMPACT:
        from .vehicle_prius import _prius
        _prius(m, lod)
    elif kind == X1:
        from .vehicle_x1 import _x1
        _x1(m, lod)
    elif kind == BUS:
        from .vehicle_transit import _bus
        _bus(m, lod)
    elif kind == TRUCK:
        from .vehicle_transit import _truck
        _truck(m, lod)
    else:
        raise ValueError('Unsupported reference vehicle kind')
    tri = np.asarray(m.I[start:], np.int64).reshape(-1, 3)
    p = np.asarray(m.V, np.float64)[tri, :3]
    keep = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) > 1e-10
    m.I[start:] = tri[keep].ravel().tolist()
    return DIMENSIONS[kind].length


def build_reference_lod(kind, level):
    if level not in (1, 2):
        raise ValueError('Reference distance LOD must be 1 or 2')
    m = _Mesh()
    build_reference(m, kind, level)
    return m.template(compact=True)
