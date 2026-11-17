"""Low-floor city bus and cab-over box truck (replaces the old box-and-cylinder versions).

Bus: 11 m two-door low-floor transit bus (side/front/rear elevations supplied): big
windscreen, destination blind, continuous side glazing, a front door ahead of the front axle
and a second door amidships on the kerb (-y) side, real wheel arches, roof A/C pod.
Truck: Isuzu N-series style cab-over, 4.5 m wheelbase, ribbed 2.2 m box with swing doors.
The rear axle is a *dual* wheel set that sits below the box floor, so both axles are visible
from both sides (the old model hid its rear wheels inside the box).
Metres, +x forward, +y left, z = 0 is the road.
"""
import numpy as np

from .vehicle_models import (
    ONE, GLASS, TRIM, P_PAINT, P_GLASS, P_CHROME, P_TRIM, P_MATTE, P_PIANO, P_FIXED, P_HEAD,
    P_TAIL, P_LEFT, P_RIGHT, P_TYRE, P_PLATE, P_METAL, TYRE, RIM, REAR_SPIN,
)
from .vehicle_coachwork import PIANO, MATTE, CHROME, SLAT, LENS
from .vehicle_reference import (
    BUS_SPEC, TRUCK_SPEC, AMBER, _coach, _wheels, _window, _handle, _seams, _mirrors, _plate,
)

HEAD_LENS = (.92, .92, .84)
GREY = (.80, .80, .78)


def _panes(C, x0, x1, z0, z1, side, target=1.25, gap=.03):
    n = max(1, int(round((x1 - x0) / target)))
    e = np.linspace(x0, x1, n + 1)
    for a, b in zip(e[:-1], e[1:]):
        _window(C, [(a + gap, z0), (b - gap, z0), (b - gap, z1), (a + gap, z1)],
                [(a + gap + .04, z0 + .04), (b - gap - .04, z0 + .04),
                 (b - gap - .04, z1 - .04), (a + gap + .04, z1 - .04)], side=side)


def _door(C, x0, x1, side, z0=.46, z1=2.66):
    xm = .5 * (x0 + x1)
    _window(C, [(x0, z0), (x1, z0), (x1, z1), (x0, z1)],
            [(x0 + .04, z0 + .10), (xm - .015, z0 + .10), (xm - .015, z1 - .05), (x0 + .04, z1 - .05)], side=side)
    _window(C, [(x0, z0), (x1, z0), (x1, z1), (x0, z1)],
            [(xm + .015, z0 + .10), (x1 - .04, z0 + .10), (x1 - .04, z1 - .05), (xm + .015, z1 - .05)], side=side)
    if not C.lod:
        C.sides[side].line([(xm, z0), (xm, z1)], .012, P_TRIM, TRIM, .008)
        for x in (xm - .07, xm + .07):
            C.sides[side].line([(x, 1.00), (x, 1.80)], .016, P_METAL, CHROME, .014)    # stainless grab rails


# ======================================================================== bus
def _bus(m, lod):
    # x, half-width, underbody z, sill (belt) z, roof-edge half-width, roof z
    sections = [(-5.300, 1.130, .420, 1.200, 1.020, 2.980), (-5.150, 1.255, .340, 1.150, 1.130, 3.100),
                (-4.600, 1.275, .300, 1.120, 1.185, 3.150), (-1.000, 1.275, .300, 1.120, 1.185, 3.150),
                (3.000, 1.275, .300, 1.120, 1.185, 3.150), (4.500, 1.272, .300, 1.120, 1.178, 3.130),
                (5.000, 1.250, .330, 1.130, 1.140, 3.040), (5.150, 1.220, .360, 1.140, 1.100, 2.960)]
    C = _coach(m, BUS_SPEC, sections, lod,
               dict(base=5.150, exponent=7., peak=.80,
                    bulge=[(.36, .22), (.60, .26), (1.25, .20), (2.00, .13), (2.70, .07), (2.96, .02)]),
               dict(base=-5.300, exponent=7., peak=1.0,
                    bulge=[(.42, .14), (1.0, .12), (2.0, .09), (2.7, .04), (2.98, .01)]),
               sweep=(4.85, 4.85), crown=.045)
    F, R = C.faces[1], C.faces[-1]

    # ---- front fascia: split windscreen, destination blind, lamps, bumper
    F.poly([(-1.17, 1.30), (1.17, 1.30), (1.12, 2.72), (-1.12, 2.72)], P_PIANO, PIANO, .005, mirror=False)
    F.poly([(-1.11, 1.36), (-.025, 1.36), (-.025, 2.67), (-1.07, 2.67)], P_GLASS, GLASS, .011, mirror=False)
    F.poly([(.025, 1.36), (1.11, 1.36), (1.07, 2.67), (.025, 2.67)], P_GLASS, GLASS, .011, mirror=False)
    F.poly([(-.95, 2.76), (.95, 2.76), (.93, 2.94), (-.93, 2.94)], P_PIANO, PIANO, .006, mirror=False)
    F.poly([(-.80, 2.79), (.80, 2.79), (.78, 2.91), (-.78, 2.91)], P_FIXED, (.95, .52, .06), .012, mirror=False)
    for s in (1, -1):
        F.poly([(s * .72, .64), (s * 1.12, .64), (s * 1.12, .90), (s * .72, .90)], P_PIANO, PIANO, .006, mirror=False)
        F.poly([(s * .78, .68), (s * 1.08, .68), (s * 1.08, .86), (s * .78, .86)], P_HEAD, HEAD_LENS, .012, mirror=False)
        F.poly([(s * .72, .94), (s * 1.12, .94), (s * 1.12, 1.03), (s * .72, 1.03)],
               P_LEFT if s > 0 else P_RIGHT, AMBER, .012, mirror=False)
        F.poly([(s * .50, .40), (s * .66, .40), (s * .66, .52), (s * .50, .52)], P_HEAD, HEAD_LENS, .010, mirror=False)
    F.poly([(-1.18, .36), (1.18, .36), (1.18, .60), (-1.18, .60)], P_MATTE, MATTE, .005, mirror=False)
    F.poly([(-.45, .72), (.45, .72), (.45, 1.12), (-.45, 1.12)], P_PIANO, PIANO, .006, mirror=False)  # grille/panel
    _plate(F, .82, width=.52, height=.12)

    # ---- rear: small window, engine louvres, tall lamp clusters
    R.poly([(-.84, 1.70), (.84, 1.70), (.80, 2.52), (-.80, 2.52)], P_PIANO, PIANO, .005, mirror=False)
    R.poly([(-.79, 1.75), (.79, 1.75), (.75, 2.47), (-.75, 2.47)], P_GLASS, GLASS, .011, mirror=False)
    R.poly([(-.95, .62), (.95, .62), (.95, 1.60), (-.95, 1.60)], P_PIANO, PIANO, .006, mirror=False)
    if lod < 2:
        for z in np.linspace(.70, 1.52, 8 if not lod else 4):
            R.surface.line([(-.90, z), (.90, z)], .018, P_MATTE, SLAT, .012)
    for s in (1, -1):
        R.poly([(s * 1.02, .85), (s * 1.17, .85), (s * 1.17, 1.60), (s * 1.02, 1.60)], P_TAIL, LENS, .010, mirror=False)
        R.poly([(s * 1.02, 1.64), (s * 1.17, 1.64), (s * 1.17, 1.80), (s * 1.02, 1.80)],
               P_LEFT if s > 0 else P_RIGHT, AMBER, .010, mirror=False)
    R.poly([(-1.19, .40), (1.19, .40), (1.19, .60), (-1.19, .60)], P_MATTE, MATTE, .005, mirror=False)
    _plate(R, .50, width=.52, height=.12)

    # ---- side glazing, doors, skirt (kerb/doors on the right, -y)
    for s in (1, -1):
        C.sides[s].patch([(-4.95, .50), (4.95, .50), (4.95, .86), (-4.95, .86)], P_FIXED, GREY, .006)
        C.sides[s].patch([(-4.95, 1.04), (4.95, 1.04), (4.95, 1.10), (-4.95, 1.10)], P_TRIM, TRIM, .005)
    _panes(C, -4.95, 4.60, 1.16, 2.70, +1)
    _panes(C, -4.95, 0.05, 1.16, 2.70, -1)
    _panes(C, 1.20, 3.68, 1.16, 2.70, -1)
    _panes(C, 4.78, 5.04, 1.16, 2.70, -1, target=.3)
    _door(C, 0.14, 1.12, -1)
    _door(C, 3.76, 4.72, -1)
    if lod < 2:
        for x in (0.14, 1.12, 3.76, 4.72):
            C.sides[-1].line([(x, .44), (x, 2.70)], .014, P_PIANO, PIANO, .010)
        C.sides[1].patch([(4.20, 1.16), (4.70, 1.16), (4.70, 2.70), (4.20, 2.70)], P_GLASS, GLASS, .012)  # driver window
    _mirrors(C, 5.00, 2.15, truck=True)

    # ---- roof: A/C pod and rear fairing
    m.box(-1.6, 0, 3.30, 3.2, 1.30, .22, GREY, P_FIXED)
    m.box(-1.6, 0, 3.425, 2.6, 1.0, .04, (.62, .63, .65), P_FIXED)
    if lod < 2:
        m.box(3.9, 0, 3.22, .90, .90, .10, GREY, P_FIXED)
    _wheels(C, 'steel')
    return C


# ======================================================================= truck
def _dual_tyre(m, ax, s, lod):
    """Second (inner) tyre of a dual rear wheel."""
    sp = TRUCK_SPEC
    wr, rr, hw = sp.tyre_radius, sp.rim_radius, sp.tyre_width / 2
    yc = s * (sp.track / 2 - sp.tyre_width - .012)
    n = (24, 12, 8)[lod]
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bead = rr * .965 / wr
    profile = ((-.88, bead), (-1., .89), (-.72, 1.), (.72, 1.), (1., .89), (.88, bead))
    rings = [np.column_stack((ax + wr * r * np.sin(a), np.full(n, yc + s * u * hw),
                              wr + wr * r * np.cos(a))) for u, r in profile]
    start = len(m.V)
    m.loft(np.stack(rings), TYRE, P_TYRE, cap_first=False, cap_last=False)
    m.disc((ax, yc + s * (hw - .006), wr), rr * .93, (0, s, 0), RIM, P_METAL, n)
    m.mark_animation(start, (ax, yc, wr), REAR_SPIN)


def _truck(m, lod):
    sp = TRUCK_SPEC
    ax_f, ax_r = sp.axles
    # box body (x < 1.88) then the cab (x > 1.90); the step between them shows the chassis
    sections = [(-3.600, 1.060, 1.000, 2.900, 1.020, 3.000), (-3.500, 1.100, 1.000, 2.950, 1.060, 3.050),
                (1.800, 1.100, 1.000, 2.950, 1.060, 3.050), (1.860, 1.100, 1.000, 2.950, 1.060, 3.050),
                (1.900, .955, .560, 1.420, .880, 2.280), (2.300, .958, .540, 1.420, .880, 2.320),
                (3.000, .958, .530, 1.420, .880, 2.325), (3.400, .955, .500, 1.410, .870, 2.310),
                (3.600, .940, .440, 1.400, .850, 2.240)]
    C = _coach(m, sp, sections, lod,
               dict(base=3.600, exponent=5., peak=.50,
                    bulge=[(.44, .10), (.60, .085), (1.30, .045), (2.10, .02), (2.24, .0)]),
               dict(base=-3.600, exponent=6., peak=1.5,
                    bulge=[(1.0, .05), (2.0, .04), (3.0, .01)]),
               sweep=(3.30, 3.40), crown=.03)
    F, R = C.faces[1], C.faces[-1]

    # ---- cab front: tall windscreen, visor, grille, lamps, bumper
    F.poly([(-.88, 1.34), (.88, 1.34), (.84, 2.14), (-.84, 2.14)], P_PIANO, PIANO, .005, mirror=False)
    F.poly([(-.83, 1.39), (-.02, 1.39), (-.02, 2.10), (-.80, 2.10)], P_GLASS, GLASS, .011, mirror=False)
    F.poly([(.02, 1.39), (.83, 1.39), (.80, 2.10), (.02, 2.10)], P_GLASS, GLASS, .011, mirror=False)
    F.poly([(-.88, 2.14), (.88, 2.14), (.86, 2.24), (-.86, 2.24)], P_PIANO, PIANO, .006, mirror=False)
    F.poly([(-.52, .78), (.52, .78), (.52, 1.22), (-.52, 1.22)], P_PIANO, PIANO, .006, mirror=False)
    if lod < 2:
        for z in np.linspace(.84, 1.16, 5 if not lod else 3):
            F.surface.line([(-.48, z), (.48, z)], .016, P_MATTE, SLAT, .011)
        F.surface.line([(-.50, 1.0), (.50, 1.0)], .026, P_CHROME, CHROME, .014)
    for s in (1, -1):
        F.poly([(s * .56, .80), (s * .92, .80), (s * .92, 1.06), (s * .56, 1.06)], P_PIANO, PIANO, .006, mirror=False)
        F.poly([(s * .60, .83), (s * .89, .83), (s * .89, 1.00), (s * .60, 1.00)], P_HEAD, HEAD_LENS, .012, mirror=False)
        F.poly([(s * .56, 1.10), (s * .92, 1.10), (s * .92, 1.19), (s * .56, 1.19)],
               P_LEFT if s > 0 else P_RIGHT, AMBER, .012, mirror=False)
    F.poly([(-.94, .38), (.94, .38), (.94, .64), (-.94, .64)], P_MATTE, MATTE, .005, mirror=False)
    _plate(F, .52, width=.46, height=.11)

    # ---- box rear: swing doors, hinges, handles
    R.poly([(-1.04, 1.06), (1.04, 1.06), (1.04, 2.96), (-1.04, 2.96)], P_PAINT, ONE, .004, mirror=False)
    if lod < 2:
        R.surface.line([(0, 1.04), (0, 2.96)], .014, P_TRIM, TRIM, .008)
        for x in (-1.0, 1.0):
            R.surface.line([(x, 1.04), (x, 2.96)], .016, P_MATTE, MATTE, .008)
        for x in (-.12, .12):
            R.surface.line([(x, 1.55), (x, 2.30)], .020, P_METAL, CHROME, .012)
        for z in (1.3, 2.0, 2.7):
            for x in (-1.0, 1.0):
                R.surface.line([(x * .97, z - .06), (x * .97, z + .06)], .03, P_METAL, CHROME, .012)
    for s in (1, -1):
        m.box(-3.60, s * 1.03, 3.00, .06, .14, .05, (.95, .55, .08), P_FIXED)         # roof marker lamps

    # ---- side glass, doors, handles, ribbed box
    for s in (1, -1):
        _window(C, [(3.42, 1.38), (2.30, 1.38), (2.30, 2.20), (3.20, 2.20)],
                [(3.34, 1.45), (2.38, 1.45), (2.38, 2.13), (3.17, 2.13)], side=s)
        if lod < 2:
            C.sides[s].line([(2.26, .66), (2.26, 2.20)], .012, P_TRIM, TRIM, .008)
            C.sides[s].line([(3.46, .66), (3.46, 2.00)], .012, P_TRIM, TRIM, .008)
        _handle(C, 2.45, 1.30, side=s)
        if lod == 0:
            for z in np.arange(1.26, 2.90, .22):
                C.sides[s].line([(-3.50, z), (1.80, z)], .008, P_TRIM, TRIM, .004)
    if lod < 2:
        C.sides[-1].line([(1.02, 1.12), (1.02, 2.90), (1.62, 2.90), (1.62, 1.12), (1.02, 1.12)], .010, P_TRIM, TRIM, .006)
    _mirrors(C, 3.30, 1.95, truck=True)

    # ---- chassis, bumpers, tank, rear lamps and the wheels (both axles on both sides)
    for s in (1, -1):
        m.box(-.80, s * .44, .74, 5.10, .09, .22, TRIM, P_TRIM)                          # frame rails
        m.box(-3.69, s * .86, .84, .10, .30, .24, TAIL_COL, P_TAIL)
        m.box(-3.73, s * .86, .62, .06, .24, .08, AMBER, P_LEFT if s > 0 else P_RIGHT)
        m.box(ax_r, s * .0, .50, .06, .06, .06, TRIM, P_TRIM)
        m.box(-3.12, s * .93, 1.10, .60, .03, .10, (.80, .10, .08), P_FIXED)             # reflective tape
    m.box(-3.64, 0, .56, .20, 2.10, .22, TRIM, P_TRIM)                                   # underride bar
    _plate(R, .56, width=.46, height=.11)
    m.box(-3.74, 0, .56, .03, .46, .11, (.86, .86, .80), P_PLATE)
    m.box(-.2, -.84, .72, 1.20, .52, .46, (.55, .56, .58), P_METAL)                      # bare aluminium fuel tank
    m.box(.9, .74, .70, .50, .30, .34, MATTE, P_MATTE)                                   # battery/air tank
    for x in (ax_f, ax_r):
        m.box(x, 0, .50, .12, 1.4, .12, TRIM, P_TRIM)                                    # axle
    _wheels(C, 'truck')
    for s in (1, -1):
        _dual_tyre(m, ax_r, s, lod)
    return C


TAIL_COL = (.60, .06, .06)
