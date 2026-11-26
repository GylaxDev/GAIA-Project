"""Procedural vehicle templates (one instanced mesh per vehicle kind).

Vertex layout (10 floats): pos3, normal3, colour3, part.   +x is forward, +z up,
z = 0 is the road surface.  Part ids (read by shaders/vehicle.py) are the per-vertex
*material id* - one instanced draw call shades every material of a car:

     0 instance paint (metallic clear-coat, takes the per-vehicle colour)
     1 glass (tinted, semi-reflective)           10 matte dark-grey plastic (arches, cladding,
     2 tyre                                          skirts, mirror bases, lower valances)
     3 headlight (warm projector, night-emissive) 11 LED signature (cool white; DRL by day,
     4 taillight (red, night-emissive)                night-emissive)
     5 chrome (legacy wheel parts are alloy)     12 piano-black gloss (pillars, grille mesh,
     6 black trim / rubber                            lamp housings)
     7 licence plate                              13 panoramic roof glass
     8 fixed-colour paint                         14 cool-white LED head / fog light
     9 satin alloy / unplated metal
                                                  15 ground glow quad (translucent, LAST)
                                                   16/17 left/right indicator lenses
                                                   22/23 lamp covers, 24 clear cabin windows

Near templates end with 12 ground-glow indices. ``glass_count`` indices immediately
after ``opaque_count`` hold transmissive lamp covers/windows at every LOD.
"""
import numpy as np

SEDAN, SUV, COMPACT, BUS, TRUCK, TAXI, VAN, PICKUP, X1 = range(9)
KINDS = 9

GLASS = (0.03, 0.04, 0.05)
TYRE = (0.025, 0.025, 0.027)
RIM = (0.62, 0.64, 0.68)
TRIM = (0.025, 0.025, 0.028)
HEAD = (0.90, 0.90, 0.80)
TAIL = (0.60, 0.06, 0.06)
PLATE = (0.86, 0.86, 0.80)
GLOW = (1.00, 0.92, 0.72)
WHITE_BODY = (0.88, 0.88, 0.85)
ONE = (1.0, 1.0, 1.0)

# material ids (vertex attribute `aPart`)
P_PAINT, P_GLASS, P_TYRE, P_HEAD, P_TAIL, P_CHROME, P_TRIM, P_PLATE, P_FIXED = range(9)
P_METAL = 9                      # satin alloy / bare metal; never a chrome mirror
P_MATTE, P_LED, P_PIANO, P_SUNROOF = 10, 11, 12, 13
P_HEAD_LED = 14                  # LED projectors, night-only; distinct from daytime signatures
LED_COLOR = (0.72, 0.88, 1.00)   # cool-blue LED signature / DRL (uniform uLedColor)
P_GLOW = 15                      # translucent pass, explicitly tested by material id
P_LEFT, P_RIGHT = 16, 17
P_LENS_CLEAR, P_LENS_RED = 22, 23   # translucent lamp lenses (own blended pass; never in the opaque range)
P_WINDOW = 24                     # real cabin transmission, using the same two-pass glazing draw
AMBER = (.94, .38, .035)
# Separate vertex metadata: pivot xyz + animation role. Material ids stay integral.
STATIC, REAR_SPIN, FRONT_SPIN, FRONT_STEER = range(4)


class VehicleTemplate(tuple):
    """Backward-compatible (vertices, indices, opaque_count) plus wheel metadata."""
    def __new__(cls, vertices, indices, opaque_count, animation=None, glass_count=0):
        v = np.ascontiguousarray(vertices, np.float32)
        i = np.ascontiguousarray(indices, np.uint32)
        result = super().__new__(cls, (v, i, int(opaque_count)))
        result.glass_count = int(glass_count)   # translucent lens/window indices after the opaque range
        result.animation = (np.zeros((len(v), 4), np.float32) if animation is None
                            else np.ascontiguousarray(animation, np.float32))
        if result.animation.shape != (len(v), 4):
            raise ValueError('vehicle animation metadata must align with vertices')
        return result


def animation_metadata(template):
    """Old inspection templates are static; runtime templates have explicit pivots."""
    animation = getattr(template, 'animation', None)
    return np.zeros((len(template[0]), 4), np.float32) if animation is None else animation


class _Mesh:
    def __init__(self):
        self.V, self.I, self.A = [], [], []

    def mark_animation(self, start, pivot, role, stop=None):
        """Tag only a wheel's vertices, not its static arch or brake caliper."""
        stop = len(self.V) if stop is None else stop
        self.A[start:stop] = [(*pivot, float(role))] * (stop - start)

    def template(self, opaque=None, compact=False, glass=0):
        v, i, a = np.asarray(self.V, np.float32), np.asarray(self.I, np.uint32), np.asarray(self.A, np.float32)
        if compact:
            used, remap = np.unique(i, return_inverse=True)
            v, a, i = v[used], a[used], remap.astype(np.uint32)
        return VehicleTemplate(v, i, len(i) if opaque is None else opaque, a, glass)

    # ---- primitives ---------------------------------------------------
    def _v(self, p, n, col, part):
        self.V.append((float(p[0]), float(p[1]), float(p[2]),
                       float(n[0]), float(n[1]), float(n[2]), *col, float(part)))
        self.A.append((0., 0., 0., float(STATIC)))
        return len(self.V) - 1

    def quad(self, a, b, c, d, col, part, hint=None):
        """Flat quad with a normal forced to face `hint` (the renderer has no culling)."""
        a, b, c, d = (np.asarray(q, np.float64) for q in (a, b, c, d))
        n = np.cross(b - a, c - a) + np.cross(c - a, d - a)
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-12 else np.array([0, 0, 1.0])
        if hint is not None and float(n @ np.asarray(hint, np.float64)) < 0:
            n = -n
        i = [self._v(q, n, col, part) for q in (a, b, c, d)]
        self.I += [i[0], i[1], i[2], i[0], i[2], i[3]]

    def triangle(self, a, b, c, col, part, hint=None):
        """Flat triangle, without a collapsed fourth corner/degenerate triangle."""
        a, b, c = (np.asarray(q, np.float64) for q in (a, b, c))
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            return
        n /= ln
        if hint is not None and float(n @ np.asarray(hint, np.float64)) < 0:
            b, c, n = c, b, -n
        self.I.extend(self._v(q, n, col, part) for q in (a, b, c))

    def box(self, cx, cy, cz, sx, sy, sz, col, part):
        x0, x1, y0, y1, z0, z1 = (cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2,
                                  cz - sz / 2, cz + sz / 2)
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        for (a, b, cc, d, n) in ((0, 3, 2, 1, (0, 0, -1)), (4, 5, 6, 7, (0, 0, 1)),
                                 (0, 4, 7, 3, (-1, 0, 0)), (1, 2, 6, 5, (1, 0, 0)),
                                 (0, 1, 5, 4, (0, -1, 0)), (3, 7, 6, 2, (0, 1, 0))):
            i = [self._v(c[k], n, col, part) for k in (a, b, cc, d)]
            self.I += [i[0], i[1], i[2], i[0], i[2], i[3]]

    def disc(self, c, radius, normal, col, part, segs=12, axis="y"):
        """Flat disc facing `normal` (axis-aligned)."""
        c = np.asarray(c, np.float64)
        ang = np.linspace(0, 2 * np.pi, segs, endpoint=False)
        ctr = self._v(c, normal, col, part)
        ring = []
        for a in ang:
            if axis == "y":
                p = c + np.array([radius * np.sin(a), 0.0, radius * np.cos(a)])
            else:
                p = c + np.array([0.0, radius * np.sin(a), radius * np.cos(a)])
            ring.append(self._v(p, normal, col, part))
        for k in range(segs):
            self.I += [ctr, ring[k], ring[(k + 1) % segs]]

    def loft(self, rings, col, part, cap_first=True, cap_last=True):
        """Smooth-shaded closed loft.  rings: (K, N, 3) array."""
        P = np.asarray(rings, np.float64)
        K, N, _ = P.shape
        ctr = P.mean(axis=1)
        base = len(self.V)
        for k in range(K):
            k0, k1 = max(k - 1, 0), min(k + 1, K - 1)
            for i in range(N):
                dk = P[k1, i] - P[k0, i]
                di = P[k, (i + 1) % N] - P[k, (i - 1) % N]
                n = np.cross(dk, di)
                ln = np.linalg.norm(n)
                n = n / ln if ln > 1e-12 else np.array([0, 0, 1.0])
                if float(n @ (P[k, i] - ctr[k])) < 0:
                    n = -n
                self._v(P[k, i], n, col, part)
        for k in range(K - 1):
            for i in range(N):
                j = (i + 1) % N
                a, b = base + k * N + i, base + k * N + j
                c, d = base + (k + 1) * N + j, base + (k + 1) * N + i
                self.I += [a, b, c, a, c, d]
        if cap_first:
            self._cap(P[0], ctr[0], (-1.0, 0.0, 0.0), col, part)
        if cap_last:
            self._cap(P[-1], ctr[-1], (1.0, 0.0, 0.0), col, part)

    def _cap(self, ring, centre, n, col, part):
        c = self._v(centre, n, col, part)
        idx = [self._v(p, n, col, part) for p in ring]
        for i in range(len(idx)):
            self.I += [c, idx[i], idx[(i + 1) % len(idx)]]


# ------------------------------------------------------------------ helpers
def _ring(x, zb, zt, hw, n, expo):
    zm, hh = (zt + zb) * 0.5, (zt - zb) * 0.5
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / expo
    y = hw * np.sign(c) * np.abs(c) ** e
    z = zm + hh * np.sign(s) * np.abs(s) ** e
    return np.stack([np.full(n, x), y, z], axis=1)


class _Body:
    """A lofted body plus the surface queries needed to hang details on it."""

    def __init__(self, mesh, sections, col=ONE, part=0, n=14, expo=4.0):
        self.m, self.n, self.expo = mesh, n, expo
        self.sec = np.asarray(sections, np.float64)        # x, zb, zt, hw
        mesh.loft(np.stack([_ring(x, zb, zt, hw, n, expo) for x, zb, zt, hw in self.sec]),
                  col, part)

    def at(self, x):
        s = self.sec
        return (float(np.interp(x, s[:, 0], s[:, 1])), float(np.interp(x, s[:, 0], s[:, 2])),
                float(np.interp(x, s[:, 0], s[:, 3])))

    def side_y(self, x, z):
        zb, zt, hw = self.at(x)
        zm, hh = (zt + zb) * 0.5, max((zt - zb) * 0.5, 1e-6)
        u = min(abs((z - zm) / hh), 1.0)
        return hw * (1.0 - u ** self.expo) ** (1.0 / self.expo)



def _side_window(m, body, s, x0, x1, z0, z1, part=1, col=GLASS, inset=0.0, nx=5, nz=3):
    """Glass panel hugging the side of a lofted body.

    The panel is tessellated and every sample takes its y from the body surface,
    so it follows the curvature instead of cutting through it.
    """
    xs = np.linspace(x0, x1, nx + 1)
    grid = []
    for x in xs:
        ztop = min(z1, body.at(x)[1] - 0.10)
        zs = np.linspace(z0, max(ztop, z0 + 0.05), nz + 1)
        grid.append([(x, s * (body.side_y(x, z) + 0.005 - inset), z) for z in zs])
    for i in range(nx):
        for j in range(nz):
            m.quad(grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1], col, part, hint=(0, s, 0))


def _on_top(m, body, x0, x1, width, off=0.012, col=GLASS, part=1):
    """Panel lying on the top surface of a lofted body between x0 and x1 (e.g. a windshield)."""
    z0, z1 = body.at(x0)[1] + off, body.at(x1)[1] + off
    m.quad((x0, -width, z0), (x0, width, z0), (x1, width, z1), (x1, -width, z1), col, part,
           hint=(0.6, 0, 1))

def _wheel(m, body, x, z, r, half_w, sign, center_y=None):
    """Tyre (lathe loft) + alloy rim + hub + arch shadow on one side of the car."""
    y_surf = body.side_y(x, z) if body is not None else 0.9
    yc = sign * (y_surf - 0.04) if center_y is None else sign * center_y
    start = len(m.V)
    u = np.array([-1.0, -0.88, 0.88, 1.0]) * half_w
    rad = np.array([0.74, 1.0, 1.0, 0.74]) * r
    a = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    rings = [np.stack([x + rr * np.sin(a), np.full(10, yc + uu), z + rr * np.cos(a)], axis=1)
             for uu, rr in zip(u, rad)]
    m.loft(np.stack(rings), TYRE, 2, cap_first=False, cap_last=False)
    yo = yc + sign * half_w * 1.0
    m.disc((x, yo, z), r * 0.66, (0, sign, 0), RIM, P_METAL, 10)
    for k in range(5):                                    # dark gaps between spokes
        ak = 2 * np.pi * (k + 0.5) / 5
        m.disc((x + 0.40 * r * np.sin(ak), yo + sign * 0.004, z + 0.40 * r * np.cos(ak)),
               r * 0.16, (0, sign, 0), TRIM, 6, 5)
    m.disc((x, yo + sign * 0.006, z), r * 0.17, (0, sign, 0), (0.20, 0.20, 0.22), 6, 8)
    m.mark_animation(start, (x, yc, z), FRONT_SPIN if x > 0 else REAR_SPIN)
    if body is not None:                                  # wheel-arch shadow
        m.disc((x, sign * (y_surf + 0.004), z + 0.05), r * 1.16, (0, sign, 0), TRIM, 6, 12)


def _seam(m, body, x, z0, z1, sign, w=0.010):
    zs = np.linspace(z0, z1, 4)
    pts = [(x, sign * (body.side_y(x, z) + 0.004), z) for z in zs]
    for i in range(3):
        a, b = pts[i], pts[i + 1]
        m.quad((a[0] - w, a[1], a[2]), (a[0] + w, a[1], a[2]),
               (b[0] + w, b[1], b[2]), (b[0] - w, b[1], b[2]), TRIM, 6, hint=(0, sign, 0))


def _plate(m, x, y, z, facing):
    m.box(x, y, z, 0.03, 0.36, 0.13, PLATE, 7)


def _lights(m, L, y_head, z_head, y_tail, z_tail, head=(0.30, 0.12), tail=(0.30, 0.11), fwd=None, rear=None):
    fx = L * 0.5 if fwd is None else fwd
    rx = -L * 0.5 if rear is None else rear
    for s in (-1, 1):
        m.box(fx - 0.07, s * y_head, z_head, 0.16, head[0], head[1], HEAD, 3)
        m.box(rx + 0.07, s * y_tail, z_tail, 0.16, tail[0], tail[1], TAIL, 4)
        part = P_LEFT if s > 0 else P_RIGHT
        m.box(fx - .025, s * y_head, z_head - head[1] * .55, .09, head[0] * .7, .055, AMBER, part)
        m.box(rx + .025, s * y_tail, z_tail - tail[1] * .55, .09, tail[0] * .7, .055, AMBER, part)


def _glasshouse(m, wb, wt, fb, rf, rr, rb, col=ONE, pillar=0.52, roof_part=0, rear_glass=True):
    """Windshield, roof, rear window, side windows and a black B pillar.

    fb/rf/rr/rb are (x, z): windshield base, roof front, roof rear, rear-window base.
    """
    ctr = np.array([(fb[0] + rb[0]) * 0.5, 0.0, (fb[1] + rf[1]) * 0.5])
    def hint(*pts):
        return np.mean(pts, axis=0) - ctr
    # windshield / rear glass / roof
    q = [(fb[0], -wb, fb[1]), (fb[0], wb, fb[1]), (rf[0], wt, rf[1]), (rf[0], -wt, rf[1])]
    m.quad(*q, GLASS, 1, hint=hint(*q))
    if rear_glass:
        q = [(rb[0], -wb, rb[1]), (rb[0], wb, rb[1]), (rr[0], wt, rr[1]), (rr[0], -wt, rr[1])]
        m.quad(*q, GLASS, 1, hint=hint(*q))
    q = [(rf[0], -wt, rf[1]), (rf[0], wt, rf[1]), (rr[0], wt, rr[1]), (rr[0], -wt, rr[1])]
    m.quad(*q, col, roof_part, hint=(0, 0, 1))
    # body-coloured A and C pillars laid just above the glass edges
    def _strip(base, top, wid, glass_hint, off=0.010):
        for s_ in (-1, 1):
            p0 = np.array([base[0], s_ * base[1], base[2]]); p1 = np.array([top[0], s_ * top[1], top[2]])
            inward = np.array([0.0, -s_ * wid, 0.0])
            n = np.array(glass_hint, np.float64); n /= np.linalg.norm(n)
            q_ = [p0 + n * off, p1 + n * off, p1 + inward + n * off, p0 + inward + n * off]
            m.quad(*q_, col, roof_part, hint=glass_hint)
    _strip((fb[0], wb, fb[1]), (rf[0], wt, rf[1]), 0.075, (-(rf[1] - fb[1]), 0.0, abs(fb[0] - rf[0])))
    if rear_glass:
        _strip((rb[0], wb, rb[1]), (rr[0], wt, rr[1]), 0.12, ((rr[1] - rb[1]), 0.0, abs(rb[0] - rr[0])))
    # side windows, split at the pillar
    for s in (-1, 1):
        def lerp(t):
            bx = fb[0] + (rb[0] - fb[0]) * t; bz = fb[1] + (rb[1] - fb[1]) * t
            tx = rf[0] + (rr[0] - rf[0]) * t; tz = rf[1] + (rr[1] - rf[1]) * t
            wb_t = wb; wt_t = wt
            return (bx, s * wb_t, bz), (tx, s * wt_t, tz)
        for (t0, t1, part, c) in ((0.0, pillar - 0.025, 1, GLASS),
                                  (pillar - 0.025, pillar + 0.025, 6, TRIM),
                                  (pillar + 0.025, 1.0, 1, GLASS)):
            b0, t_0 = lerp(t0); b1, t_1 = lerp(t1)
            m.quad(b0, b1, t_1, t_0, c, part, hint=(0, s, 0.35))
        # thin roof-edge trim above the glass line
        b0, t_0 = lerp(0.0); b1, t_1 = lerp(1.0)
        m.quad(t_0, t_1, (t_1[0], t_1[1] - s * 0.02, t_1[2] + 0.015),
               (t_0[0], t_0[1] - s * 0.02, t_0[2] + 0.015), TRIM, 6, hint=(0, s, 0.5))


def _mirrors(m, x, y, z, col=ONE):
    for s in (-1, 1):
        m.box(x, s * y, z, 0.12, 0.10, 0.08, col, 0)
        m.box(x - 0.02, s * (y - 0.07), z - 0.04, 0.05, 0.06, 0.04, TRIM, 6)


# ------------------------------------------------------------------ passenger cars
def _passenger(m, kind):
    if kind in (SEDAN, TAXI):
        L, wr, hwid = 4.70, 0.32, 1.84
        sec = [(-2.35, .40, .86, .66), (-2.25, .30, .97, .82), (-1.75, .24, 1.00, .90),
               (-0.60, .22, .96, .92), (0.60, .22, .95, .92), (1.15, .23, 1.00, .91),
               (1.85, .25, .88, .88), (2.22, .30, .76, .80), (2.35, .38, .62, .68)]
        axle_f, axle_r = 1.45, -1.35
        cab = dict(wb=.84, wt=.73, fb=(1.02, .93), rf=(0.30, 1.46), rr=(-1.18, 1.45), rb=(-1.68, .98))
        belt, seams = 0.95, (1.05, -0.28, -1.30)
    elif kind == COMPACT:
        L, wr, hwid = 3.90, 0.30, 1.74
        sec = [(-1.95, .40, .90, .66), (-1.88, .26, .98, .80), (-1.50, .23, .98, .86),
               (-0.30, .21, .92, .87), (0.55, .21, .92, .87), (0.95, .22, .94, .86),
               (1.55, .24, .82, .84), (1.85, .29, .72, .76), (1.95, .36, .60, .64)]
        axle_f, axle_r = 1.20, -1.15
        cab = dict(wb=.80, wt=.71, fb=(0.80, .89), rf=(0.12, 1.42), rr=(-1.42, 1.40), rb=(-1.82, .90))
        belt, seams = 0.91, (0.80, -0.45, -1.30)
    elif kind == SUV:
        L, wr, hwid = 4.85, 0.37, 1.96
        sec = [(-2.42, .50, 1.02, .78), (-2.34, .34, 1.12, .92), (-1.90, .30, 1.16, .97),
               (-0.30, .28, 1.12, .98), (0.80, .28, 1.14, .97), (1.30, .30, 1.18, .96),
               (2.05, .32, 1.06, .93), (2.38, .38, .90, .86), (2.43, .46, .74, .76)]
        axle_f, axle_r = 1.50, -1.45
        cab = dict(wb=.90, wt=.82, fb=(1.15, 1.10), rf=(0.38, 1.78), rr=(-2.08, 1.76), rb=(-2.28, 1.20))
        belt, seams = 1.12, (1.00, -0.10, -1.30)
    body = _Body(m, sec, ONE, 0, n=14, expo=5.0)
    _glasshouse(m, **cab, pillar=0.50 if kind != SUV else 0.46)
    cab_belt = cab["fb"][1]
    for s in (-1, 1):
        for xs in seams:
            _seam(m, body, xs, body.at(xs)[0] + 0.14, belt - 0.03, s)
    # front / rear fascia
    zb_f, zt_f, hw_f = body.at(L * 0.5 - 0.05)
    _lights(m, L, hw_f * 0.64, zb_f + (zt_f - zb_f) * 0.62, hw_f * 0.64, zt_f - 0.17 if kind != SUV else zt_f - 0.25,
            head=(0.30, 0.12), tail=(0.32, 0.12))
    m.box(L * 0.5 - 0.035, 0, zb_f + (zt_f - zb_f) * 0.35, 0.08, hw_f * 0.9, (zt_f - zb_f) * 0.26, TRIM, 6)   # grille
    m.box(L * 0.5 - 0.04, 0, zb_f + 0.05, 0.10, hw_f * 1.6, 0.11, TRIM, 6)                                   # lower valance
    zb_r, zt_r, hw_r = body.at(-L * 0.5 + 0.05)
    m.box(-L * 0.5 + 0.04, 0, zb_r + 0.06, 0.10, hw_r * 1.7, 0.11, TRIM, 6)                                  # rear valance
    _plate(m, L * 0.5 - 0.02, 0, zb_f + 0.15, 1)
    _plate(m, -L * 0.5 + 0.02, 0, zb_r + 0.20, -1)
    _mirrors(m, cab["fb"][0] - 0.12, cab["wb"] + 0.12, belt + 0.10)
    for s in (-1, 1):
        for ax in (axle_f, axle_r):
            _wheel(m, body, ax, wr, wr, 0.12 if kind != SUV else 0.13, s)
    if kind == SUV:                                       # roof rails + rear spoiler lip
        for s in (-1, 1):
            m.box((cab["rf"][0] + cab["rr"][0]) * 0.5, s * (cab["wt"] - 0.02), cab["rf"][1] + 0.035,
                  abs(cab["rf"][0] - cab["rr"][0]) * 0.96, 0.04, 0.04, TRIM, 6)
    if kind == TAXI:
        m.box(-0.05, 0, cab["rf"][1] + 0.10, 0.62, 0.24, 0.16, (1.0, 0.95, 0.65), 8)
        m.box(-0.05, 0, cab["rf"][1] + 0.02, 0.50, 0.20, 0.04, TRIM, 6)
    if kind in (SEDAN, COMPACT):                          # shark-fin antenna
        m.box(cab["rr"][0] + 0.15, 0, cab["rr"][1] + 0.035, 0.12, 0.04, 0.05, TRIM, 6)
    return L


def _pickup(m):
    L, wr = 5.40, 0.38
    sec = [(-0.55, .52, 1.10, .86), (-0.50, .34, 1.12, .96), (0.20, .30, 1.14, .98),
           (0.90, .30, 1.22, .98), (1.55, .32, 1.16, .96), (2.40, .36, 1.00, .93),
           (2.68, .42, .86, .86), (2.70, .50, .72, .74)]
    body = _Body(m, sec, ONE, 0, n=14, expo=4.0)
    # cab glass: ends at the bed wall
    _glasshouse(m, wb=.92, wt=.84, fb=(1.40, 1.16), rf=(0.66, 1.82), rr=(-0.30, 1.80), rb=(-0.52, 1.20), pillar=0.55)
    # open bed: floor, side walls, front wall, tailgate
    bed_x0, bed_x1, hw = -2.70, -0.50, 0.92
    m.box((bed_x0 + bed_x1) / 2, 0, 0.78, bed_x1 - bed_x0, 2 * hw - 0.08, 0.06, TRIM, 6)
    for s in (-1, 1):
        m.box((bed_x0 + bed_x1) / 2, s * (hw - 0.03), 0.98, bed_x1 - bed_x0, 0.10, 0.52, ONE, 0)
        m.box((bed_x0 + bed_x1) / 2, s * (hw - 0.03), 0.74, bed_x1 - bed_x0, 0.12, 0.26, ONE, 0)
        m.box((bed_x0 + bed_x1) / 2, s * (hw - 0.095), 1.00, bed_x1 - bed_x0, 0.03, 0.40, (0.08, 0.08, 0.09), 6)
    m.box(bed_x1 - 0.05, 0, 1.00, 0.10, 2 * hw, 0.52, ONE, 0)
    m.box(bed_x0 + 0.04, 0, 0.98, 0.08, 2 * hw, 0.50, ONE, 0)
    m.box(bed_x0 - 0.02, 0, 0.42, 0.10, 2 * hw - 0.1, 0.16, TRIM, 6)
    m.box(bed_x0 + 0.02, 0, 0.50, 0.05, 2 * hw - 0.2, 0.10, RIM, 5)                  # rear bumper
    for s in (-1, 1):
        m.box(bed_x0 + 0.03, s * (hw - 0.14), 0.92, 0.06, 0.14, 0.32, TAIL, 4)
        m.box(-2.02, s * (hw + 0.005), 0.60, 0.4, 0.03, 0.34, TRIM, 6)               # bed-side shadow
    zb_f, zt_f, hw_f = body.at(L / 2 - 2.70 - 0.05 + 2.70 - 0.05)
    zb_f, zt_f, hw_f = body.at(2.65)
    _lights(m, 0, hw_f * 0.66, zb_f + (zt_f - zb_f) * 0.62, 0, 0, head=(0.32, 0.13), fwd=2.70)
    m.box(2.665, 0, zb_f + (zt_f - zb_f) * 0.35, 0.08, hw_f * 0.9, (zt_f - zb_f) * 0.30, TRIM, 6)
    m.box(2.66, 0, zb_f + 0.05, 0.10, hw_f * 1.7, 0.12, TRIM, 6)
    _plate(m, 2.69, 0, zb_f + 0.16, 1)
    _plate(m, bed_x0 - 0.06, 0, 0.62, -1)
    _mirrors(m, 1.28, 1.02, 1.28)
    for s in (-1, 1):
        for xs in (1.15, 0.05):
            _seam(m, body, xs, 0.46, 1.10, s)
        for ax in (1.75, -1.55):
            _wheel(m, body, ax, wr, wr, 0.14, s)
    return 5.40


# ------------------------------------------------------------------ large vehicles
def _van(m):
    sec = [(-2.80, .55, 2.10, .86), (-2.74, .42, 2.34, 1.00), (-1.50, .38, 2.40, 1.02),
           (0.95, .38, 2.40, 1.02), (1.45, .38, 2.00, 1.00), (2.20, .40, 1.40, .96),
           (2.78, .46, 1.05, .86), (2.80, .54, .96, .76)]
    body = _Body(m, sec, ONE, 0, n=14, expo=6.0)
    _on_top(m, body, 1.47, 2.18, 0.84)
    for s_ in (-1, 1):
        _side_window(m, body, s_, 0.30, 1.52, 1.50, 2.12)
        _seam(m, body, 0.22, 0.55, 2.2, s_); _seam(m, body, -0.35, 0.55, 2.2, s_)
        _wheel(m, body, 1.85, 0.36, 0.36, 0.12, s_); _wheel(m, body, -1.75, 0.36, 0.36, 0.13, s_)
    zb, zt, hw = body.at(2.7)
    _lights(m, 0, hw * 0.62, 0.75, 0, 0, fwd=2.8, head=(0.30, 0.16))
    m.box(2.78, 0, 0.62, 0.08, hw * 1.0, 0.24, TRIM, 6); m.box(2.78, 0, 0.45, 0.10, hw * 1.8, 0.14, TRIM, 6)
    _plate(m, 2.82, 0, 0.45, 1)
    for s in (-1, 1):
        m.box(-2.78, s * 0.78, 0.95, 0.06, 0.12, 0.62, TAIL, 4)
    m.box(-2.80, 0, 0.55, 0.12, 2.0, 0.14, TRIM, 6); _plate(m, -2.86, 0, 0.60, -1)
    _mirrors(m, 1.45, 1.12, 1.9, WHITE_BODY)
    return 5.6


def _bus(m):
    L, W, wr = 11.0, 2.55, 0.50
    sec = [(-5.50, .55, 3.00, 1.10), (-5.42, .42, 3.12, 1.26), (-4.0, .38, 3.20, 1.28),
           (4.5, .38, 3.20, 1.28), (5.38, .42, 3.12, 1.24), (5.50, .55, 2.95, 1.12)]
    body = _Body(m, sec, ONE, 0, n=14, expo=7.0)
    for s in (-1, 1):
        y = 1.285
        n_w = 8
        xs = np.linspace(-4.7, 4.3, n_w + 1)
        for i in range(n_w):
            _side_window(m, body, s, xs[i] + 0.06, xs[i + 1] - 0.06, 1.40, 2.68)
        m.box(0, s * 1.282, 1.0, 10.8, 0.01, 0.06, TRIM, 6)
        m.box(0, s * 1.282, 0.74, 10.8, 0.01, 0.30, (0.62, 0.62, 0.60), 8)
        for ax in (3.8, -3.6):
            _wheel(m, body, ax, wr, wr, 0.20, s)
        m.box(5.41, s * 0.85, 0.80, 0.10, 0.30, 0.24, HEAD, 3)
        m.box(-5.41, s * 0.95, 1.00, 0.08, 0.20, 0.50, TAIL, 4)
        part = P_LEFT if s > 0 else P_RIGHT
        m.box(5.47, s * .85, .62, .06, .24, .08, AMBER, part)
        m.box(-5.47, s * .95, .68, .06, .18, .08, AMBER, part)
    m.quad((5.52, -1.10, 2.72), (5.52, 1.10, 2.72), (5.52, 1.08, 1.35), (5.52, -1.08, 1.35), GLASS, 1, hint=(1, 0, 0))
    m.quad((5.46, -0.60, 3.08), (5.46, 0.60, 3.08), (5.52, 0.60, 2.78), (5.52, -0.60, 2.78), (0.95, 0.80, 0.30), 8, hint=(1, 0, 0.4))
    m.quad((-5.52, -1.0, 2.7), (-5.52, 1.0, 2.7), (-5.52, 1.0, 1.7), (-5.52, -1.0, 1.7), GLASS, 1, hint=(-1, 0, 0))
    m.box(-0.5, 0, 3.28, 3.0, 1.5, 0.14, (0.55, 0.55, 0.57), 8)                        # roof A/C
    m.box(5.45, 0, 0.55, 0.12, 2.3, 0.18, TRIM, 6); _plate(m, 5.52, 0, 0.58, 1)
    m.box(-5.46, 0, 0.55, 0.12, 2.3, 0.18, TRIM, 6); _plate(m, -5.52, 0, 0.62, -1)
    _mirrors(m, 5.25, 1.45, 2.3)
    return L


def build_template(kind):
    m = _Mesh()
    if kind == SUV:                                   # Ford Explorer ST blueprint
        from .vehicle_coachwork import build_explorer
        build_explorer(m)
    elif kind in (SEDAN, TAXI, PICKUP, VAN, COMPACT, BUS, TRUCK, X1):  # reference fleet (COMPACT = supplied 2012 Prius)
        from .vehicle_reference import build_reference
        build_reference(m, kind)
    else:
        raise ValueError(f'unknown vehicle kind {kind}')

    # translucent ground-glow quads (night headlight pools) - must stay last
    for (x0, x1, y) in ((2.5, 7.0, 3.0), (3.0, 11.0, 1.5)):
        base = len(m.V)
        for (px, py) in ((x0, -y), (x1, -y), (x1, y), (x0, y)):
            m._v((px, py, .08), (0, 0, 1), GLOW, P_GLOW)
        m.I += [base, base + 1, base + 2, base, base + 2, base + 3]
    glass = getattr(m, 'glass_indices', 0)   # layout: [opaque][translucent lenses][12 glow indices]
    opaque_count = len(m.I) - 12 - glass
    return m.template(opaque_count, glass=glass)


def build_vehicle_lod(template, level, kind=None):
    """Simplify a vehicle without changing the instanced vertex format.

    Reference-fleet distance models rebuild the shell at lower resolution,
    then fit glazing and details to it. Other kinds retain material-preserving
    clustering; all distance models omit the translucent ground-glow pass.
    """
    if kind == SUV:
        from .vehicle_explorer import build_explorer_lod
        return build_explorer_lod(level)
    if kind in (SEDAN, TAXI, PICKUP, VAN, COMPACT, BUS, TRUCK, X1):
        from .vehicle_reference import build_reference_lod
        return build_reference_lod(kind, level)
    if level not in (1, 2):
        raise ValueError('vehicle LOD level must be 1 or 2')
    v, idx, opaque = template
    animation = animation_metadata(template)
    tri = idx[:opaque].reshape(-1, 3)
    part = v[tri[:, 0], 9].astype(np.int32)
    keep = (np.isin(part, (0, 1, 2, 3, 4, 8, 11, 13, P_HEAD_LED, P_LEFT, P_RIGHT))
            | (animation[tri[:, 0], 3] != STATIC))
    tri = tri[keep]
    used = np.unique(tri)
    src = v[used]
    anim = animation[used]
    cell = .22 if level == 1 else .55
    resolution = np.where(anim[:, 3] == STATIC, cell, .10 if level == 1 else .16)
    # Never cluster separate wheels together or average a spinning rim with an arch.
    key = np.column_stack((np.rint(src[:, :3] / resolution[:, None]).astype(np.int32),
                           src[:, 9].astype(np.int32), np.rint(anim[:, :3] * 10000).astype(np.int32),
                           anim[:, 3].astype(np.int32)))
    _, remap = np.unique(key, axis=0, return_inverse=True)
    n = int(remap.max()) + 1
    out = np.zeros((n, 10), np.float32)
    counts = np.bincount(remap)
    for j in range(10): out[:, j] = np.bincount(remap, weights=src[:, j], minlength=n) / counts
    out_anim = np.zeros((n, 4), np.float32)
    for j in range(4): out_anim[:, j] = np.bincount(remap, weights=anim[:, j], minlength=n) / counts
    out[:, 3:6] /= np.maximum(np.linalg.norm(out[:, 3:6], axis=1, keepdims=True), 1e-6)
    mapping = np.zeros(len(v), np.uint32); mapping[used] = remap
    t = mapping[tri]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    p = out[t, :3].astype(np.float64)
    t = t[np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) > 1e-10]
    _, first = np.unique(np.sort(t, axis=1), axis=0, return_index=True)
    indices = np.ascontiguousarray(t[np.sort(first)].ravel(), np.uint32)
    return VehicleTemplate(out, indices, len(indices), out_anim)
