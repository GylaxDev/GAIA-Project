"""Coachwork-style car templates: Ford Explorer ST (SUV) and Toyota Camry (sedan).

Proportions, axle positions, belt / hood / roof heights, grille, lamp and exhaust positions were
measured from the supplied blueprints (silhouette extraction + gridded front / rear / side views),
not eyeballed.  Build order:

    * ``_Coach``  - the skin: one continuous shell lofted through ~20 stations along +x.  Every
                    station is an 11-row half section (rocker, door, shoulder crease, belt,
                    tumblehome, roof/hood crown) mirrored left/right.  Hood, windscreen, roof,
                    back-light and boot are the same surface, so pillars and fastback fall out of
                    the side-view centre-line table.
    * ``_Face``   - front / rear fascia: a height-field patch closing the shell (bumper bulge,
                    grille plane, tailgate rake are tables of x-offset over height).
    * overlays    - glass, pillars, lamps, grille, vents ... are bilinear patches projected onto
                    the skin / fascia a few mm proud (one offset per layer, no z-fighting).

The Explorer's reference dimensions and surface-conforming details are implemented
in ``vehicle_explorer.py``; the common loft remains shared with the Camry.
"""
import numpy as np

from .vehicle_models import (
    ONE, GLASS, TYRE, RIM, TRIM, PLATE,
    P_PAINT, P_GLASS, P_TYRE, P_HEAD, P_TAIL, P_CHROME, P_TRIM, P_PLATE,
    P_MATTE, P_LED, P_PIANO, P_SUNROOF,
)

PIANO = (0.010, 0.010, 0.012)
MATTE = (0.070, 0.073, 0.078)
SLAT = (0.160, 0.165, 0.180)
SUNG = (0.018, 0.022, 0.032)
CHROME = (0.74, 0.76, 0.80)
SKID = (0.52, 0.53, 0.56)
LED = (0.80, 0.90, 1.00)
LAMP = (0.96, 0.96, 0.94)
LENS = (0.72, 0.05, 0.05)
BADGE = (0.05, 0.14, 0.42)
BRAKE_DISC = (0.28, 0.28, 0.30)

O_BASE, O_GLASS, O_DETAIL, O_LIGHT, O_TOP = 0.006, 0.013, 0.018, 0.022, 0.028

FR = np.array([1.0, 0.75, 0.5, 0.25, 0.0])      # top-surface columns: edge -> centre
R_BELT, R_EDGE, N_ROWS = 4, 6, 11


class _Tab:
    def __init__(self, pts):
        a = np.asarray(pts, np.float64)
        self.x, self.y = a[:, 0], a[:, 1]

    def __call__(self, x):
        return float(np.interp(x, self.x, self.y))


def _bell(x, c, w):
    return float(np.exp(-((x - c) / w) ** 2))


def _sstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ grid emitter (welded)
def _emit_grid(m, G, parts, cols, hint, cos_thr=0.88, center_col=None, ridge_x=(1e9, -1e9)):
    """Point grid -> quads with angle-limited smooth normals and welded vertices."""
    G = np.asarray(G, np.float64)
    A, B, _ = G.shape
    n = np.cross(G[1:, 1:] - G[:-1, :-1], G[:-1, 1:] - G[1:, :-1])
    ln = np.linalg.norm(n, axis=2)
    ok = ln > 1e-9
    n = n / np.where(ok, ln, 1.0)[..., None]
    ctr = 0.25 * (G[:-1, :-1] + G[1:, :-1] + G[1:, 1:] + G[:-1, 1:])
    n = np.where(((n * hint(ctr)).sum(-1) < 0)[..., None], -n, n)
    cols = np.broadcast_to(np.asarray(cols, np.float64), (A - 1, B - 1, 3))
    parts = np.broadcast_to(np.asarray(parts), (A - 1, B - 1))
    cache = m.__dict__.setdefault("_weld", {})
    for a in range(A - 1):
        for b in range(B - 1):
            if not ok[a, b] or parts[a, b] < 0:
                continue
            ids = []
            for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                pa, pb = a + da, b + db
                acc = n[a, b].copy()
                for qa, qb in ((pa - 1, pb - 1), (pa - 1, pb), (pa, pb - 1), (pa, pb)):
                    if (qa, qb) == (a, b) or qa < 0 or qb < 0 or qa >= A - 1 or qb >= B - 1 or not ok[qa, qb]:
                        continue
                    if float(n[qa, qb] @ n[a, b]) > cos_thr:
                        acc += n[qa, qb]
                l = np.linalg.norm(acc)
                nv = acc / l if l > 1e-9 else n[a, b]
                if (center_col is not None and pb == center_col
                        and not (ridge_x[0] <= G[pa, pb, 0] <= ridge_x[1])):
                    nv = nv.copy(); nv[1] = 0.0
                    l = np.linalg.norm(nv)
                    nv = nv / l if l > 1e-9 else n[a, b]
                col = cols[a, b]
                key = (tuple(np.round(G[pa, pb], 4)), tuple(np.round(nv, 2)), int(parts[a, b]),
                       tuple(np.round(col, 3)))
                i = cache.get(key)
                if i is None:
                    i = cache[key] = m._v(G[pa, pb], nv, col, float(parts[a, b]))
                ids.append(i)
            m.I += [ids[0], ids[1], ids[2], ids[0], ids[2], ids[3]]


def _const_hint(vec):
    v = np.asarray(vec, np.float64)
    return lambda c: np.broadcast_to(v, c.shape)


# ------------------------------------------------------------------ skin
class _Coach:
    def __init__(self, m, xs, T):
        self.m, self.T = m, T
        self.xs = np.asarray(xs, np.float64)
        self.P = np.stack([self.profile(x) for x in self.xs])      # (K, 11, 2)  (|y|, z)

    def ztop(self, x, f):
        T = self.T
        z = T["zc"](x) - T["ridge"](x) * f - T["crown"](x) * f * f
        if T.get("sculpt") is not None:
            z += T["sculpt"](x, f)
        return z

    def profile(self, x):
        T = self.T
        zb, zm, Wm, Wsh, Wb, zbe, Wr = (T[k](x) for k in ("zb", "zm", "Wm", "Wsh", "Wb", "zbe", "Wr"))
        zte = max(self.ztop(x, 1.0), zbe + 0.035)
        rows = [(Wm * 0.82, zb), (Wm * 0.965, zb + 0.08), (Wm, zm), (Wsh, zbe - T["shdrop"]),
                (Wb, zbe), (Wb - 0.02, zbe + min(0.012, 0.3 * (zte - zbe))), (Wr, zte)]
        for f in FR[1:]:
            rows.append((Wr * f, max(self.ztop(x, f), zte)))
        return np.asarray(rows, np.float64)

    def prof_at(self, x):
        xs = self.xs
        i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
        t = float(np.clip((x - xs[i]) / (xs[i + 1] - xs[i]), 0.0, 1.0))
        return self.P[i] * (1 - t) + self.P[i + 1] * t

    def side_zmax(self, x):
        return float(self.prof_at(x)[R_EDGE, 1]) - 0.004

    def side_y(self, x, z):
        P = self.prof_at(x)[:R_EDGE + 1]
        return float(np.interp(min(z, P[-1, 1] - 0.004), P[:, 1], P[:, 0]))

    def top_z(self, x, y):
        P = self.prof_at(x)
        return float(np.interp(abs(y), P[R_EDGE:, 0][::-1], P[R_EDGE:, 1][::-1]))

    def emit(self, matfn, cos_thr=0.88, ridge_x=(1e9, -1e9), cutouts=()):
        K = len(self.xs)
        zcen = 0.5 * (self.P[:, 0, 1] + self.P[:, -1, 1])
        zc_q = 0.5 * (zcen[:-1] + zcen[1:])[:, None]
        hint = lambda c: np.stack([np.zeros_like(c[..., 0]), c[..., 1], c[..., 2] - zc_q], -1)
        for s in (1, -1):
            G = np.zeros((K, N_ROWS, 3))
            G[:, :, 0] = self.xs[:, None]
            G[:, :, 1] = s * self.P[:, :, 0]
            G[:, :, 2] = self.P[:, :, 1]
            # Open U-shaped wheel apertures, rather than a dark decal over an
            # unbroken body.  Stations at the arch tangencies preserve the lip.
            for k, x in enumerate(self.xs):
                for ax, zc, radius in cutouts:
                    if abs(x - ax) <= radius:
                        floor = zc + np.sqrt(max(radius * radius - (x - ax) ** 2, 0.0))
                        for j in range(R_EDGE + 1):
                            if G[k, j, 2] < floor:
                                G[k, j, 1] = s * self.side_y(x, floor)
                                G[k, j, 2] = floor
            parts = np.zeros((K - 1, N_ROWS - 1), int)
            cols = np.ones((K - 1, N_ROWS - 1, 3))
            for k in range(K - 1):
                xm = 0.5 * (self.xs[k] + self.xs[k + 1])
                for j in range(N_ROWS - 1):
                    parts[k, j], cols[k, j] = matfn(xm, j, s)
            _emit_grid(self.m, G, parts, cols, hint, cos_thr, center_col=N_ROWS - 1, ridge_x=ridge_x)

    def _patch(self, mapfn, quad, part, col, hint, nu, nv, cos_thr=0.9):
        c = [np.asarray(q, np.float64) for q in quad]
        G = np.zeros((nv + 1, nu + 1, 3))
        for iv in range(nv + 1):
            v = iv / nv
            for iu in range(nu + 1):
                u = iu / nu
                G[iv, iu] = mapfn((1 - v) * ((1 - u) * c[0] + u * c[1]) + v * ((1 - u) * c[3] + u * c[2]))
        _emit_grid(self.m, G, part, col, _const_hint(hint), cos_thr)

    def side(self, quad, part, col, off, nu=4, nv=1):
        """(x, z) corners projected onto both body sides."""
        for s in (1, -1):
            def mapfn(p, s=s):
                z = min(p[1], self.side_zmax(p[0]))
                return (p[0], s * (self.side_y(p[0], z) + off), z)
            self._patch(mapfn, quad, part, col, (0, s, 0.15), nu, nv)

    def top(self, quad, part, col, off, nu=2, nv=6, mirror=False):
        """(x, y) plan corners projected onto the top surface."""
        for s in ((1, -1) if mirror else (1,)):
            q = [(a, s * b) for a, b in quad] if mirror else quad
            self._patch(lambda p: (p[0], p[1], self.top_z(p[0], p[1]) + off), q, part, col,
                        (0, 0, 1), nu, nv)


# ------------------------------------------------------------------ fascia
class _Face:
    """x = x0 + s * b(z) * (1 - |y/w(z)|^p) closing the open end of the skin."""
    def __init__(self, C, x0, sgn, z_lo, btab, p=2.5):
        self.C, self.x0, self.s, self.zlo, self.p = C, x0, sgn, z_lo, p
        self.b = _Tab(btab)
        P = C.prof_at(x0)
        self.zr, self.yr = P[:R_EDGE + 1, 1], P[:R_EDGE + 1, 0]
        pts = P[R_BELT:]
        o = np.argsort(pts[:, 0])
        self.ty, self.tz = pts[o, 0], pts[o, 1]
        self.ztop_c = float(P[-1, 1])

    def w(self, z):
        return float(np.interp(z, self.zr, self.yr))

    def ztop(self, y):
        return float(np.interp(abs(y), self.ty, self.tz))

    def pos(self, y, z, off=0.0):
        w = self.w(z)
        u = min(abs(y) / max(w, 1e-6), 1.0)
        return np.array([self.x0 + self.s * (self.b(z) * (1 - u ** self.p) + off), y, z])

    def emit(self, matfn, nz=8, ny=15, cos_thr=0.90):
        t = np.sin(np.linspace(-np.pi / 2, np.pi / 2, ny))
        G = np.zeros((nz, ny, 3))
        for i, ti in enumerate(t):
            zt = max(self.ztop(ti * self.w(self.ztop_c)), self.zlo + 0.05)
            for j in range(nz):
                z = self.zlo + (zt - self.zlo) * j / (nz - 1)
                G[j, i] = self.pos(ti * self.w(z), z)
        parts = np.zeros((nz - 1, ny - 1), int)
        cols = np.ones((nz - 1, ny - 1, 3))
        for j in range(nz - 1):
            for i in range(ny - 1):
                c = 0.25 * (G[j, i] + G[j + 1, i] + G[j + 1, i + 1] + G[j, i + 1])
                parts[j, i], cols[j, i] = matfn(c[1], c[2])
        _emit_grid(self.C.m, G, parts, cols, _const_hint((self.s, 0, 0)), cos_thr)

    def poly(self, quad, part, col, off, mirror=True, nu=1, nv=1):
        for s in ((1, -1) if mirror else (1,)):
            q = [(s * a, b) for a, b in quad] if mirror else quad
            self.C._patch(lambda p: self.pos(p[0], p[1], off), q, part, col, (self.s, 0, 0.1), nu, nv)


# ------------------------------------------------------------------ small parts
def _tube(m, x0, x1, y, z, r, part, col, segs=10):
    a = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    rings = [np.stack([np.full(segs, x), y + r * np.sin(a), z + r * np.cos(a)], 1) for x in (x0, x1)]
    m.loft(np.stack(rings), col, part, cap_first=False, cap_last=False)


def _spokes(m, x, y, z, r_in, r_out, n, s, col, part, width=0.55):
    for k in range(n):
        a = 2 * np.pi * k / n
        h = width * np.pi / n
        pts = [(x + rr * np.sin(aa), y, z + rr * np.cos(aa))
               for (rr, aa) in ((r_in, a - h * 1.4), (r_out, a - h), (r_out, a + h), (r_in, a + h * 1.4))]
        m.quad(*pts, col, part, hint=(0, s, 0))


def _wheel(C, x, r, hw, s, proud, spokes=10, rim_col=RIM, flare=0.0, arch_scale=1.10, segs=14):
    """14-gon tyre, dished alloy and dark arch (+ optional raised flare lip) following the body side."""
    m = C.m
    ys = C.side_y(x, r)
    reach = max(C.side_y(x + r * np.sin(a), min(max(r + r * np.cos(a), 0.05), 2 * r)) for a in np.linspace(0, 2 * np.pi, 24))
    y_out = max(reach, ys) + proud
    zlo = C.T["zb"](x) + 0.012
    N = 16
    ang = np.linspace(0, 2 * np.pi, N + 1)

    def at(rad, off):
        pts = []
        for a in ang:
            px, pz = x + rad * np.sin(a), max(r + rad * np.cos(a), zlo)
            pts.append((px, s * (C.side_y(px, pz) + off), pz))
        return np.asarray(pts)

    ring = at(r * arch_scale, 0.006)
    well = np.stack([np.tile((x, s * (ys + 0.006), r), (N + 1, 1)), ring], 0)
    _emit_grid(m, well, P_MATTE, (0.010, 0.010, 0.012), _const_hint((0, s, 0)), 0.5)
    if flare > 0:
        inner = at(r * arch_scale, flare * 0.55)
        lip = np.stack([inner, at(r * arch_scale + flare + 0.03, 0.004)], 0)
        _emit_grid(m, lip, P_MATTE, MATTE, _const_hint((0, s, 0.2)), 0.5)
    u = np.array([-1.0, -0.86, 0.0, 0.86, 1.0]) * hw
    rad = np.array([0.76, 0.97, 1.0, 0.97, 0.76]) * r
    a = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    yc = s * (y_out - hw)
    rings = [np.stack([x + rr * np.sin(a), np.full(segs, yc + s * uu), r + rr * np.cos(a)], 1)
             for uu, rr in zip(u, rad)]
    m.loft(np.stack(rings), TYRE, P_TYRE, cap_first=False, cap_last=False)
    yo = s * y_out
    m.disc((x, yo - s * 0.040, r), r * 0.74, (0, s, 0), TRIM, P_TRIM, segs)                 # barrel
    _spokes(m, x, yo - s * 0.022, r, r * 0.14, r * 0.74, spokes, s, rim_col, P_CHROME)
    m.disc((x, yo - s * 0.016, r), r * 0.14, (0, s, 0), (0.20, 0.20, 0.22), P_CHROME, 8)    # hub


def _mirror(m, x, y, z, s, col=ONE):
    part = P_PIANO if col is PIANO else P_PAINT
    m.box(x, s * y, z, 0.13, 0.09, 0.07, col, part)
    m.box(x + 0.03, s * (y - 0.06), z - 0.04, 0.05, 0.06, 0.03, MATTE, P_MATTE)


def _seam(m, C, x, z0, z1, s, w=0.007):
    zm = 0.5 * (z0 + z1)
    pts = [(x, s * (C.side_y(x, z) + 0.010), z) for z in (z0, zm, z1)]
    for a, b in zip(pts[:-1], pts[1:]):
        m.quad((a[0] - w, a[1], a[2]), (a[0] + w, a[1], a[2]), (b[0] + w, b[1], b[2]), (b[0] - w, b[1], b[2]),
               TRIM, P_TRIM, hint=(0, s, 0))


def _handle(m, C, x, z, s):
    y = s * (C.side_y(x, z) + 0.012)
    m.quad((x - 0.07, y, z - 0.013), (x + 0.07, y, z - 0.013), (x + 0.07, y, z + 0.013), (x - 0.07, y, z + 0.013),
           MATTE, P_MATTE, hint=(0, s, 0))


def _plate(m, F, y, z):
    p = F.pos(y, z, 0.012 * F.s)
    m.box(p[0], y, z, 0.022, 0.36, 0.13, PLATE, P_PLATE)


def _unique(*arrays):
    return np.unique(np.round(np.concatenate([np.atleast_1d(a) for a in arrays]), 3))


# ================================================================== FORD EXPLORER ST
EXPLORER_LEN, EXPLORER_W, EXPLORER_H, EXPLORER_CLEARANCE = 5.050, 2.004, 1.783, 0.208
EXPLORER_WHEELBASE, EXPLORER_TRACK, EXPLORER_MIRROR_W = 3.025, 1.699, 2.268
EXPLORER_FRONT_OVERHANG, EXPLORER_REAR_OVERHANG = 0.856, 1.169
EXPLORER_AXLE_F = EXPLORER_LEN / 2 - EXPLORER_FRONT_OVERHANG
EXPLORER_AXLE_R = EXPLORER_AXLE_F - EXPLORER_WHEELBASE
EXPLORER_TYRE_WIDTH, EXPLORER_RIM_RADIUS = 0.255, 20 * 0.0254 / 2
EXPLORER_TYRE_RADIUS = EXPLORER_RIM_RADIUS + EXPLORER_TYRE_WIDTH * 0.55
# Lamp-cluster centres, just proud of the curved lens surfaces (local metres).
EXPLORER_HEAD_SPRITE = (2.456, .705, 1.020)
EXPLORER_TAIL_SPRITE = (-2.366, .764, 1.146)


def build_explorer(m):
    from .vehicle_explorer import build_explorer as build
    return build(m)


# ================================================================== TOYOTA CAMRY
CAMRY_LEN, CAMRY_W, CAMRY_H, CAMRY_CLEARANCE = 4.90, 1.84, 1.44, 0.14


def build_camry(m, taxi=False):
    AX_F, AX_R, WR = 1.50, -1.31, 0.334
    T = dict(
        zb=_Tab([(-2.30, .30), (-2.1, .20), (-1.8, .14), (1.8, .14), (2.1, .20), (2.30, .26)]),
        zm=_Tab([(-2.30, .50), (-2.0, .46), (2.0, .46), (2.30, .48)]),
        Wm=_Tab([(-2.30, .77), (-2.2, .83), (-2.05, .895), (-1.7, .91), (1.7, .91), (2.0, .895), (2.15, .86), (2.30, .78)]),
        Wsh=_Tab([(-2.30, .78), (-2.2, .845), (-2.05, .905), (-1.7, .922), (1.7, .922), (2.0, .905), (2.15, .87), (2.30, .79)]),
        Wb=_Tab([(-2.30, .73), (-2.2, .79), (-2.05, .855), (-1.7, .868), (1.7, .868), (2.0, .85), (2.15, .82), (2.30, .74)]),
        zbe=_Tab([(-2.30, 1.07), (-1.9, 1.07), (-1.0, 1.06), (-0.2, 1.035), (0.6, 1.01), (0.95, 1.0),
                  (1.05, .90), (1.5, .89), (2.0, .81), (2.30, .70)]),
        shdrop=0.075,
        zc=_Tab([(-2.30, 1.155), (-2.22, 1.14), (-2.1, 1.125), (-1.94, 1.135), (-1.57, 1.25), (-1.27, 1.33), (-0.94, 1.40),
                 (-0.6, 1.415), (-0.1, 1.425), (0.15, 1.415), (0.35, 1.375), (0.6, 1.285), (0.8, 1.17),
                 (1.05, 1.03), (1.5, 1.016), (1.8, .971), (2.0, .927), (2.2, .86), (2.30, .826)]),
        Wr=_Tab([(-2.30, .66), (-2.0, .73), (-1.6, .72), (-1.27, .69), (-0.9, .65), (-0.1, .635), (0.3, .65),
                 (0.6, .69), (1.05, .78), (1.5, .80), (2.0, .77), (2.30, .66)]),
        crown=_Tab([(-2.30, .025), (0.0, .03), (2.3, .03)]),
        ridge=_Tab([(-2.30, 0), (1.0, 0), (1.3, .05), (1.9, .07), (2.3, .06)]),     # sharp centre hood crease
        sculpt=None,
    )
    xs = _unique([-2.30, -2.1, -1.94, -1.57, -1.27, -0.94, -0.6, -0.1, 0.15, 0.35, 0.6, 0.8, 1.05, 1.5, 1.8, 2.0, 2.2, 2.30],
                 [AX_F - 0.4, AX_F + 0.4, AX_R - 0.4, AX_R + 0.4])
    C = _Coach(m, xs, T)
    C.emit(lambda xm, j, s: (P_MATTE, MATTE) if j == 0 else (P_PAINT, ONE), ridge_x=(1.15, 2.31))

    # ---- front fascia: pointed nose, huge lower mesh grille wider at the bottom, swept blade lamps
    F = _Face(C, 2.30, +1, 0.24,
              [(.24, .04), (.30, .10), (.40, .14), (.50, .15), (.60, .14), (.70, .09), (.80, .03), (.826, 0)], p=2.4)
    F.emit(lambda y, z: (P_PAINT, ONE))
    F.poly([(0, .29), (.78, .29), (.80, .42), (0, .42)], P_PIANO, PIANO, O_GLASS, nu=3)                  # lower grille
    F.poly([(0, .42), (.80, .42), (.52, .57), (0, .57)], P_PIANO, PIANO, O_GLASS, nu=3)
    for z in (0.33, 0.37, 0.41, 0.45, 0.49, 0.53):
        hw = np.interp(z, [.29, .42, .57], [.78, .80, .52]) - 0.04
        F.poly([(-hw, z), (hw, z), (hw, z + .008), (-hw, z + .008)], P_MATTE, SLAT, O_DETAIL, mirror=False, nu=3)
    F.poly([(-.78, .245), (.78, .245), (.78, .275), (-.78, .275)], P_PIANO, PIANO, O_BASE, mirror=False, nu=3)  # splitter
    F.poly([(0, .60), (.58, .66), (.58, .735), (0, .70)], P_PIANO, PIANO, O_BASE, nu=2)                    # upper grille V
    F.poly([(-.09, .62), (.09, .62), (.09, .745), (-.09, .745)], P_CHROME, CHROME, O_TOP, mirror=False)       # badge
    wl = F.w(.80) - 0.02
    F.poly([(.30, .70), (wl, .74), (wl + .01, .835), (.40, .76)], P_PIANO, PIANO, O_BASE, nu=3)            # cat-eye housing
    F.poly([(.34, .715), (wl - .03, .75), (wl - .02, .812), (.44, .765)], P_HEAD, LAMP, O_LIGHT, nu=3)
    F.poly([(.46, .762), (wl - .01, .815), (wl - .01, .83), (.46, .775)], P_LED, LED, O_TOP, nu=3)
    C.side([(2.30, .70), (2.02, .735), (2.02, .78), (2.30, .775)], P_PIANO, PIANO, O_BASE, nu=2)            # runs into the fender
    C.side([(2.30, .735), (2.05, .752), (2.05, .768), (2.30, .76)], P_LED, LED, O_LIGHT, nu=2)
    _plate(m, F, 0.0, 0.40)

    # ---- rear: boomerang lamps wrapping the corners, lip spoiler, valance, single exhaust (car's right)
    R = _Face(C, -2.30, -1, 0.30,
              [(.30, .04), (.40, .09), (.50, .12), (.60, .13), (.70, .13), (.80, .11), (.95, .07), (1.05, .03), (1.10, 0)], p=2.4)
    R.emit(lambda y, z: (P_PAINT, ONE) if z > 0.36 else (P_MATTE, MATTE))
    wr_ = R.w(.90) - 0.01
    R.poly([(.32, .915), (wr_, .825), (wr_, .995), (.32, .975)], P_PIANO, PIANO, O_BASE, nu=4, nv=1)
    R.poly([(.34, .93), (wr_ - .02, .845), (wr_ - .02, .975), (.34, .962)], P_TAIL, LENS, O_LIGHT, nu=4)
    R.poly([(-.32, .935), (.32, .935), (.32, .950), (-.32, .950)], P_CHROME, (0.45, 0.46, 0.50), O_DETAIL, mirror=False, nu=2)
    C.side([(-2.12, .85), (-2.29, .84), (-2.29, .97), (-2.12, .955)], P_TAIL, LENS, O_LIGHT, nu=1)             # wrap onto the quarter
    R.poly([(.18, .44), (.58, .44), (.58, .47), (.18, .47)], P_TAIL, LENS, O_LIGHT, nu=2)                    # reflectors
    R.poly([(-.30, .29), (.30, .29), (.40, .42), (-.40, .42)], P_PIANO, PIANO, O_BASE, mirror=False, nu=2)   # sculpted valance
    px = R.pos(.62, .34)[0]
    _tube(m, px + 0.012, px - 0.07, -0.62, 0.34, 0.042, P_CHROME, (0.60, 0.62, 0.66))
    m.disc((px - 0.069, -0.62, 0.34), 0.033, (-1, 0, 0), (0.02, 0.02, 0.02), P_TRIM, 8, axis="x")
    _plate(m, R, 0.0, 0.74)

    # ---- greenhouse (belt rises to the rear, long fastback glass)
    C.side([(1.02, 1.015), (-1.12, 1.08), (-1.02, 1.50), (0.20, 1.50)], P_PIANO, PIANO, O_BASE, nu=5)
    C.side([(0.93, 1.02), (-0.20, 1.045), (-0.20, 1.385), (0.36, 1.33)], P_GLASS, GLASS, O_GLASS, nu=3)
    C.side([(-0.32, 1.05), (-1.10, 1.085), (-1.00, 1.33), (-0.32, 1.385)], P_GLASS, GLASS, O_GLASS, nu=3)
    C.top([(1.00, -.72), (1.00, .72), (0.20, .60), (0.20, -.60)], P_GLASS, GLASS, O_GLASS, nu=8, nv=5)
    C.top([(1.00, .71), (1.00, .78), (0.20, .64), (0.20, .59)], P_PIANO, PIANO, O_BASE, nu=1, nv=4, mirror=True)
    C.top([(-0.92, -.60), (-0.92, .60), (-1.84, .70), (-1.84, -.70)], P_GLASS, GLASS, O_GLASS, nu=8, nv=5)
    C.top([(-0.90, -.63), (-0.90, .63), (-1.88, .73), (-1.88, -.73)], P_PIANO, PIANO, O_BASE, nu=8, nv=5)
    C.side([(1.05, .17), (-1.0, .17), (-1.0, .27), (1.05, .27)], P_MATTE, MATTE, O_DETAIL, nu=3)            # side skirts
    for s in (1, -1):
        _mirror(m, 0.92, 0.95, 1.07, s, PIANO)
    if taxi:
        m.box(-0.05, 0, 1.52, 0.62, 0.24, 0.15, (1.0, 0.95, 0.65), 8)
    for s in (1, -1):
        for ax in (AX_F, AX_R):
            _wheel(C, ax, WR, 0.100, s, proud=0.010, spokes=10, flare=0.0, arch_scale=1.08)
    return CAMRY_LEN
