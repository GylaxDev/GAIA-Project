"""Procedural humanoid template for the pedestrian crowd and the on-foot player.

Two levels of detail are built from the same skeleton, so one shader animates both:

    lod 0  ~14k triangles  face (eyes, brow, nose, lips, ears), fingers, shaped hair,
                           collars, cuffs, soles, straps ... drawn within ~30 m
    lod 1  ~3k triangles   the same silhouette with fewer rings; far crowd + shadows

Every vertex carries

    position, normal, material id, joint role, joint pivot, parent pivot, accessory mask, ao

The vertex shader (shaders/pedestrian.py) poses the joints: two-bone leg IK with foot
planting, pelvis sway and rotation, counter-rotating shoulders, arm poses ... Accessories
(umbrella, phone, bag, hair styles, cap, coat, coffee, watch ...) live in the same mesh and
are shown when an instance's flag bit matches ``aAcc``.

Model space: +x forward, +y left, +z up, feet at z = 0, standing 1.78 m tall.
"""
from __future__ import annotations

import math

import numpy as np

HEIGHT = 1.78
FLOATS = 16          # pos3 normal3 mat1 role1 pivot3 parent3 acc1 ao1
STRIDE = FLOATS * 4

# ---- material ids (see shaders/pedestrian.py) -----------------------------------
(M_SKIN, M_HAIR, M_TOP, M_BOTTOM, M_SHOE, M_FIXED, M_EYE, M_CANOPY, M_METAL,
 M_PHONE, M_SCREEN, M_GLASS, M_LIMB, M_SHIN, M_CAP, M_DARK,
 M_LIP, M_SCLERA, M_IRIS, M_GLOVE, M_CUP, M_SOLE, M_BROW, M_BEARD) = range(24)

# ---- joint roles ----------------------------------------------------------------
(R_BODY, R_TORSO, R_UARM_L, R_UARM_R, R_FARM_L, R_FARM_R, R_THIGH_L, R_THIGH_R,
 R_SHIN_L, R_SHIN_R, R_COAT_L, R_COAT_R, R_HEAD, R_FOOT_L, R_FOOT_R) = range(15)

# ---- accessory / pose bits (instance flag column 7 uses the same bits) ---------
ACC_UMBRELLA = 1        # right arm raised, umbrella above the head
ACC_PHONE_EAR = 2       # left hand holds a phone to the ear
ACC_PHONE_LOOK = 4      # both hands hold a phone at chest height, head tilted down
ACC_BACKPACK = 8
ACC_COAT = 16           # long coat tails, collar
ACC_SKIRT = 32
ACC_HAIR_SHORT = 64
ACC_HAIR_LONG = 128
ACC_HAIR_BUN = 256
ACC_CAP = 512
ACC_GLASSES = 1024
ACC_BAG = 2048
ACC_IDLE = 16384        # no mesh: standing weight shift / breathing
ACC_PLAYER = 32768      # the controlled character (leather coat, fingerless gloves)
POSE_POCKETS = 65536    # hands in pockets
POSE_CROSSED = 131072   # arms folded
ACC_COFFEE = 262144     # coffee cup in the right hand (pose + mesh)
POSE_WATCH = 524288     # checking a wristwatch
POSE_TALK = 1048576     # animated conversational gestures
ACC_WATCH = 4194304     # wears a wristwatch (mesh)
ACC_BEARD = 8388608     # stubble / beard

# ---- skeleton -------------------------------------------------------------------
SPINE = np.array([0., 0., .98])
NECK = np.array([0., 0., 1.50])
SHOULDER_Y, SHOULDER_Z = .205, 1.42
ELBOW_Z, WRIST_Z, HAND_Z = 1.13, .87, .82
HIP_Y, HIP_Z, KNEE_Z, ANKLE_Z = .095, .93, .50, .085
WAIST = np.array([0., 0., 1.0])

# Arm poses shared with the GLSL: (shoulder pitch, outward roll, elbow bend, forearm swivel
# toward the midline).  Found by searching the same forward kinematics for a hand at the
# ear / chest / pocket / opposite arm.
POSE_UMBRELLA = (.62, -.06, 1.30, 0.)
POSE_EAR = (1.30, 0., 2.60, .50)
POSE_LOOK = (.25, .10, 1.75, .55)
POSE_POCKET = (-.30, .188, .836, .60)
POSE_CROSS_A = (1.00, -.075, 1.818, 1.688)      # left arm over
POSE_CROSS_B = (.70, 0., 1.673, 1.688)          # right arm under
POSE_COFFEE = (.20, .30, 1.755, .65)
POSE_WATCH_L = (.30, .063, 2.241, .688)
POSE_SWING_ROLL = .06


def _rot_pitch(q, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * q[0] - s * q[2], q[1], s * q[0] + c * q[2]])


def _rot_roll(q, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([q[0], c * q[1] - s * q[2], s * q[1] + c * q[2]])


def _rot_yaw(q, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * q[0] - s * q[1], s * q[0] + c * q[1], q[2]])


def hand_position(side, pose):
    """Hand-centre model position for an arm pose; mirrors the GLSL chain exactly."""
    pitch, roll, elbow, yaw = pose
    shoulder = np.array([0., side * SHOULDER_Y, SHOULDER_Z])
    e_pivot = np.array([0., side * SHOULDER_Y, ELBOW_Z])
    p = np.array([0., side * SHOULDER_Y, HAND_Z])
    p = e_pivot + _rot_pitch(p - e_pivot, elbow)
    p = e_pivot + _rot_yaw(p - e_pivot, -yaw * side)
    p = shoulder + _rot_pitch(p - shoulder, pitch)
    p = shoulder + _rot_roll(p - shoulder, roll * side)
    return p


def pose_unrotate_vector(side, pose, v):
    """A rest-frame direction that the forearm chain turns into ``v``."""
    pitch, roll, elbow, yaw = pose
    v = _rot_roll(np.asarray(v, float), -roll * side)
    v = _rot_pitch(v, -pitch)
    v = _rot_yaw(v, yaw * side)
    return _rot_pitch(v, -elbow)


# =====================================================================================
# mesh primitives
# =====================================================================================
class _Builder:
    def __init__(self):
        self.rows, self.tris, self.n = [], [], 0

    def add(self, prim, mat, role, pivot=(0., 0., 0.), parent=(0., 0., 0.), acc=0, ao=1.):
        pos, nrm, tris = prim
        pos = np.asarray(pos, np.float64).reshape(-1, 3)
        nrm = np.asarray(nrm, np.float64).reshape(-1, 3)
        tris = np.asarray(tris, np.int64).reshape(-1, 3)
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        face = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
        flip = np.einsum('ij,ij->i', face, nrm[tris].sum(axis=1)) < 0
        tris = tris.copy()
        tris[flip] = tris[flip][:, [0, 2, 1]]
        rows = np.zeros((len(pos), FLOATS), np.float32)
        rows[:, 0:3], rows[:, 3:6] = pos, nrm
        rows[:, 6], rows[:, 7] = mat, role
        rows[:, 8:11] = np.asarray(pivot, np.float32)
        rows[:, 11:14] = np.asarray(parent, np.float32)
        rows[:, 14] = acc
        rows[:, 15] = ao
        self.rows.append(rows)
        self.tris.append((tris + self.n).astype(np.uint32))
        self.n += len(pos)

    def finish(self):
        return np.ascontiguousarray(np.concatenate(self.rows)), np.ascontiguousarray(np.concatenate(self.tris))


def _smooth_normals(pos, tris):
    face = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    nrm = np.zeros_like(pos)
    for k in range(3):
        np.add.at(nrm, tris[:, k], face)
    return nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)


def _grid(P, closed=True, outward_from=None):
    """Mesh a (K rings, N points, 3) lattice; smooth normals; outward orientation."""
    P = np.asarray(P, np.float64)
    K, N, _ = P.shape
    pos = P.reshape(-1, 3)
    i, j = np.meshgrid(np.arange(K - 1), np.arange(N if closed else N - 1), indexing='ij')
    a = i * N + j
    b = i * N + (j + 1) % N
    c = (i + 1) * N + j
    d = (i + 1) * N + (j + 1) % N
    tris = np.concatenate((np.stack((a, c, b), -1).reshape(-1, 3), np.stack((b, c, d), -1).reshape(-1, 3)))
    nrm = _smooth_normals(pos, tris)
    ref = np.repeat(P.mean(axis=1), N, axis=0) if outward_from is None else np.asarray(outward_from)
    if float(np.einsum('ij,ij->i', nrm, pos - ref).mean()) < 0:
        nrm, tris = -nrm, tris[:, [0, 2, 1]]
    return pos, nrm, tris


def _basis(ax, u_hint, v_hint):
    u = u_hint - ax * (u_hint @ ax)
    if np.linalg.norm(u) < 1e-6:
        u = np.array([0., 0., 1.]) - ax * ax[2]
    u /= np.linalg.norm(u)
    v = v_hint - ax * (v_hint @ ax) - u * (v_hint @ u)
    if np.linalg.norm(v) < 1e-6:
        v = np.cross(ax, u)
    return u, v / np.linalg.norm(v)


def _sweep(centers, ru, rv, n=14, exp=2.0, bump=None, caps=(False, False), u_hint=(1., 0., 0.),
           v_hint=(0., 1., 0.), arc=None, cap_depth=.7):
    """Lofted tube through ``centers``. Ring angle 0 points along u (default +x), 90 deg along v (+y).

    ru / rv are the half-extents along u / v; exp > 2 squares the section; bump (K, n) pushes
    points radially (metres).  arc=(a0, a1) builds an open sheet instead of a closed ring.
    """
    centers = np.asarray(centers, np.float64)
    k = len(centers)
    ru = np.broadcast_to(np.asarray(ru, np.float64), (k,)).copy()
    rv = np.broadcast_to(np.asarray(rv, np.float64), (k,)).copy()
    bump = np.zeros((k, n)) if bump is None else np.asarray(bump, np.float64)
    axes = np.gradient(centers, axis=0) if k > 1 else np.array([[0., 0., 1.]])
    axes = axes / np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
    u_hint, v_hint = np.asarray(u_hint, float), np.asarray(v_hint, float)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False) if arc is None else np.linspace(arc[0], arc[1], n)
    c, s = np.cos(th), np.sin(th)
    e = 2. / exp
    xe, ye = np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e
    rings = []
    for i in range(k):
        u, v = _basis(axes[i], u_hint, v_hint)
        pts = centers[i] + (xe * ru[i])[:, None] * u + (ye * rv[i])[:, None] * v
        rings.append(pts + (c[:, None] * u + s[:, None] * v) * bump[i][:, None])
    rings = np.array(rings)
    start, end = [], []
    for first, on in ((True, caps[0]), (False, caps[1])):
        if not on:
            continue
        idx = 0 if first else k - 1
        ax = axes[idx] * (-1. if first else 1.)
        r0 = min(ru[idx], rv[idx])
        u, v = _basis(axes[idx], u_hint, v_hint)
        for fr, fa in ((.82, .45), (.55, .80), (.22, .97), (0., 1.)):
            ring = centers[idx] + ax * (fa * r0 * cap_depth) + (
                (xe * ru[idx])[:, None] * u + (ye * rv[idx])[:, None] * v) * fr
            (start if first else end).append(ring)
    if start:
        rings = np.concatenate((np.array(start[::-1]), rings))
    if end:
        rings = np.concatenate((rings, np.array(end)))
    return _grid(rings, closed=arc is None)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _ellipsoid(center, radii, n=14, m=9, v0=0., v1=math.pi, rot=None):
    rx, ry, rz = radii
    phis = np.linspace(v0, v1, m + 1)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rings = np.array([[(rx * math.sin(p) * math.cos(t), ry * math.sin(p) * math.sin(t), rz * math.cos(p))
                       for t in th] for p in phis])
    if rot is not None:
        rings = rings @ np.asarray(rot).T
    rings = rings + np.asarray(center, np.float64)
    ref = np.broadcast_to(rings.reshape(-1, 3).mean(0), (rings.shape[0] * rings.shape[1], 3))
    return _grid(rings, closed=True, outward_from=ref)


def _box(center, half, rot_y=0.):
    rot = _rot_y(rot_y)
    pos, nrm, tris = [], [], []
    for axis in range(3):
        for sign in (-1., 1.):
            n = np.zeros(3)
            n[axis] = sign
            a, b = [i for i in range(3) if i != axis]
            base = len(pos)
            for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis], p[a], p[b] = sign * half[axis], sa * half[a], sb * half[b]
                pos.append(rot @ p + center)
                nrm.append(rot @ n)
            tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(pos), np.array(nrm), tris


def _along(points, r0, r1=None, n=8, caps=(True, True), exp=2.0):
    """Thin round rod through 3-D points (fingers, straps, poles)."""
    points = np.asarray(points, np.float64)
    r1 = r0 if r1 is None else r1
    radii = np.linspace(r0, r1, len(points))
    return _sweep(points, radii, radii, n=n, caps=caps, exp=exp)


# =====================================================================================
# head
# =====================================================================================
_HEAD_KEYS = np.array([
    # z      cx     rf     rb     ry
    (1.548, .030, .020, .026, .024),
    (1.558, .036, .046, .046, .042),
    (1.575, .024, .074, .072, .062),
    (1.605, .013, .090, .088, .076),
    (1.640, .010, .097, .094, .083),
    (1.675, .010, .098, .096, .084),
    (1.705, .010, .097, .097, .084),
    (1.740, .008, .091, .096, .081),
    (1.768, .006, .070, .083, .066),
    (1.784, .004, .037, .047, .036),
    (1.790, .004, .000, .000, .000)])


def _head_profile(z):
    return [np.interp(z, _HEAD_KEYS[:, 0], _HEAD_KEYS[:, k]) for k in range(1, 5)]


def _gauss(x, mu, sigma):
    return np.exp(-((x - mu) / sigma) ** 2)


def _head_points(zs, n, shift=0.):
    """Ring points of the skull + face shell at heights zs -> (K, n, 3)."""
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    ang = np.abs(np.where(th > math.pi, th - 2 * math.pi, th))
    rings = []
    for z in zs:
        cx, rf, rb, ry = _head_profile(z)
        depth = np.where(c >= 0, rf, rb)
        e = 2.0 / 2.15
        x = cx + np.sign(c) * np.abs(c) ** e * depth
        y = np.sign(s) * np.abs(s) ** e * ry
        front = np.clip(c, 0, 1) ** 2
        relief = (.0050 * _gauss(z, 1.704, .011) * np.clip(c, 0, 1) ** 1.5 * (ang < 1.0)
                  - .0042 * _gauss(z, 1.679, .011) * _gauss(ang, .42, .13)
                  + .0040 * _gauss(z, 1.641, .014) * _gauss(ang, .78, .22)
                  + .0050 * _gauss(z, 1.566, .010) * front
                  + .0030 * _gauss(z, 1.745, .018) * front)
        relief = np.where(c > 0, relief, 0.)
        ring = np.zeros((n, 3))
        ring[:, 0] = x + (relief + shift) * c
        ring[:, 1] = y + (relief + shift) * s
        ring[:, 2] = z
        rings.append(ring)
    return np.array(rings)


def _head_mesh(n, nz):
    zs = np.concatenate(([1.5485], np.linspace(1.556, 1.782, nz), [1.7895]))
    return _grid(_head_points(zs, n), closed=True)


def _hair_cap(n, nz, thickness, hairline_low, back_low, style_extra=0., front_drop=0.):
    """A shell over the skull: thick above the hairline, sunk inside the head below it."""
    zs = np.linspace(1.585, 1.7895, nz)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(th), np.sin(th)
    a = np.abs(np.where(th > math.pi, th - 2 * math.pi, th))
    rings = []
    for z in zs:
        ring = _head_points([z], n)[0].copy()
        hl = np.interp(a, [0., .55, 1.1, 1.65, 2.3, math.pi], [1.748 - front_drop, 1.728 - front_drop * .8, 1.700 - front_drop * .4, 1.675, hairline_low, back_low])
        t = np.clip((z - hl) / .018, 0., 1.)
        t = t * t * (3. - 2. * t)
        top = np.clip((z - 1.70) / .08, 0., 1.)
        thick = thickness * (.75 + .5 * top) + style_extra * np.clip(-c, 0, 1)
        off = t * thick - (1. - t) * .007
        ring[:, 0] += off * c
        ring[:, 1] += off * s
        ring[:, 2] += off * .35 * top
        rings.append(ring)
    return _grid(np.array(rings), closed=True)


# =====================================================================================
# the template
# =====================================================================================
def build_procedural_template(lod=0):
    """-> (vertices float32 [N, 16], indices uint32 [M]).  lod 0 = detailed, 1 = distant."""
    hi = lod == 0
    B = _Builder()
    nl = 14 if hi else 8            # limb ring points
    nt = 22 if hi else 10           # torso ring points
    steps = 6 if hi else 3

    def add(prim, mat, role, pivot=(0., 0., 0.), parent=(0., 0., 0.), acc=0, ao=1.):
        B.add(prim, mat, role, pivot, parent, acc, ao)

    def profile(keys, k):
        keys = np.asarray(keys, float)
        order = np.argsort(keys[:, 0])
        zs = np.linspace(keys[0, 0], keys[-1, 0], k)          # keep the caller's top -> bottom order
        return zs, np.interp(zs, keys[order, 0], keys[order, 1])

    # ---------------------------------------------------------------- pelvis + trousers
    zs = np.linspace(1.045, .80, 5 if hi else 3)
    rx = np.interp(zs, [.80, .90, .98, 1.045], [.112, .126, .121, .108])
    ry = np.interp(zs, [.80, .90, .98, 1.045], [.150, .170, .165, .148])
    c = np.stack((np.full(len(zs), -.004), np.zeros(len(zs)), zs), -1)
    add(_sweep(c, rx, ry, nt, exp=2.3, caps=(False, True)), M_BOTTOM, R_BODY, ao=.9)

    # ---------------------------------------------------------------- torso (shirt / jacket)
    keys_z = [.96, 1.02, 1.10, 1.20, 1.30, 1.37, 1.42, 1.455, 1.475]
    tz = np.array(keys_z) if hi else np.array([.96, 1.10, 1.30, 1.42, 1.475])
    rx = np.interp(tz, keys_z, [.120, .109, .108, .117, .130, .124, .106, .082, .060])
    ry = np.interp(tz, keys_z, [.163, .150, .152, .160, .170, .180, .188, .150, .072])
    cx = np.interp(tz, keys_z, [-.003, 0., 0., .003, .010, .008, .003, 0., .012])
    th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
    bump = np.zeros((len(tz), nt))
    for i, z in enumerate(tz):
        bump[i] = (.006 * _gauss(z, 1.30, .05) * np.clip(np.cos(th), 0, 1) ** 2
                   + .006 * _gauss(z, 1.33, .05) * np.clip(-np.cos(th), 0, 1) ** 2 * (np.abs(np.sin(th)) > .25))
    c = np.stack((cx, np.zeros(len(tz)), tz), -1)
    add(_sweep(c, rx, ry, nt, exp=2.35, bump=bump), M_TOP, R_TORSO, SPINE, ao=.95)
    add(_sweep(np.array([[.014, 0, 1.462], [.016, 0, 1.482]]), [.058, .054], [.062, .058], nt,
               exp=2.1), M_TOP, R_TORSO, SPINE, ao=.75)

    # ---------------------------------------------------------------- neck + head
    head = dict(role=R_HEAD, pivot=NECK, parent=SPINE)
    c = np.stack((np.linspace(.004, .026, 5), np.zeros(5), np.linspace(1.40, 1.56, 5)), -1)
    add(_sweep(c, [.052, .049, .046, .045, .046], [.050, .048, .046, .045, .046], nl), M_SKIN, ao=.6, **head)
    add(_head_mesh(28 if hi else 12, 26 if hi else 10), M_SKIN, **head)

    if hi:
        for sgn in (-1., 1.):                                          # ears
            add(_ellipsoid((.004, sgn * .084, 1.672), (.011, .0065, .026), 10, 7, rot=_rot_y(-.12)),
                M_SKIN, ao=.8, **head)
        for sgn in (-1., 1.):                                          # eyes + lids + brows
            ex, ey, ez = .0875, sgn * .0345, 1.6775
            add(_ellipsoid((ex, ey, ez), (.0132, .0134, .0128), 12, 8), M_SCLERA, **head)
            add(_ellipsoid((ex + .0092, ey + sgn * .0006, ez), (.0060, .0074, .0074), 10, 6), M_IRIS, **head)
            add(_ellipsoid((ex + .0128, ey + sgn * .0006, ez), (.0016, .0034, .0034), 6, 4), M_EYE, **head)
            add(_ellipsoid((ex - .0020, ey, ez + .0075), (.0138, .0140, .0072), 12, 6, 0., 1.75), M_SKIN, ao=.9, **head)
            add(_sweep(np.array([[.092, sgn * .018, 1.7155], [.1005, sgn * .040, 1.7175], [.094, sgn * .066, 1.7105]]),
                       [.0030, .0034, .0026], [.0026, .0028, .0022], 6), M_BROW, **head)
        add(_sweep(np.array([[.097, 0., 1.698], [.106, 0., 1.672], [.1165, 0., 1.6545]]),
                   [.0045, .0070, .0090], [.0060, .0078, .0098], 10, caps=(False, True)), M_SKIN, **head)
        for sgn in (-1., 1.):
            add(_ellipsoid((.1090, sgn * .0118, 1.6480), (.0072, .0066, .0072), 8, 6), M_SKIN, **head)
            add(_ellipsoid((.1150, sgn * .0058, 1.6430), (.0028, .0032, .0018), 6, 4), M_DARK, **head)
        add(_ellipsoid((.1052, 0., 1.6090), (.0062, .0225, .0050), 12, 6), M_LIP, **head)
        add(_ellipsoid((.1048, 0., 1.5965), (.0072, .0200, .0066), 12, 6), M_LIP, **head)
    else:
        for sgn in (-1., 1.):
            add(_ellipsoid((.088, sgn * .034, 1.678), (.010, .011, .010), 6, 4), M_SCLERA, **head)
            add(_ellipsoid((.096, sgn * .034, 1.678), (.004, .005, .005), 6, 4), M_EYE, **head)
        add(_ellipsoid((.114, 0., 1.655), (.014, .012, .024), 6, 5), M_SKIN, **head)

    nh, nzh = (40, 34) if hi else (14, 10)
    add(_hair_cap(nh, nzh, .0105, 1.668, 1.642), M_HAIR, acc=ACC_HAIR_SHORT, **head)
    add(_hair_cap(nh, nzh, .0135, 1.640, 1.585, style_extra=.006), M_HAIR, acc=ACC_HAIR_LONG, **head)
    zl = np.linspace(1.68, 1.33, 9 if hi else 5)
    cl = np.stack((np.interp(zl, [1.33, 1.5, 1.68], [-.060, -.074, -.080]), np.zeros(len(zl)), zl), -1)
    flare = np.interp(zl, [1.33, 1.5, 1.68], [.060, .045, .040])
    wid = np.interp(zl, [1.33, 1.5, 1.68], [.092, .088, .090])
    add(_sweep(cl, flare, wid, 20 if hi else 10, arc=(math.pi * .52, math.pi * 1.48), exp=2.2),
        M_HAIR, acc=ACC_HAIR_LONG, ao=.7, **head)
    add(_ellipsoid((-.086, 0., 1.790), (.044, .042, .040), 12, 7), M_HAIR, acc=ACC_HAIR_BUN, **head)
    if hi:
        add(_along(np.array([[-.062, 0., 1.775], [-.100, 0., 1.790], [-.112, 0., 1.810]]), .020, .014, 10),
            M_HAIR, acc=ACC_HAIR_BUN, **head)
        nb = 28
        P = _head_points(np.linspace(1.553, 1.604, 5), nb, shift=.0042)
        thb = np.linspace(0, 2 * math.pi, nb, endpoint=False)
        wrapped = np.where(thb > math.pi, thb - 2 * math.pi, thb)
        order = np.argsort(wrapped)
        order = order[np.abs(wrapped[order]) < 2.05]           # contiguous, left ear -> chin -> right ear
        sl = P[:, order]
        ref = np.broadcast_to(np.array([.01, 0., 1.6]), (sl.shape[0] * sl.shape[1], 3))
        add(_grid(sl, closed=False, outward_from=ref), M_BEARD, acc=ACC_BEARD, **head)

    add(_hair_cap(nh, nzh, .0135, 1.690, 1.650, style_extra=.003, front_drop=.030), M_CAP, acc=ACC_CAP, **head)
    brim_c = np.array([[.088, 0., 1.722], [.118, 0., 1.716], [.146, 0., 1.699]])
    add(_sweep(brim_c, [.0060, .0055, .0050], [.078, .074, .062], 20 if hi else 10, arc=(-math.pi * .55, math.pi * .55),
               u_hint=(0., 0., 1.), v_hint=(0., 1., 0.)), M_CAP, acc=ACC_CAP, **head)
    add(_ellipsoid((-.002, 0., 1.795), (.008, .008, .006), 6, 4), M_CAP, acc=ACC_CAP, **head)
    for sgn in (-1., 1.):                                              # glasses
        add(_ellipsoid((.1045, sgn * .0345, 1.6785), (.0030, .0215, .0160), 14, 6, 0., math.pi), M_GLASS,
            acc=ACC_GLASSES, **head)
        add(_along(np.array([[.1035, sgn * .0555, 1.6785], [.060, sgn * .090, 1.6790], [.000, sgn * .092, 1.6740]]),
                   .0018, .0018, 6), M_DARK, acc=ACC_GLASSES, **head)
    add(_along(np.array([[.1052, -.0130, 1.6790], [.1062, 0., 1.6830], [.1052, .0130, 1.6790]]), .0016, .0016, 6),
        M_DARK, acc=ACC_GLASSES, **head)

    # ---------------------------------------------------------------- arms
    ua_keys = [(1.42, .060), (1.37, .058), (1.30, .053), (1.22, .047), (1.15, .041), (1.13, .040)]
    fa_keys = [(1.13, .040), (1.07, .044), (.99, .039), (.92, .032), (.87, .029)]
    for side in (1., -1.):
        left = side > 0
        y = side * SHOULDER_Y
        sh = np.array([0., y, SHOULDER_Z])
        el = np.array([0., y, ELBOW_Z])
        ua = R_UARM_L if left else R_UARM_R
        fa = R_FARM_L if left else R_FARM_R

        add(_ellipsoid(sh + [0., -side * .004, -.012], (.056, .055, .058), 12 if hi else 8, 7 if hi else 5), M_TOP, ua, sh)
        zs, r = profile(ua_keys, steps + 1)
        cxs = np.interp(zs, [1.13, 1.42], [-.002, .0])
        c = np.stack((cxs, np.full(len(zs), y), zs), -1)
        top_n = max(2, int(round(len(zs) * .55)))
        add(_sweep(c[:top_n], r[:top_n] * 1.03, r[:top_n] * 1.03, nl), M_TOP, ua, sh, ao=.85)
        add(_sweep(c[top_n - 1:], r[top_n - 1:], r[top_n - 1:] * .96, nl), M_LIMB, ua, sh, ao=.9)
        add(_ellipsoid(el, (.041, .041, .043), 10 if hi else 6, 6 if hi else 4), M_LIMB, fa, el, sh)
        zs, r = profile(fa_keys, steps + 1)
        c = np.stack((np.interp(zs, [.87, 1.13], [.004, -.002]), np.full(len(zs), y), zs), -1)
        add(_sweep(c, r, r * 1.12, nl, caps=(False, True)), M_LIMB, fa, el, sh)

        c = np.stack((np.zeros(3), np.full(3, y), np.array([.875, .835, .79])), -1)       # palm
        add(_sweep(c, [.017, .018, .016], [.0285, .0305, .0295], 10 if hi else 6), M_SKIN, fa, el, sh)
        if hi:
            for k, (yy, ln) in enumerate(((.0225, .078), (.0075, .088), (-.0075, .082), (-.0225, .066))):
                ys = y - side * yy
                pts = np.array([[.002, ys, .792], [.010, ys, .792 - ln * .45], [.021, ys, .792 - ln * .78],
                                [.026, ys, .792 - ln * .98]])
                add(_along(pts, .0088 - k * .0003, .0066, 6), M_SKIN, fa, el, sh, ao=.9)
            thumb = np.array([[.010, y + side * .028, .850], [.027, y + side * .030, .826],
                              [.040, y + side * .027, .805], [.046, y + side * .023, .791]])
            add(_along(thumb, .0102, .0074, 6), M_SKIN, fa, el, sh, ao=.9)
            add(_sweep(c, [.0205, .0215, .0195], [.0325, .0345, .0335], 12), M_GLOVE, fa, el, sh, acc=ACC_PLAYER)
            for yy in (.0225, .0075, -.0075, -.0225):
                ys = y - side * yy
                add(_along(np.array([[.004, ys, .792], [.011, ys, .760]]), .0098, .0095, 6),
                    M_GLOVE, fa, el, sh, acc=ACC_PLAYER)
        else:
            add(_ellipsoid((.012, y, .765), (.020, .030, .045), 8, 5), M_SKIN, fa, el, sh)
        if left:                                                      # wristwatch
            add(_sweep(np.array([[0., y, .895], [0., y, .875]]), [.0345, .0345], [.0385, .0385], nl),
                M_DARK, fa, el, sh, acc=ACC_WATCH)
            add(_box(np.array([0., y + .036, .886]), (.020, .004, .017)), M_METAL, fa, el, sh, acc=ACC_WATCH)

    # ---------------------------------------------------------------- legs and feet
    th_keys = [(.93, .094), (.86, .092), (.76, .084), (.64, .071), (.54, .060), (.50, .056)]
    sh_keys = [(.50, .056), (.44, .056), (.36, .058), (.26, .050), (.16, .041), (.10, .037)]
    for side in (1., -1.):
        left = side > 0
        hy = side * HIP_Y
        hip = np.array([0., hy, HIP_Z])
        knee = np.array([0., hy, KNEE_Z])
        ankle = np.array([0., hy, ANKLE_Z])
        th_r = R_THIGH_L if left else R_THIGH_R
        sn = R_SHIN_L if left else R_SHIN_R
        ft = R_FOOT_L if left else R_FOOT_R
        zs, r = profile(th_keys, steps + 2)
        c = np.stack((np.interp(zs, [.5, .93], [.0, .004]), np.full(len(zs), hy), zs), -1)
        add(_sweep(c, r * 1.02, r, nl), M_BOTTOM, th_r, hip, ao=.9)
        add(_ellipsoid(knee, (.058, .058, .060), 10 if hi else 6, 6 if hi else 4), M_BOTTOM, sn, knee, hip)
        zs, r = profile(sh_keys, steps + 2)
        c = np.stack((np.interp(zs, [.1, .36, .5], [.004, -.010, .0]), np.full(len(zs), hy), zs), -1)
        add(_sweep(c, r, r * .97, nl), M_SHIN, sn, knee, hip, ao=.95)
        add(_sweep(np.array([[.004, hy, .135], [.004, hy, .095]]), [.048, .054], [.047, .053], nl),
            M_BOTTOM, sn, knee, hip, ao=.7)
        # shoe: upper, sole, tongue, collar. Foot role: pivot = ankle, parent = knee.
        fx = np.array([-.080, -.055, -.010, .045, .100, .150, .190, .207])
        fz = np.array([.052, .054, .056, .043, .034, .027, .021, .017])
        fr = np.array([.030, .040, .044, .040, .034, .028, .021, .008])
        fw = np.array([.032, .040, .046, .047, .046, .042, .033, .014])
        c = np.stack((fx, np.full(len(fx), hy), fz), -1)
        add(_sweep(c, fr, fw, 14 if hi else 8, u_hint=(0., 0., 1.), caps=(True, True)), M_SHOE, ft, ankle, knee, ao=.9)
        sx = np.array([-.090, -.040, .040, .120, .180, .212])
        c = np.stack((sx, np.full(6, hy), np.full(6, .011)), -1)
        add(_sweep(c, [.0115] * 6, [.034, .043, .048, .047, .040, .018], 14 if hi else 8, u_hint=(0., 0., 1.),
                   caps=(True, True)), M_SOLE, ft, ankle, knee)
        if hi:
            add(_ellipsoid((.040, hy, .090), (.052, .030, .017), 10, 6, rot=_rot_y(-.5)), M_SHOE, ft, ankle, knee)
        add(_sweep(np.array([[-.004, hy, .098], [-.004, hy, .112]]), [.040, .036], [.040, .036], nl),
            M_SHOE, ft, ankle, knee, ao=.65)

        coat = R_COAT_L if left else R_COAT_R                          # coat tails / skirt
        nz = 6 if hi else 3
        zs = np.linspace(1.00, .56, nz)
        t = np.linspace(0, 1, nz)
        c = np.stack((np.full(nz, -.002), np.full(nz, side * .090), zs), -1)
        tht = np.linspace(0, 2 * math.pi, nl, endpoint=False)
        vent = .035 * t[:, None] ** 1.5 * np.clip(-np.cos(tht), 0, 1)[None, :]
        add(_sweep(c, .134 + .010 * t, .106 + .004 * t, nl, exp=2.2, bump=vent), M_TOP, coat, WAIST,
            acc=ACC_COAT, ao=.8)
        zs = np.linspace(1.0, .70, 4 if hi else 3)
        t = np.linspace(0, 1, len(zs))
        c = np.stack((np.zeros(len(zs)), np.full(len(zs), side * .090), zs), -1)
        add(_sweep(c, .124 + .07 * t, .110 + .06 * t, nl), M_BOTTOM, coat, WAIST, acc=ACC_SKIRT, ao=.8)

    add(_sweep(np.array([[.002, 0., 1.450], [.000, 0., 1.490], [-.004, 0., 1.530]]), [.088, .084, .094],
               [.098, .092, .100], 18 if hi else 8, arc=(math.pi * .30, math.pi * 1.70), exp=2.2),
        M_TOP, R_TORSO, SPINE, acc=ACC_COAT, ao=.6)                    # popped coat collar, open at the front

    # ---------------------------------------------------------------- accessories
    add(_box(np.array([-.172, 0., 1.27]), (.070, .125, .185)), M_FIXED, R_TORSO, SPINE, acc=ACC_BACKPACK)
    add(_box(np.array([-.232, 0., 1.17]), (.016, .095, .085)), M_FIXED, R_TORSO, SPINE, acc=ACC_BACKPACK)
    for sgn in (-1., 1.):
        add(_along(np.array([[-.12, sgn * .105, 1.43], [.0, sgn * .125, 1.42], [.05, sgn * .122, 1.28],
                             [-.02, sgn * .130, 1.15], [-.12, sgn * .115, 1.14]]), .017, .017, 6),
            M_FIXED, R_TORSO, SPINE, acc=ACC_BACKPACK)
    add(_box(np.array([.02, .272, .97]), (.058, .042, .072)), M_FIXED, R_TORSO, SPINE, acc=ACC_BAG)
    add(_along(np.array([[.105, -.12, 1.43], [.112, .0, 1.30], [.092, .13, 1.16], [.060, .262, 1.04]]), .0085, .0085, 6),
        M_FIXED, R_TORSO, SPINE, acc=ACC_BAG)

    hand = hand_position(-1., POSE_UMBRELLA)                           # umbrella
    top = np.array([.14, -.10, 2.06])
    add(_along(np.array([hand + [0., 0., -.04], hand + [0., 0., .02], top]), .0075, .0075, 8), M_METAL, R_TORSO,
        SPINE, acc=ACC_UMBRELLA)
    add(_along(np.array([hand + [0., 0., -.04], hand + [0., .028, -.07], hand + [0., .05, -.04]]), .0085, .0085, 8),
        M_DARK, R_TORSO, SPINE, acc=ACC_UMBRELLA)
    add(_ellipsoid(top + [0., 0., -.03], (.64, .64, .24), 16 if hi else 10, 5 if hi else 3, 0., 1.45), M_CANOPY,
        R_TORSO, SPINE, acc=ACC_UMBRELLA)
    if hi:
        for k in range(8):
            a = k * math.pi / 4
            pts = np.array([top + [0., 0., .205], top + [.34 * math.cos(a), .34 * math.sin(a), .10],
                            top + [.64 * math.cos(a), .64 * math.sin(a), -.075]])
            add(_along(pts, .0035, .0030, 4), M_METAL, R_TORSO, SPINE, acc=ACC_UMBRELLA)
    add(_along(np.array([top + [0., 0., .20], top + [0., 0., .27]]), .006, .004, 6), M_METAL, R_TORSO, SPINE,
        acc=ACC_UMBRELLA)

    el, sh = np.array([0., SHOULDER_Y, ELBOW_Z]), np.array([0., SHOULDER_Y, SHOULDER_Z])      # phone at the ear
    add(_box(np.array([0., SHOULDER_Y, HAND_Z + .02]), (.034, .0045, .07)), M_PHONE, R_FARM_L, el, sh, ACC_PHONE_EAR)
    add(_box(np.array([0., SHOULDER_Y - .0052, HAND_Z + .02]), (.031, .0008, .066)), M_SCREEN, R_FARM_L, el, sh,
        ACC_PHONE_EAR)

    mid = (hand_position(1., POSE_LOOK) + hand_position(-1., POSE_LOOK)) * .5 + [0., 0., .025]   # texting
    tilt = -.70
    normal = np.array([math.sin(tilt), 0., math.cos(tilt)])
    add(_box(mid, (.075, .036, .004), tilt), M_PHONE, R_TORSO, SPINE, acc=ACC_PHONE_LOOK)
    add(_box(mid + normal * .0047, (.070, .032, .0008), tilt), M_SCREEN, R_TORSO, SPINE, acc=ACC_PHONE_LOOK)

    er, sr = np.array([0., -SHOULDER_Y, ELBOW_Z]), np.array([0., -SHOULDER_Y, SHOULDER_Z])      # coffee
    axis = pose_unrotate_vector(-1., POSE_COFFEE, (0., 0., 1.))
    base = np.array([0., -SHOULDER_Y, HAND_Z + .02]) - axis * .035
    cup = np.array([base + axis * h for h in (0., .02, .07, .095, .102, .108)])
    add(_sweep(cup, [.026, .030, .038, .040, .042, .033], [.026, .030, .038, .040, .042, .033], 14 if hi else 8,
               caps=(True, True)), M_CUP, R_FARM_R, er, sr, ACC_COFFEE)
    add(_along(np.array([base + axis * .090, base + axis * .092]), .040, .043, 14 if hi else 8),
        M_DARK, R_FARM_R, er, sr, ACC_COFFEE)

    return B.finish()


def build_pedestrian_template(lod=0, variant=0):
    """Load the supplied GLB-derived anatomy and clothing (20-float weighted vertices).

    The procedural builder above is retained for the offline carried-prop bake.
    Gameplay always uses the imported male/female/player meshes.
    """
    from .human_assets import human_template
    return human_template(lod, variant)


def shader_constants():
    """GLSL constant block shared with shaders/pedestrian.py."""
    def v3(a):
        return f"vec3({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f})"

    def v4(a):
        return f"vec4({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}, {a[3]:.4f})"
    out = [f"const vec3 SPINE = {v3(SPINE)};", f"const vec3 NECK = {v3(NECK)};",
           f"const vec3 WAIST = {v3(WAIST)};", f"const float MODEL_HEIGHT = {HEIGHT:.3f};",
           f"const float HIP_Z0 = {HIP_Z:.3f}, HIP_Y0 = {HIP_Y:.3f}, ANKLE_H = {ANKLE_Z:.3f};",
           f"const float L_THIGH = {HIP_Z - KNEE_Z:.3f}, L_SHIN = {KNEE_Z - ANKLE_Z:.3f};",
           f"const float SWING_ROLL = {POSE_SWING_ROLL:.3f};"]
    for name, pose in (("UMBRELLA", POSE_UMBRELLA), ("EAR", POSE_EAR), ("LOOK", POSE_LOOK),
                       ("POCKET", POSE_POCKET), ("CROSS_A", POSE_CROSS_A), ("CROSS_B", POSE_CROSS_B),
                       ("COFFEE", POSE_COFFEE), ("WATCH", POSE_WATCH_L)):
        out.append(f"const vec4 POSE_{name} = {v4(pose)};")
    return "\n".join(out) + "\n"
