"""NumPy mirror of shaders/pedestrian.py ``boneMatrix`` + linear-blend skinning (24-float asset format).

Used to verify the baked rig without a GPU: it reuses the project's own gait / pose maths
(pedestrian_gait.gait, pedestrian_pose.params) and builds the fifteen joint matrices exactly like GLSL.
"""
from __future__ import annotations
import math, sys, os
import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(PKG))
_pkgname = os.path.basename(PKG)
import importlib
G = importlib.import_module(f'{_pkgname}.peds.pedestrian_gait')
PP = importlib.import_module(f'{_pkgname}.peds.pedestrian_pose')
M = importlib.import_module(f'{_pkgname}.peds.pedestrian_models')
E = importlib.import_module(f'{_pkgname}.peds.eye_animation')
AN = importlib.import_module(f'{_pkgname}.peds.human_anatomy')
F = importlib.import_module(f'{_pkgname}.peds.human_faces')


def facial_rest(vertices, t=0., seed=.3, gaze=0., flags=0, variant=None):
    """Eye rotations and lid deformation before skeletal skinning (positions, normals)."""
    p, n = vertices[:, :3].copy(), vertices[:, 3:6].copy()
    if vertices.shape[1] < 24:
        return p, n
    rig = vertices[:, 20:24]
    eyeball = (rig[:, 3] > .5) & (rig[:, 3] < 1.5)
    yaw_angle, pitch_angle = E.gaze_angles(t, seed, gaze, bool(flags & 4))
    for side in (1., -1.):
        mask = eyeball & (np.sign(rig[:, 1]) == side)
        convergence = side * .025 if flags & 4 else 0.
        r = (yaw(yaw_angle - convergence) @ pitch(pitch_angle))[:3, :3]
        p[mask] = rig[mask, :3] + (p[mask] - rig[mask, :3]) @ r.T
        n[mask] = n[mask] @ r.T
    lids = rig[:, 3] >= 2.
    blink = E.blink_closure(t, seed)
    follow = 3. - rig[lids, 3]
    old = p[lids, 1:3] - rig[lids, 1:3]
    p[lids, 2] -= vertices[lids, 19] * blink
    offset = p[lids, 1:3] - rig[lids, 1:3]
    front = np.sqrt(np.maximum(AN.EYE_RADIUS ** 2 - (offset ** 2).sum(1), 0.))
    old_front = np.sqrt(np.maximum(AN.EYE_RADIUS ** 2 - (old ** 2).sum(1), 0.))
    p[lids, 0] += (front - old_front) * .72 * follow * follow
    normal = np.column_stack((np.maximum(front, .002) / .72, offset))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    f = (blink * follow)[:, None]
    n[lids] = n[lids] * (1 - f) + normal * f
    n[lids] /= np.maximum(np.linalg.norm(n[lids], axis=1, keepdims=True), 1e-12)
    if variant is not None:
        head = (vertices[:, 2] > 1.53) & ~eyeball
        n[head] = F.identity_normals(p[head], n[head], seed, variant)
        p[head] = F.identity_positions(p[head], seed, variant)
        p[eyeball] += F.identity_positions(rig[eyeball, :3], seed, variant) - rig[eyeball, :3]
    return p, n


def T(t):
    m = np.eye(4); m[:3, 3] = t; return m


def pitch(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4); m[0, 0] = c; m[0, 2] = -s; m[2, 0] = s; m[2, 2] = c; return m


def roll(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4); m[1, 1] = c; m[1, 2] = -s; m[2, 1] = s; m[2, 2] = c; return m


def yaw(a):
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4); m[0, 0] = c; m[0, 1] = -s; m[1, 0] = s; m[1, 1] = c; return m


def around(p, r):
    return T(p) @ r @ T(-np.asarray(p))


def scale_around(p, s):
    m = np.eye(4); m[0, 0] = m[1, 1] = m[2, 2] = s
    return T(p) @ m @ T(-np.asarray(p))


def joint_matrices(speed=1.4, t=0., flags=0, height=1.78, seed=.3, gaze=0., nod=0., head_scale=1., phase=None):
    """-> (15,4,4) matrices for a pedestrian at ``speed`` m/s (or a given cycle ``phase`` in radians)."""
    S, sf, rate = G.gait(speed, height)
    if phase is None:
        phase = rate * t
    P = PP.params(phase, S, sf, flags, t, seed, gaze, nod)
    zp = np.array([0, 0, M.HIP_Z])
    pel = T([0, P['sway'], P['hz'] - M.HIP_Z]) @ around(zp, roll(P['roll'])) @ around(zp, yaw(P['yaw']))
    spine = pel @ around(M.SPINE, pitch(-P['lean'])) @ around(M.SPINE, roll(P['spine_roll'])) @ around(M.SPINE, yaw(P['twist']))
    out = np.tile(np.eye(4), (15, 1, 1))
    out[0] = pel
    out[1] = spine
    out[12] = spine @ around(M.NECK, yaw(P['head_yaw'])) @ around(M.NECK, pitch(-P['head_nod'])) @ scale_around(M.NECK, head_scale)
    for side, ua, fa in ((1., 2, 4), (-1., 3, 5)):
        pose = P['poses'][side]
        sh = np.array([0, side * M.SHOULDER_Y, M.SHOULDER_Z]); el = np.array([0, side * M.SHOULDER_Y, M.ELBOW_Z])
        upper = around(sh, roll(pose[1] * side)) @ around(sh, pitch(pose[0]))
        out[ua] = spine @ upper
        out[fa] = spine @ upper @ around(el, yaw(-pose[3] * side)) @ around(el, pitch(pose[2]))
    for side, th, sn, ft in ((1., 6, 8, 13), (-1., 7, 9, 14)):
        L = P['legs'][side]
        hip = np.array([0, side * M.HIP_Y, M.HIP_Z])
        thigh = T(L['hip'] - hip) @ around(hip, roll(L['roll'])) @ around(hip, pitch(L['th']))
        knee = around([0, side * M.HIP_Y, M.KNEE_Z], pitch(-L['beta']))
        out[th] = thigh
        out[sn] = thigh @ knee
        out[ft] = thigh @ knee @ around([0, side * M.HIP_Y, M.ANKLE_Z], pitch(L['rho'] + (L['th'] - L['beta']) - (L['th'] - L['beta'])))
    # coat tails
    skirt = bool(flags & M.ACC_SKIRT)
    sn_ = max(S, .05)
    for side, j in ((1., 10), (-1., 11)):
        L = P['legs'][side]
        trail = -(.06 + .50 * sf * P['on']) * (.35 if skirt else 1.)
        f = (.30 if skirt else .45) * L['th'] * P['on']
        out[j] = pel @ around(M.WAIST, pitch(trail + f))
    return out, P


def skin(v, joints, weights, mats):
    """v (n,3), joints (n,4) int, weights (n,4) -> posed positions."""
    h = np.concatenate([v, np.ones((len(v), 1))], 1)
    out = np.zeros_like(v)
    for k in range(4):
        w = weights[:, k]
        m = mats[joints[:, k].astype(int)]
        out += w[:, None] * np.einsum('nij,nj->ni', m, h)[:, :3]
    return out


def skin_normals(n, joints, weights, mats):
    out = np.zeros_like(n)
    for k in range(4):
        w = weights[:, k]
        m = mats[joints[:, k].astype(int)][:, :3, :3]
        out += w[:, None] * np.einsum('nij,nj->ni', m, n)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
