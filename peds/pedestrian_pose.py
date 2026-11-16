"""NumPy reference of the vertex-shader pose chain (developer tool + executable spec).

``pose_vertices`` takes the template vertex array and one instance's animation state and
returns posed positions / normals in model space.  shaders/pedestrian.py implements the same
maths in GLSL; this version lets the animation be checked and previewed without a GPU.
"""
from __future__ import annotations

import math

import numpy as np

from . import pedestrian_gait as G
from . import pedestrian_models as M

HIP_Z0, HIP_Y0 = M.HIP_Z, M.HIP_Y
KNEE = np.array([0., 0., M.KNEE_Z])


def _pitch(p, n, pivot, a):
    c, s = math.cos(a), math.sin(a)
    q = p - pivot
    p[:] = pivot + np.stack((c * q[:, 0] - s * q[:, 2], q[:, 1], s * q[:, 0] + c * q[:, 2]), -1)
    n[:] = np.stack((c * n[:, 0] - s * n[:, 2], n[:, 1], s * n[:, 0] + c * n[:, 2]), -1)


def _roll(p, n, pivot, a):
    c, s = math.cos(a), math.sin(a)
    q = p - pivot
    p[:] = pivot + np.stack((q[:, 0], c * q[:, 1] - s * q[:, 2], s * q[:, 1] + c * q[:, 2]), -1)
    n[:] = np.stack((n[:, 0], c * n[:, 1] - s * n[:, 2], s * n[:, 1] + c * n[:, 2]), -1)


def _yaw(p, n, pivot, a):
    c, s = math.cos(a), math.sin(a)
    q = p - pivot
    p[:] = pivot + np.stack((c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1], q[:, 2]), -1)
    n[:] = np.stack((c * n[:, 0] - s * n[:, 1], s * n[:, 0] + c * n[:, 1], n[:, 2]), -1)


def params(phase, S, sf, flags, t, seed=.3, gaze=0., nod=0.):
    """Everything the shader derives from the instance row."""
    on = G.smoothstep(.05, .25, S)
    idle = 1. if flags & M.ACC_IDLE else 0.
    breathe = math.sin(t * 1.9 + seed * 40.)
    idle_sway = .03 * math.sin(t * .8 + seed * 20.) * idle
    sway, yaw, roll, hz = G.pelvis(phase, S, sf, idle_sway)
    u0 = (phase / (2 * math.pi)) % 1.
    legs = {}
    for side, u in ((1., u0), (-1., (u0 + .5) % 1.)):
        x, y, z, psi = G.foot_target(u, side, S, sf, idle_sway / .03)
        hip = np.array([-side * HIP_Y0 * math.sin(yaw) * 1.0,
                        sway + side * HIP_Y0 * math.cos(yaw) * math.cos(roll),
                        hz + side * HIP_Y0 * math.cos(yaw) * math.sin(roll)])
        th1, beta, rl, _ = G.leg_solve(hip, (x, y, z))
        legs[side] = dict(x=x, th=th1, beta=beta, roll=rl, rho=psi - (th1 - beta), hip=hip)
    gain = (.30 + .42 * sf) * on
    sn = max(S, .05)
    arm = {}
    for side in (1., -1.):
        a = -legs[side]['x'] / sn * gain                    # opposite the leg on the same side
        arm[side] = a * (.75 if a > 0 else 1.15)            # swings further back than forward
    elbow = {side: .24 + 1.30 * sf * on + .30 * max(arm[side], 0.) + idle * .04 for side in (1., -1.)}
    poses = {side: np.array([arm[side], M.POSE_SWING_ROLL, elbow[side], 0.]) for side in (1., -1.)}
    if flags & M.ACC_UMBRELLA: poses[-1.] = np.array(M.POSE_UMBRELLA)
    if flags & M.ACC_PHONE_EAR: poses[1.] = np.array(M.POSE_EAR)
    if flags & M.ACC_PHONE_LOOK: poses[1.] = poses[-1.] = np.array(M.POSE_LOOK)
    if flags & M.POSE_POCKETS: poses[1.] = poses[-1.] = np.array(M.POSE_POCKET)
    if flags & M.POSE_CROSSED: poses[1.], poses[-1.] = np.array(M.POSE_CROSS_A), np.array(M.POSE_CROSS_B)
    if flags & M.ACC_COFFEE: poses[-1.] = np.array(M.POSE_COFFEE)
    if flags & M.POSE_WATCH: poses[1.] = np.array(M.POSE_WATCH_L)
    if flags & M.POSE_TALK:
        poses[-1.] = np.array([.55 + .20 * math.sin(t * 2.7 + seed * 9.), .12, 1.30 + .35 * math.sin(t * 3.9 + seed * 4.),
                               .25 + .20 * math.sin(t * 2.1 + seed * 2.)])
        if not (flags & (M.ACC_PHONE_EAR | M.POSE_POCKETS | M.POSE_CROSSED | M.POSE_WATCH)):
            poses[1.] = np.array([.28 + .08 * math.sin(t * 2.2 + 1.), .08, .95 + .15 * math.sin(t * 3.1), .10])
    twist = -1.6 * yaw
    lean = .02 + .17 * sf * on + idle * .012 * breathe
    head_yaw = gaze - .7 * (yaw + twist)
    head_nod = nod + (.55 if flags & M.ACC_PHONE_LOOK else 0.) + idle * .03 * breathe - .5 * lean
    return dict(on=on, sway=sway, yaw=yaw, roll=roll, hz=hz, legs=legs, poses=poses, twist=twist, lean=lean,
                spine_roll=-.8 * rollp(roll), head_yaw=head_yaw, head_nod=head_nod, sf=sf, idle=idle,
                breathe=breathe, seed=seed, t=t)


def rollp(r):
    return r


def pose_vertices(v, phase, S, sf, flags=0, t=0., seed=.3, gaze=0., nod=0.):
    P = params(phase, S, sf, flags, t, seed, gaze, nod)
    pos = v[:, 0:3].astype(np.float64).copy()
    nrm = v[:, 3:6].astype(np.float64).copy()
    role = v[:, 7].astype(int)
    pivot = v[:, 8:11].astype(np.float64)
    parent = v[:, 11:14].astype(np.float64)
    acc = v[:, 14].astype(int)
    hidden = (acc != 0) & ((acc & flags) == 0)
    hip0 = {s: np.array([0., s * HIP_Y0, HIP_Z0]) for s in (1., -1.)}
    sh0 = {s: np.array([0., s * M.SHOULDER_Y, M.SHOULDER_Z]) for s in (1., -1.)}
    el0 = {s: np.array([0., s * M.SHOULDER_Y, M.ELBOW_Z]) for s in (1., -1.)}
    ZP = np.array([0., 0., HIP_Z0])

    def sub(mask):
        return pos[mask], nrm[mask]

    def spine(p, n):
        _yaw(p, n, M.SPINE, P['twist'])
        _roll(p, n, M.SPINE, P['spine_roll'])
        _pitch(p, n, M.SPINE, -P['lean'])

    def pelvis(p, n):
        _yaw(p, n, ZP, P['yaw'])
        _roll(p, n, ZP, P['roll'])
        p += np.array([0., P['sway'], P['hz'] - HIP_Z0])

    def run(mask, fn):
        if not mask.any():
            return
        p, n = pos[mask].copy(), nrm[mask].copy()
        fn(p, n)
        pos[mask], nrm[mask] = p, n

    run(role == M.R_BODY, lambda p, n: pelvis(p, n))
    run(role == M.R_TORSO, lambda p, n: (spine(p, n), pelvis(p, n)))

    def head(p, n):
        _pitch(p, n, M.NECK, -P['head_nod'])
        _yaw(p, n, M.NECK, P['head_yaw'])
        spine(p, n)
        pelvis(p, n)
    run(role == M.R_HEAD, head)

    for side, ua, fa in ((1., M.R_UARM_L, M.R_FARM_L), (-1., M.R_UARM_R, M.R_FARM_R)):
        pose = P['poses'][side]

        def arm_up(p, n, pose=pose, side=side):
            _pitch(p, n, sh0[side], pose[0])
            _roll(p, n, sh0[side], pose[1] * side)
            spine(p, n)
            pelvis(p, n)

        def arm_fore(p, n, pose=pose, side=side):
            _pitch(p, n, el0[side], pose[2])
            _yaw(p, n, el0[side], -pose[3] * side)
            arm_up(p, n)
        run(role == ua, arm_up)
        run(role == fa, arm_fore)

    for side, th, sn, ft, ct in ((1., M.R_THIGH_L, M.R_SHIN_L, M.R_FOOT_L, M.R_COAT_L),
                                 (-1., M.R_THIGH_R, M.R_SHIN_R, M.R_FOOT_R, M.R_COAT_R)):
        L = P['legs'][side]
        shift = L['hip'] - hip0[side]

        def leg(p, n, chain, L=L, shift=shift, side=side):
            if chain >= 2:
                _pitch(p, n, np.array([0., side * HIP_Y0, M.ANKLE_Z]), L['rho'])
            if chain >= 1:
                _pitch(p, n, np.array([0., side * HIP_Y0, M.KNEE_Z]), -L['beta'])
            _pitch(p, n, hip0[side], L['th'])
            _roll(p, n, hip0[side], L['roll'])
            p += shift
        run(role == th, lambda p, n: leg(p, n, 0))
        run(role == sn, lambda p, n: leg(p, n, 1))
        run(role == ft, lambda p, n: leg(p, n, 2))

        def tail(p, n, side=side, L=L, skirt_mask=None):
            pass
        for skirt in (False, True):
            m = (role == ct) & (((acc & M.ACC_SKIRT) != 0) == skirt)
            f = (.30 if skirt else .45) * L['th'] * P['on']
            trail = -(.06 + .50 * sf * P['on']) * (.35 if skirt else 1.)
            fl = .05 * sf * math.sin(t * 8. + seed * 30. + (0. if side > 0 else 1.7))
            run(m, lambda p, n, a=trail + f + fl * (.5 if skirt else 1.) + P['idle'] * .01 * P['breathe']:
                (_pitch(p, n, M.WAIST, a), pelvis(p, n)))
    pos[hidden] = np.nan
    return pos, nrm, P
