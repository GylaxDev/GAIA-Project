"""CPU preview of assets/humans/human_citizens.npz (no GPU needed).

    python tools/preview_citizens.py sheet.png                 14 citizens, front + 3/4 + side
    python tools/preview_citizens.py sheet.png --pose walk     widest-stance frame of the shader gait (ported solveGait)
    python tools/preview_citizens.py sheet.png --pose walk --slot 3 --phases 0 .25 .5 .75   gait cycle strip
    python tools/preview_citizens.py sheet.png --pose walk --stride .5 --sf .8              faster (jog-like) gait
    python tools/preview_citizens.py sheet.png --props         include umbrella/bag/etc. meshes
    python tools/preview_citizens.py sheet.png --slot 3        one citizen, large

Unlit-ish Lambert shading of the baked texture through a z-buffer. The walk pose is a CPU port of the shader's
gait (foot trajectory, two-bone leg IK, pelvis drop/sway/yaw, arm swing, spine lean, head nod, coat follow-through),
enough to catch bad skin weights.  Skeleton lengths are derived from the joint heights in this file and the
stride numbers are plain-walk defaults, so it matches the game closely but not to the millimetre.

Rotation conventions mirror shaders/pedestrian.py exactly (pitchM / rollM / boneMatrix):
  * pitch(a) > 0 swings anything hanging below the pivot FORWARD (+x): thighs, upper arms, forearms.
  * knee flexion f is pitch(-f) (shin goes back); the head nods down with pitch(-nod); the spine leans forward
    with pitch(-lean).
  * arms and head are composed under the spine matrix (gSpineM * ...), thighs/shins/feet are not.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(PKG, 'assets/humans/human_citizens.npz')
HIP_Y, HIP_Z, KNEE_Z, ANKLE_Z = .095, .93, .50, .085
SH_Y, SH_Z, EL_Z = .205, 1.42, 1.13
NECK = np.array([0., 0., 1.50])
(R_BODY, R_TORSO, R_UARM_L, R_UARM_R, R_FARM_L, R_FARM_R, R_THIGH_L, R_THIGH_R, R_SHIN_L, R_SHIN_R,
 R_COAT_L, R_COAT_R, R_HEAD, R_FOOT_L, R_FOOT_R) = range(15)


def pitch(a):
    """Same matrix as the shader's pitchM (GLSL is column-major): x' = c*x - s*z, z' = s*x + c*z."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def roll(a):
    """Same matrix as the shader's rollM: y' = c*y - s*z, z' = s*y + c*z."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def around(pivot, rot):
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = np.asarray(pivot) - rot @ np.asarray(pivot)
    return m


SPINE = np.array([0., 0., .98])
WAIST = np.array([0., 0., 1.0])
SWING_ROLL = .06
# Skeleton lengths taken from the preview's own joint heights; the stride numbers are a plain walk.
HIP_Y0, HIP_Z0, ANKLE_H = HIP_Y, HIP_Z, ANKLE_Z
L_THIGH, L_SHIN = HIP_Z - KNEE_Z, KNEE_Z - ANKLE_Z
L_LEGMAX = .99 * (L_THIGH + L_SHIN)
S_CAP = .36
STRIDE, SF = .30, .35               # half stride length (m) and stride factor: walk .30/.35, jog ~.50/.8


def yaw(a):
    """Same matrix as the shader's yawM: x' = c*x - s*y, y' = s*x + c*y."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def translate(t):
    m = np.eye(4)
    m[:3, 3] = t
    return m


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0., 1.)
    return t * t * (3. - 2. * t)


def foot_target(u, side, S, sf):
    """Port of footTarget(): heel-to-toe stance line, lifted swing.  -> x, y, z, psi (absolute foot pitch)."""
    d = .62 - .37 * sf ** .8
    front, rear = .85 * S, 1.15 * S
    lift = .055 + .16 * sf
    if u < d:
        t = u / d
        x, z = front + (-rear - front) * t, 0.
        psi = .28 * (1. - sstep(0., .22, t)) - .75 * sstep(.55, 1., t)
    else:
        t = (u - d) / (1. - d)
        x = -rear + (front + rear) * sstep(0., 1., t)
        z = lift * np.sin(np.pi * t ** .85)
        psi = -.75 + 1.05 * sstep(0., .85, t)
    rollh = ANKLE_H * np.cos(psi) + (.07 * np.sin(psi) if psi > 0. else .17 * np.sin(-psi))
    return x, side * (HIP_Y0 + .012), rollh + z, psi


def solve_leg(hip, foot):
    """Port of solveLeg(): two-bone IK -> thigh pitch, knee flexion, hip roll, foot pitch relative to the shin."""
    ax, ay, az = foot[0] - hip[0], foot[1] - hip[1], hip[2] - foot[2]
    rl = np.arctan2(ay, az)
    dy = np.hypot(az, ay)
    d = np.hypot(ax, dy)
    dc = np.clip(d, .12, L_LEGMAX)
    cg = np.clip((L_THIGH ** 2 + dc ** 2 - L_SHIN ** 2) / (2. * L_THIGH * dc), -1., 1.)
    ck = np.clip((L_THIGH ** 2 + L_SHIN ** 2 - dc ** 2) / (2. * L_THIGH * L_SHIN), -1., 1.)
    th = np.arctan2(ax, dy) + np.arccos(cg)
    beta = np.pi - np.arccos(ck)
    return th, beta, rl, foot[3] - (th - beta)


def walk_pose(phase=.05, S=None, sf=None, skirt=False):
    """Joint matrices computed like shaders/pedestrian.py solveGait() + boneMatrix().

    phase in [0, 1): 0 is the left foot's heel strike; the default .05 is the widest stance.
    """
    S, sf = STRIDE if S is None else S, SF if sf is None else sf
    k = min(S / S_CAP, 1.)
    u0 = phase % 1.
    shift = np.cos(2 * np.pi * (u0 - .31))
    sway = .028 * k * shift
    g_yaw = -.075 * k * np.cos(2 * np.pi * u0)
    g_roll = .030 * k * shift
    fl = foot_target(u0, 1., S, sf)
    fr = foot_target((u0 + .5) % 1., -1., S, sf)
    reach = [f[2] + np.sqrt(max(L_LEGMAX ** 2 - f[0] ** 2 - (f[1] - (s * HIP_Y0 + sway)) ** 2, .01))
             for f, s in ((fl, 1.), (fr, -1.))]
    g_z = max(min(HIP_Z0 - .008, min(reach)), HIP_Z0 - .16)
    cy, sy, cr, sr = np.cos(g_yaw), np.sin(g_yaw), np.cos(g_roll), np.sin(g_roll)
    hips = {1.: np.array([-HIP_Y0 * sy, sway + HIP_Y0 * cy * cr, g_z + HIP_Y0 * cy * sr]),
            -1.: np.array([HIP_Y0 * sy, sway - HIP_Y0 * cy * cr, g_z - HIP_Y0 * cy * sr])}
    legs = {1.: solve_leg(hips[1.], fl), -1.: solve_leg(hips[-1.], fr)}

    gain, sn = .30 + .42 * sf, max(S, .05)
    lean = .02 + .17 * sf
    twist = -1.6 * g_yaw
    spine_roll = -.8 * g_roll
    head_yaw = -.7 * (g_yaw + twist)
    nod = -.5 * lean
    zp = [0, 0, HIP_Z0]
    pelvis = translate([0, sway, g_z - HIP_Z0]) @ around(zp, roll(g_roll)) @ around(zp, yaw(g_yaw))
    spine = pelvis @ around(SPINE, pitch(-lean)) @ around(SPINE, roll(spine_roll)) @ around(SPINE, yaw(twist))

    m = [np.eye(4) for _ in range(15)]
    m[R_BODY], m[R_TORSO] = pelvis, spine
    for side, foot in ((1., fl), (-1., fr)):
        left = side > 0
        th, flex, rl, wf = legs[side]
        hip = np.array([0, side * HIP_Y, HIP_Z])
        thigh = translate(hips[side] - hip) @ around(hip, roll(rl)) @ around(hip, pitch(th))
        shin = thigh @ around([0, side * HIP_Y, KNEE_Z], pitch(-flex))
        sole = shin @ around([0, side * HIP_Y, ANKLE_Z], pitch(wf))
        k3 = (R_THIGH_L, R_SHIN_L, R_FOOT_L) if left else (R_THIGH_R, R_SHIN_R, R_FOOT_R)
        m[k3[0]], m[k3[1]], m[k3[2]] = thigh, shin, sole
        # arms swing opposite the leg on their side; further back than forward
        a = -foot[0] / sn * gain
        a *= .65 if a > 0 else 1.15
        e = .24 + (-.09 + 1.20 * max(sf, 0.) ** 1.3) + .35 * max(a, 0.)      # relaxed walk, folds with the swing
        w = .08 + .30 * max(a, 0.)                                            # forearm toward the midline
        sh, el = [0, side * SH_Y, SH_Z], [0, side * SH_Y, EL_Z]
        upper = around(sh, roll(SWING_ROLL * side)) @ around(sh, pitch(a))
        arm = (R_UARM_L, R_FARM_L) if left else (R_UARM_R, R_FARM_R)
        m[arm[0]] = spine @ upper
        m[arm[1]] = spine @ upper @ around(el, yaw(-w * side)) @ around(el, pitch(e))
        trail = -(.06 + .50 * sf) * (.35 if skirt else 1.)
        tail = trail + (.30 if skirt else .45) * th
        m[R_COAT_L if left else R_COAT_R] = pelvis @ around(WAIST, pitch(tail))
    m[R_HEAD] = spine @ around(NECK, yaw(head_yaw)) @ around(NECK, pitch(-nod))
    return m


def skin(v, mats):
    out = np.zeros((len(v), 3))
    nrm = np.zeros((len(v), 3))
    home = np.c_[v[:, :3], np.ones(len(v))]
    for k in range(4):
        w = v[:, 12 + k:13 + k]
        mk = np.stack([mats[int(j)] for j in v[:, 8 + k]])           # (N, 4, 4)
        out += w * np.einsum('nij,nj->ni', mk, home)[:, :3]
        nrm += w * np.einsum('nij,nj->ni', mk[:, :3, :3], v[:, 3:6])
    return out, nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)


def rasterize(pos, nrm, uv, layer, tris, tex, yaw, size=(300, 520), scale=270., light=(.4, .5, .75)):
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])           # yaw about z, then look down -x from +x
    p, n = pos @ rot.T, nrm @ rot.T
    w, h = size
    sx = w * .5 + p[:, 1] * scale
    sy = h - 16 - p[:, 2] * scale
    depth = p[:, 0]
    img = np.full((h, w, 3), 255, np.float32)
    zbuf = np.full((h, w), -1e9, np.float32)
    ld = np.array(light) / np.linalg.norm(light)
    th, tw = tex.shape[1:3]
    for a, b, c3 in tris:
        xs, ys = sx[[a, b, c3]], sy[[a, b, c3]]
        x0, x1 = int(max(np.floor(xs.min()), 0)), int(min(np.ceil(xs.max()), w - 1))
        y0, y1 = int(max(np.floor(ys.min()), 0)), int(min(np.ceil(ys.max()), h - 1))
        if x1 < x0 or y1 < y0:
            continue
        det = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(det) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
        l0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / det
        l1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / det
        l2 = 1. - l0 - l1
        inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not inside.any():
            continue
        z = l0 * depth[a] + l1 * depth[b] + l2 * depth[c3]
        view = zbuf[y0:y1 + 1, x0:x1 + 1]
        take = inside & (z > view)
        if not take.any():
            continue
        view[take] = z[take]
        u = (l0 * uv[a, 0] + l1 * uv[b, 0] + l2 * uv[c3, 0])[take]
        v = (l0 * uv[a, 1] + l1 * uv[b, 1] + l2 * uv[c3, 1])[take]
        col = tex[int(layer[a])][np.clip((v * th).astype(int), 0, th - 1), np.clip((u * tw).astype(int), 0, tw - 1), :3]
        nn = (l0[take, None] * n[a] + l1[take, None] * n[b] + l2[take, None] * n[c3])
        nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-9)
        lam = np.abs(nn @ ld)                                    # double sided
        img[y0:y1 + 1, x0:x1 + 1][take] = col * (.55 + .5 * lam[:, None])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def render_slot(data, slot, yaw, pose=None, props=False, size=(300, 520), scale=270.):
    v, idx = data[f'v{slot}'], data[f'i{slot}'].reshape(-1, 3)
    body_tris = data['meta']['identities'][slot]['triangles']
    tris = idx if props else idx[:body_tris]
    if pose is None:
        pos, nrm = v[:, :3].astype(np.float64), v[:, 3:6].astype(np.float64)
    else:
        pos, nrm = skin(v.astype(np.float64), pose)
    return rasterize(pos, nrm, v[:, 17:19], v[:, 19], tris, data['textures'], yaw, size, scale)


def load():
    z = np.load(PATH, allow_pickle=False)
    data = {k: z[k] for k in z.files if k != 'metadata'}
    data['meta'] = json.loads(str(z['metadata']))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--pose', choices=('rest', 'walk'), default='rest')
    ap.add_argument('--props', action='store_true')
    ap.add_argument('--slot', type=int)
    ap.add_argument('--yaws', type=float, nargs='*', default=[0., 45., 90.])
    ap.add_argument('--stride', type=float, help='half stride length in m (default .30)')
    ap.add_argument('--sf', type=float, help='stride factor 0..1 (default .35)')
    ap.add_argument('--phases', type=float, nargs='*',
                    help='gait cycle fractions (0 = left heel strike); one column per phase at the first yaw')
    args = ap.parse_args()
    data = load()
    gait = dict(S=args.stride, sf=args.sf)
    slots = [args.slot] if args.slot is not None else range(len(data['meta']['identities']))
    size, scale = ((600, 1040), 540.) if args.slot is not None else ((300, 520), 270.)
    if args.pose == 'walk' or args.phases:                       # a side-on stride is wider than a standing figure
        size = (int(size[0] * 1.5), size[1])
    tiles = []
    for slot in slots:
        skirt = data['meta']['identities'][slot]['variant'] == 'female'
        if args.phases:
            yaw_ = np.radians(args.yaws[0] if args.yaws else 90.)
            row = [render_slot(data, slot, yaw_, walk_pose(ph, skirt=skirt, **gait), args.props, size, scale) for ph in args.phases]
        else:
            pose = walk_pose(skirt=skirt, **gait) if args.pose == 'walk' else None
            row = [render_slot(data, slot, np.radians(y), pose, args.props, size, scale) for y in args.yaws]
        tile = Image.new('RGB', (size[0] * len(row), size[1]), 'white')
        for k, im in enumerate(row):
            tile.paste(im, (k * size[0], 0))
        ImageDraw.Draw(tile).text((6, 4), f'slot {slot} {data["meta"]["identities"][slot]["variant"]}', fill=(0, 0, 0))
        tiles.append(tile)
    cols = 1 if args.slot is not None else 2
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new('RGB', (tiles[0].width * cols, tiles[0].height * rows), 'white')
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * t.width, (k // cols) * t.height))
    sheet.save(args.out)
    print('wrote', args.out, sheet.size)


if __name__ == '__main__':
    main()
