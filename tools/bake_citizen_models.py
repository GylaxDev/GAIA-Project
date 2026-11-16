"""Bake assets/humans/human_citizens.npz: the 14 supplied citizen NPCs as rigged runtime meshes.

    python tools/bake_citizen_models.py            (NumPy + SciPy + Pillow)

Input  assets/humans/reference_citizen/citizen_source.npz (+ textures/, inspection.json)
       assets/humans/human_models.npz chain via human_assets (skeleton weights, carried props)
Output assets/humans/human_citizens.npz

Pipeline per citizen
  1. weld the per-corner FBX data into an indexed mesh (position, normal, uv and material)
  2. rotate so the character faces +x (the pack faces -y), stand the soles on z = 0 and scale to
     the 1.78 m gameplay model (the instance row scales it again per pedestrian)
  3. re-pose the A-pose arms so they hang beside the body like the shader's rest skeleton
  4. skin: every vertex copies the joint influences of the nearest bare-body vertices of the
     existing male/female template, so the supplied meshes bend with the same 15 joints;
     skirts and coat hems that stand away from the legs follow the coat joints instead
  5. append the carried props (umbrella, phone, backpack, bag, coffee, watch) from the old template
  6. 24-float vertices, material M_CITIZEN, texture layer in the blink slot (2k body, 2k+1 clothes)
7. relax the hands (tools/relax_citizen_hands.py): the pack's fingers are fanned wide and ride the forearm
   joint, so they are gathered and slightly curled; otherwise they sweep forward like a reaching hand in the gait

The two source textures of each citizen are resampled to TEXTURE_SIZE and stored as one RGBA array.
Raise TEXTURE_SIZE to 1024 for the full-resolution pack (about 150 MiB of GPU memory).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from relax_citizen_hands import CURL, GATHER, HAND_LENGTH, relax_hands     # tools/ is sys.path[0] when run as a script

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(PKG))          # so the package imports as its own directory name
PKG_NAME = os.path.basename(PKG)

REF = os.path.join(PKG, 'assets/humans/reference_citizen')
OUT = os.path.join(PKG, 'assets/humans/human_citizens.npz')
TEXTURE_SIZE = 512
HEIGHT = 1.78
SH_Y, SH_Z = .205, 1.42
HANG_ROLL = .06                 # shader rest roll (POSE_SWING_ROLL): arms hang very slightly outward
M_CITIZEN = 35
(R_BODY, R_TORSO, R_UARM_L, R_UARM_R, R_FARM_L, R_FARM_R, R_THIGH_L, R_THIGH_R, R_SHIN_L, R_SHIN_R,
 R_COAT_L, R_COAT_R, R_HEAD, R_FOOT_L, R_FOOT_R) = range(15)
PROP_BITS = (1, 2, 4, 8, 2048, 262144, 4194304)   # umbrella, phones, backpack, bag, coffee, watch


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0., 1.)
    return t * t * (3. - 2. * t)


# ------------------------------------------------------------------------------ source mesh
def load_citizen(src, index, info):
    p, n, uv = src[f'mesh{index}_p'], src[f'mesh{index}_n'], src[f'mesh{index}_uv']
    f, tri_mat = src[f'mesh{index}_f'].astype(np.int64), src[f'mesh{index}_material']
    corners = f.reshape(-1)
    p, n, uv = p[corners].astype(np.float64), n[corners].astype(np.float64), uv[corners].astype(np.float64)
    mat = np.repeat(tri_mat, 3).astype(np.float64)
    key = np.concatenate([np.round(p * 1e5), np.round(n * 1e3), np.round(uv * 1e5), mat[:, None]], axis=1)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    names = [m['name'] for m in info['materials']]
    body_slot = [k for k, name in enumerate(names) if 'BODY' in name.upper()]
    if len(body_slot) != 1:
        raise ValueError(f'{info["name"]}: cannot tell body and clothing materials apart: {names}')
    return (p[first], n[first], uv[first], mat[first].astype(int), inverse.reshape(-1, 3),
            body_slot[0], [m['images'][0] for m in info['materials']])


def stand_up(p, n):
    """Face +x, soles on z = 0, 1.78 m tall, legs centred on the origin."""
    rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])       # pack faces -y -> +x (proper rotation)
    p, n = p @ rot.T, n @ rot.T
    p = p - [0., 0., p[:, 2].min()]
    scale = HEIGHT / p[:, 2].max()
    p = p * scale
    calf = p[(p[:, 2] > .30) & (p[:, 2] < .55)]
    p = p - [calf[:, 0].mean(), (calf[:, 1].min() + calf[:, 1].max()) * .5, 0.]
    return p, n, scale


def hang_arms(p, n):
    """Rotate each A-pose arm about its shoulder until it hangs beside the body.

    Also returns how much each vertex belongs to an arm (0..1, measured on the A-pose), which the
    skinning step needs: hanging hands sit right beside the thighs and must not share their weights.
    """
    src_p, p, n = p, p.copy(), n.copy()
    angles, arm_weight = {}, np.zeros(len(p))
    for side in (1., -1.):
        s = np.array([0., side * SH_Y, SH_Z])
        arm = (side * src_p[:, 1] > .12) & (src_p[:, 2] < 1.28) & (src_p[:, 2] > .55)
        if arm.sum() < 20:
            angles[side] = 0.
            continue
        tip = src_p[arm][np.argsort(side * src_p[arm, 1])[-max(8, arm.sum() // 60):]].mean(axis=0)
        d = tip - s
        current = np.arctan2(side * d[1], -d[2])                      # angle from straight down, + = outward
        angles[side] = float(current)
        rel = src_p - s
        ry, rz = side * rel[:, 1], rel[:, 2]                          # outward-positive y
        t = ry * np.sin(current) - rz * np.cos(current)
        perp = np.abs(ry * np.cos(current) + rz * np.sin(current))   # in-plane; elbows bend forward in x
        # the arm is whatever lies near the shoulder->fingertip line and outside the torso's flank;
        # below the hip line the flank is further out (thighs), so the lateral gate moves outward
        lo = .19 + .05 * smooth(1.10, .95, src_p[:, 2])
        w = smooth(.02, .11, t) * (1. - smooth(.13, .22, perp)) * smooth(lo, lo + .06, side * src_p[:, 1])
        arm_weight = np.maximum(arm_weight, w)
        if current < .10:                                             # already hanging: weights only
            continue
        a = -(current - HANG_ROLL) * w                                # swing the arm toward the body
        ca, sa = np.cos(a), np.sin(a)
        ny, nz = ry * ca - rz * sa, ry * sa + rz * ca
        move = w > 0
        p[move, 1] = side * (SH_Y + ny[move])
        p[move, 2] = SH_Z + nz[move]
        nrel_y, nrel_z = side * n[:, 1], n[:, 2]
        n[move, 1] = side * (nrel_y[move] * ca[move] - nrel_z[move] * sa[move])
        n[move, 2] = nrel_y[move] * sa[move] + nrel_z[move] * ca[move]
    return p, n, angles, arm_weight


# ------------------------------------------------------------------------------ skinning
def template_skin(ha, variant):
    """Bare-body template vertices split into an arm set and a rest-of-body set."""
    v, _ = ha.human_template(0, variant)
    body = (v[:, 23] == 0) & (v[:, 7] == 0)
    pos, joints, weights = (v[body, :3].astype(np.float64), v[body, 8:12].astype(np.int64),
                            v[body, 12:16].astype(np.float64))
    arm_share = (weights * np.isin(joints, (R_UARM_L, R_UARM_R, R_FARM_L, R_FARM_R))).sum(axis=1)
    sets = {}
    for name, pick in (('arm', arm_share > .5), ('body', arm_share <= .5)):
        sets[name] = (cKDTree(pos[pick]), joints[pick], weights[pick])
    return sets


def transfer_weights(p, subset, k=6):
    tree, joints, weights = subset
    dist, nearest = tree.query(p, k=k)
    mix = 1. / (dist + 2e-3) ** 2
    dense = np.zeros((len(p), 15))
    for c in range(k):
        for j in range(4):
            np.add.at(dense, (np.arange(len(p)), joints[nearest[:, c], j]), mix[:, c] * weights[nearest[:, c], j])
    return dense / dense.sum(axis=1, keepdims=True), dist[:, 0]


def skirt_to_coat(p, dense, gap):
    """Skirt and coat hems that stand off the legs follow the coat joints like the old garments."""
    away = smooth(.05, .10, gap)
    drop = smooth(.97, .90, p[:, 2]) * smooth(.45, .56, p[:, 2])
    amount = np.clip(away * drop, 0., 1.)
    left = smooth(-.05, .05, p[:, 1])
    dense = dense * (1. - amount)[:, None]
    dense[:, R_COAT_L] += amount * left
    dense[:, R_COAT_R] += amount * (1. - left)
    return dense


def top_four(dense):
    order = np.argsort(-dense, axis=1)[:, :4]
    w = np.take_along_axis(dense, order, axis=1)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
    return order.astype(np.float32), w.astype(np.float32)


# ------------------------------------------------------------------------------ textures
def load_texture(name):
    im = Image.open(os.path.join(REF, 'textures', name)).convert('RGB')
    im = im.resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.LANCZOS)
    rgba = np.concatenate([np.asarray(im, np.uint8), np.full((TEXTURE_SIZE, TEXTURE_SIZE, 1), 255, np.uint8)], axis=2)
    return np.ascontiguousarray(rgba[::-1])           # FBX v = 0 is the bottom row; GL row 0 is v = 0


# ------------------------------------------------------------------------------ props
def carried_props(ha):
    v, idx = ha.human_template(0, ha.MALE)
    keep = np.isin(v[:, 7].astype(np.int64), PROP_BITS)
    remap = -np.ones(len(v), np.int64)
    remap[keep] = np.arange(keep.sum())
    tris = idx.reshape(-1, 3)
    tris = tris[keep[tris].all(axis=1)]
    return np.array(v[keep], np.float32), remap[tris].astype(np.uint32)


# ------------------------------------------------------------------------------ main
def main():
    import importlib
    ha = importlib.import_module(f'{PKG_NAME}.peds.human_assets')
    hf = importlib.import_module(f'{PKG_NAME}.peds.human_faces')
    src = np.load(os.path.join(REF, 'citizen_source.npz'), allow_pickle=False)
    inspection = json.load(open(os.path.join(REF, 'inspection.json')))
    source_sha = hashlib.sha256(open(os.path.join(REF, 'citizen_source.npz'), 'rb').read()).hexdigest()
    order = [('male', i) for i in hf.CITIZEN_MALE] + [('female', i) for i in hf.CITIZEN_FEMALE]
    skins = {'male': template_skin(ha, ha.MALE), 'female': template_skin(ha, ha.FEMALE)}
    prop_v, prop_i = carried_props(ha)

    arrays, textures, report = {}, [], []
    for slot, (variant, index) in enumerate(order):
        info = inspection['objects'][index]
        p, n, uv, mat, tris, body_slot, images = load_citizen(src, index, info)
        p, n, scale = stand_up(p, n)
        p, n, angles, arm = hang_arms(p, n)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        on_arm, gap_arm = transfer_weights(p, skins[variant]['arm'])
        on_body, gap = transfer_weights(p, skins[variant]['body'])
        dense = arm[:, None] * on_arm + (1. - arm[:, None]) * on_body
        dense = skirt_to_coat(p, dense, gap * (1. - arm))
        joints, weights = top_four(dense)
        layer = 2 * slot + np.where(mat == body_slot, 0, 1)
        v = np.zeros((len(p), 24), np.float32)
        v[:, 0:3], v[:, 3:6], v[:, 6] = p, n, M_CITIZEN
        v[:, 8:12], v[:, 12:16], v[:, 16] = joints, weights, 1.
        v[:, 17:19], v[:, 19] = uv, layer
        v, _ = relax_hands(v, len(v))                      # props are appended after this, untouched
        body_img = images[body_slot]
        cloth_img = images[1 - body_slot]
        textures += [load_texture(body_img), load_texture(cloth_img)]
        full_v = np.concatenate([v, prop_v])
        full_i = np.concatenate([tris.reshape(-1).astype(np.uint32), prop_i.reshape(-1) + len(v)])
        arrays[f'v{slot}'], arrays[f'i{slot}'] = full_v, full_i
        report.append(dict(slot=slot, variant=variant, source_index=index, name=info['name'],
                           vertices=int(len(v)), triangles=int(len(tris)), prop_vertices=int(len(prop_v)),
                           scale=float(scale), arm_angle_deg=[round(np.degrees(angles[1.]), 1),
                                                              round(np.degrees(angles[-1.]), 1)],
                           body_texture=body_img, clothes_texture=cloth_img,
                           max_gap_to_body=float(gap.max())))
        print(f'{slot:2d} {variant:6s} {info["name"]:20s} {len(tris):5d} tris  arms {report[-1]["arm_angle_deg"]}  '
              f'gap<= {gap.max():.3f}')
    metadata = dict(version=1, vertex_floats=24, material=M_CITIZEN, texture_size=TEXTURE_SIZE,
                    texture_layers='2*slot = body, 2*slot+1 = clothes', source_sha256=source_sha,
                    identities=report, props=int(len(prop_v)),
                    hands_relaxed=dict(gather=GATHER, curl=CURL, hand_length=HAND_LENGTH),
                    rights=('Citizen NPC Pack 1 Character Lowpoly Mobile, 00amza (Sketchfab, listed CC BY 4.0). '
                            'Licence and ownership NOT verified; see assets/humans/CITIZEN_MODELS.md.'))
    np.savez_compressed(OUT, textures=np.stack(textures), metadata=json.dumps(metadata), **arrays)
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KiB')


if __name__ == '__main__':
    main()
