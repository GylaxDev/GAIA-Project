"""Relax the open, fanned A-pose hands of the baked citizens (assets/humans/human_citizens.npz).

    python tools/relax_citizen_hands.py              edit the asset in place (idempotent; skips if already done)
    python tools/relax_citizen_hands.py --check      report what would change, write nothing
    python tools/relax_citizen_hands.py --force      apply again on top of an already relaxed asset

Why: the supplied pack models hands with the fingers spread wide in the front/back plane. They have no
finger joints in the 15-joint gameplay skeleton and ride the forearm joint, so as soon as the elbow flexes in
the gait the whole fan sweeps forward like a "reaching" hand.  This pulls the fingers together (like a relaxed
hand) and curls the fingertips a little toward the palm.  Only vertices that are
    * body geometry (not carried props),
    * skin texture (even layer: 2*slot = body),
    * dominated by a forearm joint (R_FARM_L / R_FARM_R),
    * and below the wrist
are moved; skin weights, UVs, indices and every other vertex stay byte-identical.  The edit is recorded in the
metadata (`hands_relaxed`) so the tool is safe to run twice.  bake_citizen_models.py calls relax_hands() as its
last mesh step, so a rebake keeps the result.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(PKG, 'assets/humans/human_citizens.npz')
R_FARM_L, R_FARM_R = 4, 5
HAND_LENGTH = .19            # wrist -> fingertip of the 1.78 m model
GATHER = .55                 # fingers end up this much narrower (front/back) than the baked fan
CURL = .022                  # fingertip travel toward the palm (m)
PALM = .05                   # the palm (wrist .. wrist-PALM) is left alone so the hand stays attached to the cuff


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0., 1.)
    return t * t * (3. - 2. * t)


def relax_hands(v, body_vertices):
    """Return (new_v, report).  v is the (N, 24) vertex array of one citizen; the first body_vertices rows are the
    body (props are appended after them).  Pure function: v is not modified."""
    out = v.copy()
    body = out[:body_vertices]
    joints, weights = body[:, 8:12].astype(int), body[:, 12:16]
    dominant = joints[np.arange(len(body)), weights.argmax(1)]
    skin = body[:, 19].astype(int) % 2 == 0
    report = {}
    for joint, side in ((R_FARM_L, 1.), (R_FARM_R, -1.)):
        arm = (dominant == joint) & skin & (body[:, 7] == 0) & (body[:, 2] < .96)
        if not arm.any():
            continue
        tip = float(body[arm, 2].min())
        wrist = tip + HAND_LENGTH
        hand = arm & (body[:, 2] < wrist - PALM * .25)
        if not hand.any():
            continue
        z = body[hand, 2]
        depth = np.clip((wrist - PALM - z) / (HAND_LENGTH - PALM), 0., 1.)       # 0 at the palm, 1 at the tips
        along = smooth(0., 1., depth)
        xc = float(np.median(body[hand & (body[:, 2] < wrist - .07), 0]))
        k = 1. - GATHER * along
        body[hand, 0] = xc + (body[hand, 0] - xc) * k
        body[hand, 1] -= side * CURL * depth ** 1.5                                # toward the palm (medial)
        report['L' if side > 0 else 'R'] = dict(vertices=int(hand.sum()), tip_z=round(tip, 3), wrist_z=round(wrist, 3),
                                                centre_x=round(xc, 3))
    out[:body_vertices] = body
    return out, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    z = np.load(PATH, allow_pickle=False)
    arrays = {k: z[k] for k in z.files if k != 'metadata'}
    meta = json.loads(str(z['metadata']))
    if meta.get('hands_relaxed') and not args.force:
        print('hands already relaxed; use --force to apply again')
        return
    for slot, ident in enumerate(meta['identities']):
        body_vertices = ident['vertices']
        v = arrays[f'v{slot}']
        new, report = relax_hands(v, body_vertices)
        moved = float(np.abs(new[:, :3] - v[:, :3]).max())
        print(f'slot {slot:2d} {ident["variant"]:6s} {report}  max move {moved * 100:.1f} cm')
        arrays[f'v{slot}'] = new.astype(np.float32)
    if args.check:
        return
    meta['hands_relaxed'] = dict(gather=GATHER, curl=CURL, hand_length=HAND_LENGTH)
    np.savez_compressed(PATH, metadata=json.dumps(meta), **arrays)
    print('wrote', PATH, os.path.getsize(PATH) // 1024, 'KiB')


if __name__ == '__main__':
    main()
