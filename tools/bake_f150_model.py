"""Bake assets/vehicles/f150_2015.npz from the exported source dumps.

The supplied 2015 Ford F-150 Lariat Super Cab FBX is exported by Blender into
three per-LOD ``source_dumpN.npz`` files (one vertex record per triangle corner,
with normals, UVs, object ids and material ids).  This tool maps the source
materials onto GAIA's ten-float instanced vehicle format:

  * exterior paint / bare body panels -> P_PAINT (instance colour)
  * the original source cabin stays in the mesh and is recoloured as a light
    tan Lariat interior, shaded by the source diffuse atlas luminance
  * glass -> P_WINDOW, ordered last and counted in ``metadata['glass']``
  * wheel assemblies get per-wheel pivots: front spin + steer, rear spin,
    front calipers steer (the source has no rear calipers)
  * lamp internals keep the source reflector geometry; the C-clamp light guide
    is the LED signature, the projector cards are warm headlamps, and the rear
    glow cards become tail/brake/reverse emitters behind the red covers

Run from the repository root:

    python tools/bake_f150_model.py --source <dir with source_dump0..2.npz>
                                    --textures <dir with file2.png, file5.png>
                                    --out assets/vehicles/f150_2015.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from vehicles.vehicle_models import (  # noqa: E402
    P_PAINT, P_TYRE, P_HEAD, P_TAIL, P_CHROME, P_TRIM, P_PLATE, P_FIXED,
    P_METAL, P_MATTE, P_PIANO, P_WINDOW, P_LENS_CLEAR, P_LENS_RED, P_LEFT, P_RIGHT,
    P_LED, STATIC, REAR_SPIN, FRONT_SPIN, FRONT_STEER, ONE, GLASS, TRIM, PLATE,
    HEAD, TAIL, LED_COLOR, AMBER,
)
from vehicles.vehicle_reference import F150  # noqa: E402

# --- source transform (metres; +x forward, +y left, z = 0 at the road) -------
WHEELBASE_SRC = 1.5399726033210754 + 1.5363284945487976
SCALE = F150.wheelbase / WHEELBASE_SRC
FRONT_AXLE_SRC = 1.5399726033210754
SHIFT = F150.front_axle - FRONT_AXLE_SRC * SCALE
GROUND = 0.09134405106306076

# --- finishes (sRGB; the shader linearises) ---------------------------------
DARK_PLASTIC = (0.045, 0.047, 0.050)
PLASTIC_TRIM = (0.075, 0.076, 0.080)
CHROME_WHITE = (0.82, 0.83, 0.84)
PIANO = (0.018, 0.018, 0.022)
TAN = (0.44, 0.34, 0.235)
TAN_LIGHT = (0.56, 0.46, 0.335)
TAN_DARK = (0.28, 0.215, 0.15)
DASH = (0.075, 0.076, 0.082)
HEADLINER = (0.56, 0.54, 0.48)
CARPET = (0.050, 0.050, 0.055)
CALIPER = (0.20, 0.205, 0.20)
REVERSE_LENS = (0.86, 0.87, 0.84)


def transform(pos, nrm):
    p = np.asarray(pos, np.float64)
    out = np.empty_like(p)
    out[:, 0] = p[:, 0] * SCALE + SHIFT
    out[:, 1] = p[:, 1] * SCALE
    out[:, 2] = (p[:, 2] - GROUND) * SCALE
    return out, np.asarray(nrm, np.float64)


def load_textures(folder):
    out = {}
    for name in ('file2', 'file5'):
        path = folder / f'{name}.png'
        out[name] = np.asarray(Image.open(path).convert('RGB'), np.float32) / 255.0
    return out


def sample_tex(img, uvs):
    h, w, _ = img.shape
    u = np.mod(uvs[..., 0], 1.0)
    v = np.mod(uvs[..., 1], 1.0)
    xx = np.clip((u * (w - 1)).round().astype(np.int32), 0, w - 1)
    yy = np.clip(((1.0 - v) * (h - 1)).round().astype(np.int32), 0, h - 1)
    return img[yy, xx]


def luminance(rgb):
    return rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114


def sample_tri(img, uvs):
    """Per-triangle albedo: the mean of the three corner texels."""
    return sample_tex(img, uvs).mean(1)


def components(tris, points):
    """Triangle-connected components -> list of triangle indices."""
    key = np.round(points.reshape(-1, 3) / 2e-4).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    parent = np.arange(int(inv.max()) + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for k in range(len(tris)):
        a, b, c = inv[k * 3:k * 3 + 3]
        for u, v in ((a, b), (b, c)):
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[ru] = rv
    groups = defaultdict(list)
    for k in range(len(tris)):
        groups[find(int(inv[k * 3]))].append(k)
    return list(groups.values())


def add_box(V, I, A, lo, hi, colour, part, role=STATIC, pivot=(0, 0, 0)):
    """Append one axis-aligned box with per-corner vertices."""
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    corners = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    faces = ((0, 3, 2, 1, (0, 0, -1)), (4, 5, 6, 7, (0, 0, 1)),
             (0, 4, 7, 3, (-1, 0, 0)), (1, 2, 6, 5, (1, 0, 0)),
             (0, 1, 5, 4, (0, -1, 0)), (3, 7, 6, 2, (0, 1, 0)))
    base = len(V)
    for f in faces:
        quad, n = f[:4], np.asarray(f[4], np.float64)
        for k in (0, 1, 2, 0, 2, 3):
            V.append(np.r_[corners[quad[k]], n, colour, part].astype(np.float32))
            A.append(np.r_[pivot, role].astype(np.float32))
    I.extend(base + np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7,
                              8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15,
                              16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23],
                             np.int64))


def build_lod(lod, tex, pivots):
    folder = Path(tex['_folder'])
    d = np.load(folder / f'source_dump{lod}.npz')
    pos = d['pos'].astype(np.float64)
    nrm = d['nrm'].astype(np.float64)
    uv = d['uv'].astype(np.float64)
    tri = d['tri'].astype(np.int64)
    tobj = d['tri_obj'].astype(np.int64)
    tmat = d['tri_mat'].astype(np.int64)
    objs = [str(x) for x in d['obj_names']]
    mats = [str(x) for x in d['materials']]
    off = d['obj_slot_offsets'].astype(np.int64)

    # reject zero-area triangles before anything counts indices
    pa = pos[tri]
    keep = np.linalg.norm(np.cross(pa[:, 1] - pa[:, 0], pa[:, 2] - pa[:, 0]), axis=1) > 1e-10
    tri, tobj, tmat = tri[keep], tobj[keep], tmat[keep]

    p, n = transform(pos, nrm)
    uv_tri = uv[tri]
    cent = p[tri].mean(1)
    fn = n[tri].mean(1)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    x, y, z = cent[:, 0], cent[:, 1], cent[:, 2]
    mat = np.array([mats[off[tobj[i]] + tmat[i]] for i in range(len(tmat))])

    T = len(tri)
    part = np.full(T, P_MATTE, np.int8)
    colour = np.tile(np.asarray(DARK_PLASTIC, np.float32), (T, 1))
    role = np.zeros(T, np.int8)
    pivot = np.zeros((T, 3), np.float32)
    for oi, name in enumerate(objs):
        sel = tobj == oi
        role[sel] = pivots[name][0]
        pivot[sel] = pivots[name][1]

    def put(mask, pid, rgb):
        part[mask] = pid
        colour[mask] = rgb

    # --------------------------------------------------------------- interior
    paint = np.isin(mat, ('CarPaint', 'CP'))
    plastic = np.isin(mat, ('CP_PLastic', 'plastic_Det'))
    box = (x > -0.78) & (x < 1.58) & (np.abs(y) < 0.96) & (z > 0.80) & (z < 1.88)
    ref = np.array([0.34, 0.0, 1.14])
    inward = np.einsum('ij,ij->i', fn, ref[None, :] - cent) > 0.0
    deep = (x > -0.62) & (x < 1.05) & (np.abs(y) < 0.80) & (z > 0.80) & (z < 1.66)

    wheelwell = np.zeros(T, bool)
    for wx, wy in ((1.985, .8466), (1.985, -.8582), (-1.698, .8329), (-1.698, -.8311)):
        wc = np.array([wx, wy, .4376])
        r = np.hypot(x - wx, z - .4376)
        toward = np.einsum('ij,ij->i', fn, wc[None, :] - cent) > 0.0
        wheelwell |= (r < 0.60) & (np.abs(y) > 0.45) & (z < 0.98) & toward

    interior = (paint | plastic) & box & (deep | inward) & ~wheelwell

    # The top of the dashboard sits above the ``deep`` volume and faces up, so
    # the generic test above leaves it as body paint.  It is visible through the
    # windscreen; keep it dark like the rest of the dash instead of taking the
    # instance colour.
    dash_top = (paint & (x > 1.02) & (x < 1.52) & (np.abs(y) < .90)
                & (z > 1.12) & (z < 1.45) & (fn[:, 2] > .45))
    interior |= dash_top

    # interior component roles (shell / steering / seats / headrests / floor)
    ids = np.flatnonzero(interior)
    role_name = np.full(T, '', dtype=object)
    if len(ids):
        comps = components(tri[ids], p[tri[ids]])
        for comp in comps:
            a = ids[np.asarray(comp, np.int64)]
            c = cent[a]
            if len(a) >= 400:
                name = 'shell'
            elif len(a) >= 100:
                name = 'steering'
            elif np.ptp(c[:, 0]) > 1.5:
                name = 'floor'
            elif c[:, 2].min() > 1.30:
                name = 'headrest'
            elif np.ptp(c[:, 0]) < 0.80 and c[:, 2].max() > 1.20:
                name = 'seat'
            else:
                name = 'trim'
            role_name[a] = name

    def base_colour(a):
        c = cent[a]
        ax, ay, az = c[:, 0], np.abs(c[:, 1]), c[:, 2]
        name = role_name[a]
        out = np.tile(np.asarray(DASH, np.float32), (len(a), 1))
        out[:] = np.where((name == 'seat')[:, None], TAN, out)
        out = np.where((name == 'headrest')[:, None], TAN_LIGHT, out)
        out = np.where((name == 'steering')[:, None], DASH, out)
        out = np.where((name == 'trim')[:, None], DARK_PLASTIC, out)
        fl = name == 'floor'
        out = np.where(fl[:, None] & (az < 1.0)[:, None], CARPET, out)
        out = np.where(fl[:, None] & (az >= 1.0)[:, None], TAN, out)
        sh = name == 'shell'
        out = np.where(sh[:, None] & (az > 1.70)[:, None], HEADLINER, out)
        out = np.where(sh[:, None] & (az < 0.98)[:, None], CARPET, out)
        out = np.where(sh[:, None] & (ax > 1.08)[:, None], DASH, out)
        out = np.where(sh[:, None] & (ax < -0.55)[:, None], DASH, out)
        door = sh & (ay > 0.72)
        out = np.where(door[:, None] & (az > 1.30)[:, None], DASH, out)
        out = np.where(door[:, None] & (az > 1.04)[:, None] & (az <= 1.30)[:, None], TAN, out)
        out = np.where(door[:, None] & (az <= 1.04)[:, None], CARPET, out)
        return out

    # shade interior finishes with the diffuse atlas's baked-in shadows
    shade = luminance(sample_tex(tex['file2'], uv_tri)).mean(1)
    shade = 0.62 + 0.38 * np.clip(shade * 1.5, 0.0, 1.0)

    m = paint & interior
    base = np.tile(np.asarray(DASH, np.float32), (T, 1))
    if m.any():
        base[m] = base_colour(np.flatnonzero(m))
    colour[m] = base[m] * shade[m][:, None]
    part[m] = P_MATTE

    # dark upper dash / cowl cover, regardless of the component split above
    put(paint & dash_top, P_MATTE, DASH)

    m = (mat == 'CP_PLastic') & interior
    if m.any():
        base[m] = base_colour(np.flatnonzero(m)) * 0.85
        colour[m] = base[m]
    part[(mat == 'CP_PLastic') & interior] = P_MATTE

    m = (mat == 'plastic_Det') & interior
    if m.any():
        base[m] = np.minimum(base_colour(np.flatnonzero(m)) * 0.70, 0.35)
        colour[m] = base[m]
    part[m] = P_MATTE

    m = (mat == 'CP_Chrome') & box & (deep | inward)
    put(m, P_CHROME, CHROME_WHITE)

    # ---------------------------------------------------------------- exterior
    m = paint & ~interior
    put(m, P_PAINT, ONE)
    m = (mat == 'CarPaint') & ~interior
    put(m, P_PAINT, ONE)

    m = paint & ~interior & wheelwell
    put(m, P_MATTE, (0.030, 0.031, 0.033))

    m = (mat == 'CP_Chrome') & ~((mat == 'CP_Chrome') & box & (deep | inward))
    put(m, P_CHROME, CHROME_WHITE)

    m = (mat == 'CP_PLastic') & ~interior
    put(m, P_MATTE, PLASTIC_TRIM)

    m = (mat == 'plastic_Det') & ~interior
    put(m, P_MATTE, DARK_PLASTIC)

    m = mat == 'cab_Color'
    if m.any():
        rgb = sample_tri(tex['file2'], uv_tri[m])
        colour[m] = np.clip(rgb, 0.0, 0.90)
    part[m] = P_MATTE

    m = mat == 'plate'
    put(m, P_PLATE, PLATE)

    m = mat == 'rims'
    put(m, P_TRIM, TRIM)
    # Rim_Color's atlas UVs land in unrelated green/black texels; the alloy
    # itself is a uniform satin silver, so do not sample the body atlas here.
    m = mat == 'Rim_Color'
    put(m, P_METAL, (0.58, 0.59, 0.61))

    m = mat == 'tire'
    if m.any():
        rgb = np.clip(sample_tri(tex['file5'], uv_tri[m]), 0.0, 0.55)
        colour[m] = rgb
    part[m] = P_TYRE

    m = mat == 'under'
    put(m, P_TRIM, TRIM)

    # ------------------------------------------------------------ detail atlas
    m = mat == 'details_R'
    caliper = m & np.array([('caliper' in objs[o]) for o in tobj])
    put(caliper, P_FIXED, CALIPER)

    front_lamp = (x > 2.35) & (np.abs(y) > 0.52) & (z > 0.72) & (z < 1.25)
    rear_badge = (x < -2.82) & (np.abs(y) < 0.80) & (z > 0.90) & (z < 1.20)
    front_badge = (x > 2.82) & (np.abs(y) < 0.20) & (z > 0.84) & (z < 1.10)

    sel = m & front_lamp & ~caliper
    if sel.any():
        rgb = sample_tri(tex['file2'], uv_tri[sel])
        lum = luminance(rgb)
        bright = sel.copy()
        bright[sel] = lum > 0.30
        put(bright, P_CHROME, CHROME_WHITE)
        put(sel & ~bright, P_PIANO, PIANO)

    sel = m & rear_badge & ~caliper
    if sel.any():
        rgb = sample_tri(tex['file2'], uv_tri[sel])
        blue = (rgb[:, 2] > np.maximum(rgb[:, 0] * 1.22, 0.28)) & (rgb[:, 2] > rgb[:, 1] * 1.10)
        blue_tri = sel.copy()
        blue_tri[sel] = blue
        part[blue_tri] = P_FIXED
        colour[blue_tri] = np.clip(rgb[blue], 0.0, 1.0)
        silver = sel & ~blue_tri
        put(silver, P_METAL, np.clip(rgb[~blue].mean(0), 0.35, 0.9))

    sel = m & front_badge & ~caliper
    if sel.any():
        rgb = sample_tri(tex['file2'], uv_tri[sel])
        blue = (rgb[:, 2] > np.maximum(rgb[:, 0] * 1.22, 0.28)) & (rgb[:, 2] > rgb[:, 1] * 1.10)
        blue_tri = sel.copy()
        blue_tri[sel] = blue
        part[blue_tri] = P_FIXED
        colour[blue_tri] = np.clip(rgb[blue], 0.0, 1.0)
        silver = sel & ~blue_tri
        put(silver, P_METAL, np.clip(rgb[~blue].mean(0), 0.35, 0.9))

    sel = m & ~caliper & ~front_lamp & ~rear_badge & ~front_badge
    put(sel, P_MATTE, PLASTIC_TRIM)

    m = mat == 'details_sh'
    rear_lens = (x < -2.45) & (np.abs(y) > 0.75) & (z > 0.70) & (z < 1.25)
    third_brake = (x < -0.60) & (x > -0.80) & (z > 1.75)
    mirror_lamp = (x > 0.95) & (x < 1.20) & (np.abs(y) > 1.15) & (z > 1.20) & (z < 1.35)
    front_badge = (x > 2.84) & (np.abs(y) < 0.18) & (z > 0.85) & (z < 1.08)
    side_badge = (x > 1.35) & (x < 1.75) & (np.abs(y) > 0.92) & (z > 1.10) & (z < 1.24)
    put(m & (rear_lens | third_brake), P_LENS_RED, (0.62, 0.075, 0.055))
    lm = m & mirror_lamp
    part[lm & (y < 0)] = P_RIGHT
    part[lm & (y >= 0)] = P_LEFT
    colour[lm] = AMBER
    sel = m & front_badge
    if sel.any():
        rgb = sample_tri(tex['file2'], uv_tri[sel])
        colour[sel] = np.clip(rgb, 0.0, 1.0)
    part[sel] = P_FIXED
    sel = m & side_badge
    if sel.any():
        rgb = sample_tri(tex['file2'], uv_tri[sel])
        bright = rgb.mean(1) > 0.45
        chrome_tri = sel.copy()
        chrome_tri[sel] = bright
        put(chrome_tri, P_CHROME, CHROME_WHITE)
        put(sel & ~chrome_tri, P_PIANO, PIANO)
    other = m & ~(rear_lens | third_brake | mirror_lamp | front_badge | side_badge)
    put(other, P_CHROME, CHROME_WHITE)

    m = mat == 'glows'
    front_glow = m & (x > 0.0)
    if front_glow.any():
        # The C-clamp light guide is the LED daytime signature; the two tiny
        # cards over the projectors are the warm headlamp emitters.
        comps = components(tri[front_glow], p[tri[front_glow]])
        fids = np.flatnonzero(front_glow)
        for comp in comps:
            a = fids[np.asarray(comp, np.int64)]
            if len(a) <= 3:
                part[a] = P_HEAD
                colour[a] = HEAD
            else:
                part[a] = P_LED
                colour[a] = LED_COLOR
    reverse = m & np.array([('reverse' in objs[o]) for o in tobj])
    put(reverse & ~front_glow, P_FIXED, REVERSE_LENS)
    rest = m & ~front_glow & ~reverse
    put(rest, P_TAIL, TAIL)

    # glass is translucent and must be the last block
    put(mat == 'windiws', P_WINDOW, GLASS)

    # ------------------------------------------------------- extra cabin detail
    V = np.empty((T * 3, 10), np.float32)
    V[:, :3] = p[tri].reshape(-1, 3)
    V[:, 3:6] = n[tri].reshape(-1, 3)
    V[:, 6:9] = np.repeat(colour, 3, axis=0)
    V[:, 9] = np.repeat(part, 3)
    A = np.repeat(np.column_stack((pivot, role.astype(np.float32))), 3, axis=0).astype(np.float32)
    I = np.arange(T * 3, dtype=np.int64)

    # the source cabin has no centre console between the front buckets
    Vx, Ix, Ax = [], [], []
    add_box(Vx, Ix, Ax, (0.22, -0.105, 0.875), (0.92, 0.105, 1.000), TAN_DARK, P_MATTE)
    add_box(Vx, Ix, Ax, (0.46, -0.095, 1.000), (0.76, 0.095, 1.035), TAN, P_MATTE)
    add_box(Vx, Ix, Ax, (0.86, -0.055, 1.000), (0.92, 0.055, 1.075), PIANO, P_PIANO)
    V = np.concatenate((V, np.asarray(Vx, np.float32)))
    I = np.concatenate((I, np.asarray(Ix, np.int64)))
    A = np.concatenate((A, np.asarray(Ax, np.float32)))

    # ----------------------------------------------------------- order & counts
    translucent = np.concatenate((np.isin(part, (P_LENS_CLEAR, P_LENS_RED, P_WINDOW)),
                                  np.zeros(len(Ix) // 3, bool)))
    order = np.concatenate((np.flatnonzero(~translucent), np.flatnonzero(translucent)))
    V = V.reshape(-1, 3, 10)[order].reshape(-1, 10)
    A = A.reshape(-1, 3, 4)[order].reshape(-1, 4)
    I = np.arange(len(V), dtype=np.uint32)

    glass_count = 3 * int(translucent.sum())
    return V, I, A, glass_count, int(translucent.sum()), int(T), cent, part


def wheel_pivots(d0):
    pos = d0['pos'].astype(np.float64)
    tri = d0['tri']
    tobj = d0['tri_obj']
    objs = [str(x) for x in d0['obj_names']]
    p, _ = transform(pos, d0['nrm'])

    def centre(key):
        oi = next(i for i, o in enumerate(objs) if o.endswith(key))
        q = p[tri[tobj == oi]].reshape(-1, 3)
        return ((q[:, 0].min() + q[:, 0].max()) / 2,
                (q[:, 1].min() + q[:, 1].max()) / 2,
                (q[:, 2].min() + q[:, 2].max()) / 2)

    fl, fr = centre('WheelFL_tires'), centre('WheelFR_tires')
    bl, br = centre('WheelBL_tires'), centre('WheelBR_tires')

    pivots = {}
    for name in objs:
        if name.endswith('WheelFL_tires') or name.endswith('WheelFL_detail'):
            pivots[name] = (FRONT_SPIN, fl)
        elif name.endswith('WheelFR_tires') or name.endswith('WheelFR_detail'):
            pivots[name] = (FRONT_SPIN, fr)
        elif name.endswith('WheelBL_tires') or name.endswith('WheelBL_detail'):
            pivots[name] = (REAR_SPIN, bl)
        elif name.endswith('WheelBR_tires') or name.endswith('WheelBR_detail'):
            pivots[name] = (REAR_SPIN, br)
        elif 'caliper_FL' in name:
            pivots[name] = (FRONT_STEER, fl)
        elif 'caliper_FR' in name:
            pivots[name] = (FRONT_STEER, fr)
        else:
            pivots[name] = (STATIC, (0.0, 0.0, 0.0))
    return pivots, (fl, fr, bl, br)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='folder containing source_dump0..2.npz')
    ap.add_argument('--textures', required=True, help='folder containing file2.png and file5.png')
    ap.add_argument('--out', default=str(PKG / 'assets/vehicles/f150_2015.npz'))
    args = ap.parse_args(argv)

    src = Path(args.source)
    tex = load_textures(Path(args.textures))
    tex['_folder'] = str(src)

    d0 = np.load(src / 'source_dump0.npz')
    pivots, wheels = wheel_pivots(d0)
    print('wheel pivots', {k: tuple(round(v, 4) for v in p) for k, (r, p) in pivots.items() if r})

    lods = []
    info = None
    for lod in range(3):
        V, I, A, glass_count, glass_tris, tris, cent, part = build_lod(lod, tex, pivots)
        lods.append((V, I, A, glass_count))
        print('lod', lod, 'verts', len(V), 'tris', len(I) // 3, 'glass tris', glass_tris,
              'glass idx', glass_count)
        info = (tris, cent, part)

    # measured hard points straight from LOD0's transformed bounds
    pos = d0['pos'].astype(np.float64)
    tri = d0['tri']
    tobj = d0['tri_obj']
    objs = [str(x) for x in d0['obj_names']]
    p, _ = transform(pos, d0['nrm'])
    q = p[tri].reshape(-1, 3)
    wheel_objs = [i for i, o in enumerate(objs) if 'Wheel' in o or 'caliper' in o]
    body = ~np.isin(tobj, wheel_objs)
    body_q = p[tri[body]].reshape(-1, 3)
    mirror_box = ((np.abs(q[:, 1]) > 1.05) & (q[:, 0] > 0.70) & (q[:, 0] < 1.60) & (q[:, 2] > 1.05))
    mirror_width = 2 * np.abs(q[:, 1]).max()
    body_points = q[~mirror_box]
    body_width = 2 * np.abs(body_points[:, 1]).max() if len(body_points) else F150.width
    height = q[:, 2].max() - q[:, 2].min()
    length = q[:, 0].max() - q[:, 0].min()
    clearance = body_q[:, 2].min() - q[:, 2].min()

    # lamp anchors: headlamp projectors and rear lens centres
    c0 = p[tri].mean(1)
    head = p[tri[(c0[:, 0] > 2.35) & (c0[:, 1] > 0.52) & (c0[:, 2] > 0.72) & (c0[:, 2] < 1.25)]]
    tail = p[tri[(c0[:, 0] < -2.45) & (c0[:, 1] > 0.75) & (c0[:, 2] > 0.70) & (c0[:, 2] < 1.25)]]
    head_anchor = head.reshape(-1, 3).mean(0) if len(head) else (2.55, .75, 1.0)
    tail_anchor = tail.reshape(-1, 3).mean(0) if len(tail) else (-2.70, .88, 1.0)

    metadata = {
        'version': 1,
        'source_file': '2015-ford-f-150-lariat-super-cab.zip',
        'source_credit': 'User-supplied 2015 Ford F-150 Lariat Super Cab FBX',
        'triangles': [len(v[1]) // 3 for v in lods],
        'glass': [v[3] for v in lods],
        'head_anchor': [float(v) for v in head_anchor],
        'tail_anchor': [float(v) for v in tail_anchor],
        'dimensions': {
            'name': F150.name,
            'length': F150.length,
            'width': F150.width,
            'height': F150.height,
            'wheelbase': F150.wheelbase,
            'front_axle': F150.front_axle,
            'track': F150.track,
            'tyre_width': F150.tyre_width,
            'tyre_radius': F150.tyre_radius,
            'rim_radius': F150.rim_radius,
            'clearance': F150.clearance,
            'mirror_width': F150.mirror_width,
            'arch_margin': F150.arch_margin,
            'floor_half': F150.floor_half,
            'open_axles': list(F150.open_axles),
            'rear_track': F150.rear_track,
        },
        'interior': 'original source cabin, recoloured tan leather / dark dash, clear glazing',
        'windows': 'clear lightly tinted transmission',
        'measured': {
            'length': float(length), 'body_width': float(body_width),
            'mirror_width': float(mirror_width), 'height': float(height),
            'clearance': float(clearance),
        },
    }
    print('measured', {k: round(v, 4) for k, v in metadata['measured'].items()})
    print('anchors', np.round(head_anchor, 4), np.round(tail_anchor, 4))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, metadata=np.array(json.dumps(metadata)),
                        **{f'v{i}': v[0] for i, v in enumerate(lods)},
                        **{f'i{i}': v[1] for i, v in enumerate(lods)},
                        **{f'a{i}': v[2] for i, v in enumerate(lods)})
    print('SAVED', out, out.stat().st_size)


if __name__ == '__main__':
    main()
