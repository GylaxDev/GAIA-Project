"""Bake assets/humans/human_models.npz from the supplied HumanLowPolyBaseMesh.fbx.

    python -m <package>.tools.bake_human_models  [fbx]  (or run the file directly)

Pipeline: FBX bodies -> rig.Rig (skin weights, re-pose onto the shader skeleton) -> clothing painted
onto the body surface by cutting it along hems/necklines -> shells for coat / blazer / hoodie /
skirt / hair / cap / mask -> LOD1 by decimating the body and re-running the same builder ->
accessories (umbrella, phone, bag, ...) carried over from the previous asset -> 20-float vertices.
Vertex: position3, normal3, material, accessory, joints4, weights4, ao, u, v, blink.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fbx_bodies as FB
import rig as RG
import meshops as mo
from meshops import smoothstep

PKG = os.path.dirname(HERE)
ASSET = os.path.join(PKG, 'assets/humans/human_models.npz')
DEFAULT_FBX = os.path.join(PKG, 'assets/humans/HumanLowPolyBaseMesh.fbx')

# material ids (pedestrian_models / human_assets)
(M_SKIN, M_HAIR, M_TOP, M_BOTTOM, M_SHOE, M_FIXED, M_EYE, M_CANOPY, M_METAL, M_PHONE, M_SCREEN, M_GLASS, M_LIMB, M_SHIN,
 M_CAP, M_DARK, M_LIP, M_SCLERA, M_IRIS, M_GLOVE, M_CUP, M_SOLE, M_BROW, M_BEARD) = range(24)
M_LEATHER, M_KNIT, M_LINING, M_STITCH, M_ZIP, M_SHIRT, M_NAIL, M_MASK = range(24, 32)
ACC_COAT, ACC_SKIRT, ACC_HAIR_SHORT, ACC_HAIR_LONG, ACC_HAIR_BUN, ACC_CAP = 16, 32, 64, 128, 256, 512
ACC_FORMAL, ACC_HOOD, ACC_MASK = 4096, 8192, 2097152
KEEP_OLD_ACC = (1, 2, 4, 8, 1024, 2048, 262144, 4194304)     # umbrella, phones, backpack, glasses, bag, coffee, watch
J_COAT = (RG.R_COAT_L, RG.R_COAT_R)

HEM, SLEEVE_Z = 1.00, 1.235


class Layer:
    def __init__(self, m, mat, acc=0, off=0., kind='body'):
        self.m, self.mat, self.acc, self.off, self.kind = m, mat, acc, off, kind


def F_(m):
    v = m['v']; W = m['W']
    return v, W, W[:, 2] + W[:, 3] + W[:, 4] + W[:, 5]


class Builder:
    def __init__(self, body, variant, lod, neck_x):
        self.body, self.variant, self.lod, self.xn = body, variant, lod, neck_x
        self.layers = []
        head = body['v'][body['v'][:, 2] > 1.58]
        self.hc = np.array([head[:, 0].mean() - .005, 0., 1.70])
        self.eye_z = 1.676 if variant != 'female' else 1.667
        self.mouth_z = 1.600 if variant != 'female' else 1.598

    # ---------------------------------------------------------------- painted base layer (shirt / trousers / shoes)
    def base(self, player=False):
        M0 = self.body
        hem = .84 if player else HEM
        sleeve = .91 if player else SLEEVE_Z
        top_mat, top_off = (M_LEATHER, .012) if player else (M_TOP, .005)

        def hem_f(m):
            v, W, a = F_(m); return v[:, 2] - hem + 12 * np.clip(a - .1, 0, None)
        up, lo = mo.split(M0, hem_f(M0))
        up_a, head = mo.split(up, up['v'][:, 2] * -1 + 1.565)          # >=0 below 1.565
        # neckline
        def neck_f(m):
            v = m['v']; return np.maximum(np.hypot(v[:, 0] - self.xn - .008, v[:, 1]) - .097, 1.455 - v[:, 2])
        shirt_a, neck = mo.split(up_a, neck_f(up_a) + 0 * up_a['v'][:, 2])
        def sl_f(m):
            v, W, a = F_(m); return v[:, 2] - sleeve + 12 * np.clip(.92 - a, 0, None)
        shirt, limbs = mo.split(shirt_a, sl_f(shirt_a))
        hand_cut = .90
        limb, hands = (mo.split(limbs, limbs['v'][:, 2] - hand_cut) if limbs is not None else (None, None))
        self.add(shirt, top_mat, 0, top_off)
        self.add(limb, M_LEATHER if player else M_LIMB, 0, .012 if player else .004)
        self.add(hands, M_GLOVE if player else M_SKIN, 0, .003 if player else 0.)
        self.add(neck, M_KNIT if player else M_SKIN, 0, .006 if player else 0.)
        # lower body
        knee = .485
        bot, legs = mo.split(lo, lo['v'][:, 2] - knee)
        def ank_f(m, h):
            v = m['v']; return v[:, 2] - h - .6 * v[:, 0]
        shoe_h = .20 if player else .115
        shin, shoe = mo.split(legs, ank_f(legs, shoe_h))
        self.add(bot, M_BOTTOM, 0, .006, kind='bottom')
        self.add(shin, M_SHIN, 0, .005)
        if shoe is not None:
            sole, upper = mo.split(shoe, .02 - shoe['v'][:, 2])
            self.add(upper, M_SHOE, 0, .008, kind='shoe'); self.add(sole, M_SOLE, 0, .008, kind='shoe')
        # head details painted on the skin
        self.paint_head(head)

    def paint_head(self, head):
        if head is None:
            return
        hv = head['v']
        pieces = [head]
        def cut(piece, field, mat):
            inside, rest = mo.split(piece, field(piece['v']))
            if inside is not None:
                self.add(inside, mat, 0, 0.0004 if mat in (M_SCLERA,) else 0.)
            return rest
        if self.lod == 0:
            front = lambda v: (v[:, 0] - (self.hc[0] + .035))          # face only
            for sgn in (1, -1):
                head = self._cut_ellipse(head, (sgn * .045, self.eye_z), (.0175, .0062), M_SCLERA, offset=.0004)
                head = self._cut_ellipse(head, (sgn * .046, self.eye_z + .023), (.026, .0045), M_BROW, tilt=-sgn * .12)
            head = self._cut_ellipse(head, (0., self.mouth_z + .001), (.025, .0075), M_LIP)
            self._irises()
        self.add(head, M_SKIN, 0, 0.)

    def _cut_ellipse(self, piece, c, ab, mat, offset=0., tilt=0.):
        v = piece['v']
        y = v[:, 1] - c[0]; z = v[:, 2] - c[1]
        yy = y * np.cos(tilt) - z * np.sin(tilt); zz = y * np.sin(tilt) + z * np.cos(tilt)
        field = 1. - (yy / ab[0]) ** 2 - (zz / ab[1]) ** 2
        field = np.where(v[:, 0] > self.hc[0] + .035, field, -1.)
        inside, rest = mo.split(piece, field)
        if inside is not None:
            self.add(inside, mat, 0, offset)
        return rest if rest is not None else piece

    def _irises(self):
        bv = self.body['v']; tree = cKDTree(bv[:, 1:3])
        for sgn in (1, -1):
            c = np.array([sgn * .045, self.eye_z])
            _, i = tree.query(c, k=12)
            front = bv[i][:, 0].max()
            for r, mat, off in ((.0062, M_IRIS, .0008), (.0027, M_EYE, .0012)):
                dv, dt = mo.flat_disc(np.array([front, c[0], c[1]]), [1, 0, 0], r, 20, off)
                # dv is built in the plane normal to +x; fan winding made outward below
                W = np.zeros((len(dv), 15)); W[:, RG.R_HEAD] = 1.
                self.add(mo.mesh(dv, dt, W), mat, 0, 0., kind='decal')

    def add(self, m, mat, acc, off, kind='body'):
        if m is None or not len(m['t']):
            return
        self.layers.append(Layer(m, mat, acc, off, kind))

    # ---------------------------------------------------------------- shells
    def torso_shell(self, hem, sleeve, mat, acc, off, kind='shell', add=True):
        M0 = self.body
        def f(m):
            v, W, a = F_(m); return v[:, 2] - hem + 12 * np.clip(a - .1, 0, None)
        up, _ = mo.split(M0, f(M0))
        up, _ = mo.split(up, 1.565 - up['v'][:, 2])
        up, _ = mo.split(up, np.maximum(np.hypot(up['v'][:, 0] - self.xn - .008, up['v'][:, 1]) - .104, 1.455 - up['v'][:, 2]))
        a_ = F_(up)[2]
        sh, _ = mo.split(up, up['v'][:, 2] - sleeve + 12 * np.clip(.92 - a_, 0, None))
        return sh

    def coat(self, hem, sleeve, mat, acc, off, flare=.05):
        """Open coat: shell over torso+arms, thigh tubes flare outwards and take the coat-tail joints."""
        sh = self.torso_shell(hem, sleeve, mat, acc, off)
        if sh is None:
            return
        v, t = sh['v'].copy(), sh['t']
        n = mo.smooth_field(v, t, mo.vnormals(v, t), 2); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        hi = smoothstep(.95, 1.05, v[:, 2])
        v = v + n * (off * hi)[:, None]
        # below the waist the two thigh tubes are inflated onto one flared elliptical hull (a closed coat skirt)
        s_ = smoothstep(1.0, hem, v[:, 2])
        phi = np.arctan2(v[:, 1], v[:, 0] + 1e-9)
        rx, ry = .12 + .07 * s_ + off, .185 + .085 * s_ + off
        rh = 1. / np.sqrt((np.cos(phi) / rx) ** 2 + (np.sin(phi) / ry) ** 2)
        r = np.hypot(v[:, 0], v[:, 1]) + 1e-9
        k = smoothstep(1.06, .93, v[:, 2])
        rn = r + k * (np.maximum(rh, r) - r)
        v[:, 0] *= rn / r; v[:, 1] *= rn / r
        sh = dict(sh, v=v); apply_coat_weights(sh, hem)
        self.layers.append(Layer(sh, mat, acc, 0., 'raw'))

    def skirt(self, mat, acc):
        seg, rings = 28, 7
        v, W = [], []
        for i in range(rings):
            s = i / (rings - 1)
            z = 1.0 - .42 * s
            ry = .165 + .13 * s; rx = .125 + .10 * s
            for j in range(seg):
                a = 2 * np.pi * j / seg
                v.append((rx * np.cos(a), ry * np.sin(a), z))
        v = np.array(v)
        for i in range(rings):
            s = i / (rings - 1)
            for j in range(seg):
                w = np.zeros(15)
                wc = smoothstep(0., .5, s) * .9
                w[RG.R_BODY] = 1 - wc
                w[RG.R_COAT_L if np.sin(2 * np.pi * j / seg) >= 0 else RG.R_COAT_R] = wc
                W.append(w)
        t = []
        for i in range(rings - 1):
            for j in range(seg):
                a = i * seg + j; b = i * seg + (j + 1) % seg
                t += [(a, a + seg, b), (b, a + seg, b + seg)]
        self.add(mo.mesh(v, np.array(t), np.array(W), mo.uv_planar(v)), mat, acc, 0., kind='skirt')

    def hair(self, style):
        M0 = self.body
        hc = self.hc
        def line(v):
            ang = np.arctan2(v[:, 1], v[:, 0] - hc[0]); c = np.cos(ang)
            base = 1.700 + .052 * np.maximum(c, 0) - (.085 if style == 'short' else .09) * np.maximum(-c, 0)
            if style == 'long':
                back = np.maximum(-c, 0) ** .7
                base = np.where(c < 0, 1.700 - back * .42 + .0, base)
            return v[:, 2] - base
        m, _ = mo.split(M0, np.where(M0['v'][:, 2] > (1.25 if style == 'long' else 1.52), line(M0['v']), -1.))
        if style == 'long':
            m, _ = mo.split(m, .085 - np.where(m['v'][:, 2] < 1.60, np.abs(m['v'][:, 1]), 0.))
        if m is None:
            return
        acc = {'short': ACC_HAIR_SHORT, 'long': ACC_HAIR_LONG, 'bun': ACC_HAIR_BUN}[style]
        self.add(m, M_HAIR, acc if style != 'bun' else ACC_HAIR_BUN, .010, kind='hair')
        if style == 'bun':
            v, t = mo.uv_sphere(np.array([hc[0] - .085, 0, 1.752]), .045)
            W = np.zeros((len(v), 15)); W[:, RG.R_HEAD] = 1
            self.add(mo.mesh(v, t, W, mo.uv_planar(v)), M_HAIR, ACC_HAIR_BUN, 0., kind='raw')

    def cap(self, acc=ACC_CAP, mat=M_CAP):
        M0 = self.body; hc = self.hc
        m, _ = mo.split(M0, np.where(M0['v'][:, 2] > 1.5, M0['v'][:, 2] - 1.700, -1.))
        self.add(m, mat, acc, .014, kind='hair')
        # brim: fan of quads in front
        n = 9
        ang = np.linspace(-1.25, 1.25, n)
        v = []
        for r in (.092, .165):
            for a in ang:
                v.append((hc[0] + r * np.cos(a), r * np.sin(a) * .95, 1.708 - (.012 if r > .1 else 0)))
        v = np.array(v); t = []
        for j in range(n - 1):
            t += [(j, j + 1, n + j), (j + 1, n + j + 1, n + j)]
            t += [(j, n + j, j + 1), (j + 1, n + j, n + j + 1)]
        W = np.zeros((len(v), 15)); W[:, RG.R_HEAD] = 1
        self.add(mo.mesh(v, np.array(t), W, mo.uv_planar(v)), mat, acc, 0., kind='raw')

    def mask(self, acc=ACC_MASK):
        M0 = self.body; hc = self.hc
        f = np.minimum(M0['v'][:, 2] - 1.585, 1.664 - M0['v'][:, 2])
        f = np.minimum(f, .074 - np.abs(M0['v'][:, 1]))
        f = np.where(M0['v'][:, 0] > hc[0] + .005, f, -1.)
        m, _ = mo.split(M0, f)
        self.add(m, M_MASK, acc, .007, kind='shell')

    def hood_collar(self):
        M0 = self.body; hc = self.hc
        f = np.minimum(M0['v'][:, 2] - 1.49, 1.77 - M0['v'][:, 2])
        f = np.where(M0['v'][:, 0] < hc[0] - .01, f, -1.)
        m, _ = mo.split(M0, f)
        self.add(m, M_KNIT, ACC_HOOD, .028, kind='shell')

    # ---------------------------------------------------------------- final geometry
    def finalize(self):
        """-> dict of float arrays: v (n,3), nrm, mat, acc, W (n,15), uv, ao, tris."""
        V, N, MAT, ACC, WW, UV, TR = [], [], [], [], [], [], []
        base = 0
        for L in self.layers:
            m = L.m; v = m['v'].copy(); t = m['t']
            n = mo.vnormals(v, t)
            if L.kind in ('body', 'shoe', 'shell', 'hair', 'bottom') and L.off:
                ns = mo.smooth_field(v, t, n, 2)
                ns /= np.linalg.norm(ns, axis=1, keepdims=True) + 1e-12
                if L.kind == 'hair':
                    top = smoothstep(1.62, 1.75, v[:, 2])
                    d = L.off * (.8 + .9 * top)
                else:
                    d = L.off
                v = v + ns * np.broadcast_to(np.asarray(d), (len(v),))[:, None]
            if L.kind == 'shoe':
                v[:, 2] = np.maximum(v[:, 2], 0.)
            if L.kind == 'bottom':
                crotch = (np.abs(v[:, 1]) < .09) & (v[:, 2] > .78) & (v[:, 2] < 1.0) & (v[:, 0] > .05)
                v[crotch, 0] = .05 + (v[crotch, 0] - .05) * .35
            Wm = m['W'].copy()
            if L.kind == 'coat':
                pass
            if L.kind == 'shell' and L.acc in (ACC_COAT, 0) and False:
                pass
            nrm = mo.vnormals(v, t)
            if L.kind == 'decal':
                nrm = np.tile([1., 0, 0], (len(v), 1))
            uv = m['uv'] if (L.kind in ('body', 'shoe', 'bottom', 'shell', 'coat', 'hair') and m['uv'].any()) else mo.uv_planar(v)
            V.append(v); N.append(nrm); MAT.append(np.full(len(v), L.mat, float)); ACC.append(np.full(len(v), L.acc, float))
            WW.append(Wm); UV.append(uv); TR.append(t + base); base += len(v)
        return dict(v=np.concatenate(V), n=np.concatenate(N), mat=np.concatenate(MAT), acc=np.concatenate(ACC),
                    W=np.concatenate(WW), uv=np.concatenate(UV), t=np.concatenate(TR))


def apply_coat_weights(m, hem):
    """Give the part of a coat shell below the waist the coat-tail joints (rotate about the waist)."""
    v, W = m['v'], m['W'].copy()
    wc = smoothstep(1.0, hem + .12, v[:, 2]) * 0 + smoothstep(1.0, .80, v[:, 2]) * .9
    side = np.where(v[:, 1] >= 0, RG.R_COAT_L, RG.R_COAT_R)
    W *= (1 - wc)[:, None]
    W[np.arange(len(v)), side] += wc
    m['W'] = W


# ====================================================================================== assembly
def top4(W):
    idx = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, idx, 1)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
    return idx.astype(np.float32), w.astype(np.float32)


def prepare_body(fbx, variant, lod):
    b = FB.load_body(fbx, variant)
    wt = b['vsrc'][b['tris']]
    R = RG.Rig(variant, b['weld_v'], wt)
    W = R.weights()
    vr = R.repose()
    if lod == 0:
        t = b['tris']; vs = vr[b['vsrc']]
        a3 = .5 * np.linalg.norm(np.cross(vs[t[:, 1]] - vs[t[:, 0]], vs[t[:, 2]] - vs[t[:, 0]]), axis=1).sum()
        u = b['uv']
        au = .5 * np.abs((u[t[:, 1], 0] - u[t[:, 0], 0]) * (u[t[:, 2], 1] - u[t[:, 0], 1])
                         - (u[t[:, 2], 0] - u[t[:, 0], 0]) * (u[t[:, 1], 1] - u[t[:, 0], 1])).sum()
        sc = np.sqrt(a3 / max(au, 1e-9))                     # detail tiles stay world-sized on the UV atlas
        body = mo.mesh(vs, t, W[b['vsrc']], b['uv'] * sc)
    else:
        import fast_simplification as fs
        # decimate in the A-pose scan (arms clear of the hips), then carry the result onto the rest skeleton
        pts, faces = fs.simplify(R.v.astype(np.float32), wt.astype(np.int32), target_reduction=.86)
        _, nn = cKDTree(R.v).query(pts)
        Wd = W[nn]
        body = mo.mesh(R.repose(pts.astype(float), Wd), faces, Wd, None)
    return body, R


def build_variant(fbx, variant, lod, old):
    kind = 'male' if variant in ('male', 'player') else 'female'
    body, R = prepare_body(fbx, 'male' if variant == 'player' else variant, lod)
    B_ = Builder(body, kind, lod, R.neck_x)
    player = variant == 'player'
    B_.base(player)
    if player:
        B_.coat(.58, .91, M_LEATHER, 0, .016, flare=.055)
        B_.hair('short'); B_.cap(); B_.mask()
    else:
        B_.coat(.52, .91, M_BOTTOM, ACC_COAT, .020, flare=.06)
        B_.add(B_.torso_shell(.84, .91, M_BOTTOM, ACC_FORMAL, .012), M_BOTTOM, ACC_FORMAL, .012, 'shell')
        B_.add(B_.torso_shell(.90, .91, M_KNIT, ACC_HOOD, .013), M_KNIT, ACC_HOOD, .013, 'shell')
        B_.hood_collar()
        B_.skirt(M_BOTTOM, ACC_SKIRT)
        B_.hair('short'); B_.hair('long'); B_.hair('bun'); B_.cap()
    g = B_.finalize()
    ov, oi = old
    acc = ov[:, 7].astype(np.int64)
    keep = np.zeros(len(ov), bool)
    for b in KEEP_OLD_ACC:
        keep |= (acc & b) != 0
    keep &= (acc & ~np.int64(sum(KEEP_OLD_ACC))) == 0
    tri = oi.reshape(-1, 3)
    tk = keep[tri].all(1)
    used = np.unique(tri[tk]); remap = -np.ones(len(ov), np.int64); remap[used] = np.arange(len(used))
    return g, (ov[used], remap[tri[tk]]), R


def pack(g, acc_part, lod):
    n = len(g['v'])
    J, Wt = top4(g['W'])
    out = np.zeros((n, 20), np.float32)
    out[:, 0:3] = g['v']; out[:, 3:6] = g['n']; out[:, 6] = g['mat']; out[:, 7] = g['acc']
    out[:, 8:12] = J; out[:, 12:16] = Wt
    out[:, 16] = 1.0; out[:, 17:19] = g['uv']; out[:, 19] = 0.
    ao_v, ao_t = acc_part
    idx = np.concatenate([g['t'].ravel(), (ao_t + n).ravel()]).astype(np.uint32)
    return np.concatenate([out, ao_v.astype(np.float32)]), idx


def main(fbx=DEFAULT_FBX):
    with np.load(ASSET, allow_pickle=False) as d:
        old = {k: d[k] for k in d.files}
    new = dict(old)
    meta = json.loads(str(old['metadata']))
    meshes = {}
    for variant in ('male', 'female', 'player'):
        for lod in (0, 1):
            print('baking', variant, lod, flush=True)
            g, accp, R = build_variant(fbx, variant, lod, (old[f'{variant}_v{lod}'], old[f'{variant}_i{lod}']))
            v, i = pack(g, accp, lod)
            new[f'{variant}_v{lod}'], new[f'{variant}_i{lod}'] = v, i
            meshes[f'{variant}_lod{lod}'] = dict(vertices=int(len(v)), triangles=int(len(i) // 3))
    meta.update(anatomy='HumanLowPolyBaseMesh.fbx male/female base bodies, re-posed and skinned to the gameplay skeleton',
                source='HumanLowPolyBaseMesh.fbx', meshes=meshes, source_has_animation_clips=False)
    meta.pop('source_sha256', None)
    new['metadata'] = np.array(json.dumps(meta, sort_keys=True))
    np.savez_compressed(ASSET, **new)
    print('wrote', ASSET)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FBX)
