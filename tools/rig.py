"""Rig the supplied base bodies onto the shader's fifteen-joint gameplay skeleton.

* landmarks (shoulder/elbow/wrist/hip/knee/ankle) measured on the A-pose scans (hand-verified)
* skin weights: geometric region seeds -> graph diffusion, hinge joints (elbow, knee, ankle,
  neck, waist) blended analytically along the bone axis
* re-pose: every bone gets a rotation + axial scale that carries the scan joints onto the
  shader's rest-pose pivots (arms hang straight, legs straight and parallel)
Joint ids follow pedestrian_models.R_*.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

R_BODY, R_TORSO, R_UARM_L, R_UARM_R, R_FARM_L, R_FARM_R, R_THIGH_L, R_THIGH_R, R_SHIN_L, R_SHIN_R, \
    R_COAT_L, R_COAT_R, R_HEAD, R_FOOT_L, R_FOOT_R = range(15)

# ---- canonical (shader) rest skeleton, metres ---------------------------------------------------
SPINE_Z, NECK_Z, WAIST_Z = .98, 1.50, 1.0
SH_Y, SH_Z, EL_Z, WR_Z = .205, 1.42, 1.13, .87
HIP_Y, HIP_Z, KNEE_Z, ANKLE_Z = .095, .93, .50, .085


def canon(side):
    return dict(S=np.array([0, side * SH_Y, SH_Z]), E=np.array([0, side * SH_Y, EL_Z]), W=np.array([0, side * SH_Y, WR_Z]),
                H=np.array([0, side * HIP_Y, HIP_Z]), K=np.array([0, side * HIP_Y, KNEE_Z]), A=np.array([0, side * HIP_Y, ANKLE_Z]))


# ---- scan landmarks (left side, after the x shift); hand verified on overlays -------------------
LANDMARKS = {
    'male': dict(dx=.05,
                 S=(-.050, .212, 1.425), E=(-.035, .358, 1.178), W=(.128, .462, 1.040), T=(.205, .525, .885),
                 H=(-.045, .095, .93), K=(-.035, .121, .50), A=(-.058, .143, .085)),
    'female': dict(dx=-.02,
                   S=(-.045, .200, 1.415), E=(-.040, .330, 1.185), W=(.125, .445, 1.040), T=(.200, .490, .900),
                   H=(.010, .095, .93), K=(.040, .120, .50), A=(.010, .146, .085)),
}


def mirror(p):
    p = np.array(p, float); p[1] = -p[1]; return p


def _adjacency(n, tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.
    return A


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0., 1.)
    return t * t * (3. - 2. * t)


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0., 1.)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1), t


class Rig:
    def __init__(self, variant, welded, wtris):
        """welded: (n,3) scan vertices (project frame, UNshifted); wtris: (m,3) triangles over them."""
        L = LANDMARKS[variant]
        self.variant = variant
        self.v = welded.copy()
        self.v[:, 0] += L['dx']
        self.lm = {s: {k: (np.array(L[k]) if s > 0 else mirror(L[k])) for k in 'SEWTHKA'} for s in (1, -1)}
        self._auto_legs()
        self.tris = wtris
        self.n = len(self.v)

    def _auto_legs(self):
        """Hip / knee / ankle joint centres from slab centroids of each leg (scan frame after the x shift)."""
        v = self.v
        for s in (1, -1):
            leg = v[(v[:, 1] * s > .03) & (v[:, 1] * s < .22) & (v[:, 2] < 1.0)]

            def slab(z0, z1):
                q = leg[(leg[:, 2] > z0) & (leg[:, 2] < z1)]
                return q[:, 0].mean(), np.abs(q[:, 1]).mean() * s
            hx, hy = slab(.90, .96); kx, ky = slab(.47, .53); ax_, ay = slab(.095, .125)
            m = self.lm[s]
            m['H'] = np.array([hx, hy, HIP_Z]); m['K'] = np.array([kx, ky, KNEE_Z]); m['A'] = np.array([ax_, ay, ANKLE_Z])

    # ------------------------------------------------------------------ weights
    def weights(self):
        v, n = self.v, self.n
        W = np.zeros((n, 15))
        z = v[:, 2]
        side = np.where(v[:, 1] >= 0, 1, -1)
        label = np.full(n, R_TORSO)
        label[z < WAIST_Z] = R_BODY
        label[z >= 1.58] = R_HEAD
        # legs below the hip joint
        lm = self.lm
        for s in (1, -1):
            sel = (side == s) & (z < HIP_Z - .005)
            thigh, shin, foot = (R_THIGH_L, R_SHIN_L, R_FOOT_L) if s > 0 else (R_THIGH_R, R_SHIN_R, R_FOOT_R)
            label[sel] = thigh
            label[sel & (z < KNEE_Z)] = shin
            a = lm[s]['A']
            ft = sel & (z - a[2] - .6 * (v[:, 0] - a[0]) < 0)
            label[ft] = foot
        # arms: tapered capsules around S-E, E-W, W-T (+ deltoid ball)
        for s in (1, -1):
            m = lm[s]
            uarm, farm = (R_UARM_L, R_FARM_L) if s > 0 else (R_UARM_R, R_FARM_R)
            d1, t1 = seg_dist(v, m['S'], m['E']); d2, t2 = seg_dist(v, m['E'], m['W']); d3, t3 = seg_dist(v, m['W'], m['T'])
            r1 = .056 + .03 * (1 - t1); r2 = .050 + .01 * (1 - t2); r3 = .070
            ball = np.linalg.norm(v - m['S'], axis=1)
            in_arm = (d1 < r1) | (d2 < r2) | (d3 < r3) | ((ball < .095) & (side == s) & (z < 1.56))
            in_arm &= (side == s) & (z > .75)
            # keep the trunk wall out of the capsule near the armpit
            near_trunk = (np.abs(v[:, 1]) < .165) & (z < m['S'][2] - .02)
            in_arm &= ~near_trunk
            in_arm |= (side == s) & (np.abs(v[:, 1]) > .255) & (z < 1.38)      # hands, thumbs, elbow tips
            e1 = m['E'] - m['S']; e1 /= np.linalg.norm(e1)
            e2 = m['W'] - m['E']; e2 /= np.linalg.norm(e2)
            nb = e1 + e2; nb /= np.linalg.norm(nb)
            is_fore = ((v - m['E']) @ nb) > 0
            label[in_arm] = np.where(is_fore[in_arm], farm, uarm)
        # one-hot -> diffusion
        W[np.arange(n), label] = 1.
        A = _adjacency(n, self.tris)
        deg = np.asarray(A.sum(1)).ravel()
        D = sp.diags(1. / np.maximum(deg, 1))
        P = D @ A
        for it in range(36):
            W = .5 * W + .5 * (P @ W)
        self.W = W
        self._hinges()
        self._neck()
        self._finish()
        return self.W

    def _hinges(self):
        v, W = self.v, self.W
        z = v[:, 2]
        side = np.where(v[:, 1] >= 0, 1, -1)
        for s in (1, -1):
            m = self.lm[s]
            thigh, shin, foot, uarm, farm = ((R_THIGH_L, R_SHIN_L, R_FOOT_L, R_UARM_L, R_FARM_L) if s > 0
                                             else (R_THIGH_R, R_SHIN_R, R_FOOT_R, R_UARM_R, R_FARM_R))
            # knee: the two weights of {thigh, shin} get their total split by smoothstep along z
            sel = (side == s) & (z < .93)
            tot = W[:, thigh] + W[:, shin]
            hw = .055
            f = smoothstep(m['K'][2] - hw, m['K'][2] + hw, z)           # 0 below knee .. 1 above
            ankle_tot = W[:, foot]
            keep = sel & (tot > 1e-4)
            W[keep, thigh] = tot[keep] * f[keep]
            W[keep, shin] = tot[keep] * (1 - f[keep])
            # ankle: split {shin, foot} by the tilted plane through the ankle joint
            a = m['A']
            sd = (z - a[2] - .6 * (v[:, 0] - a[0])) / np.sqrt(1 + .36)
            g = smoothstep(-.022, .022, sd)                               # 0 foot .. 1 shin
            tot2 = W[:, shin] + W[:, foot]
            k2 = sel & (z < .40) & (tot2 > 1e-4)
            W[k2, shin] = tot2[k2] * g[k2]
            W[k2, foot] = tot2[k2] * (1 - g[k2])
            # elbow: split {uarm, farm} along the bisector plane through the elbow
            e1 = (m['E'] - m['S']); e1 /= np.linalg.norm(e1)
            e2 = (m['W'] - m['E']); e2 /= np.linalg.norm(e2)
            nb = e1 + e2; nb /= np.linalg.norm(nb)
            se = (v - m['E']) @ nb
            h = smoothstep(-.050, .050, se)                               # 0 upper .. 1 fore
            tot3 = W[:, uarm] + W[:, farm]
            k3 = (side == s) & (tot3 > 1e-4) & (se > -.20)
            # only touch the part that is already mostly arm (keep shoulder blend from diffusion)
            ua_only = k3 & (W[:, uarm] + W[:, farm] > .5) & (np.linalg.norm(v - m['S'], axis=1) > .14)
            W[ua_only, uarm] = tot3[ua_only] * (1 - h[ua_only])
            W[ua_only, farm] = tot3[ua_only] * h[ua_only]
            # near the elbow also pull the farm weight from the diffusion tail
        self.W = W

    def _neck(self):
        """Head / torso split: the neck column ramps smoothly, everything forward of it (jaw, chin) is head."""
        v, W = self.v, self.W
        z = v[:, 2]
        sel = (z > 1.50) & (z < 1.56) & (np.abs(v[:, 1]) < .06)
        xn = v[sel][:, 0].mean()
        self.neck_x = xn
        r = np.hypot(v[:, 0] - xn, v[:, 1])
        col = (r < .078) & (z > 1.44) & (z < 1.66) & (v[:, 0] < xn + .062)
        ramp = smoothstep(NECK_Z + .02, 1.64, z)
        arm = W[:, R_UARM_L] + W[:, R_UARM_R] + W[:, R_FARM_L] + W[:, R_FARM_R]
        k = col & (arm < .2)
        W[k, :] = 0
        W[k, R_HEAD] = ramp[k]
        W[k, R_TORSO] = 1 - ramp[k]
        head = (z >= 1.58) | ((z > 1.50) & (v[:, 0] > xn + .062) & (np.abs(v[:, 1]) < .12))
        head &= ~k
        W[head, :] = 0
        W[head, R_HEAD] = 1.
        self.W = W

    def _finish(self):
        W = self.W
        W = np.clip(W, 0, None)
        W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
        self.W = W

    # ------------------------------------------------------------------ re-pose to the shader rest pose
    def bone_transforms(self):
        """-> dict joint -> (R 3x3, S 3x3 (axial scale in the scan frame), src_origin, dst_origin)."""
        T = {}
        ident = (np.eye(3), np.eye(3), np.zeros(3), np.zeros(3))

        def bone(a, b, A, B):
            """rigid-ish map of segment a->b onto A->B: rotate axis, scale along axis."""
            u = b - a; L = np.linalg.norm(u); u /= L
            U = B - A; LL = np.linalg.norm(U); U /= LL
            v = np.cross(u, U); c = u @ U
            if np.linalg.norm(v) < 1e-9:
                R = np.eye(3)
            else:
                K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + K + K @ K / (1 + c)
            S = np.eye(3) + (LL / L - 1.) * np.outer(u, u)
            return (R, S, a, A)

        for s in (1, -1):
            m, c = self.lm[s], canon(s)
            uarm, farm, thigh, shin, foot = ((R_UARM_L, R_FARM_L, R_THIGH_L, R_SHIN_L, R_FOOT_L) if s > 0
                                             else (R_UARM_R, R_FARM_R, R_THIGH_R, R_SHIN_R, R_FOOT_R))
            T[uarm] = bone(m['S'], m['E'], c['S'], c['E'])
            T[farm] = bone(m['E'], m['W'], c['E'], c['W'])
            T[thigh] = bone(m['H'], m['K'], c['H'], c['K'])
            T[shin] = bone(m['K'], m['A'], c['K'], c['A'])
            # foot: keep orientation, just carry the ankle onto the canonical ankle
            T[foot] = (np.eye(3), np.eye(3), m['A'], c['A'])
        # trunk: pure translation so the spine pivot axis passes through x = 0
        x_mid = .5 * (self.lm[1]['H'][0] + self.lm[-1]['H'][0])
        T[R_BODY] = (np.eye(3), np.eye(3), np.array([x_mid, 0, HIP_Z]), np.array([0, 0, HIP_Z]))
        T[R_TORSO] = (np.eye(3), np.eye(3), np.array([self.neck_x, 0, SPINE_Z]), np.array([0, 0, SPINE_Z]))
        T[R_HEAD] = (np.eye(3), np.eye(3), np.array([self.neck_x, 0, NECK_Z]), np.array([0, 0, NECK_Z]))
        for j in (R_COAT_L, R_COAT_R):
            T[j] = T[R_BODY]
        return T

    def repose(self, v=None, W=None):
        v = self.v if v is None else v
        W = self.W if W is None else W
        out = np.zeros_like(v)
        T = self.bone_transforms()
        for j in range(15):
            w = W[:, j]
            if not w.any():
                continue
            R, S, a, A = T[j]
            out += w[:, None] * (A + (v - a) @ (R @ S).T)
        return out
