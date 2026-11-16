"""Small mesh toolkit for the human baker: plane/field splitting with attribute interpolation,
vertex normals, normal offsets, smoothing, primitive builders."""
from __future__ import annotations
import numpy as np

ATTRS = ('v', 'W', 'uv')


def mesh(v, t, W, uv=None):
    return dict(v=np.asarray(v, float), t=np.asarray(t, np.int64), W=np.asarray(W, float),
                uv=np.zeros((len(v), 2)) if uv is None else np.asarray(uv, float))


def compact(m):
    used = np.unique(m['t'])
    remap = -np.ones(len(m['v']), np.int64); remap[used] = np.arange(len(used))
    return dict(v=m['v'][used], W=m['W'][used], uv=m['uv'][used], t=remap[m['t']])


def split(m, f):
    """Cut mesh along the zero set of per-vertex field ``f``. -> (f>=0 part, f<0 part), both compacted."""
    v, W, uv, t = m['v'], m['W'], m['uv'], m['t']
    f = np.asarray(f, float)
    ft = f[t]
    pos = ft >= 0
    n = pos.sum(1)
    P = [t[n == 3]]; N = [t[n == 0]]
    nv = [v]; nW = [W]; nuv = [uv]
    cache = {}
    base = len(v)
    newv, newW, newuv = [], [], []

    def cut(a, b):
        key = (a, b) if a < b else (b, a)
        if key in cache:
            return cache[key]
        fa, fb = f[a], f[b]
        s = fa / (fa - fb)
        i = base + len(newv)
        newv.append(v[a] + s * (v[b] - v[a])); newW.append(W[a] + s * (W[b] - W[a])); newuv.append(uv[a] + s * (uv[b] - uv[a]))
        cache[key] = i
        return i
    pp, nn = [], []
    for ti in np.nonzero((n == 1) | (n == 2))[0]:
        tri = t[ti]; ps = pos[ti]
        k = n[ti]
        # rotate so the odd vertex (alone on its side) is first
        odd = int(np.argmax(ps)) if k == 1 else int(np.argmin(ps))
        a, b, c = tri[odd], tri[(odd + 1) % 3], tri[(odd + 2) % 3]
        ab, ac = cut(a, b), cut(a, c)
        if k == 1:
            pp.append((a, ab, ac)); nn += [(ab, b, c), (ab, c, ac)]
        else:
            nn.append((a, ab, ac)); pp += [(ab, b, c), (ab, c, ac)]
    if newv:
        nv.append(np.array(newv)); nW.append(np.array(newW)); nuv.append(np.array(newuv))
    big = dict(v=np.concatenate(nv), W=np.concatenate(nW), uv=np.concatenate(nuv))
    out = []
    for lst, extra in ((P, pp), (N, nn)):
        tt = np.concatenate([x.reshape(-1, 3) for x in lst] + ([np.array(extra, np.int64)] if extra else []))
        out.append(compact(dict(big, t=tt)) if len(tt) else None)
    return out[0], out[1]


def concat(ms):
    ms = [m for m in ms if m is not None and len(m['t'])]
    off = np.cumsum([0] + [len(m['v']) for m in ms])
    return dict(v=np.concatenate([m['v'] for m in ms]), W=np.concatenate([m['W'] for m in ms]),
                uv=np.concatenate([m['uv'] for m in ms]), t=np.concatenate([m['t'] + o for m, o in zip(ms, off)]))


def vnormals(v, t):
    fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, t[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def smooth_field(v, t, x, iters=6):
    """Laplacian-smooth a per-vertex array over the mesh (positions merged by coordinates)."""
    n = len(v)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]); e = np.concatenate([e, e[:, ::-1]])
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.
    D = 1. / np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    x = np.array(x, float)
    for _ in range(iters):
        x = .5 * x + .5 * (sp.diags(D) @ (A @ x))
    return x


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0., 1.)
    return t * t * (3. - 2. * t)


def uv_planar(v):
    return np.stack([.7 * v[:, 0] + .7 * v[:, 1], v[:, 2]], 1)


def uv_sphere_like(v):
    return uv_planar(v)


def flat_disc(center, normal, r, seg=24, offset=0.):
    """Triangle fan disc."""
    normal = np.asarray(normal, float); normal /= np.linalg.norm(normal)
    a = np.cross(normal, [0, 0, 1.]); a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array([1., 0, 0])
    b = np.cross(normal, a)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = center + offset * normal + r * (np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b)
    v = np.vstack([center + offset * normal, ring])
    t = np.array([(0, 1 + i, 1 + (i + 1) % seg) for i in range(seg)])
    return v, t


def uv_sphere(center, radius, rings=8, seg=14):
    v = [center + [0, 0, radius]]
    for i in range(1, rings):
        th = np.pi * i / rings
        for j in range(seg):
            ph = 2 * np.pi * j / seg
            v.append(center + radius * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]))
    v.append(center + [0, 0, -radius])
    v = np.array(v); t = []
    for j in range(seg):
        t.append((0, 1 + j, 1 + (j + 1) % seg))
    for i in range(rings - 2):
        for j in range(seg):
            a = 1 + i * seg + j; b = 1 + i * seg + (j + 1) % seg
            t += [(a, a + seg, b), (b, a + seg, b + seg)]
    last = len(v) - 1
    for j in range(seg):
        a = 1 + (rings - 2) * seg + j; b = 1 + (rings - 2) * seg + (j + 1) % seg
        t.append((a, last, b))
    return v, np.array(t)
