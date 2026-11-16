"""Minimal binary-FBX reader (no SDK needed) + conversion of the two supplied base bodies
into the project frame: +X forward, +Y left, +Z up, metres, feet on z = 0, 1.78 m tall."""
from __future__ import annotations
import struct, zlib
import numpy as np


class _Node:
    __slots__ = ('name', 'props', 'children')

    def __init__(self, name):
        self.name, self.props, self.children = name, [], []

    def first(self, name):
        for c in self.children:
            if c.name == name:
                return c

    def find(self, name):
        return [c for c in self.children if c.name == name]


def _prop(f):
    t = f.read(1).decode('latin1')
    if t == 'Y': return struct.unpack('<h', f.read(2))[0]
    if t == 'C': return bool(f.read(1)[0])
    if t == 'I': return struct.unpack('<i', f.read(4))[0]
    if t == 'F': return struct.unpack('<f', f.read(4))[0]
    if t == 'D': return struct.unpack('<d', f.read(8))[0]
    if t == 'L': return struct.unpack('<q', f.read(8))[0]
    if t in 'fdilb':
        n, enc, clen = struct.unpack('<III', f.read(12))
        raw = f.read(clen)
        if enc == 1:
            raw = zlib.decompress(raw)
        return np.frombuffer(raw, dtype={'f': '<f4', 'd': '<f8', 'i': '<i4', 'l': '<i8', 'b': 'u1'}[t], count=n)
    if t == 'S':
        return f.read(struct.unpack('<I', f.read(4))[0]).decode('utf8', 'replace')
    if t == 'R':
        return f.read(struct.unpack('<I', f.read(4))[0])
    raise ValueError('bad FBX property type ' + t)


def _node(f, v64):
    end, nprops, _ = struct.unpack('<QQQ' if v64 else '<III', f.read(24 if v64 else 12))
    nl = f.read(1)[0]
    if end == 0:
        return None
    n = _Node(f.read(nl).decode('latin1'))
    n.props = [_prop(f) for _ in range(nprops)]
    while f.tell() < end:
        c = _node(f, v64)
        if c is None:
            break
        n.children.append(c)
    f.seek(end)
    return n


def read_fbx(path):
    with open(path, 'rb') as f:
        f.read(23)
        version = struct.unpack('<I', f.read(4))[0]
        root = _Node('root')
        while True:
            n = _node(f, version >= 7500)
            if n is None:
                break
            root.children.append(n)
    return root


def read_meshes(path):
    """-> {model name: dict(v (n,3), polys [[...]], uv (m,2), uvi per polygon-vertex)} in FBX axes (Y up, +Z forward)."""
    root = read_fbx(path)
    obj = root.first('Objects')
    names = {m.props[0]: m.props[1].split('\x00')[0] for m in obj.find('Model')}
    owner = {}
    for c in root.first('Connections').children:
        if c.props[0] == 'OO' and c.props[2] in names:
            owner[c.props[1]] = names[c.props[2]]
    out = {}
    for g in obj.find('Geometry'):
        v = g.first('Vertices').props[0].reshape(-1, 3).astype(np.float64)
        polys, cur = [], []
        for i in g.first('PolygonVertexIndex').props[0]:
            if i < 0:
                cur.append(int(~i)); polys.append(cur); cur = []
            else:
                cur.append(int(i))
        uvl = g.first('LayerElementUV')
        out[owner.get(g.props[0], str(g.props[0]))] = dict(
            v=v, polys=polys, uv=uvl.first('UV').props[0].reshape(-1, 2), uvi=uvl.first('UVIndex').props[0])
    return out


# (model name in the FBX, x pivot of that model) -> variant
SOURCES = {'male': ('Male1', 7.8995684859), 'female': ('LOD7', -7.0)}
HEIGHT = 1.78


def load_body(path, variant):
    """Welded, triangulated body in the project frame.

    Returns dict(v, tris, uv, vsrc): v (n,3) positions; tris (m,3) with per-corner UV splits
    already applied (vertices are duplicated along UV seams; ``vsrc`` maps each back to the
    welded position index so adjacency/weights can be shared).
    """
    name, cx = SOURCES[variant]
    m = read_meshes(path)[name]
    fv = m['v']
    p = np.stack([fv[:, 2], fv[:, 0] - cx, fv[:, 1]], 1)          # (x fwd, y left, z up)
    z0 = p[:, 2].min()
    scale = HEIGHT / (p[:, 2].max() - z0)
    p = (p - [0., 0., z0]) * scale
    # triangulate quads along the shorter diagonal; split vertices per UV corner
    corner_pos, corner_uv, tris = [], [], []
    k = 0
    cuv = m['uv']
    uvi = m['uvi']
    for poly in m['polys']:
        n = len(poly)
        base = k
        for j, vi in enumerate(poly):
            corner_pos.append(vi); corner_uv.append(cuv[uvi[k + j]])
        if n == 3:
            tris.append((base, base + 1, base + 2))
        elif n == 4:
            d02 = np.linalg.norm(p[poly[0]] - p[poly[2]]); d13 = np.linalg.norm(p[poly[1]] - p[poly[3]])
            if d02 <= d13:
                tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
            else:
                tris += [(base, base + 1, base + 3), (base + 1, base + 2, base + 3)]
        else:
            for j in range(1, n - 1):
                tris.append((base, base + j, base + j + 1))
        k += n
    corner_pos = np.asarray(corner_pos); corner_uv = np.asarray(corner_uv); tris = np.asarray(tris)
    # merge corners that share position index and (rounded) uv
    key = np.concatenate([corner_pos[:, None].astype(np.float64), np.round(corner_uv, 6)], 1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    vsrc = corner_pos[first]
    return dict(v=p[vsrc], tris=inv[tris], uv=corner_uv[first], vsrc=vsrc, weld_v=p, scale=scale)
