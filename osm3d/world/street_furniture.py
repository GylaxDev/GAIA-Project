"""Shared, metre-sized street furniture; baked geometry, no per-frame modelling.

Furniture uses the city's 17-float stream. Material IDs in style.y are 3 wood,
4 weathered paint, 5 concrete, 6 bare fasteners and 7 dry leaves. Lamp templates export
only position/normal/part and retain the existing nine-float instance format.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

STRIDE = 17
LAMP_LENS_DROP = .30
IRON = np.array([.19, .23, .22])
STEEL = np.array([.46, .48, .47])
WOOD = np.array([.48, .39, .29])


class _Builder:
    """Small loft/tube builder with metric UVs and optional smooth normals."""

    def __init__(self):
        self.vertices, self.indices, self.count = [], [], 0

    def face(self, points, color, material=4., *, part=7., seed=.37, normals=None, ao=1.):
        p = np.asarray(points, np.float64)
        n = np.cross(p[1] - p[0], p[2] - p[0])
        length = float(np.linalg.norm(n))
        if length < 1e-10:
            return
        n /= length
        axis = int(np.argmax(np.abs(n)))
        uv_axes = ((1, 2), (0, 2), (0, 1))[axis]
        block = np.zeros((len(p), STRIDE), np.float32)
        block[:, :3], block[:, 3:6], block[:, 6:9] = p, n if normals is None else normals, color
        block[:, 9], block[:, 10:13] = ao, (seed, material, 0.)
        block[:, 13:15], block[:, 16] = p[:, uv_axes], part
        self.vertices.append(block)
        self.indices.append(np.array([(self.count, self.count + i, self.count + i + 1)
                                      for i in range(1, len(p) - 1)], np.uint32).ravel())
        self.count += len(p)

    def loft(self, rings, color, material=4., *, part=7., seed=.37, caps=True, ao=1.):
        rings = np.asarray(rings, float)
        for lower, upper in zip(rings, rings[1:]):
            for j in range(len(lower)):
                k = (j + 1) % len(lower)
                self.face((lower[j], lower[k], upper[k], upper[j]), color, material,
                          part=part, seed=seed, ao=ao)
        if caps:
            self.face(rings[0][::-1], color, material, part=part, seed=seed, ao=ao)
            self.face(rings[-1], color, material, part=part, seed=seed, ao=ao)

    def box(self, center, size, color, material=4., *, bevel=.008, part=7., seed=.37,
            rotation=None, ao=1.):
        hx, hy, hz = np.asarray(size, float) * .5
        b = min(bevel, hx * .35, hy * .35, hz * .35)

        def ring(z, inset):
            x, y = hx - inset, hy - inset
            c = max(b - inset * .5, 1e-5)
            return np.array([(x-c, -y, z), (x, -y+c, z), (x, y-c, z), (x-c, y, z),
                             (-x+c, y, z), (-x, y-c, z), (-x, -y+c, z), (-x+c, -y, z)])

        rings = np.array([ring(-hz, b), ring(-hz+b, 0.), ring(hz-b, 0.), ring(hz, b)])
        if rotation is not None:
            rings = rings @ np.asarray(rotation).T
        self.loft(rings + center, color, material, part=part, seed=seed, ao=ao)

    def tube(self, points, radius, color, material=4., *, sides=8, part=7., seed=.37, caps=True):
        p = np.asarray(points, float)
        radii = np.broadcast_to(radius, (len(p),))
        keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-8]
        p, radii = p[keep], radii[keep]
        if len(p) < 2:
            return
        delta = np.diff(p, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        segments = delta / lengths[:, None]
        tangent = np.vstack((segments[0], segments[:-1] + segments[1:], segments[-1]))
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
        arc = np.r_[0., np.cumsum(lengths)]
        slopes = np.gradient(radii, arc)
        angles = np.arange(sides) * (math.tau / sides)
        rings, ns, u = [], [], None
        for j, axis in enumerate(tangent):
            # Transport one cross-section frame through bends: adjacent segments
            # share their end ring instead of leaving cracks at every elbow.
            if u is not None:
                u = u - axis * np.dot(u, axis)
            if u is None or np.linalg.norm(u) < 1e-6:
                helper = np.array([0., 0., 1.]) if abs(axis[2]) < .95 else np.array([0., 1., 0.])
                u = np.cross(helper, axis)
            u /= np.linalg.norm(u)
            v = np.cross(axis, u)
            radial = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
            rings.append(p[j] + radii[j] * radial)
            normals = radial - axis * slopes[j]
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            ns.append(normals)
        for j, (lower, upper) in enumerate(zip(rings, rings[1:])):
            for k in range(sides):
                k1 = (k + 1) % sides
                self.face((lower[k], lower[k1], upper[k1], upper[k]), color, material,
                          part=part, seed=seed, normals=[ns[j][k], ns[j][k1], ns[j+1][k1], ns[j+1][k]])
        if caps:
            self.face(rings[0][::-1], color, material, part=part, seed=seed)
            self.face(rings[-1], color, material, part=part, seed=seed)

    def lathe(self, profile, color, material=4., *, sides=12, part=7., seed=.37, smooth=False):
        """Revolve an oriented (radius, z) profile, including hollow rims."""
        profile = np.asarray(profile, float)
        delta = np.diff(profile, axis=0)
        segment = np.c_[delta[:, 1], -delta[:, 0]]
        segment /= np.maximum(np.linalg.norm(segment, axis=1, keepdims=True), 1e-8)
        if smooth:
            ring_normal = np.vstack((segment[0], segment[:-1]+segment[1:], segment[-1]))
            ring_normal /= np.maximum(np.linalg.norm(ring_normal, axis=1, keepdims=True), 1e-8)
            if profile[-1, 0] < 1e-8: ring_normal[-1] = (0., 1.)
        angles = np.arange(sides) * (math.tau / sides)
        radial = np.c_[np.cos(angles), np.sin(angles)]
        for index, ((r0, z0), (r1, z1)) in enumerate(zip(profile, profile[1:])):
            dr, dz = r1-r0, z1-z0
            length = math.hypot(dr, dz)
            if length < 1e-8:
                continue
            lower, upper = np.c_[radial*r0, np.full(sides, z0)], np.c_[radial*r1, np.full(sides, z1)]
            lower_n = upper_n = np.c_[radial * (dz / length), np.full(sides, -dr / length)]
            if smooth:
                lower_n = np.c_[radial*ring_normal[index, 0], np.full(sides, ring_normal[index, 1])]
                upper_n = np.c_[radial*ring_normal[index+1, 0], np.full(sides, ring_normal[index+1, 1])]
            for j in range(sides):
                k = (j + 1) % sides
                # Apex triangles must not start with a repeated zero-radius edge.
                ids = (0, 2, 3) if r0 < 1e-8 else ((0, 1, 2) if r1 < 1e-8 else (0, 1, 2, 3))
                pts = np.array([lower[j], lower[k], upper[k], upper[j]])
                ns = np.array([lower_n[j], lower_n[k], upper_n[k], upper_n[j]])
                self.face(pts[list(ids)], color, material, part=part, seed=seed, normals=ns[list(ids)])

    def mesh(self):
        v = np.ascontiguousarray(np.vstack(self.vertices), np.float32)
        i = np.ascontiguousarray(np.concatenate(self.indices), np.uint32)
        return v, i


def _rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


@lru_cache(maxsize=2)
def bench_template(variant=0):
    """Slatted park bench with reclined back, cast end frames and curved arms."""
    b = _Builder()
    for j in range(7):
        y = -.255 + j * .077
        color = WOOD * (.91 + j * .023) if variant == 0 else IRON * (1.02 + j * .018)
        b.box((0., y, .475 + .04*y*y), (1.96, .065, .042), color,
              3. if variant == 0 else 4., seed=.17 + j*.071, bevel=.007)
    for j in range(5):
        z = .615 + j * .081
        y = .225 + (z-.55) * .22
        color = WOOD * (1.04 - j * .032) if variant == 0 else IRON * (1.12 - j*.023)
        b.box((0., y, z), (1.92, .044, .066), color, 3. if variant == 0 else 4.,
              rotation=_rotation_x(-.215), seed=.52 + j*.069, bevel=.007)
    for x in (-.73, .73):
        for y in (-.225, .225):
            b.box((x, y, .025), (.15, .115, .05), IRON, bevel=.012, ao=.72)
            b.tube([(x, y, .035), (x, y*.78, .42)], .028, IRON, sides=8)
        b.tube([(x, -.285, .42), (x, -.10, .435), (x, .23, .435), (x, .32, .96)],
               .028, IRON, sides=8)
        b.tube([(x, -.23, .44), (x, -.23, .68), (x, -.19, .735),
                (x, .16, .735), (x, .29, .70)], .023, IRON, sides=8)
    b.tube([(-.74, .0, .28), (.74, .0, .28)], .018, IRON, sides=6)
    # Flush hex bolt heads: actual small faces, not oversized cubes.
    angles = np.arange(6) * math.tau / 6.
    for x in (-.73, .73):
        for j in range(7):
            y = -.255+j*.077
            b.face(np.c_[x+.008*np.cos(angles), y+.008*np.sin(angles),
                         np.full(6, .498+.04*y*y)], STEEL, 6.)
    return b.mesh()


@lru_cache(maxsize=2)
def bin_template(variant=0):
    """Open tapered litter basket, or a ribbed hooded municipal bin."""
    b = _Builder()
    if variant == 0:
        b.lathe([(0., .025), (.245, .025), (.27, .07), (.27, .13), (.24, .15), (0., .15)], IRON)
        # A recessed liner is visible through the cage; the mouth stays open.
        b.lathe([(.235, .14), (.286, .69), (.265, .69), (.22, .17), (0., .17)],
                np.array([.08, .09, .085]), sides=16)
        for j in range(16):
            a = j * math.tau / 16.
            radial = np.array([math.cos(a), math.sin(a)])
            b.tube([np.r_[radial*.263, .105], np.r_[radial*.326, .86]], .0125, IRON, sides=5, caps=False)
        b.lathe([(.328, .83), (.346, .846), (.346, .888), (.312, .895), (.311, .852), (.328, .83)],
                IRON, sides=16)
        b.lathe([(.285, .33), (.294, .34), (.296, .372), (.287, .38)], IRON, sides=16)
        for side in (-1., 1.):
            b.tube([(side*.30, -.10, .68), (side*.365, -.10, .73),
                     (side*.365, .10, .73), (side*.30, .10, .68)], .013, IRON, sides=6, caps=False)
    else:
        color = np.array([.25, .29, .265])
        b.box((0., 0., .06), (.56, .56, .12), IRON, bevel=.018)
        b.box((0., 0., .43), (.515, .515, .68), color, bevel=.025)
        for side in (-1., 1.):
            for j in range(5):
                b.box((-.20+j*.10, side*.264, .43), (.023, .019, .56), color*.85, bevel=.004)
            b.box((side*.257, 0., .83), (.046, .52, .25), color, bevel=.01)
        b.box((0., .25, .83), (.47, .045, .25), color, bevel=.01)
        b.box((0., 0., .754), (.46, .46, .022), np.array([.075, .08, .073]), bevel=.004)
        b.box((0., 0., .979), (.59, .58, .075), color, bevel=.025)
        b.box((0., -.278, .65), (.15, .016, .045), np.array([.58, .60, .55]), bevel=.005)
    v, i = b.mesh()
    if variant == 0: v[:, 2] -= .025
    return v, i


@lru_cache(maxsize=1)
def hydrant_template():
    """Cast barrel, domed bonnet, three capped outlets and retaining chains."""
    b = _Builder()
    red = np.array([.52, .19, .115])
    b.lathe([(0., 0.), (.185, 0.), (.205, .045), (.205, .07), (.14, .105),
              (.128, .15), (.128, .61), (.168, .63), (.168, .678),
             (.153, .70)], red)
    b.lathe([(.153, .70), (.143, .746), (.11, .791), (.057, .825), (0., .834)], red, smooth=True)
    b.lathe([(.169, .645), (.169, .652)], red*.58)
    for side in (-1., 1.):
        b.tube([(side*.085, 0., .49), (side*.19, 0., .49), (side*.211, 0., .49)],
               [.074, .074, .087], red, sides=10, caps=False)
        b.tube([(side*.213, 0., .49), (side*.222, 0., .49)], .085, red*.56, sides=10, caps=False)
        b.tube([(side*.224, 0., .49), (side*.251, 0., .49)], [.087, .077], red, sides=10)
        b.tube([(side*.252, 0., .49), (side*.272, 0., .49)], .025, STEEL, 6., sides=6)
        b.tube([(side*.24, -.065, .49), (side*.265, -.09, .40),
                (side*.20, -.13, .31), (side*.12, -.10, .38)], .0055,
               np.array([.30, .28, .235]), sides=4)
    # The large steamer fitting faces the carriageway; the two smaller hose
    # caps are perpendicular to it. Each has a separate seam and wrench nut.
    b.tube([(0., -.09, .48), (0., -.19, .48), (0., -.215, .48)],
           [.102, .102, .117], red, sides=12, caps=False)
    b.tube([(0., -.217, .48), (0., -.227, .48)], .112, red*.55, sides=12, caps=False)
    b.tube([(0., -.229, .48), (0., -.261, .48)], [.117, .099], red, sides=12)
    b.tube([(0., -.262, .48), (0., -.284, .48)], .031, STEEL, 6., sides=6)
    b.tube([(0., 0., .83), (0., 0., .879)], .029, STEEL, 6., sides=6)
    for j in range(4):
        a = (j+.5)*math.tau/4.
        x, y = .145*math.cos(a), .145*math.sin(a)
        b.tube([(x, y, .674), (x, y, .690)], .010, STEEL, 6., sides=6)
        x, y = .174*math.cos(a), .174*math.sin(a)
        b.tube([(x, y, .067), (x, y, .084)], .013, STEEL, 6., sides=6)
    return b.mesh()


@lru_cache(maxsize=2)
def bollard_template(variant=0):
    b = _Builder()
    color = np.array([.53, .52, .48]) if variant == 0 else IRON
    if variant == 0:
        b.lathe([(0., 0.), (.18, 0.), (.18, .59), (.165, .62), (0., .62)], color, 5., sides=12)
    else:
        b.lathe([(0., 0.), (.11, 0.), (.11, .07), (.075, .10), (.075, .70),
                 (.05, .745), (0., .76)], color, sides=10)
        b.lathe([(.077, .52), (.077, .57)], np.array([.73, .70, .53]), sides=10)
    return b.mesh()


@lru_cache(maxsize=1)
def streetpole_template():
    """Tapered shaft, cast base, curved tube arm and bevelled cobra-head housing.

    Parts: 0 height-scaled shaft, 1 arm-length-scaled tube, 2 housing,
    3 metre-sized base, 4 top collar, 5 recessed lens, 6/7 post-top frame/lens.
    Only lens parts emit. The emitter remains exactly 30 cm below the arm datum.
    """
    b = _Builder()
    b.tube([(0., 0., .025), (0., 0., 1.)], [.125, .073], IRON, sides=12, part=0.)
    b.lathe([(0., .0), (.205, .0), (.22, .055), (.205, .105), (.16, .16),
             (.145, .34), (.13, .40)], IRON, sides=12, part=3.)
    b.box((0., -.137, .245), (.105, .025, .15), IRON*.72, bevel=.012, part=3.)
    b.lathe([(.072, -.20), (.104, -.185), (.104, -.115), (.075, -.10)], IRON, part=4.)
    b.tube([(0., 0., -.145), (.18, 0., -.01), (.40, 0., .028),
            (.69, 0., .012), (1., 0., -.11)], [.052, .052, .048, .042, .038],
           IRON, sides=8, part=1.)
    b.tube([(0., 0., -.61), (.17, 0., -.38), (.43, 0., -.055)], .018, IRON, sides=6, part=1.)
    # The lens is an inset face, not the glowing underside of an oversized box.
    b.box((0., 0., -.194), (.78, .34, .185), np.array([.34, .36, .36]),
          bevel=.045, part=2.)
    b.box((-.035, 0., -.286), (.59, .263, .018), np.array([.12, .14, .14]), bevel=.007, part=2.)
    b.face([(-.27, .103, -LAMP_LENS_DROP), (.22, .103, -LAMP_LENS_DROP),
            (.25, .07, -LAMP_LENS_DROP), (.25, -.07, -LAMP_LENS_DROP),
            (.22, -.103, -LAMP_LENS_DROP), (-.27, -.103, -LAMP_LENS_DROP)],
           np.ones(3), part=5.)
    for y in (-.085, 0., .085):
        b.box((-.12, y, -.09), (.35, .011, .025), np.array([.25, .27, .27]), bevel=.003, part=2.)
    # Legacy three-column light arrays still get a proper post-top lantern.
    b.lathe([(0., -.51), (.19, -.51), (.23, -.46), (.18, -.41)], IRON, part=6.)
    b.lathe([(.24, -.03), (.27, .015), (.20, .09), (0., .15)], IRON, part=6.)
    for x, y in ((-.15, -.15), (.15, -.15), (.15, .15), (-.15, .15)):
        b.tube([(x, y, -.44), (x, y, .0)], .013, IRON, sides=5, part=6.)
    b.lathe([(.115, -.42), (.145, -.05)], np.array([.76, .77, .71]), sides=8, part=7.)
    v, i = b.mesh()
    return np.ascontiguousarray(np.c_[v[:, :6], v[:, 16]], np.float32), i
