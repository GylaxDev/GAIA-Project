"""Shared, deterministic broadleaf trees with a real woody branch hierarchy.

Levels 0-2 are geometric LODs of one skeleton; level 3 is a camera-facing
impostor quad that samples a baked four-view atlas. Every template keeps the
existing 13-float foliage contract, so cached eight-float tree instances need no
rebuild.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

TREE_CENTER_Z = 4.65
TREE_RADIUS = 6.5       # includes the shared shader's bounded instance deformation
TREE_HEIGHT = 10.2
# Optimized: lower pixel thresholds for earlier LOD transitions (was 200, 65)
TREE_LOD_PIXELS = (150., 40.)
# Below this projected height the geometric crown is replaced by a two-triangle
# impostor.  Its square world window is TREE_BILLBOARD_SIZE wide and tall, which
# covers the deformed crown (actual bounds are about 8.3 x 9.6 m).
TREE_BILLBOARD_PIXELS = 28.
TREE_BILLBOARD_SIZE = TREE_HEIGHT


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _point(path, t):
    """Interpolate a branch by its normalized segment parameter."""
    t = float(np.clip(t, 0., 1.)) * (len(path) - 1)
    i = min(int(t), len(path) - 2)
    return path[i] * (1. - (t - i)) + path[i + 1] * (t - i)


@dataclass
class _Limb:
    path: np.ndarray
    radius: float
    tips: list
    sprays: list


def _skeleton():
    rng = np.random.default_rng(731)
    trunk = np.array([[0., 0., 0.], [.015, -.01, .20], [.045, .015, .8],
                      [.025, .035, 2.05], [-.035, .04, 3.25], [.08, .04, 4.6],
                      [.18, .055, 5.75], [.13, .15, 6.9], [.21, .16, 7.95],
                      [.13, .17, 8.5]])
    radii = np.array([.43, .33, .245, .215, .18, .135, .09, .056, .028, .008])
    limbs = []
    for i in range(17):
        # Irregular scaffold limbs spread through a rounded deciduous crown.
        # Three steeper forks fill its centre instead of making a bare funnel.
        leader_fork = i >= 14
        z = (3.25 + (i - 14) * .42 if leader_fork else 2.65 + i * .235)
        z += rng.uniform(-.19, .18)
        start = np.array([np.interp(z, trunk[:, 2], trunk[:, 0]),
                          np.interp(z, trunk[:, 2], trunk[:, 1]), z])
        angle = i * 2.39996323 + rng.uniform(-.32, .32)
        radial = np.array([math.cos(angle), math.sin(angle), 0.])
        side = np.array([-radial[1], radial[0], 0.])
        reach = (1.05 if leader_fork else 2.65 - max(0, i - 5) * .135)
        reach += rng.uniform(-.25, .24)
        rise = (7.55 - z if leader_fork else 2.45 - i * .023) + rng.uniform(-.25, .24)
        end = start + radial * reach + np.array([0., 0., rise])
        path = np.array([start,
                         start + radial * reach * .20 + side * .14 + [0., 0., rise * .32],
                         start + radial * reach * .67 - side * .13 + [0., 0., rise * .77], end])
        tips, sprays = [], []
        for j in range(4):
            attach = _point(path, .43 + j * .145)
            a = angle + (-.70 if j % 2 == 0 else .65) + rng.uniform(-.20, .20)
            length = rng.uniform(.75, 1.10) * (1. - max(0, i - 7) * .035)
            direction = np.array([math.cos(a) * length, math.sin(a) * length,
                                  rng.uniform(.48, .78)])
            tip = attach + direction
            secondary = np.array([attach, attach + direction * .52 - [0., 0., .09], tip])
            fork_tips = []
            for sign in (-1., 1.):
                fork_start = _point(secondary, .72)
                fa = a + sign * rng.uniform(.60, .95)
                fd = np.array([math.cos(fa), math.sin(fa), rng.uniform(.35, .65)]) * rng.uniform(.32, .48)
                fork_tip = fork_start + fd
                fork_tips.append((np.array([fork_start, fork_tip]), .008))
                sprays.append((fork_tip, _unit(fd), rng.uniform(.86, 1.10)))
            tips.append((secondary, .032 * (1. - j * .10), fork_tips))
            sprays.append((tip, _unit(direction), rng.uniform(.95, 1.12)))
            sprays.append((_point(secondary, .70), _unit(direction), rng.uniform(.86, 1.02)))
        sprays.append((end, _unit(path[-1] - path[-2]), 1.05))
        limbs.append(_Limb(path, .108 - min(i, 13) * .0036, tips, sprays))
    # A few fine terminal sprays close the leader without hiding its structure.
    leader = [(trunk[-2] + [dx, dy, dz], _unit([dx, dy, .7]), .90)
              for dx, dy, dz in ((-.45, .12, .18), (.40, -.14, .28), (.05, .42, .55))]
    return trunk, radii, limbs, leader


class _Builder:
    def __init__(self):
        self.vertices, self.indices = [], []

    def branch(self, path, radii, sides, seed):
        path, radii = np.asarray(path), np.asarray(radii)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        tangents = np.empty_like(path)
        tangents[0], tangents[-1] = path[1] - path[0], path[-1] - path[-2]
        tangents[1:-1] = path[2:] - path[:-2]
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
        # Parallel transport avoids rotating the bark seam at a curved joint.
        ref = np.array([1., 0., 0.]) if abs(tangents[0, 0]) < .8 else np.array([0., 1., 0.])
        axis = _unit(ref - tangents[0] * np.dot(ref, tangents[0]))
        base = len(self.vertices)
        for k, (center, radius, tangent) in enumerate(zip(path, radii, tangents)):
            axis = _unit(axis - tangent * np.dot(axis, tangent))
            other = np.cross(tangent, axis)
            taper = ((radii[max(k - 1, 0)] - radii[min(k + 1, len(path) - 1)]) /
                     max(arc[min(k + 1, len(path) - 1)] - arc[max(k - 1, 0)], 1e-6))
            for j in range(sides + 1):
                angle = j * math.tau / sides
                radial = axis * math.cos(angle) + other * math.sin(angle)
                # Shallow fluting is strongest at the root collar, not a round pole.
                flute = 1. + .065 * math.sin(angle * 3. + seed * 7.) * math.exp(-arc[k] * .8)
                pos = center + radial * radius * flute
                normal = _unit(radial + tangent * taper)
                color = np.array([.115, .103, .088]) * (.90 + .14 * seed)
                self.vertices.append((*pos, *normal, *color, 0.,
                                      j / sides * math.tau * radii[0], arc[k], seed))
        for k in range(len(path) - 1):
            for j in range(sides):
                a = base + k * (sides + 1) + j
                b = a + sides + 1
                self.indices.extend((a, a + 1, b + 1, a, b + 1, b))

    def spray(self, center, direction, size, seed, level):
        # Each twig carries differently tilted, gently folded leaf sprays.
        # There is no camera-facing rotation and no full-crown vertical plane.
        center = np.asarray(center)
        direction = _unit(direction)
        radial = _unit((center - [0., 0., 6.2]) / [3., 3., 2.8])
        ref = np.array([0., 0., 1.]) if abs(direction[2]) < .88 else np.array([0., 1., 0.])
        right = _unit(np.cross(direction, ref))
        plane = _unit(np.cross(right, direction))
        angle = seed * math.tau
        right = right * math.cos(angle) + plane * math.sin(angle)
        plane = _unit(np.cross(right, direction))
        if np.dot(plane, radial) < 0.:
            plane = -plane
        width, length = .48 * size, .67 * size
        center = center + direction * (.10 * size)
        columns = (0., .5, 1.) if level == 0 else (0., 1.)
        base = len(self.vertices)
        shade = np.clip(.79 + .17 * np.linalg.norm(center[:2]) / 3.5
                        + .05 * (center[2] - 5.9) + .07 * (seed - .5), .72, 1.04)
        for v in (0., 1.):
            for u in columns:
                fold = (1. - (u * 2. - 1.) ** 2) * .16 * size
                p = center + right * ((u * 2. - 1.) * width) + direction * ((v * 2. - 1.) * length) + plane * fold
                # Volume normals retain crown depth from either side of a spray.
                outward = _unit((p - [0., 0., 6.2]) / [3., 3., 2.8])
                normal = _unit(outward * .48 + radial * .08 + plane * .44)
                self.vertices.append((*p, *normal, shade, shade, shade, 1., u, v, seed))
        n = len(columns)
        for j in range(n - 1):
            a = base + j
            self.indices.extend((a, a + 1, a + n + 1, a, a + n + 1, a + n))

    def cluster(self, center, direction, size, seed, level):
        """Equal-area near/middle sprays; coarser clusters share the same sites."""
        rng = np.random.default_rng(int(seed * 65536) + 91)
        for _ in range((5, 3, 2)[level]):
            offset = rng.uniform(-.17, .17, 3) * size
            axis = _unit(direction * .45 + _unit(rng.normal(size=3)) * .55 + [0., 0., .16])
            roll = float(rng.random())
            self.spray(center + offset, axis, size * (.64, .82, 1.04)[level], roll, level)

    def finish(self):
        return (np.ascontiguousarray(self.vertices, np.float32),
                np.ascontiguousarray(self.indices, np.uint32))


def build_tree_template(level=0):
    """Return (vertices, indices) at near/middle/far detail (0, 1, 2)."""
    if level not in (0, 1, 2):
        raise ValueError('tree LOD must be 0, 1 or 2')
    trunk, radii, limbs, leader = _skeleton()
    b = _Builder()
    keep = (np.arange(len(trunk)) if level == 0 else
            np.array([0, 1, 2, 3, 5, 7, 9]) if level == 1 else np.array([0, 2, 3, 6, 9]))
    b.branch(trunk[keep], radii[keep], (12, 8, 6)[level], .37)
    if level < 2:
        for i in range(5):
            a = i * math.tau / 5. + .2
            direction = np.array([math.cos(a), math.sin(a), 0.])
            path = np.array([[0., 0., .22], direction * .38 + [0., 0., .09],
                             direction * .67 + [0., 0., .018]])
            if level == 1:
                path = path[[0, 2]]
            b.branch(path, np.linspace(.13, .010, len(path)), 5 if level == 0 else 4, i / 6.)
    for i, limb in enumerate(limbs):
        if level == 0:
            path, sides = limb.path, 8
        elif level == 1:
            path, sides = limb.path[[0, 2, 3]], 5
        else:
            path, sides = limb.path[[0, 3]], 3
        if level < 2 or i % 3 != 1:
            b.branch(path, np.linspace(limb.radius, .012, len(path)), sides, (i * .173) % 1.)
        if level == 0:
            for j, (path, radius, forks) in enumerate(limb.tips):
                b.branch(path, [radius, radius * .56, .006], 5, (i * .173 + j * .13) % 1.)
                for fork, r in forks:
                    b.branch(fork, [r, .0025], 3, (i * .11 + j * .21) % 1.)
            for j, (pos, direction, size) in enumerate(limb.sprays):
                seed = (i * .231 + j * .137 + .17) % 1.
                b.cluster(pos, direction, size, seed, level)
        elif level == 1:
            for j, (path, radius, _) in enumerate(limb.tips):
                b.branch(path[[0, 2]], [radius, .005], 4, (i * .173 + j * .13) % 1.)
            for j, (pos, direction, size) in enumerate(limb.sprays):
                b.cluster(pos, direction, size, (i * .231 + j * .137 + .17) % 1., level)
        else:
            for j, (pos, direction, size) in enumerate(limb.sprays):
                if j % 2 == 0:
                    b.cluster(pos, direction, size, (i * .231 + j * .137 + .17) % 1., level)
    for i, (pos, direction, size) in enumerate(leader):
        b.cluster(pos, direction, size, (.17 + i * .27) % 1., level)
    return b.finish()


def build_tree_billboard_template():
    """Return the 13-float impostor quad used by tree LOD 3.

    ``aPos.x`` is the horizontal corner and ``aPos.z`` the vertical one, so the
    shared vertex shader can expand it in world space.  ``aPart = 2`` selects the
    impostor atlas and the camera-facing transform.
    """
    half = TREE_BILLBOARD_SIZE * .5
    vertices = np.array([
        [-half, 0., 0., 0., 0., 1., 1., 1., 1., 2., 0., 0., 0.],
        [half, 0., 0., 0., 0., 1., 1., 1., 1., 2., 1., 0., 0.],
        [half, 0., TREE_BILLBOARD_SIZE, 0., 0., 1., 1., 1., 1., 2., 1., 1., 0.],
        [-half, 0., TREE_BILLBOARD_SIZE, 0., 0., 1., 1., 1., 1., 2., 0., 1., 0.],
    ], np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    return vertices, indices


def tree_lod_levels(scales, *, pixel_scale=None, distances=None, texel=None):
    """Projected height selects detail in either the scene or a light-space map.

    Returns 0=high, 1=medium, 2=low geometry, 3=2D impostor.  Shadow-map callers
    clamp 3 back to 2: a camera-facing card is not a useful caster.
    """
    heights = np.asarray(scales) * TREE_HEIGHT
    pixels = heights / max(float(texel), 1e-6) if texel is not None else heights * pixel_scale / np.maximum(distances, 1.)
    return np.where(pixels > TREE_LOD_PIXELS[0], 0,
                    np.where(pixels > TREE_LOD_PIXELS[1], 1,
                             np.where(pixels > TREE_BILLBOARD_PIXELS, 2, 3)))
