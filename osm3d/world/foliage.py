"""Broadleaf spray textures and coverage-preserving mips, generated at startup."""
from __future__ import annotations

import math

import numpy as np

LEAF_TILE = 512
LEAF_ALPHA_CUTOFF = .45
# Linear reflectances, also used by newly generated eight-float tree instances.
FOLIAGE_PALETTE = np.array([
    (.108, .142, .082), (.088, .124, .083), (.124, .150, .095),
    (.097, .137, .100), (.134, .151, .091), (.081, .116, .080),
], np.float64)


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _value_noise(size: int, freq: int, rng: np.random.Generator) -> np.ndarray:
    """Tileable-enough smooth value noise in [0, 1]."""
    grid = rng.random((freq + 2, freq + 2), dtype=np.float32)
    coords = np.linspace(0.0, freq, size, endpoint=False, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords, indexing="xy")
    ix = np.floor(gx).astype(np.int32)
    iy = np.floor(gy).astype(np.int32)
    fx = gx - ix
    fy = gy - iy
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    a = grid[iy, ix]
    b = grid[iy, ix + 1]
    c = grid[iy + 1, ix]
    d = grid[iy + 1, ix + 1]
    return ((a + (b - a) * ux) * (1.0 - uy)
            + (c + (d - c) * ux) * uy)


def _composite(rgb, material, opacity, region, alpha, color, detail):
    """Over composite in premultiplied space; no black fringes on leaf edges."""
    old = opacity[region]
    a = alpha[..., None]
    rgb[region] = color * a + rgb[region] * (1. - a)
    material[region] = detail * a + material[region] * (1. - a)
    opacity[region] = alpha + old * (1. - alpha)


def _leaf(rgb, material, opacity, u, v, center, angle, rx, ry, rng, grain):
    inner = len(u)
    ca, sa = math.cos(angle), math.sin(angle)
    ex, ey = abs(ca) * rx + abs(sa) * ry, abs(sa) * rx + abs(ca) * ry
    x0, x1 = max(0, int((center[0] - ex) * inner) - 2), min(inner, int((center[0] + ex) * inner) + 3)
    y0, y1 = max(0, int((center[1] - ey) * inner) - 2), min(inner, int((center[1] + ey) * inner) + 3)
    if x1 <= x0 or y1 <= y0:
        return
    region = np.s_[y0:y1, x0:x1]
    du, dv = u[region] - center[0], v[region] - center[1]
    x, y = (du * ca + dv * sa) / rx, (-du * sa + dv * ca) / ry
    # Pointed broad leaves, not circular blobs or noise-cut crown silhouettes.
    edge = np.abs(x) ** 1.65 + np.abs(y) ** 1.35
    aa = min(.40, 1.35 / (inner * min(rx, ry)))
    alpha = 1. - _smoothstep(1. - aa, 1. + aa, edge)
    vein = np.exp(-np.square(x / .09))
    laterals = np.exp(-np.square((np.abs(x) * .55 + y * 3.5 - np.round(np.abs(x) * .55 + y * 3.5)) / .065))
    shade = rng.uniform(.67, .96) * (1. - .08 * np.abs(x) + .035 * vein - .025 * laterals)
    shade *= .96 + .08 * grain[region]
    color = shade[..., None] * np.array([rng.uniform(.96, 1.02), 1., rng.uniform(.91, 1.02)])
    # Each leaf has its own tilt, curved lamina and raised central vein.
    nx = rng.uniform(-.43, .43) + x * .28 + np.sign(x) * vein * .06
    ny = rng.uniform(-.44, .44) + y * .20
    tx, ty = nx * ca - ny * sa, nx * sa + ny * ca
    detail = np.stack([np.clip(tx * .5 + .5, .05, .95),
                       np.clip(ty * .5 + .5, .05, .95),
                       np.clip(.89 - .10 * np.abs(x) - .04 * laterals, .65, 1.),
                       np.full_like(x, rng.uniform(.66, .84))], axis=-1)
    _composite(rgb, material, opacity, region, alpha, color, detail)


def _stem(rgb, material, opacity, u, v, start, end, width):
    delta = end - start
    t = np.clip(((u - start[0]) * delta[0] + (v - start[1]) * delta[1]) /
                max(float(delta @ delta), 1e-8), 0., 1.)
    dist = np.sqrt(np.square(u - start[0] - delta[0] * t) + np.square(v - start[1] - delta[1] * t))
    alpha = 1. - _smoothstep(width, width + 1. / len(u), dist)
    _composite(rgb, material, opacity, np.s_[:, :], alpha,
               np.array([.61, .45, .29]), np.array([.5, .5, .76, .94]))


def build_leaf_textures(tile: int = LEAF_TILE, margin: int = 6, seed: int = 31):
    """Return albedo/alpha and tangent-normal/cavity/roughness 2x2 RGBA atlases.

    RGB is a linear modulation of the instance reflectance, not an sRGB tree
    color. RG in the second texture encodes the leaf normal's signed XY; B is
    local cavity visibility and A is roughness. Only the first atlas clips.
    """
    if tile < 32 or tile & (tile - 1) or margin < 0 or tile - 2 * margin < 16:
        raise ValueError('leaf tile must be a power of two >= 32 with a valid margin')
    atlas = np.zeros((tile * 2, tile * 2, 4), np.uint8)
    atlas[..., :3] = [184, 184, 174]  # harmless unassociated RGB in transparent gutters
    normals = np.empty_like(atlas)
    normals[:] = [128, 128, 230, 199]
    inner = tile - 2 * margin
    coords = (np.arange(inner, dtype=np.float32) + .5) / inner
    u, v = np.meshgrid(coords, coords, indexing='xy')
    for variant in range(4):
        rng = np.random.default_rng(seed + variant * 101)
        rgb = np.zeros((inner, inner, 3), np.float32)
        material = np.zeros((inner, inner, 4), np.float32)
        opacity = np.zeros((inner, inner), np.float32)
        grain = _value_noise(inner, 65, rng)
        foot, leader = np.array([.47, .07]), np.array([.52, .89])
        _stem(rgb, material, opacity, u, v, foot, leader, .003)
        twigs = []
        for j in range(8):
            sign = -1. if j % 2 == 0 else 1.
            height = .18 + (j // 2) * .165 + rng.uniform(-.025, .025)
            start = foot + (leader - foot) * height
            reach = .30 - (j // 2) * .035 + rng.uniform(-.025, .025)
            end = start + [sign * reach, rng.uniform(.19, .27)]
            twigs.append((start, end))
            _stem(rgb, material, opacity, u, v, start, end, .0018)
        # Overlapping leaf pairs follow twigs, with irregular gaps at their ends.
        for start, end in twigs:
            along = end - start
            across = np.array([-along[1], along[0]]) / np.linalg.norm(along)
            for j in range(8):
                t = .13 + j * .108 + rng.uniform(-.022, .022)
                sign = -1. if j % 2 else 1.
                center = start + along * t + across * sign * rng.uniform(.025, .041)
                # _leaf's Y axis follows the long axis, so subtract pi/2.
                angle = math.atan2(along[1], along[0]) - math.pi * .5 + sign * .62 + rng.uniform(-.20, .20)
                _leaf(rgb, material, opacity, u, v, center, angle,
                      rng.uniform(.032, .045), rng.uniform(.045, .064), rng, grain)
        for j in range(9):
            center = foot + (leader - foot) * (.27 + j * .075)
            center += [(-1. if j % 2 else 1.) * .042, .012]
            _leaf(rgb, material, opacity, u, v, center, (-.55 if j % 2 else .55),
                  rng.uniform(.034, .044), rng.uniform(.046, .062), rng, grain)
        opaque = opacity > 1e-6
        rgb[opaque] /= opacity[opaque, None]
        material[opaque] /= opacity[opaque, None]
        rgb[~opaque] = [184 / 255., 184 / 255., 174 / 255.]
        material[~opaque] = [128 / 255., 128 / 255., 230 / 255., 199 / 255.]
        ox, oy = variant % 2 * tile + margin, variant // 2 * tile + margin
        atlas[oy:oy + inner, ox:ox + inner, :3] = np.rint(np.clip(rgb, 0., 1.) * 255.).astype(np.uint8)
        atlas[oy:oy + inner, ox:ox + inner, 3] = np.rint(opacity * 255.).astype(np.uint8)
        normals[oy:oy + inner, ox:ox + inner] = np.rint(np.clip(material, 0., 1.) * 255.).astype(np.uint8)
    return atlas, normals


def build_leaf_atlas(tile: int = LEAF_TILE, margin: int = 6, seed: int = 31) -> np.ndarray:
    """Compatibility entry point returning just the albedo/alpha atlas."""
    return build_leaf_textures(tile, margin, seed)[0]


def _coverage_alpha(alpha, target):
    """Scale each tile's mip alpha to retain its cutout coverage at 0.45."""
    # One texel cannot express fractional cutout coverage. Keep a nonempty
    # terminal spray instead of choosing the numerically closer empty tile;
    # otherwise every subpixel crown vanishes in both colour and caster passes.
    if alpha.size == 1 and target > 0. and float(alpha.max()) > 0.:
        return np.full_like(alpha, .65)
    low, high, best, error = 0., 8., alpha, math.inf
    for _ in range(18):
        scale = (low + high) * .5
        candidate = np.rint(np.clip(alpha * scale, 0., 1.) * 255.) / 255.
        coverage = float(np.mean(candidate >= LEAF_ALPHA_CUTOFF))
        if abs(coverage - target) < error:
            best, error = candidate, abs(coverage - target)
        if coverage < target:
            low = scale
        else:
            high = scale
    return best


def build_leaf_mipmaps(albedo, material):
    """Alpha-weighted color/normal mips, without shrinking distant leafy crowns.

    The two atlas tiles in each axis are reduced independently for alpha
    coverage. Ordinary glGenerateMipmap averages sparse leaves below the clip
    threshold and makes a tree disappear rather than become finer at distance.
    """
    side = albedo.shape[0]
    if albedo.shape != (side, side, 4) or material.shape != albedo.shape or side & (side - 1):
        raise ValueError('leaf mipmaps require matching square power-of-two RGBA atlases')
    color_levels, material_levels = [np.ascontiguousarray(albedo)], [np.ascontiguousarray(material)]
    tile = side // 2
    coverage = [float(np.mean(albedo[y:y + tile, x:x + tile, 3] / 255. >= LEAF_ALPHA_CUTOFF))
                for y in (0, tile) for x in (0, tile)]
    while side > 1:
        color, detail = color_levels[-1].astype(np.float32) / 255., material_levels[-1].astype(np.float32) / 255.
        weights = color[..., 3:4]
        def reduce(a):
            return a.reshape(side // 2, 2, side // 2, 2, -1).mean(axis=(1, 3))
        alpha = reduce(weights)
        rgb = reduce(color[..., :3] * weights) / np.maximum(alpha, 1e-6)
        aux = reduce(detail * weights) / np.maximum(alpha, 1e-6)
        empty = alpha[..., 0] <= 1e-6
        rgb[empty] = [184 / 255., 184 / 255., 174 / 255.]
        aux[empty] = [128 / 255., 128 / 255., 230 / 255., 199 / 255.]
        side //= 2
        if side >= 2:
            tile = side // 2
            for variant in range(4):
                x, y = variant % 2 * tile, variant // 2 * tile
                region = np.s_[y:y + tile, x:x + tile, 0]
                alpha[region] = _coverage_alpha(alpha[region], coverage[variant])
        color_levels.append(np.ascontiguousarray(np.rint(np.clip(np.concatenate((rgb, alpha), axis=-1), 0., 1.) * 255.), np.uint8))
        material_levels.append(np.ascontiguousarray(np.rint(np.clip(aux, 0., 1.) * 255.), np.uint8))
    return color_levels, material_levels
