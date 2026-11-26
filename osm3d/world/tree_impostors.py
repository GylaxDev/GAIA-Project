"""CPU-side mip chain for the four-view tree impostor atlas.

The atlas itself is rasterised on the GPU at startup (``Renderer``), because the
production tree template is far too heavy for a Python triangle loop.  Mip
generation is pure numpy so it can be tested without a GL context, and it keeps
each view's cutout coverage instead of letting distant billboards dissolve.
"""
from __future__ import annotations

import math

import numpy as np

from .foliage import LEAF_ALPHA_CUTOFF

TREE_IMPOSTOR_VIEWS = 4
TREE_IMPOSTOR_TILE = 256
TREE_IMPOSTOR_ATLAS = TREE_IMPOSTOR_TILE * 2


def _coverage_alpha(alpha, target):
    """Scale a tile's mip alpha so its cutout coverage stays close to ``target``."""
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


def build_tree_impostor_mipmaps(atlas):
    """Return coverage-preserving RGBA mips for a square 2x2 impostor atlas."""
    atlas = np.asarray(atlas)
    side = atlas.shape[0]
    if atlas.shape != (side, side, 4) or side < 2 or side & (side - 1):
        raise ValueError('tree impostor atlas must be a square power-of-two RGBA image')
    levels = [np.ascontiguousarray(atlas)]
    tile = side // 2
    coverage = [float(np.mean(atlas[y:y + tile, x:x + tile, 3] / 255. >= LEAF_ALPHA_CUTOFF))
                for y in (0, tile) for x in (0, tile)]
    while side > 1:
        color = levels[-1].astype(np.float32) / 255.
        weights = color[..., 3:4]
        def reduce(a):
            return a.reshape(side // 2, 2, side // 2, 2, -1).mean(axis=(1, 3))
        alpha = reduce(weights)
        rgb = reduce(color[..., :3] * weights) / np.maximum(alpha, 1e-6)
        rgb[alpha[..., 0] <= 1e-6] = 0.
        side //= 2
        if side >= 2:
            tile = side // 2
            for variant in range(4):
                x, y = variant % 2 * tile, variant // 2 * tile
                region = np.s_[y:y + tile, x:x + tile, 0]
                alpha[region] = _coverage_alpha(alpha[region], coverage[variant])
        levels.append(np.ascontiguousarray(
            np.rint(np.clip(np.concatenate((rgb, alpha), axis=-1), 0., 1.) * 255.), np.uint8))
    return levels
