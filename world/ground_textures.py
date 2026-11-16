"""Small, seamless ground-material tiles, baked once rather than per fragment.

One opaque RGBA8 array holds four ordinary 2D textures. R is material tone,
GB are signed height gradients, and A is thatch/aggregate/pore coverage. The
shader supplies restrained linear palettes. No displacement, cutouts, blades,
terrain samplers or per-chunk texture storage are needed.
"""
from __future__ import annotations

import math

import numpy as np

GROUND_TILE = 512
GROUND_METRES = 2.0
GROUND_SLOPE_LIMIT = .40
GROUND_LAYERS = ('grass', 'asphalt', 'concrete', 'soil')


def _smoothstep(lo, hi, value):
    t = np.clip((value - lo) / (hi - lo), 0., 1.)
    return t * t * (3. - 2. * t)


def _periodic_noise(size, cells, rng):
    """Pixel-centred value noise with a genuinely periodic lattice in both axes."""
    nx, ny = (cells, cells) if np.ndim(cells) == 0 else cells
    grid = rng.random((ny, nx), dtype=np.float32)
    x = (np.arange(size, dtype=np.float32) + .5) * (nx / size)
    y = (np.arange(size, dtype=np.float32) + .5) * (ny / size)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - ix, y - iy
    ux, uy = fx * fx * (3. - 2. * fx), fy * fy * (3. - 2. * fy)
    a = grid[iy[:, None] % ny, ix[None, :] % nx]
    b = grid[iy[:, None] % ny, (ix[None, :] + 1) % nx]
    c = grid[(iy[:, None] + 1) % ny, ix[None, :] % nx]
    d = grid[(iy[:, None] + 1) % ny, (ix[None, :] + 1) % nx]
    return (a + (b - a) * ux) * (1. - uy[:, None]) + (c + (d - c) * ux) * uy[:, None]


def _strand(tone, height, thatch, rng):
    """Bake a short bowed, tapered fibre, wrapping its footprint across tile edges."""
    size = len(tone)
    pixels = size / GROUND_METRES
    center = rng.uniform(0., size, 2)
    angle = rng.uniform(0., math.tau)
    ca, sa = math.cos(angle), math.sin(angle)
    half = rng.uniform(.012, .032) * pixels
    width = rng.uniform(.0012, .0024) * pixels
    bend = rng.uniform(-.18, .18)
    radius = half + width + abs(bend) * half + 2.
    x = np.arange(math.floor(center[0] - radius), math.ceil(center[0] + radius) + 1)
    y = np.arange(math.floor(center[1] - radius), math.ceil(center[1] + radius) + 1)
    dx, dy = x[None, :] + .5 - center[0], y[:, None] + .5 - center[1]
    along, across = dx * ca + dy * sa, -dx * sa + dy * ca
    s = np.clip(along, -half, half)
    distance = np.sqrt((along - s)**2 + (across - bend * s * s / half)**2)
    taper = .85 - .45 * np.abs(s) / half
    w = width * taper
    # Integrate subpixel widths instead of turning each strand into a bright pixel.
    alpha = (1. - _smoothstep(np.maximum(0., w - .7), w + .7, distance)) * np.minimum(1., w / .7)
    region = np.ix_(y % size, x % size)
    dry = rng.random() > .93
    value = rng.uniform(.25, .88)
    tone[region] = tone[region] * (1. - alpha * .78) + value * alpha * .78
    height[region] += alpha * rng.uniform(.0006, .0014)
    if dry:
        thatch[region] = np.maximum(thatch[region], alpha * .8)


def _encode(tone, height, feature):
    pixels = len(tone) / GROUND_METRES
    gx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * (.5 * pixels)
    gy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * (.5 * pixels)
    # 128 is exactly flat, including in the terminal mip; no quantization tilt.
    gradients = 128. + np.stack((gx, gy), axis=-1) * (127. / GROUND_SLOPE_LIMIT)
    tile = np.empty((*tone.shape, 4), np.uint8)
    tile[..., 0] = np.rint(np.clip(tone, 0., 1.) * 255.).astype(np.uint8)
    tile[..., 1:3] = np.rint(np.clip(gradients, 1., 255.)).astype(np.uint8)
    tile[..., 3] = np.rint(np.clip(feature, 0., 1.) * 255.).astype(np.uint8)
    return tile


def build_ground_textures(size=GROUND_TILE, seed=73):
    """Return four seamless, opaque material-detail tiles; no image dependencies."""
    if size < 32 or size & (size - 1):
        raise ValueError('ground tile must be a power of two >= 32')
    result = []
    for layer in range(len(GROUND_LAYERS)):
        rng = np.random.default_rng(seed + layer * 101)
        def noise(cells):
            return _periodic_noise(size, cells, rng)
        if layer == 0:
            clumps, grain = noise(9), noise(74)
            tone = .51 + (clumps - .5) * .30 + (grain - .5) * .32
            height = (grain - .5) * .0015
            feature = _smoothstep(.76, .94, noise(18)) * .25
            for _ in range(5200):
                _strand(tone, height, feature, rng)
        elif layer == 1:
            aggregate, fine = noise(118), noise(210)
            feature = _smoothstep(.64, .86, fine)
            tone = .49 + (aggregate - .5) * .85 + (fine - .5) * .38 + (noise(20) - .5) * .12
            height = (aggregate - .5) * .0024 + (fine - .5) * .0012
        elif layer == 2:
            grain, brushed = noise(180), noise((5, 205))
            feature = _smoothstep(.73, .92, noise(151))
            tone = .51 + (grain - .5) * .38 + (brushed - .5) * .18 + (noise(14) - .5) * .13 - feature * .12
            height = (grain - .5) * .0012 + (brushed - .5) * .0005 - feature * .001
        else:
            grain, stones = noise(153), noise(43)
            feature = _smoothstep(.68, .89, stones)
            tone = .50 + (grain - .5) * .66 + (stones - .5) * .28 + feature * .08
            height = (grain - .5) * .0024 + feature * .002
        result.append(_encode(tone, height, feature))
    return np.ascontiguousarray(result, dtype=np.uint8)


def build_ground_mipmaps(tiles):
    """Linear, per-layer area averages: distant turf stays opaque and stable."""
    tiles = np.asarray(tiles)
    if (tiles.ndim != 4 or tiles.shape[0] != len(GROUND_LAYERS) or tiles.shape[-1] != 4
            or tiles.shape[1] != tiles.shape[2] or not tiles.shape[1]
            or tiles.shape[1] & (tiles.shape[1] - 1) or tiles.dtype != np.uint8):
        raise ValueError('ground mips require four square power-of-two RGBA8 tiles')
    levels = [np.ascontiguousarray(tiles)]
    data = tiles.astype(np.float32)
    side = tiles.shape[1]
    while side > 1:
        data = data.reshape(len(GROUND_LAYERS), side // 2, 2, side // 2, 2, 4).mean(axis=(2, 4))
        side //= 2
        levels.append(np.ascontiguousarray(np.rint(data), dtype=np.uint8))
    return levels
