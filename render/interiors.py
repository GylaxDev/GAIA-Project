"""Historical photographic assets for offline comparison and baker checks.

RGB is sRGB-encoded, normalized photographic shading, not an albedo map. Alpha
contains only selected practical light sources. Each layer is an octahedral
direction map, which uses half the texels of a comparable lat-long panorama.
"""
from pathlib import Path

import numpy as np

INTERIOR_SIZE = 1024
INTERIOR_ROOMS = ('cafe', 'retail', 'market', 'arcade', 'lobby', 'office', 'living', 'bedroom')
INTERIOR_ASSET = Path(__file__).resolve().parents[1] / 'assets' / 'interiors' / 'rooms.npz'


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, np.float32)
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)


def linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0., 1.)
    return np.where(rgb <= .0031308, rgb * 12.92, 1.055 * rgb**(1. / 2.4) - .055)


def lightfield_mips(rgba):
    """Filter linear RGB and source area separately, without cross-room bleeding."""
    data = np.asarray(rgba, np.float32)
    while True:
        encoded = data.copy()
        encoded[..., :3] = linear_to_srgb(data[..., :3])
        yield np.ascontiguousarray(np.rint(np.clip(encoded, 0., 1.) * 255.), np.uint8)
        size = data.shape[1]
        if size == 1:
            break
        data = data.reshape(len(data), size // 2, 2, size // 2, 2, 4).mean(axis=(2, 4))


def load_interior_textures():
    """Load the historical baked mip chain for offline comparisons."""
    with np.load(INTERIOR_ASSET, allow_pickle=False) as asset:
        if int(asset['version']) != 1:
            raise ValueError('Unsupported interior texture version')
        levels = []
        for level in range(INTERIOR_SIZE.bit_length()):
            data = asset[f'level_{level}']
            size = max(1, INTERIOR_SIZE >> level)
            if data.dtype != np.uint8 or data.shape != (len(INTERIOR_ROOMS), size, size, 4):
                raise ValueError(f'Invalid interior mip {level}')
            levels.append(np.ascontiguousarray(data))
        return levels
