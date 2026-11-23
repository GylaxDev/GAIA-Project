"""Supplied 2015 Ford F-150 Lariat Super Cab, baked from the source FBX.

The high-definition pickup is converted to a versioned runtime asset,
``assets/vehicles/f150_2015.npz`` (see tools/bake_f150_model.py): three LODs with
baked vertex colours, the original body panels, mirrors, bed, alloy wheels and
brake hardware.  Unlike the earlier distance model, the original cabin is kept:
front buckets, rear bench, dashboard, steering wheel and door cards are visible
through the windows, which use the renderer's real transmission material (24).

Body paint is tagged material 0 so it keeps taking the per-vehicle colour; glass,
chrome, lamps and trim map onto GAIA's existing ten-float, single-draw instanced
material ids.  Each wheel assembly carries its own pivot/role (rear spin, front
spin, front steer for the calipers).

Metres, +x forward, +y left, z = 0 is the road, origin between the axles.  The
model is scaled to the F-150's 3683 mm wheelbase; rolling radius is 437.6 mm.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .vehicle_models import VehicleTemplate
from .vehicle_lamps import bake_lamp_coords
from .vehicle_reference import F150

_MODEL = Path(__file__).resolve().parents[1] / 'assets/vehicles/f150_2015.npz'


@lru_cache(maxsize=1)
def _templates():
    """Load and validate the baked F-150 at all three LODs."""
    with np.load(_MODEL, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata['version'] != 1:
            raise ValueError('Unsupported F-150 model asset version')
        for key in ('length', 'width', 'height', 'wheelbase', 'front_axle', 'track',
                    'tyre_width', 'tyre_radius', 'rim_radius', 'mirror_width',
                    'clearance'):
            if abs(metadata['dimensions'][key] - getattr(F150, key)) > 1e-7:
                raise ValueError('F-150 hard points changed; rebuild with tools/bake_f150_model.py')
        result = tuple(VehicleTemplate(bake_lamp_coords(data[f'v{lod}'], data[f'i{lod}']),
                                       data[f'i{lod}'],
                                       len(data[f'i{lod}']) - metadata['glass'][lod],
                                       data[f'a{lod}'], metadata['glass'][lod])
                       for lod in range(3))
    for v, idx, _ in result:
        v.setflags(write=False)
        idx.setflags(write=False)
    for template in result:
        template.animation.setflags(write=False)
    return result


def _f150(m, lod):
    """Append one baked F-150 LOD to a mesh without copying its arrays."""
    if lod not in (0, 1, 2):
        raise ValueError('F-150 LOD must be 0, 1 or 2')
    template = _templates()[lod]
    base = len(m.V)
    m.V.extend(template[0].tolist())
    m.A.extend(template.animation.tolist())
    m.I.extend((template[1][:template[2]].astype(np.int64) + base).tolist())
    m.I.extend((template[1][template[2]:].astype(np.int64) + base).tolist())
    m.glass_indices = template.glass_count
