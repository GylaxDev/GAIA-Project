"""2020 BMW X1 xDrive18d (F48), baked from the supplied FBX.

The high-definition exterior is converted to a versioned runtime asset,
``assets/vehicles/x1_2020.npz`` (see tools/bake_x1_model.py): three exterior LODs
with baked vertex colours, the original body panels, M-sport bumpers, LED lamp
internals, split alloy wheels and brake hardware.  Body paint is tagged material 0
so it keeps taking the per-vehicle colour; glass, chrome, lamps and trim map onto
GAIA's existing ten-float, single-draw instanced material ids.  Each wheel assembly
carries its own pivot/role (rear spin, front spin, front steer for calipers).

Metres, +x forward, +y left, z = 0 is the road, origin at mid-length.  The model is
scaled to the F48's 2670 mm wheelbase; the cabin and clear lamp covers are not baked
(glass is drawn opaque, so the cabin is never visible).
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .vehicle_models import VehicleTemplate
from .vehicle_lamps import bake_lamp_coords
from .vehicle_reference import X1_SPEC

_MODEL = Path(__file__).resolve().parents[1] / 'assets/vehicles/x1_2020.npz'


@lru_cache(maxsize=1)
def _templates():
    """Load and validate the baked X1 exterior at all three LODs."""
    with np.load(_MODEL, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata['version'] != 1:
            raise ValueError('Unsupported X1 model asset version')
        for key in ('length', 'width', 'height', 'wheelbase', 'front_axle', 'track',
                    'tyre_width', 'tyre_radius', 'rim_radius', 'mirror_width', 'clearance'):
            if abs(metadata['dimensions'][key] - getattr(X1_SPEC, key)) > 1e-7:
                raise ValueError('X1 hard points changed; rebuild with tools/bake_x1_model.py')
        result = tuple(VehicleTemplate(bake_lamp_coords(data[f'v{lod}'], data[f'i{lod}']),
                                       data[f'i{lod}'], len(data[f'i{lod}']),
                                       data[f'a{lod}']) for lod in range(3))
    for v, idx, _ in result:
        v.setflags(write=False)
        idx.setflags(write=False)
    for template in result:
        template.animation.setflags(write=False)
    return result


def _x1(m, lod):
    """Append one baked X1 LOD to a mesh without copying its arrays."""
    if lod not in (0, 1, 2):
        raise ValueError('X1 LOD must be 0, 1 or 2')
    template = _templates()[lod]
    base = len(m.V)
    m.V.extend(template[0].tolist())
    m.A.extend(template.animation.tolist())
    m.I.extend((template[1].astype(np.int64) + base).tolist())
