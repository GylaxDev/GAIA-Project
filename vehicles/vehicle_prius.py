"""2012 Toyota Prius, adapted from the user's OBJ (original model: TAI0504).

The baked exterior follows the supplied panels, triangular glazing, split hatch,
headlamps and alloy wheels. See tools/bake_prius_model.py for reproducible asset
conversion, bounded near-mesh refinement and aperture-preserving distance LODs.
Manufacturer/plate/wheel-centre decals and hidden interior trim are not imported.
All three meshes retain GAIA's ten-float material layout and wheel metadata.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .vehicle_models import VehicleTemplate
from .vehicle_lamps import bake_lamp_coords
from .vehicle_reference import PRIUS

_MODEL = Path(__file__).resolve().parents[1] / 'assets/vehicles/prius_2012.npz'


@lru_cache(maxsize=1)
def _templates():
    with np.load(_MODEL, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata['version'] != 1:
            raise ValueError('Unsupported Prius model asset version')
        for key in ('length', 'width', 'height', 'wheelbase', 'front_axle', 'track',
                    'tyre_width', 'tyre_radius', 'rim_radius', 'mirror_width'):
            if abs(metadata['dimensions'][key] - getattr(PRIUS, key)) > 1e-7:
                raise ValueError('Prius hard points changed; rebuild with tools/bake_prius_model.py')
        result = tuple(VehicleTemplate(bake_lamp_coords(data[f'v{lod}'], data[f'i{lod}']),
                                       data[f'i{lod}'], len(data[f'i{lod}']),
                                       data[f'a{lod}']) for lod in range(3))
    for v, idx, _ in result:
        v.setflags(write=False)
        idx.setflags(write=False)
    for template in result:
        template.animation.setflags(write=False)
    return result


def _prius(m, lod):
    if lod not in (0, 1, 2):
        raise ValueError('Prius LOD must be 0, 1 or 2')
    template = _templates()[lod]
    base = len(m.V)
    m.V.extend(template[0].tolist())
    m.A.extend(template.animation.tolist())
    m.I.extend((template[1].astype(np.int64) + base).tolist())
