"""User-supplied 2020 Camry, unbranded, with its original clear-window cabin."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .vehicle_models import VehicleTemplate, TRIM, P_TRIM, P_FIXED
from .vehicle_lamps import bake_lamp_coords
from .vehicle_reference import CAMRY

_MODEL = Path(__file__).resolve().parents[1] / 'assets/vehicles/camry_2020.npz'


@lru_cache(maxsize=1)
def _templates():
    with np.load(_MODEL, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata['version'] != 1:
            raise ValueError('Unsupported Camry model asset version')
        for key in ('length', 'width', 'height', 'wheelbase', 'front_axle', 'track',
                    'tyre_width', 'tyre_radius', 'rim_radius', 'mirror_width'):
            if abs(metadata['dimensions'][key] - getattr(CAMRY, key)) > 1e-7:
                raise ValueError('Camry hard points changed; rebuild with tools/bake_camry_model.py')
        result = tuple(VehicleTemplate(bake_lamp_coords(data[f'v{lod}'], data[f'i{lod}']),
                                       data[f'i{lod}'], len(data[f'i{lod}']) - metadata['glass'][lod],
                                       data[f'a{lod}'], metadata['glass'][lod]) for lod in range(3))
    for v, idx, _ in result:
        v.setflags(write=False)
        idx.setflags(write=False)
    for template in result:
        template.animation.setflags(write=False)
    return result


def _camry(m, lod, taxi=False):
    if lod not in (0, 1, 2):
        raise ValueError('Camry LOD must be 0, 1 or 2')
    template = _templates()[lod]
    base = len(m.V)
    m.V.extend(template[0].tolist())
    m.A.extend(template.animation.tolist())
    m.I.extend((template[1][:template[2]].astype(np.int64) + base).tolist())
    # The taxi shares exactly the same exterior/cabin; only its blank roof sign differs.
    if taxi:
        m.box(-.18, 0, 1.505, .57, .235, .130, (.98, .87, .44), P_FIXED)
        m.box(-.18, 0, 1.448, .48, .195, .035, TRIM, P_TRIM)
    m.I.extend((template[1][template[2]:].astype(np.int64) + base).tolist())
    m.glass_indices = template.glass_count
