"""FBX-derived bodies with independently fitted, four-influence clothing layers.

The archived human_models.npz is unchanged. tools/bake_human_layers.py builds
the separate body/garment assembly; the face and eye bakers fit the references.

Vertex: position3, normal3, material, accessory, joints4, weights4,
        occlusion, tiling u, tiling v, eyelid closure, eye pivot3, eye role (24 floats).
All variants share the gameplay skeleton and use real skin-weight blending.

Pedestrians use the 14 supplied citizen meshes (human_citizens.npz, built by
tools/bake_citizen_models.py) when that asset exists: they are rigged to the same skeleton and
selected by the instance seed, so each pedestrian keeps one identity.  The player keeps the
FBX-derived layered body.  Without the asset the old male/female templates are used.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from .human_anatomy import ANATOMY_VERSION

FLOATS = 24
STRIDE = FLOATS * 4
MALE, FEMALE, PLAYER = range(3)
VARIANTS = ('male', 'female', 'player')
MODEL_PATH = Path(__file__).resolve().parents[1] / 'assets/humans/human_models.npz'
EYE_PATH = MODEL_PATH.with_name('human_eyes.npz')
LAYER_PATH = MODEL_PATH.with_name('human_layers.npz')
FACE_PATH = MODEL_PATH.with_name('human_faces.npz')
NPC_ALBEDO_PATH = MODEL_PATH.with_name('human_npc_albedo.npz')
CITIZEN_PATH = MODEL_PATH.with_name('human_citizens.npz')
VERTEX_ATTRIBUTES = ((0, 3, 0), (1, 3, 3), (2, 2, 6), (3, 4, 8), (4, 4, 12), (5, 4, 16), (6, 4, 20))
M_REAL_SCLERA, M_REAL_IRIS = 32, 33
M_FACE = 34
M_CITIZEN = 35        # supplied citizen NPC surface; texture layer rides in the blink slot


@lru_cache(maxsize=1)
def _faces():
    with np.load(FACE_PATH, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        if meta['version'] != 1 or meta['vertex_floats'] != FLOATS:
            raise ValueError('Face format changed; rebuild with tools/bake_human_faces.py')
        if meta.get('anatomy_version') != ANATOMY_VERSION:
            raise ValueError('Face fit changed; rebuild faces, then eyes')
        if meta['layers_sha256'] != hashlib.sha256(LAYER_PATH.read_bytes()).hexdigest():
            raise ValueError('Garments changed; rebuild faces, then eyes')
        meshes = {(name, lod): (np.ascontiguousarray(data[f'{name}_v{lod}'], np.float32),
                                np.ascontiguousarray(data[f'{name}_i{lod}'], np.uint32))
                  for name in VARIANTS for lod in (0, 1)}
        textures = np.ascontiguousarray(data['textures'], np.uint8)
    textures.setflags(write=False)
    for v, i in meshes.values():
        v.setflags(write=False); i.setflags(write=False)
    return meshes, textures, meta


def face_textures():
    return _faces()[1]


@lru_cache(maxsize=1)
def render_face_textures():
    """Original three maps plus optional albedo-only NPC identity layers.

    Separate storage preserves the original face hash/eye fit and player maps.
    Absence falls back to procedural NPC pigment; a stale bake is rejected.
    """
    from .human_faces import CITIZEN_MALE, CITIZEN_FEMALE
    base = face_textures()
    if not NPC_ALBEDO_PATH.exists():
        return base
    with np.load(NPC_ALBEDO_PATH, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        if meta.get('version') != 1 or meta.get('face_sha256') != hashlib.sha256(FACE_PATH.read_bytes()).hexdigest():
            raise ValueError('NPC albedo fit changed; rebuild with tools/bake_citizen_albedo.py')
        expected = [('male', k) for k in CITIZEN_MALE] + [('female', k) for k in CITIZEN_FEMALE]
        if [(r['variant'], r['source_index']) for r in meta['identities']] != expected:
            raise ValueError('NPC identity order changed; rebuild with tools/bake_citizen_albedo.py')
        textures = data['textures']
        if textures.dtype != np.uint8 or textures.shape != (len(expected), *base.shape[1:]):
            raise ValueError('Invalid NPC albedo texture array')
        result = np.ascontiguousarray(np.concatenate((base, textures)), np.uint8)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def _eyes():
    with np.load(EYE_PATH, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata['version'] != 1 or metadata['vertex_floats'] != FLOATS:
            raise ValueError('Eye asset format changed; rebuild with tools/bake_human_eyes.py')
        if metadata.get('anatomy_version') != ANATOMY_VERSION:
            raise ValueError('Eye fit changed; rebuild with tools/bake_human_eyes.py')
        if metadata.get('face_sha256') != hashlib.sha256(FACE_PATH.read_bytes()).hexdigest():
            raise ValueError('Face fit changed; rebuild with tools/bake_human_eyes.py')
        meshes = {variant: (np.ascontiguousarray(data[f'{variant}_v'], np.float32),
                            np.ascontiguousarray(data[f'{variant}_i'], np.uint32)) for variant in VARIANTS}
        textures = np.ascontiguousarray(data['textures'], np.uint8)
    textures.setflags(write=False)
    for v, i in meshes.values():
        v.setflags(write=False); i.setflags(write=False)
    return meshes, textures, metadata


def eye_textures():
    return _eyes()[1]


def eye_metadata():
    return _eyes()[2]

# Actual garment surfaces, not colour bands on skin.
M_LEATHER, M_KNIT, M_LINING, M_STITCH, M_ZIP, M_SHIRT, M_NAIL, M_MASK = range(24, 32)
ACC_FORMAL, ACC_HOOD = 4096, 8192
ACC_MASK = 2097152


@lru_cache(maxsize=1)
def _asset():
    with np.load(MODEL_PATH, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata['version'] != 1 or metadata['vertex_floats'] not in (20, FLOATS):
            raise ValueError('Human asset format changed; rebuild with tools/bake_human_models.py')
        with np.load(LAYER_PATH, allow_pickle=False) as layers:
            layer_meta = json.loads(str(layers['metadata']))
            if layer_meta['version'] != 1 or layer_meta['vertex_floats'] != FLOATS:
                raise ValueError('Garment format changed; rebuild with tools/bake_human_layers.py')
            if layer_meta['source_sha256'] != hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest():
                raise ValueError('Body asset changed; rebuild with tools/bake_human_layers.py')
        meshes = {}
        for variant in VARIANTS:
            for lod in (0, 1):
                v, idx = _faces()[0][variant, lod]
                # Every LOD has the fitted globe and lids; the reference head
                # has real socket holes, unlike the earlier painted base mesh.
                ev, ei = _eyes()[0][variant]
                idx = np.concatenate((idx, ei + len(v)))
                v = np.concatenate((v, ev))
                v.setflags(write=False)
                idx.setflags(write=False)
                meshes[variant, lod] = v, idx
        detail = np.ascontiguousarray(data['material_detail'], np.uint8)
        detail.setflags(write=False)
    metadata = dict(metadata, vertex_floats=FLOATS, clothing=layer_meta, faces=_faces()[2], anatomy_version=ANATOMY_VERSION, eyes=eye_metadata(), meshes={
        f'{variant}_lod{lod}': dict(vertices=len(v), triangles=len(i) // 3)
        for (variant, lod), (v, i) in meshes.items()})
    return meshes, detail, metadata


def human_template(lod=0, variant=MALE):
    if lod not in (0, 1) or variant not in range(len(VARIANTS)):
        raise ValueError('Human template requires LOD 0/1 and a male/female/player variant')
    return _asset()[0][VARIANTS[variant], lod]


# ---------------------------------------------------------------- citizen NPC meshes
@lru_cache(maxsize=1)
def _citizens():
    """-> (meshes, textures, metadata) or None when the optional citizen asset is absent.

    meshes[k] is the (vertices, indices) pair of identity k: the 8 male citizens, then the 6
    female ones, in human_faces.CITIZEN_MALE / CITIZEN_FEMALE order.  Props (umbrella, phone,
    backpack, bag, coffee, watch) are part of every mesh and switch on by accessory bit.
    """
    if not CITIZEN_PATH.exists():
        return None
    from .human_faces import CITIZEN_MALE, CITIZEN_FEMALE
    with np.load(CITIZEN_PATH, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        if meta.get('version') != 1 or meta.get('vertex_floats') != FLOATS or meta.get('material') != M_CITIZEN:
            raise ValueError('Citizen asset format changed; rebuild with tools/bake_citizen_models.py')
        expected = [('male', k) for k in CITIZEN_MALE] + [('female', k) for k in CITIZEN_FEMALE]
        if [(r['variant'], r['source_index']) for r in meta['identities']] != expected:
            raise ValueError('Citizen identity order changed; rebuild with tools/bake_citizen_models.py')
        meshes = [(np.ascontiguousarray(data[f'v{k}'], np.float32), np.ascontiguousarray(data[f'i{k}'], np.uint32))
                  for k in range(len(expected))]
        textures = np.ascontiguousarray(data['textures'], np.uint8)
    if textures.ndim != 4 or textures.shape[0] != 2 * len(expected) or textures.shape[3] != 4:
        raise ValueError('Invalid citizen texture array')
    textures.setflags(write=False)
    for v, i in meshes:
        v.setflags(write=False)
        i.setflags(write=False)
    return meshes, textures, meta


def citizens_available():
    return _citizens() is not None


def citizen_textures():
    """RGBA8 array (2 per identity: body, clothes) or None."""
    c = _citizens()
    return None if c is None else c[1]


def citizen_metadata():
    c = _citizens()
    return None if c is None else c[2]


def runtime_face_textures():
    """Face maps for the renderer. The 14 photographic NPC albedo layers only serve the old
    Aiden-head NPCs, so they are not uploaded (about 75 MiB) while the citizen meshes are active."""
    return face_textures() if citizens_available() else render_face_textures()


def slot_count():
    """Distinct pedestrian meshes: every citizen plus the player, or the legacy 3 variants."""
    c = _citizens()
    return 3 if c is None else len(c[0]) + 1


def mesh_slots(variants, seeds):
    """Mesh slot per instance (vectorised). Citizens are picked by the stable instance seed within
    the male/female group; the player is always the last slot."""
    variants = np.clip(np.asarray(variants).astype(np.int64), 0, len(VARIANTS) - 1)
    c = _citizens()
    if c is None:
        return variants
    from .human_faces import CITIZEN_MALE, CITIZEN_FEMALE
    n_m, n_f = len(CITIZEN_MALE), len(CITIZEN_FEMALE)
    seeds = np.clip(np.asarray(seeds, np.float32), 0., 1.)
    male = np.minimum((seeds * n_m).astype(np.int64), n_m - 1)
    female = n_m + np.minimum((seeds * n_f).astype(np.int64), n_f - 1)
    return np.where(variants == PLAYER, n_m + n_f, np.where(variants == FEMALE, female, male))


def slot_template(lod, slot):
    """(vertices, indices) for a renderer mesh slot. Citizens are 1.4-2.6k triangles, already below
    the old distance mesh, so both LODs share one mesh (the identical arrays let callers share GPU buffers)."""
    c = _citizens()
    if c is None:
        return human_template(lod, slot)
    if slot == len(c[0]):
        return human_template(lod, PLAYER)
    return c[0][slot]


def material_detail():
    return _asset()[1]


def model_metadata():
    return _asset()[2]
