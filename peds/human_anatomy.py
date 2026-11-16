"""Non-destructive adult facial proportions shared by the body and eye baker.

The supplied scan, skin weights, garments and UVs remain the source of truth.
All LODs use the same smooth deformation; the source NPZ is never rewritten.
"""
import numpy as np

ANATOMY_VERSION = 2
EYE_RADIUS = .0135
EYE_HALF_WIDTH = .0122
EYE_HALF_HEIGHT = .0042


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0., 1.)
    return t * t * (3. - 2. * t)


def reshape_positions(points, variant):
    p = np.asarray(points, float)
    x, y, z = p.T
    head = _smooth(1.52, 1.59, z)
    front = _smooth(.015, .050, x) * head
    eye_z = 1.667 if variant == 'female' else 1.676
    mouth_z = 1.598 if variant == 'female' else 1.600
    q = p.copy()
    # Bring the sockets towards a 65 mm interpupillary distance, reducing the
    # oversized aperture without moving the temples or collapsing the bridge.
    for side in (1., -1.):
        dy, dz = y - side * .045, z - eye_z
        socket = (1. - _smooth(.018, .043, np.hypot(dy, dz))) * front
        q[:, 1] += (-side * .0095 - .24 * dy) * socket
        q[:, 2] += (.003 - .24 * dz) * socket
    q[:, 1] *= 1. - .08 * head
    # Less inflated cheeks; a more restrained nose tip and narrower lip span.
    cheek = np.exp(-((np.abs(y) - .047) / .025) ** 2 - ((z - 1.635) / .032) ** 2) * front
    q[:, 0] -= .0035 * cheek
    nose = np.exp(-(y / .020) ** 2 - ((z - 1.643) / .025) ** 2) * front
    q[:, 0] -= .007 * nose
    mouth = np.exp(-(y / .037) ** 4 - ((z - mouth_z) / .012) ** 2) * front
    q[:, 1] -= .10 * y * mouth
    q[:, 2] += .0035 * mouth
    chin = np.exp(-(y / .038) ** 2 - ((z - 1.569) / .018) ** 2) * front
    q[:, 0] += .0025 * chin
    q[:, 2] -= .003 * chin
    jaw = np.exp(-((z - 1.582) / .035) ** 2) * head
    q[:, 1] *= 1. + (.025 if variant != 'female' else -.025) * jaw
    # Lower the high hairline, keeping the crown/standing height unchanged.
    forehead = np.exp(-((z - 1.737) / .022) ** 2) * front
    q[:, 2] -= .012 * forehead
    return q


def eye_landmarks(variant):
    z = 1.667 if variant == 'female' else 1.676
    return reshape_positions(np.array([[.075, .045, z], [.075, -.045, z]]), variant)[:, 1:3]


def reshape_body(vertices, indices, variant):
    """Copy and reshape the head, transforming normals and retaining rig/UV data."""
    v = np.array(vertices, np.float32, copy=True)
    p = vertices[:, :3].astype(float)
    v[:, :3] = reshape_positions(p, variant)
    head = p[:, 2] > 1.52
    # Inverse-transpose deformation Jacobian, not the position transform, is
    # required for correct highlights on the reshaped face and accessories.
    sample = p[head]
    jac = np.empty((len(sample), 3, 3))
    for axis in range(3):
        delta = np.eye(3)[axis] * .00002
        jac[:, :, axis] = (reshape_positions(sample + delta, variant)
                          - reshape_positions(sample - delta, variant)) / .00004
    n = np.linalg.solve(jac.transpose(0, 2, 1), v[head, 3:6, None]).squeeze(-1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    v[head, 3:6] = n
    # The offset hair shells inherited a several-millimetre crack along the
    # centre UV split. Close that cosmetic part without touching skin or caps.
    hair = (v[:, 6] == 1) & (v[:, 2] > 1.69)
    y = v[hair, 1].astype(float)
    t = np.clip((np.abs(y) - .004) / .010, 0., 1.)
    scale = .2 + .8 * t * t * (3. - 2. * t)
    slope = scale + np.abs(y) * .8 * 6. * t * (1. - t) / .010
    v[hair, 1] = y * scale
    hair_n = v[hair, 3:6].astype(float)
    hair_n[:, 1] /= np.maximum(slope, .2)
    hair_n /= np.maximum(np.linalg.norm(hair_n, axis=1, keepdims=True), 1e-12)
    v[hair, 3:6] = hair_n
    # The old eyebrows were angular dark paint patches. Restore those surfaces
    # to skin; the shader supplies softly tapered, strand-textured brows.
    skin = np.isin(v[:, 6].astype(int), (0, 16, 17, 22)) & (v[:, 7] == 0)
    v[(v[:, 6] == 22) & (v[:, 7] == 0), 6] = 0
    tri = np.asarray(indices).reshape(-1, 3)
    tri = tri[skin[tri].all(1) & head[tri].all(1)]
    # Weld shading across UV and painted-material seams, not geometry. This
    # eliminates the polygonal lip/brow outlines without erasing facial relief.
    if len(tri):
        ids, remap = np.unique(tri, return_inverse=True)
        faces = remap.reshape(-1, 3)
        _, inverse = np.unique(np.round(v[ids, :3], 5), axis=0, return_inverse=True)
        inverse = inverse.ravel()
        q = v[ids, :3][faces].astype(float)
        face_n = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
        normals = np.zeros((int(inverse.max()) + 1, 3))
        for k in range(3):
            np.add.at(normals, inverse[faces[:, k]], face_n)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        valid = np.linalg.norm(normals[inverse], axis=1) > .5
        v[ids[valid], 3:6] = normals[inverse[valid]]
    return v
