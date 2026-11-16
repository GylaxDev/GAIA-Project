"""Seeded geometric identities, shared by the visible and shadow shaders.

The eye/socket region moves rigidly: varying the face must not stretch the
imported iris or stop the lids meeting. Shapes are stable across time and LODs.
"""
import numpy as np
from .human_anatomy import _smooth, eye_landmarks

FACE_VERSION = 4
# Restrained variation around the detailed reference fit, not caricature morphs.
# The previous 1.8 gain allowed almost 20% jaw-width and 14% mouth-width changes.
# At .25: lateral deformation stays under 4% at every vertex across 257 seeds;
# identity is carried primarily by the supplied albedo masks.
NPC_FACE_VARIATION = .25
# The detailed reference head carries a taller crown/hair mass than the supplied
# portrait pack. Above the brow, ease only NPCs toward the pack's measured
# eye-to-hair-top span. The player and detailed topology remain untouched.
NPC_FOREHEAD_BROW = .029
NPC_FOREHEAD_SCALE = .60
CITIZEN_MALE = (0, 2, 3, 5, 7, 9, 10, 11)
CITIZEN_FEMALE = (1, 4, 6, 8, 12, 13)


def npc_identity_layer(seed, variant='male'):
    """Match the shader's stable normalized-seed albedo choice; never Aiden."""
    if variant == 'player':
        return 0
    ids = CITIZEN_FEMALE if variant == 'female' else CITIZEN_MALE
    offset = len(CITIZEN_MALE) if variant == 'female' else 0
    return 3 + offset + min(int(np.clip(np.float32(seed), 0, 1) * len(ids)), len(ids) - 1)


def identity_positions(points, seed, variant='male'):
    p = np.asarray(points, float)
    x, y, z = p.T
    c = np.sin(float(seed)*np.array([37., 61., 43., 79., 53., 97., 67., 89.])
               + [1., 2., 4., 3., 5., 7., 6., 8.])
    # Identity comes primarily from the supplied faces, with tightly bounded
    # geometric variation. Never perturb the player's authored likeness.
    c *= 0. if variant == 'player' else NPC_FACE_VARIATION
    player = 0.
    head = _smooth(1.53, 1.59, z)
    front = _smooth(.012, .045, x)
    eye_z = eye_landmarks('female' if variant == 'female' else 'male')[0, 1]
    jaw = np.exp(-((z-1.583)/.027)**2)
    mouth = np.exp(-((z-1.602)/.012)**2) * front
    d = np.zeros_like(p)
    d[:, 1] = y * (.045*c[0] + (.065*c[1]+.065*player)*jaw - .08*c[3]*mouth) * head
    nose = np.exp(-(y/.017)**2 - ((z-1.643)/.024)**2)*front
    chin = np.exp(-(y/.035)**2 - ((z-1.567)/.018)**2)*front
    cheek = np.exp(-((np.abs(y)-.052)/.021)**2 - ((z-1.630)/.024)**2)*front
    d[:, 0] = head*((.003*c[2]+.001*player)*nose + (.003*c[4]+.002*player)*chin + .002*c[5]*cheek)
    # The entire aperture, including outer lids, shares one rigid offset.
    r = np.hypot(np.abs(y)-.03266, z-eye_z)
    socket = (1-_smooth(.024, .040, r))*front*head
    side = 2*_smooth(-.012, .012, y)-1
    move = np.column_stack((np.full(len(p), .0007*c[5]), side*.0022*c[6], np.full(len(p), .0014*c[7])))
    q = p + d*(1-socket[:, None]) + move*socket[:, None]
    if variant != 'player':
        brow = eye_z + NPC_FOREHEAD_BROW
        q[:, 2] -= (1-NPC_FOREHEAD_SCALE)*np.maximum(q[:, 2]-brow, 0.)
    return q


def identity_normals(points, normals, seed, variant='male'):
    p = np.asarray(points, float)
    jac = np.empty((len(p), 3, 3))
    for k in range(3):
        delta = np.eye(3)[k]*.0001
        jac[:, :, k] = (identity_positions(p+delta, seed, variant)-identity_positions(p-delta, seed, variant))/.0002
    n = np.linalg.solve(jac.transpose(0, 2, 1), np.asarray(normals, float)[..., None]).squeeze(-1)
    return n/np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


FACE_GLSL = """
vec4 gFaceA, gFaceB;
float faceSquare(float x) { return x*x; }
void faceIdentity() {
    gFaceA = sin(iBody.w * vec4(37.0,61.0,43.0,79.0) + vec4(1.0,2.0,4.0,3.0));
    gFaceB = sin(iBody.w * vec4(53.0,97.0,67.0,89.0) + vec4(5.0,7.0,6.0,8.0));
    float scale = iDyn.y > 1.5 ? 0.0 : __NPC_VARIATION__;
    gFaceA *= scale; gFaceB *= scale;
}
vec3 identityPosition(vec3 p) {
    float player = 0.0;
    float head = sstep(1.53,1.59,p.z), front = sstep(0.012,0.045,p.x);
    float ez = iDyn.y > 0.5 && iDyn.y < 1.5 ? __FEMALE_EYE__ : __MALE_EYE__;
    float jaw = exp(-faceSquare((p.z-1.583)/0.027));
    float mouth = exp(-faceSquare((p.z-1.602)/0.012))*front;
    vec3 d = vec3(0.0);
    d.y = p.y * (0.045*gFaceA.x + (0.065*gFaceA.y+0.065*player)*jaw - 0.08*gFaceA.w*mouth)*head;
    float nose = exp(-faceSquare(p.y/0.017)-faceSquare((p.z-1.643)/0.024))*front;
    float chin = exp(-faceSquare(p.y/0.035)-faceSquare((p.z-1.567)/0.018))*front;
    float cheek = exp(-faceSquare((abs(p.y)-0.052)/0.021)-faceSquare((p.z-1.630)/0.024))*front;
    d.x = head*((0.003*gFaceA.z+0.001*player)*nose+(0.003*gFaceB.x+0.002*player)*chin+0.002*gFaceB.y*cheek);
    float r = length(vec2(abs(p.y)-0.03266,p.z-ez));
    float socket = (1.0-sstep(0.024,0.040,r))*front*head;
    float side = 2.0*sstep(-0.012,0.012,p.y)-1.0;
    vec3 move = vec3(0.0007*gFaceB.y,side*0.0022*gFaceB.z,0.0014*gFaceB.w);
    vec3 q = p + d*(1.0-socket) + move*socket;
    if (iDyn.y < 1.5) {
        float brow = ez + __NPC_BROW__;
        q.z -= (1.0-__NPC_FOREHEAD_SCALE__) * max(q.z-brow, 0.0);
    }
    return q;
}
vec3 identityNormal(vec3 p, vec3 n) {
    const float e = 0.0001;
    mat3 J = mat3((identityPosition(p+vec3(e,0,0))-identityPosition(p-vec3(e,0,0)))/(2.0*e),
                  (identityPosition(p+vec3(0,e,0))-identityPosition(p-vec3(0,e,0)))/(2.0*e),
                  (identityPosition(p+vec3(0,0,e))-identityPosition(p-vec3(0,0,e)))/(2.0*e));
    return normalize(transpose(inverse(J))*n);
}
""".replace('__NPC_VARIATION__', f'{NPC_FACE_VARIATION:.2f}').replace(
    '__NPC_BROW__', f'{NPC_FOREHEAD_BROW:.3f}').replace(
    '__NPC_FOREHEAD_SCALE__', f'{NPC_FOREHEAD_SCALE:.2f}').replace(
    '__FEMALE_EYE__', f'{eye_landmarks("female")[0, 1]:.7f}').replace(
    '__MALE_EYE__', f'{eye_landmarks("male")[0, 1]:.7f}')
