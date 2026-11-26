# --- shaders/pedestrian.py ---
"""Instanced humans with seeded facial geometry and fitted clothing layers.

The gameplay skeleton is reconstructed in the vertex shader as fifteen joint
matrices.  Vertices carry four joint indices and weights, so shoulders, knees,
neck and coat tails bend smoothly instead of moving as rigid blocks.  The
authored gait remains: two-bone leg IK with heel-to-toe planting, pelvis sway
and counter-rotation, arm swings and the umbrella / phone / coffee / pocket
poses.  Real garment geometry supplies seams, lapels, boots and hair; material
microdetail (pores, leather grain, twill, knit ribs, hair) lives in a small
mipmapped texture array shared by every character.

Vertex (24 floats):
    0  position3    3  normal3        6  material, accessory
    8  joints4      12 weights4       16 ao, u, v, lid closure
   20  eye pivot3, role (0 body, -1 garment, -2 hair/prop, 1 eye, 2..3 lid)
Citizen NPC vertices (material M_CITIZEN) carry their uv in 17..18 and the layer of the
citizen texture array in slot 19 (even = body/skin, odd = clothes); they skip the shared-head
face morph and the animated-eye logic, which belong to the player's detailed head.

Instance row (28 floats, see walker.PED_STRIDE):
    0  x y z heading               8  height, width, head scale, seed
    4  phase, half stride, speed   12 skin, hair, top, bottom palette index
       factor, flags               16 shoes, accent, sleeve style, trouser style
                                    20 bulk, visibility, gaze yaw, head nod
                                    24 turn lean, variant, -, -
"""
from ..peds.pedestrian_models import (shader_constants, M_SKIN, M_HAIR, M_TOP, M_BOTTOM, M_SHOE, M_FIXED, M_EYE,
                                 M_CANOPY, M_METAL, M_PHONE, M_SCREEN, M_GLASS, M_LIMB, M_SHIN, M_CAP, M_DARK,
                                 M_LIP, M_SCLERA, M_IRIS, M_GLOVE, M_CUP, M_SOLE, M_BROW, M_BEARD,
                                 ACC_COAT, ACC_SKIRT, ACC_IDLE, ACC_PLAYER, ACC_BEARD, SHOULDER_Y, SHOULDER_Z,
                                 ELBOW_Z, HIP_Y, HIP_Z, KNEE_Z, ANKLE_Z)
from ..peds.human_assets import (M_LEATHER, M_KNIT, M_LINING, M_STITCH, M_ZIP, M_SHIRT, M_NAIL, M_MASK,
                            M_REAL_SCLERA, M_REAL_IRIS, M_FACE, M_CITIZEN, ACC_FORMAL, ACC_HOOD, ACC_MASK)
from ..peds.pedestrian_gait import S_CAP, LMAX
from ..peds.eye_animation import EYE_ANIM_GLSL
from ..peds.human_faces import FACE_GLSL, CITIZEN_MALE, CITIZEN_FEMALE
from ..peds.human_anatomy import EYE_RADIUS, EYE_HALF_WIDTH, EYE_HALF_HEIGHT, eye_landmarks, reshape_positions
from .hdr import HDR_CONV, HDR_OUT
from .streetspill import STREET_SPILL
from .materials import MATERIAL_LIGHT
from .shadow import SHADOW_PCF

PED_INPUTS = """
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aMatAcc;              // material id, accessory bits
layout(location=3) in vec4 aJoints;
layout(location=4) in vec4 aWeights;
layout(location=5) in vec4 aSurface;              // ao, tiling u, tiling v, blink weight
layout(location=6) in vec4 aEyeRig;               // local eye pivot3, eye/lid role

layout(location=8)  in vec4 iPosRot;
layout(location=9)  in vec4 iAnim;
layout(location=10) in vec4 iBody;
layout(location=11) in vec4 iLook0;
layout(location=12) in vec4 iLook1;
layout(location=13) in vec4 iExtra;
layout(location=14) in vec4 iDyn;
uniform float uTime;
"""

PED_ANIM = shader_constants() + f"""
const float PI = 3.14159265;
const float S_CAP = {S_CAP:.5f};
const float L_LEGMAX = {LMAX:.5f};
const float EYE_RADIUS = {EYE_RADIUS:.6f};
const int V_LIMB = {M_LIMB}, V_SHIN = {M_SHIN}, V_HAIR = {M_HAIR};
const float CITIZEN_MATERIAL = {M_CITIZEN}.0;
const float SH_Y = {SHOULDER_Y:.4f}, SH_Z = {SHOULDER_Z:.4f}, EL_Z = {ELBOW_Z:.4f};
const float HIP_Y = {HIP_Y:.4f}, HIP_Z = {HIP_Z:.4f}, KNEE = {KNEE_Z:.4f}, ANKLE = {ANKLE_Z:.4f};
const int R_BODY = 0, R_TORSO = 1, R_UARM_L = 2, R_UARM_R = 3, R_FARM_L = 4, R_FARM_R = 5,
          R_THIGH_L = 6, R_THIGH_R = 7, R_SHIN_L = 8, R_SHIN_R = 9, R_COAT_L = 10, R_COAT_R = 11,
          R_HEAD = 12, R_FOOT_L = 13, R_FOOT_R = 14;
""" + """
float sstep(float a, float b, float x) { float t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
""" + EYE_ANIM_GLSL + FACE_GLSL + """
// ---- foot trajectory: stance line, heel -> toe roll, lifted swing --------------------------
float dutyOf(float sf) { return 0.62 - 0.37 * pow(sf, 0.8); }

vec4 footTarget(float u, float side, float S, float sf, float idleShift) {
    float on = sstep(0.05, 0.25, S);
    float d = dutyOf(sf);
    float front = 0.85 * S, rear = 1.15 * S;
    float lift = (0.055 + 0.16 * sf) * on;
    float x, z, psi;
    if (u < d) {
        float t = u / d;
        x = front + (-rear - front) * t;
        z = 0.0;
        psi = (0.28 * (1.0 - sstep(0.0, 0.22, t)) - 0.75 * sstep(0.55, 1.0, t)) * on;
    } else {
        float t = (u - d) / (1.0 - d);
        x = -rear + (front + rear) * sstep(0.0, 1.0, t);
        z = lift * sin(PI * pow(t, 0.85));
        psi = (-0.75 + 1.05 * sstep(0.0, 0.85, t)) * on;
    }
    float unload = clamp(-side * idleShift, 0.0, 1.0) * (1.0 - on);
    x += 0.055 * unload;
    float rollH = ANKLE_H * cos(psi) + (psi > 0.0 ? 0.07 * sin(psi) : 0.17 * sin(-psi));
    return vec4(x, side * (HIP_Y0 + 0.012 + 0.012 * unload), rollH + z, psi);
}

// two-bone IK: (thigh pitch, knee flexion, hip roll, foot pitch relative to the shin)
vec4 solveLeg(vec3 hip, vec4 foot) {
    float ax = foot.x - hip.x, ay = foot.y - hip.y, az = hip.z - foot.z;
    float rl = atan(ay, az);
    float dy = sqrt(az * az + ay * ay);
    float d = sqrt(ax * ax + dy * dy);
    float dc = clamp(d, 0.12, L_LEGMAX);
    float cg = clamp((L_THIGH * L_THIGH + dc * dc - L_SHIN * L_SHIN) / (2.0 * L_THIGH * dc), -1.0, 1.0);
    float ck = clamp((L_THIGH * L_THIGH + L_SHIN * L_SHIN - dc * dc) / (2.0 * L_THIGH * L_SHIN), -1.0, 1.0);
    float th = atan(ax, dy) + acos(cg);
    float beta = PI - acos(ck);
    return vec4(th, beta, rl, foot.w - (th - beta));
}

// ---- per-instance pose state (computed once per vertex invocation) ------------------------
float gOn, gIdle, gBreathe, gSway, gYaw, gRoll, gZ, gLean, gTwist, gSpineRoll, gHeadYaw, gHeadNod;
float gTailL, gTailR;
vec3 gHipL, gHipR;
vec4 gLegL, gLegR, gPoseL, gPoseR;

void solveGait() {
    float phase = iAnim.x, S = iAnim.y, sf = iAnim.z, seed = iBody.w;
    int flags = int(iAnim.w + 0.5);
    gOn = sstep(0.05, 0.25, S);
    gIdle = ((flags & 16384) != 0) ? 1.0 : 0.0;
    gBreathe = sin(uTime * 1.9 + seed * 40.0);
    float idleSway = 0.03 * sin(uTime * 0.8 + seed * 20.0) * gIdle;
    float k = min(S / S_CAP, 1.0);
    float u0 = fract(phase / (2.0 * PI));
    float shift = cos(2.0 * PI * (u0 - 0.31));
    gSway = 0.028 * k * shift + idleSway * (1.0 - gOn);
    gYaw = -0.075 * k * cos(2.0 * PI * u0);
    gRoll = 0.030 * k * shift;
    float idleShift = idleSway / 0.03;
    vec4 fl = footTarget(u0, 1.0, S, sf, idleShift);
    vec4 fr = footTarget(fract(u0 + 0.5), -1.0, S, sf, idleShift);
    float dyL = fl.y - (HIP_Y0 + gSway), dyR = fr.y - (-HIP_Y0 + gSway);
    float reachL = fl.z + sqrt(max(L_LEGMAX * L_LEGMAX - fl.x * fl.x - dyL * dyL, 0.01));
    float reachR = fr.z + sqrt(max(L_LEGMAX * L_LEGMAX - fr.x * fr.x - dyR * dyR, 0.01));
    gZ = max(min(HIP_Z0 - 0.008, min(reachL, reachR)), HIP_Z0 - 0.16);
    float cy = cos(gYaw), sy = sin(gYaw), cr = cos(gRoll), sr = sin(gRoll);
    gHipL = vec3(-HIP_Y0 * sy, gSway + HIP_Y0 * cy * cr, gZ + HIP_Y0 * cy * sr);
    gHipR = vec3( HIP_Y0 * sy, gSway - HIP_Y0 * cy * cr, gZ - HIP_Y0 * cy * sr);
    gLegL = solveLeg(gHipL, fl);
    gLegR = solveLeg(gHipR, fr);

    // arms swing opposite the leg on their side; further back than forward
    float gain = (0.30 + 0.42 * sf) * gOn, sn = max(S, 0.05);
    float aL = -fl.x / sn * gain, aR = -fr.x / sn * gain;
    aL *= (aL > 0.0 ? 0.65 : 1.15);
    aR *= (aR > 0.0 ? 0.65 : 1.15);
    // Elbows are relaxed in a walk and fold up with speed (jog/run keep the old ~90 degrees). They also work
    // with the swing: nearly straight as the arm goes back, bending as it comes forward. A constant bend with the
    // forearms held out in front reads as a zombie. The forearms angle slightly toward the midline (pose.w).
    float eBase = 0.24 + (-0.09 + 1.20 * pow(max(sf, 0.0), 1.3)) * gOn + gIdle * 0.04;
    float eL = eBase + 0.35 * max(aL, 0.0), eR = eBase + 0.35 * max(aR, 0.0);
    float wL = (0.08 + 0.30 * max(aL, 0.0)) * gOn, wR = (0.08 + 0.30 * max(aR, 0.0)) * gOn;
    gPoseL = vec4(aL, SWING_ROLL, eL, wL);
    gPoseR = vec4(aR, SWING_ROLL, eR, wR);
    if ((flags & 1) != 0)       gPoseR = POSE_UMBRELLA;
    if ((flags & 2) != 0)       gPoseL = POSE_EAR;
    if ((flags & 4) != 0)       { gPoseL = POSE_LOOK; gPoseR = POSE_LOOK; }
    if ((flags & 65536) != 0)   { gPoseL = POSE_POCKET; gPoseR = POSE_POCKET; }
    if ((flags & 131072) != 0)  { gPoseL = POSE_CROSS_A; gPoseR = POSE_CROSS_B; }
    if ((flags & 262144) != 0)  gPoseR = POSE_COFFEE;
    if ((flags & 524288) != 0)  gPoseL = POSE_WATCH;
    if ((flags & 1048576) != 0) {
        gPoseR = vec4(0.55 + 0.20 * sin(uTime * 2.7 + seed * 9.0), 0.12, 1.30 + 0.35 * sin(uTime * 3.9 + seed * 4.0),
                      0.25 + 0.20 * sin(uTime * 2.1 + seed * 2.0));
        if ((flags & (2 | 65536 | 131072 | 524288)) == 0)
            gPoseL = vec4(0.28 + 0.08 * sin(uTime * 2.2 + 1.0), 0.08, 0.95 + 0.15 * sin(uTime * 3.1), 0.10);
    }
    gTwist = -1.6 * gYaw;
    gLean = 0.02 + 0.17 * sf * gOn + gIdle * 0.012 * gBreathe;
    gSpineRoll = -0.8 * gRoll + iDyn.x;
    gHeadYaw = iExtra.z - 0.7 * (gYaw + gTwist);
    gHeadNod = iExtra.w + (((flags & 4) != 0) ? 0.55 : 0.0) + gIdle * 0.03 * gBreathe - 0.5 * gLean;

    // coat tails / skirt follow the thigh that is under them
    bool skirt = (flags & 32) != 0;
    float trail = -(0.06 + 0.50 * sf * gOn) * (skirt ? 0.35 : 1.0);
    float followL = (skirt ? 0.30 : 0.45) * gLegL.x * gOn;
    float followR = (skirt ? 0.30 : 0.45) * gLegR.x * gOn;
    float flutter = 0.05 * sf * (skirt ? 0.5 : 1.0);
    gTailL = trail + followL + flutter * sin(uTime * 8.0 + seed * 30.0) + gIdle * 0.01 * gBreathe;
    gTailR = trail + followR + flutter * sin(uTime * 8.0 + seed * 30.0 + 1.7) + gIdle * 0.01 * gBreathe;
}

bool accessoryHidden() {
    int acc = int(aMatAcc.y + 0.5);
    int flags = int(iAnim.w + 0.5), mat = int(aMatAcc.x + 0.5);
    // Sleeve/trouser *meshes* disappear for short styles; they never become skin.
    bool outer = (flags & (16 | 4096 | 8192 | 32768)) != 0;
    if (mat == V_LIMB && iLook1.z < 0.5 && !outer) return true;
    if (mat == V_SHIN && iLook1.w > 0.5) return true;
    // Do not stack hair and cap shells through one another.
    if (mat == V_HAIR && (flags & 512) != 0 && acc != 512) return true;
    return acc != 0 && (int(iAnim.w + 0.5) & acc) == 0;
}

// ---- joint matrices -----------------------------------------------------------------------
mat4 translateM(vec3 t) { mat4 m = mat4(1.0); m[3] = vec4(t, 1.0); return m; }
mat4 pitchM(float a) { float c = cos(a), s = sin(a); return mat4(c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1); }
mat4 rollM(float a)  { float c = cos(a), s = sin(a); return mat4(1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1); }
mat4 yawM(float a)   { float c = cos(a), s = sin(a); return mat4(c,s,0,0, -s,c,0,0, 0,0,1,0, 0,0,0,1); }
mat4 around(vec3 pivot, mat4 r) { return translateM(pivot) * r * translateM(-pivot); }
mat4 scaleAround(vec3 pivot, float s) {
    return translateM(pivot) * mat4(s,0,0,0, 0,s,0,0, 0,0,s,0, 0,0,0,1) * translateM(-pivot);
}

mat4 gPelvisM, gSpineM;

void buildBaseMatrices() {
    vec3 zp = vec3(0.0, 0.0, HIP_Z0);
    gPelvisM = translateM(vec3(0.0, gSway, gZ - HIP_Z0))
             * around(zp, rollM(gRoll)) * around(zp, yawM(gYaw));
    mat4 spineLocal = around(SPINE, pitchM(-gLean))
                    * around(SPINE, rollM(gSpineRoll)) * around(SPINE, yawM(gTwist));
    gSpineM = gPelvisM * spineLocal;
}

mat4 boneMatrix(int jo) {
    if (jo == R_BODY) return gPelvisM;
    if (jo == R_TORSO) return gSpineM;
    if (jo == R_HEAD) {
        mat4 local = around(NECK, yawM(gHeadYaw)) * around(NECK, pitchM(-gHeadNod))
                   * scaleAround(NECK, iBody.z);
        return gSpineM * local;
    }
    if (jo >= R_UARM_L && jo <= R_FARM_R) {
        bool left = (jo == R_UARM_L || jo == R_FARM_L);
        float side = left ? 1.0 : -1.0;
        vec4 pose = left ? gPoseL : gPoseR;
        vec3 sh = vec3(0.0, side * SH_Y, SH_Z), el = vec3(0.0, side * SH_Y, EL_Z);
        mat4 upper = around(sh, rollM(pose.y * side)) * around(sh, pitchM(pose.x));
        if (jo == R_UARM_L || jo == R_UARM_R) return gSpineM * upper;
        mat4 fore = around(el, yawM(-pose.w * side)) * around(el, pitchM(pose.z));
        return gSpineM * upper * fore;
    }
    if (jo >= R_THIGH_L && jo <= R_FOOT_R) {
        bool left = (jo == R_THIGH_L || jo == R_SHIN_L || jo == R_FOOT_L);
        float side = left ? 1.0 : -1.0;
        vec4 leg = left ? gLegL : gLegR;
        vec3 hip = vec3(0.0, side * HIP_Y, HIP_Z);
        vec3 shift = (left ? gHipL : gHipR) - hip;
        mat4 thigh = translateM(shift) * around(hip, rollM(leg.z)) * around(hip, pitchM(leg.x));
        if (jo == R_THIGH_L || jo == R_THIGH_R) return thigh;
        mat4 knee = around(vec3(0.0, side * HIP_Y, KNEE), pitchM(-leg.y));
        if (jo == R_SHIN_L || jo == R_SHIN_R) return thigh * knee;
        return thigh * knee * around(vec3(0.0, side * HIP_Y, ANKLE), pitchM(leg.w));
    }
    if (jo == R_COAT_L || jo == R_COAT_R)
        return gPelvisM * around(WAIST, pitchM(jo == R_COAT_L ? gTailL : gTailR));
    return mat4(1.0);
}

float bellAt(vec3 a, vec3 b, float w) { vec3 d = (a - b) / w; return exp(-dot(d, d)); }
vec2 gEyeOffset;

void poseAndSkin(out vec3 p, out vec3 n, out float crease) {
    solveGait();
    buildBaseMatrices();
    vec3 rest = aPos, restNormal = aNormal;
    if (aEyeRig.w > 0.5 && aEyeRig.w < 1.5) {
        vec2 gaze = gazeAngles(uTime, iBody.w, iExtra.z, (int(iAnim.w + 0.5) & 4) != 0);
        // Both eyes share a fixation; near phone viewing adds slight convergence.
        if ((int(iAnim.w + 0.5) & 4) != 0) gaze.x -= sign(aEyeRig.y) * 0.025;
        mat3 rot = mat3(yawM(gaze.x) * pitchM(gaze.y));
        rest = aEyeRig.xyz + rot * (rest - aEyeRig.xyz);
        restNormal = rot * restNormal;
    } else if (aEyeRig.w >= 2.0) {
        float blink = blinkClosure(uTime, iBody.w);
        float follow = 3.0 - aEyeRig.w;
        vec2 oldOffset = aPos.yz - aEyeRig.yz;
        rest.z -= aSurface.w * blink;
        vec2 offset = rest.yz - aEyeRig.yz;
        float oldFront = sqrt(max(EYE_RADIUS * EYE_RADIUS - dot(oldOffset, oldOffset), 0.0));
        float front = sqrt(max(EYE_RADIUS * EYE_RADIUS - dot(offset, offset), 0.0));
        rest.x += (front - oldFront) * 0.72 * follow * follow;
        restNormal = normalize(mix(aNormal, normalize(vec3(max(front, 0.002) / 0.72, offset)), blink * follow));
    }
    gEyeOffset = rest.yz - aEyeRig.yz;
    if (aPos.z > 1.53 && aMatAcc.x < CITIZEN_MATERIAL - 0.5) {      // citizens carry their own heads
        faceIdentity();
        if (aEyeRig.w > 0.5 && aEyeRig.w < 1.5) {
            // Rigid globe motion preserves the authored corneal/iris dimensions.
            rest += identityPosition(aEyeRig.xyz) - aEyeRig.xyz;
        } else {
            restNormal = identityNormal(rest, restNormal);
            rest = identityPosition(rest);
        }
    }
    vec3 sp = vec3(0.0), sn = vec3(0.0);
    float total = 0.0;
    for (int k = 0; k < 4; ++k) {
        float w = aWeights[k];
        if (w <= 0.0001) continue;
        mat4 M = boneMatrix(int(aJoints[k] + 0.5));
        sp += w * (M * vec4(rest, 1.0)).xyz;
        sn += w * (mat3(M) * restNormal);
        total += w;
    }
    if (total < 0.0001) {
        sp = (gPelvisM * vec4(rest, 1.0)).xyz;
        sn = mat3(gPelvisM) * restNormal;
    }
    p = sp;
    n = normalize(sn);

    vec3 shL = vec3(0.0, SH_Y, SH_Z), shR = vec3(0.0, -SH_Y, SH_Z);
    vec3 hipL = vec3(0.0, HIP_Y, HIP_Z), hipR = vec3(0.0, -HIP_Y, HIP_Z);
    vec3 kneeL = vec3(0.0, HIP_Y, KNEE), kneeR = vec3(0.0, -HIP_Y, KNEE);
    float c = (abs(gLean) * 3.0 + abs(gTwist) * 2.0) * bellAt(rest, vec3(0.0, 0.0, 1.02), 0.12);
    c += 0.35 * (abs(gPoseL.x) + abs(gPoseR.x))
       * (bellAt(rest, shL, 0.09) + bellAt(rest, shR, 0.09));
    c += clamp(abs(gLegL.x) * 1.4, 0.0, 1.0) * bellAt(rest, hipL, 0.10)
       + clamp(abs(gLegR.x) * 1.4, 0.0, 1.0) * bellAt(rest, hipR, 0.10);
    c += clamp(gLegL.y / 1.5, 0.0, 1.0) * bellAt(rest, kneeL, 0.09)
       + clamp(gLegR.y / 1.5, 0.0, 1.0) * bellAt(rest, kneeR, 0.09);
    c += 0.5 * clamp(abs(gTailL), 0.0, 1.0) * bellAt(rest, vec3(0.0, 0.09, 1.0), 0.14)
       + 0.5 * clamp(abs(gTailR), 0.0, 1.0) * bellAt(rest, vec3(0.0, -0.09, 1.0), 0.14);
    crease = clamp(c, 0.0, 1.0);
}

vec3 pedWorld(vec3 p) {
    float h = iBody.x / MODEL_HEIGHT;
    float width = iBody.y * mix(1.0, iExtra.x, 0.35);
    vec3 q = vec3(p.xy * (h * width), p.z * h);
    float c = cos(iPosRot.w), s = sin(iPosRot.w);
    return iPosRot.xyz + vec3(c * q.x - s * q.y, s * q.x + c * q.y, q.z);
}
vec3 pedNormal(vec3 n) {
    float c = cos(iPosRot.w), s = sin(iPosRot.w);
    return vec3(c * n.x - s * n.y, s * n.x + c * n.y, n.z);
}
"""

PED_DISSOLVE = """
void pedDissolve(float vis) {
    if (vis >= 1.0) return;
    float d = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    if (d > vis) discard;
}
"""

def _face_info(variant):
    eye = eye_landmarks(variant)[0]
    mouth = reshape_positions([[.085, 0., 1.598 if variant == 'female' else 1.600]], variant)[0, 2]
    if variant == 'female':
        # The fitted detailed head shifts the entire female face down by the
        # eye-height difference. The old body landmark put lip tint 7 mm high.
        mouth = 1.6035 + eye[1] - eye_landmarks('male')[0, 1]
    return f'vec3({eye[1]:.6f}, {mouth:.6f}, {eye[0]:.6f})'


FACE_INFO = f'const vec3 MALE_FACE = {_face_info("male")}, FEMALE_FACE = {_face_info("female")};\n'

PED_VS = ("#version 330 core\n" + PED_INPUTS + PED_ANIM + FACE_INFO + """
uniform mat4 uView, uProj;
uniform vec3 uCamPos;
out vec3 vWorld;
out vec3 vShadowRelative;
out vec3 vNormal;
out vec3 vRest;
out vec2 vUV;
out vec2 vEyeOffset;
out float vLocalZ;
out float vCrease;
out float vAO;
flat out float vMat;
flat out vec4 vLook0;
flat out vec4 vLook1;
flat out vec3 vBodyInfo;
flat out int vFlags;
flat out float vVis;
flat out float vBlink;
flat out vec3 vFaceInfo;
flat out float vFaceRole;
flat out float vLayer;
void main() {
    if (accessoryHidden() || iExtra.y <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    vec3 p, n;
    float crease;
    poseAndSkin(p, n, crease);
    vec3 wp = pedWorld(p);
    vWorld = wp;
    vShadowRelative = wp - uCamPos;
    vNormal = pedNormal(n);
    vRest = aPos;
    vUV = aSurface.yz;
    vEyeOffset = gEyeOffset;
    vLocalZ = p.z;
    vCrease = crease;
    vAO = aSurface.x;
    vMat = aMatAcc.x;
    vLook0 = iLook0; vLook1 = iLook1;
    vBodyInfo = vec3(iBody.w, iBody.x, iDyn.y);
    vFlags = int(iAnim.w + 0.5);
    vVis = clamp(iExtra.y, 0.0, 1.0);
    vBlink = aEyeRig.w > 0.5 && aEyeRig.w < 1.5 ? blinkClosure(uTime, iBody.w) : 0.0;
    vFaceInfo = iDyn.y > 0.5 && iDyn.y < 1.5 ? FEMALE_FACE : MALE_FACE;
    vFaceRole = aEyeRig.w;
    vLayer = aSurface.w;
    gl_Position = uProj * uView * vec4(wp, 1.0);
}
""")

PED_SHADOW_VS = ("#version 330 core\n" + PED_INPUTS + PED_ANIM + """
uniform mat4 uLightSpaceMatrix;
flat out float vVis;
void main() {
    if (accessoryHidden() || iExtra.y <= 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    vec3 p, n;
    float crease;
    poseAndSkin(p, n, crease);
    vVis = clamp(iExtra.y, 0.0, 1.0);
    gl_Position = uLightSpaceMatrix * vec4(pedWorld(p), 1.0);
}
""")

PED_SHADOW_FS = ("#version 330 core\nflat in float vVis;\n" + PED_DISSOLVE + """
void main() { pedDissolve(vVis); }
""")

# sRGB display colours; converted to linear reflectance in the fragment shader.
PED_PALETTES = """
const vec3 SKIN_TONES[8] = vec3[8](
    vec3(0.91, 0.77, 0.70), vec3(0.87, 0.70, 0.61), vec3(0.79, 0.60, 0.49), vec3(0.70, 0.49, 0.38),
    vec3(0.58, 0.38, 0.29), vec3(0.47, 0.30, 0.23), vec3(0.36, 0.23, 0.18), vec3(0.27, 0.17, 0.14));
const vec3 HAIR_TONES[8] = vec3[8](
    vec3(0.06, 0.05, 0.05), vec3(0.15, 0.09, 0.06), vec3(0.30, 0.18, 0.09), vec3(0.42, 0.19, 0.10),
    vec3(0.78, 0.64, 0.38), vec3(0.62, 0.62, 0.63), vec3(0.86, 0.86, 0.85), vec3(0.11, 0.08, 0.06));
const vec3 TOP_TONES[16] = vec3[16](
    vec3(0.88, 0.88, 0.86), vec3(0.05, 0.05, 0.06), vec3(0.18, 0.18, 0.20), vec3(0.08, 0.12, 0.28),
    vec3(0.62, 0.10, 0.10), vec3(0.28, 0.32, 0.18), vec3(0.72, 0.55, 0.12), vec3(0.10, 0.45, 0.46),
    vec3(0.45, 0.45, 0.46), vec3(0.20, 0.30, 0.48), vec3(0.72, 0.62, 0.48), vec3(0.35, 0.07, 0.12),
    vec3(0.10, 0.28, 0.16), vec3(0.78, 0.45, 0.55), vec3(0.78, 0.36, 0.10), vec3(0.13, 0.11, 0.095));
const vec3 BOTTOM_TONES[16] = vec3[16](
    vec3(0.18, 0.25, 0.40), vec3(0.08, 0.10, 0.18), vec3(0.04, 0.04, 0.05), vec3(0.28, 0.28, 0.30),
    vec3(0.55, 0.47, 0.33), vec3(0.07, 0.10, 0.20), vec3(0.25, 0.27, 0.17), vec3(0.28, 0.18, 0.10),
    vec3(0.70, 0.62, 0.50), vec3(0.80, 0.80, 0.78), vec3(0.30, 0.08, 0.10), vec3(0.14, 0.14, 0.15),
    vec3(0.35, 0.45, 0.60), vec3(0.10, 0.22, 0.14), vec3(0.60, 0.48, 0.30), vec3(0.07, 0.075, 0.09));
const vec3 SHOE_TONES[8] = vec3[8](
    vec3(0.85, 0.85, 0.84), vec3(0.05, 0.05, 0.05), vec3(0.35, 0.35, 0.36), vec3(0.20, 0.12, 0.07),
    vec3(0.08, 0.10, 0.22), vec3(0.55, 0.10, 0.10), vec3(0.45, 0.30, 0.15), vec3(0.05, 0.045, 0.04));
const vec3 ACCENT_TONES[16] = vec3[16](
    vec3(0.04, 0.04, 0.05), vec3(0.08, 0.12, 0.28), vec3(0.62, 0.10, 0.10), vec3(0.14, 0.40, 0.20),
    vec3(0.85, 0.70, 0.15), vec3(0.45, 0.45, 0.46), vec3(0.35, 0.15, 0.45), vec3(0.85, 0.40, 0.10),
    vec3(0.10, 0.45, 0.46), vec3(0.85, 0.50, 0.60), vec3(0.30, 0.18, 0.10), vec3(0.88, 0.88, 0.86),
    vec3(0.18, 0.20, 0.22), vec3(0.50, 0.12, 0.20), vec3(0.20, 0.30, 0.55), vec3(0.03, 0.03, 0.035));
const vec3 IRIS_TONES[6] = vec3[6](
    vec3(0.20, 0.11, 0.05), vec3(0.30, 0.19, 0.08), vec3(0.18, 0.12, 0.07),
    vec3(0.20, 0.38, 0.55), vec3(0.22, 0.40, 0.25), vec3(0.36, 0.40, 0.42));

float hash11(float x) { return fract(sin(x * 127.1) * 43758.5453); }
float hash31(vec3 p) { p = fract(p * 0.1031); p += dot(p, p.zyx + 31.32); return fract((p.x + p.y) * p.z); }
float vnoise(vec3 p) {
    vec3 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash31(i), hash31(i + vec3(1, 0, 0)), f.x), mix(hash31(i + vec3(0, 1, 0)), hash31(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash31(i + vec3(0, 0, 1)), hash31(i + vec3(1, 0, 1)), f.x), mix(hash31(i + vec3(0, 1, 1)), hash31(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
// bump mapping from a height field using screen-space derivatives (no tangents needed)
vec3 bumpNormal(vec3 N, vec3 p, float h, float strength) {
    vec3 dpx = dFdx(p), dpy = dFdy(p);
    float dhx = dFdx(h), dhy = dFdy(h);
    vec3 r1 = cross(dpy, N), r2 = cross(N, dpx);
    float det = dot(dpx, r1);
    vec3 grad = sign(det) * (dhx * r1 + dhy * r2);
    return normalize(abs(det) * N - strength * grad);
}
"""

PED_FS = ("""
#version 330 core
in vec3 vWorld;
in vec3 vNormal;
in vec3 vRest;
in vec2 vUV;
in vec2 vEyeOffset;
in float vLocalZ;
in float vCrease;
in float vAO;
flat in float vMat;
flat in vec4 vLook0;
flat in vec4 vLook1;
flat in vec3 vBodyInfo;
flat in int vFlags;
flat in float vVis;
flat in float vBlink;
flat in vec3 vFaceInfo;
flat in float vFaceRole;
flat in float vLayer;
uniform sampler2DArray uMaterialDetail;
uniform sampler2DArray uEyeTextures;
uniform sampler2DArray uFaceTextures;
uniform sampler2DArray uCitizenTextures;
uniform vec3  uCamPos, uSunDir, uFogColor;
uniform vec3  uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uNightFactor, uExposure;
uniform float uWetness, uRain;
out vec4 FragColor;
""") + HDR_CONV + HDR_OUT + MATERIAL_LIGHT + STREET_SPILL + SHADOW_PCF + PED_DISSOLVE + PED_PALETTES + f"""
const int MAT_SKIN = {M_SKIN}, MAT_HAIR = {M_HAIR}, MAT_TOP = {M_TOP}, MAT_BOTTOM = {M_BOTTOM},
          MAT_SHOE = {M_SHOE}, MAT_FIXED = {M_FIXED}, MAT_EYE = {M_EYE}, MAT_CANOPY = {M_CANOPY},
          MAT_METAL = {M_METAL}, MAT_PHONE = {M_PHONE}, MAT_SCREEN = {M_SCREEN}, MAT_GLASS = {M_GLASS},
          MAT_LIMB = {M_LIMB}, MAT_SHIN = {M_SHIN}, MAT_CAP = {M_CAP}, MAT_DARK = {M_DARK},
          MAT_LIP = {M_LIP}, MAT_SCLERA = {M_SCLERA}, MAT_IRIS = {M_IRIS}, MAT_GLOVE = {M_GLOVE},
          MAT_CUP = {M_CUP}, MAT_SOLE = {M_SOLE}, MAT_BROW = {M_BROW}, MAT_BEARD = {M_BEARD},
          MAT_LEATHER = {M_LEATHER}, MAT_KNIT = {M_KNIT}, MAT_LINING = {M_LINING},
          MAT_STITCH = {M_STITCH}, MAT_ZIP = {M_ZIP}, MAT_SHIRT = {M_SHIRT}, MAT_NAIL = {M_NAIL},
          MAT_MASK = {M_MASK}, MAT_REAL_SCLERA = {M_REAL_SCLERA}, MAT_REAL_IRIS = {M_REAL_IRIS}, MAT_FACE = {M_FACE},
          MAT_CITIZEN = {M_CITIZEN};
const int ACC_MASK_BIT = {ACC_MASK}, ACC_BEARD_BIT = {ACC_BEARD};
const int NPC_ALBEDO_FIRST = 3, NPC_MALE_ALBEDOS = {len(CITIZEN_MALE)}, NPC_FEMALE_ALBEDOS = {len(CITIZEN_FEMALE)};
const vec2 EYE_APERTURE = vec2({EYE_HALF_WIDTH:.6f}, {EYE_HALF_HEIGHT:.6f});
""" + """
vec4 detailAt(int layer, vec2 uv) {
    return texture(uMaterialDetail, vec3(uv / 0.08, float(layer)));
}

float facePatch(vec2 yz, vec2 centre, vec2 size) {
    vec2 d = (yz - centre) / size;
    return exp(-dot(d, d));
}

vec3 skinTint(vec3 base, vec3 rest, float seed, float frontFace, float detailFade) {
    float mottling = vnoise(rest * 115.0 + seed * 17.0) - 0.5;
    float melanin = vnoise(rest * 29.0 + seed * 7.0) - 0.5;
    vec3 tint = base * (1.0 + 0.06 * mottling + 0.06 * melanin);
    vec2 yz = vec2(abs(rest.y), rest.z);
    float cheek = facePatch(yz, vec2(vFaceInfo.z * 1.35, vFaceInfo.x - 0.037), vec2(0.021, 0.025));
    float nose = facePatch(yz, vec2(0.0, vFaceInfo.x - 0.034), vec2(0.013, 0.023));
    float ear = facePatch(yz, vec2(0.073, vFaceInfo.x - 0.016), vec2(0.013, 0.035));
    float blood = 0.42 * cheek * frontFace + 0.25 * nose * frontFace + 0.35 * ear;
    tint *= mix(vec3(1.0), vec3(1.05, 0.91, 0.88), clamp(blood, 0.0, 0.65));
    float underEye = facePatch(yz, vec2(vFaceInfo.z, vFaceInfo.x - 0.010), vec2(0.015, 0.006));
    tint *= 1.0 - vec3(0.065, 0.060, 0.035) * underEye * frontFace;
    // Sparse, softly filtered pigmentation, not an unfiltered noise normal.
    vec2 cells = rest.yz * 700.0;
    vec2 cell = floor(cells);
    float mark = hash31(vec3(cell, seed * 11.0));
    vec2 jitter = vec2(hash31(vec3(cell, 1.0)), hash31(vec3(cell, 2.0))) * 0.5 + 0.25;
    float radius = length(fract(cells) - jitter);
    float aa = max(fwidth(radius), 0.02);
    float freckle = (1.0 - smoothstep(0.14 - aa, 0.14 + aa, radius)) * step(0.984, mark);
    freckle *= frontFace * cheek * detailFade * step(hash11(seed * 33.0), 0.45);
    return tint * (1.0 - 0.20 * freckle);
}

float eyebrowMask(vec3 rest) {
    float outward = (abs(rest.y) - vFaceInfo.z) / 0.020;
    float taper = 1.0 - smoothstep(0.65, 1.0, abs(outward));
    float line = vFaceInfo.x + 0.020 + 0.003 * (1.0 - outward * outward) - 0.002 * max(outward, 0.0);
    float thickness = mix(0.0026, 0.0012, clamp(outward * 0.5 + 0.5, 0.0, 1.0));
    float d = (rest.z - line) / thickness;
    return exp(-d * d) * taper;
}

// Energy-conserving dielectric GGX, shared by skin, eyes and garments. Skin
// receives a second, lower-roughness oil lobe instead of a plastic Phong shine.
vec3 humanGGX(vec3 N, vec3 V, vec3 L, float r, float F0) {
    vec3 H = normalize(L + V);
    float nl = max(dot(N, L), 0.0), nv = max(dot(N, V), 0.001);
    float nh = max(dot(N, H), 0.0), vh = max(dot(V, H), 0.0);
    float a = max(r * r, 0.002), a2 = a * a;
    float denom = nh * nh * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159265 * denom * denom);
    float gv = 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
    float gl = 2.0 * nl / max(nl + sqrt(a2 + (1.0 - a2) * nl * nl), 0.001);
    float F = F0 + (1.0 - F0) * pow(1.0 - vh, 5.0);
    return vec3(D * gv * gl * F / max(4.0 * nl * nv, 0.001));
}

vec3 surfaceMappedNormal(vec3 N, vec3 position, vec2 uv, vec3 normalTex, float strength) {
    vec3 dx = dFdx(position), dy = dFdy(position);
    vec2 tx = dFdx(uv), ty = dFdy(uv);
    float det = tx.x * ty.y - tx.y * ty.x;
    if (abs(det) < 1e-10) return N;
    vec3 T = normalize((dx * ty.y - dy * tx.y) / det);
    T = normalize(T - N * dot(N, T));
    vec3 B = normalize((-dx * ty.x + dy * tx.x) / det);
    vec3 mapped = normalTex * 2.0 - 1.0;
    mapped.xy *= strength;
    return normalize(mat3(T, B, N) * mapped);
}

vec3 eyeMappedNormal(vec3 N, vec3 position, vec2 uv, vec3 normalTex) {
    return surfaceMappedNormal(N, position, uv, normalTex, 0.35);
}

void main() {
    pedDissolve(vVis);
    int mat = int(vMat + 0.5);
    if (mat == MAT_FACE && vFaceRole < 0.5 && vRest.x > 0.035) {
        // Authored socket interior polygons must not occlude the independent
        // animated globe. Keep a fixed opening; animated lid patches close it.
        vec2 opening = vec2(abs(vRest.y)-vFaceInfo.z, vRest.z-vFaceInfo.x)/EYE_APERTURE;
        if (dot(opening, opening) < 1.0) discard;
    }
    if (mat == MAT_REAL_SCLERA || mat == MAT_REAL_IRIS) {
        // Keep the complete imported eyeball seated behind the socket. Gaze
        // rotates the globe, not this fixed head-local eyelid opening.
        vec2 aperture = vEyeOffset / (EYE_APERTURE * vec2(1.0, max(1.0 - vBlink, 0.00001)));
        if (dot(aperture, aperture) > 1.0) discard;
        if (vBlink > 0.999) discard;
    }
    // The supplied shell has a transmissive cornea, not an opaque white cap.
    // Reveal its actual inset iris geometry; shading supplies wet highlights.
    if (mat == MAT_REAL_SCLERA && texture(uEyeTextures, vec3(vUV, 3.0)).r > 0.5) discard;
    bool player = (vFlags & 32768) != 0;
    vec3 Ng = normalize(vNormal);
    vec3 V = normalize(uCamPos - vWorld);
    float dist = length(uCamPos - vWorld);
    bool backface = dot(Ng, V) < 0.0;
    vec3 N = backface ? -Ng : Ng;
    vec3 L = normalize(uSunDir);
    float sh = shadowPCF(vWorld, N, L);
    float seed = vBodyInfo.x;

    float shade = 0.92 + 0.16 * hash11(seed * 91.7);
    vec3 skin   = SKIN_TONES[int(clamp(vLook0.x, 0.0, 7.0) + 0.5)];
    vec3 hair   = HAIR_TONES[int(clamp(vLook0.y, 0.0, 7.0) + 0.5)];
    vec3 top    = TOP_TONES[int(clamp(vLook0.z, 0.0, 15.0) + 0.5)] * shade;
    vec3 bottom = BOTTOM_TONES[int(clamp(vLook0.w, 0.0, 15.0) + 0.5)] * shade;
    vec3 shoe   = SHOE_TONES[int(clamp(vLook1.x, 0.0, 7.0) + 0.5)];
    vec3 accent = ACCENT_TONES[int(clamp(vLook1.y, 0.0, 15.0) + 0.5)];
    bool longSleeve = vLook1.z > 0.5;
    bool shorts = vLook1.w > 0.5;
    int iris = int(hash11(seed * 5.31) * 5.99);
    float lipT = hash11(seed * 3.7);
    float fade = 1.0 - smoothstep(4.0, 26.0, dist);

    vec3 c = skin;
    float gloss = 0.0, f0 = 0.04;
    float sssAmt = 0.0, sheen = 0.0, wrapAmt = 0.0;
    float height = 0.0, cavity = 1.0, rough = 0.8;
    bool cloth = false, leather = false;
    bool skinSurface = mat == MAT_SKIN || mat == MAT_LIP || mat == MAT_FACE;
    float frontFace = smoothstep(0.015, 0.050, vRest.x) * smoothstep(1.53, 1.59, vRest.z);
    float oilMask = frontFace * (facePatch(vRest.yz, vec2(0.0, vFaceInfo.x - 0.029), vec2(0.016, 0.028))
                    + 0.6 * facePatch(vRest.yz, vec2(0.0, vFaceInfo.x + 0.043), vec2(0.042, 0.026)));
    oilMask = clamp(oilMask, 0.0, 1.0);

    if (mat == MAT_CITIZEN) {
        // Supplied citizen: baked albedo. Even layers are the body (face, hands, hair), odd
        // layers the outfit. Per-pedestrian shade and the crease folds below keep a crowd of
        // one identity from looking cloned; there are no source normal or roughness maps.
        bool bodyLayer = mod(vLayer, 2.0) < 0.5;
        vec3 albedoTex = texture(uCitizenTextures, vec3(vUV, vLayer)).rgb;
        if (bodyLayer) {
            c = albedoTex * (1.0 + 0.5 * (shade - 1.0));
            gloss = 0.30; f0 = 0.028; sssAmt = 0.55; wrapAmt = 0.15; rough = 0.58;
        } else {
            c = albedoTex * shade;
            cloth = true; sheen = 0.16; gloss = 0.05; rough = 0.80;
        }
    } else if (mat == MAT_FACE) {
        vec3 reference = texture(uFaceTextures, vec3(vUV, 0.0)).rgb;
        vec3 detailNormal = texture(uFaceTextures, vec3(vUV, 1.0)).rgb;
        float specular = texture(uFaceTextures, vec3(vUV, 2.0)).r;
        // Player retains the supplied diffuse likeness. NPC albedo is a separate
        // masked citizen transfer; the normal/specular layers remain shared.
        c = skinTint(skin, vRest, seed, frontFace, fade);
        if (player) c = mix(skin, reference * 1.12, smoothstep(1.535, 1.57, vRest.z));
        float lip = facePatch(vRest.yz, vec2(0.0, vFaceInfo.y), vec2(0.026, 0.005));
        float beard = smoothstep(1.54, 1.565, vRest.z)*(1.0-smoothstep(1.595, 1.645, vRest.z))*frontFace;
        float moustache = facePatch(vRest.yz, vec2(0.0, vFaceInfo.y+0.012), vec2(0.021, 0.004))*frontFace;
        if (!player) {
            vec4 citizen = vec4(0.0);
            if (textureSize(uFaceTextures, 0).z >= NPC_ALBEDO_FIRST+NPC_MALE_ALBEDOS+NPC_FEMALE_ALBEDOS) {
                bool female = vBodyInfo.z > 0.5 && vBodyInfo.z < 1.5;
                float count = female ? float(NPC_FEMALE_ALBEDOS) : float(NPC_MALE_ALBEDOS);
                float identity = min(floor(clamp(seed, 0.0, 1.0)*count), count-1.0);
                float layer = float(NPC_ALBEDO_FIRST) + (female ? float(NPC_MALE_ALBEDOS) : 0.0) + identity;
                citizen = texture(uFaceTextures, vec3(vUV, layer));
            }
            float fallback = 1.0-citizen.a;
            c *= mix(vec3(1.0), vec3(1.02, 0.84, 0.83), lip*frontFace*fallback);
            float strands = detailAt(5, vUV).g;
            float brow = eyebrowMask(vRest)*frontFace;
            c = mix(c, hair*(0.55+0.35*strands), brow*(0.70+0.25*strands)*fallback);
            if ((vFlags & ACC_BEARD_BIT) != 0)
                c = mix(c, hair*0.85, max(beard, moustache)*(0.22+0.13*hash11(seed*31.0))*(0.55+0.60*strands)*fallback);
            c = mix(c, skin*citizen.rgb/0.65, citizen.a);
        }
        // Normal detail follows the UV atlas through skinning and seeded shape
        // changes; no tangent vertex attributes or runtime image decoding.
        // Keep the same detail map, without exaggerating its authored facial
        // folds on NPC identities. The player strength is unchanged.
        float normalStrength = player ? 0.80 : 0.35;
        N = surfaceMappedNormal(N, vWorld, vUV, detailNormal, normalStrength*fade);
        gloss = 0.65; f0 = 0.028; sssAmt = 0.8; wrapAmt = 0.18;
        rough = clamp(0.64-0.20*specular-0.08*oilMask, 0.42, 0.68);
    } else if (mat == MAT_SKIN) {
        vec4 d = detailAt(0, vUV);
        c = skinTint(skin, vRest, seed, frontFace, fade) * (0.98 + 0.10 * (d.r - 0.5));
        float beard = 0.0;
        if ((vFlags & ACC_BEARD_BIT) != 0)
            beard = smoothstep(1.53, 1.56, vRest.z) * (1.0-smoothstep(1.58, 1.644, vRest.z))
                  * smoothstep(0.004, 0.040, abs(vRest.y)) * frontFace;
        float moustache = facePatch(vRest.yz, vec2(0.0, vFaceInfo.y+0.014), vec2(0.020,0.004));
        if ((vFlags & ACC_BEARD_BIT) != 0) beard = max(beard, 0.6*moustache*frontFace);
        float stubble = detailAt(5, vUV*2.3).g;
        c = mix(c, hair * 0.80, (player ? 0.36 : 0.24)*beard*(0.70+0.60*stubble));
        float brow = eyebrowMask(vRest) * frontFace;
        float strands = detailAt(5, vUV).g;
        c = mix(c, hair * (0.55 + 0.35 * strands), brow * (0.65 + 0.25 * strands));
        gloss = 0.85; f0 = 0.028;
        sssAmt = 0.8; wrapAmt = 0.18;
        height = (d.g - 0.60) * 0.00020 * fade + brow * (strands - 0.5) * 0.00012 * fade;
        cavity = d.a; rough = clamp(0.48 + 0.20 * d.b - 0.12 * oilMask + 0.08 * brow, 0.36, 0.70);
    } else if (mat == MAT_HAIR) {
        vec4 d = detailAt(5, vUV);
        float strand = d.g;
        c = hair * (0.55 + 0.70 * strand) * (0.86 + 0.20 * d.r);
        gloss = 0.35; f0 = 0.05;
        wrapAmt = 0.2; height = (strand - 0.5) * 0.00045 * fade; cavity = d.a;
    } else if (mat == MAT_BROW) {
        c = hair * 0.8;
    } else if (mat == MAT_BEARD) {
        c = mix(skin * 0.88, hair * 0.9, 0.55);
        gloss = 0.05; wrapAmt = 0.3; sssAmt = 0.4;
    } else if (mat == MAT_LIP) {
        vec4 d = detailAt(0, vUV);
        c = skin * mix(vec3(1.05, 0.83, 0.84), vec3(1.08, 0.76, 0.78), lipT);
        float lipLine = exp(-pow((vRest.z - vFaceInfo.y) / 0.0012, 2.0));
        c *= (0.98 + 0.04 * d.r) * (1.0 - 0.12 * lipLine);
        gloss = 0.75; f0 = 0.028; sssAmt = 0.65; wrapAmt = 0.15; rough = 0.36 + 0.10 * d.b;
        height = (d.g - 0.60) * 0.00014 * fade; cavity = d.a;
    } else if (mat == MAT_REAL_SCLERA || mat == MAT_REAL_IRIS) {
        c = texture(uEyeTextures, vec3(vUV, 0.0)).rgb;
        if (mat == MAT_REAL_IRIS) {
            // Keep the imported iris fibres, limbal ring and black pupil while
            // choosing green for Aiden and a stable eye colour for each NPC.
            float fibres = dot(c, vec3(0.2126, 0.7152, 0.0722));
            // The citizen reference uses subdued brown irises, not the bright
            // green/blue inspection palette. Retain the separate detailed eye.
            vec3 irisTint = player ? vec3(0.22, 0.30, 0.25) : IRIS_TONES[int(clamp(floor(seed*3.0), 0.0, 2.0))];
            c = mix(c, irisTint*clamp(fibres/0.32, 0.0, 1.6), 0.92);
        }
        float eyeRough = texture(uEyeTextures, vec3(vUV, 2.0)).g;
        N = eyeMappedNormal(N, vWorld, vUV,
            texture(uEyeTextures, vec3(vUV, mat == MAT_REAL_IRIS ? 4.0 : 1.0)).rgb);
        gloss = 1.0; f0 = 0.025;
        rough = mat == MAT_REAL_IRIS ? 0.09 : 0.12 + 0.12 * eyeRough;
        wrapAmt = mat == MAT_REAL_SCLERA ? 0.15 : 0.05;
    } else if (mat == MAT_SCLERA) {
        c = vec3(0.86, 0.83, 0.80); gloss = 0.9; rough = 0.12; f0 = 0.05; wrapAmt = 0.15;
    } else if (mat == MAT_IRIS) {
        c = IRIS_TONES[iris]; gloss = 1.0; rough = 0.09; f0 = 0.05;
    } else if (mat == MAT_EYE) {
        c = vec3(0.012); gloss = 1.0; rough = 0.09; f0 = 0.05;
    } else if (mat == MAT_NAIL) {
        c = skin * vec3(1.05, 0.93, 0.88); gloss = 0.5; rough = 0.28; f0 = 0.03;
    } else if (mat == MAT_LEATHER) {
        vec4 d = detailAt(1, vUV);
        vec3 base = player ? vec3(0.28, 0.17, 0.105) : mix(vec3(0.075, 0.063, 0.052), top, 0.48);
        c = base * (0.78 + 0.42 * d.r);
        leather = true; cloth = true; sheen = 0.07;
        gloss = 0.36; f0 = 0.05;
        height = (d.g - 0.5) * 0.00075 * fade;
        cavity = d.a; rough = d.b;
    } else if (mat == MAT_KNIT) {
        vec4 d = detailAt(4, vUV);
        c = (player ? vec3(0.64, 0.62, 0.55) : top) * (0.90 + 0.18 * d.r);
        cloth = true; sheen = 0.28;
        gloss = 0.05;
        height = (d.g - 0.5) * 0.00055 * fade; cavity = d.a; rough = d.b;
    } else if (mat == MAT_SHIRT) {
        vec4 d = detailAt(2, vUV);
        c = mix(top, vec3(0.86, 0.84, 0.80), player ? 0.0 : 0.45) * (0.92 + 0.12 * d.r);
        cloth = true; sheen = 0.20;
        gloss = 0.04;
        height = (d.g - 0.5) * 0.00032 * fade; cavity = d.a; rough = d.b;
    } else if (mat == MAT_TOP) {
        vec4 d = detailAt(2, vUV);
        c = top * (0.92 + 0.13 * d.r);
        cloth = true; sheen = 0.22;
        gloss = 0.04;
        height = (d.g - 0.5) * 0.00042 * fade; cavity = d.a; rough = d.b;
        float herring = step(0.5, fract(vRest.z * 38.0 + vRest.y * 7.0)) * 0.035;
        c *= 1.0 - herring;
    } else if (mat == MAT_BOTTOM || mat == MAT_SHIN) {
        vec4 d = detailAt(3, vUV);
        c = bottom * (0.92 + 0.13 * d.r);
        cloth = true; sheen = 0.12;
        gloss = 0.04;
        height = (d.g - 0.5) * 0.00050 * fade; cavity = d.a; rough = d.b;
        bool denim = (vLook0.w < 1.5 || (vLook0.w > 11.5 && vLook0.w < 12.5)) && !player;
        if (denim) {
            float wear = 0.5 + 0.5 * sin(vRest.z * 14.0 + vRest.y * 9.0);
            c = mix(c, c * 1.55 + vec3(0.015), 0.28 * wear * smoothstep(0.30, 0.80, vRest.z));
        }
    } else if (mat == MAT_LIMB) {
        vec4 d = detailAt(2, vUV);
        c = top * (0.92 + 0.13 * d.r);
        cloth = true; sheen = 0.20;
        gloss = 0.04;
        height = (d.g - 0.5) * 0.00040 * fade; cavity = d.a; rough = d.b;
    } else if (mat == MAT_LINING) {
        vec4 d = detailAt(2, vUV);
        c = mix(vec3(0.32, 0.22, 0.12), accent, 0.38) * (0.90 + 0.16 * d.r);
        cloth = true; sheen = 0.30;
        gloss = 0.10;
        height = (d.g - 0.5) * 0.00036 * fade; cavity = d.a; rough = d.b;
    } else if (mat == MAT_STITCH) {
        c = mix(accent, vec3(0.54, 0.48, 0.38), 0.58);
        gloss = 0.05; cloth = true; sheen = 0.18;
    } else if (mat == MAT_ZIP) {
        c = vec3(0.42, 0.42, 0.41); gloss = 0.7; rough = 0.23; f0 = 0.48;
    } else if (mat == MAT_MASK) {
        vec4 d = detailAt(2, vUV);
        c = mix(vec3(0.065, 0.070, 0.078), top, 0.12) * (0.90 + 0.18 * d.r);
        cloth = true; sheen = 0.16;
        gloss = 0.06; rough = d.b;
        height = (d.g - 0.5) * 0.00040 * fade; cavity = d.a; rough = d.b;
    } else if (mat == MAT_SHOE) {
        vec4 d = detailAt(1, vUV);
        c = shoe * (0.82 + 0.30 * d.r);
        gloss = 0.16; f0 = 0.05;
        height = (d.g - 0.5) * 0.00055 * fade; cavity = d.a; rough = d.b;
    } else if (mat == MAT_SOLE) {
        c = mix(vec3(0.85), shoe * 0.6, step(0.5, 1.0 - shoe.r)); gloss = 0.1; rough = 0.70;
    } else if (mat == MAT_FIXED) {
        vec4 d = detailAt(2, vUV);
        c = accent * (0.85 + 0.22 * d.r); gloss = 0.08; rough = d.b; cloth = true; sheen = 0.14;
        height = (d.g - 0.5) * 0.00045 * fade; cavity = d.a;
    } else if (mat == MAT_CANOPY) {
        c = accent; gloss = 0.08 + 0.45 * uWetness; rough = 0.40; f0 = 0.05;
    } else if (mat == MAT_METAL) {
        c = vec3(0.45); gloss = 0.8; rough = 0.21; f0 = 0.5;
    } else if (mat == MAT_PHONE) {
        c = vec3(0.02); gloss = 0.7; rough = 0.16; f0 = 0.06;
    } else if (mat == MAT_SCREEN) {
        c = vec3(0.03); gloss = 0.5; rough = 0.18; f0 = 0.05;
    } else if (mat == MAT_GLASS) {
        c = vec3(0.008); gloss = 1.0; rough = 0.09; f0 = 0.08;
    } else if (mat == MAT_CAP) {
        vec4 d = detailAt(4, vUV);
        c = accent * (0.88 + 0.16 * d.r); cloth = true; sheen = 0.12;
        height = (d.g - 0.5) * 0.00035 * fade; cavity = d.a;
    } else if (mat == MAT_DARK) {
        c = vec3(0.02); gloss = 0.2; rough = 0.50;
    } else if (mat == MAT_GLOVE) {
        vec4 d = detailAt(1, vUV);
        c = vec3(0.065, 0.055, 0.048) * (0.82 + 0.34 * d.r);
        leather = true; cloth = true; sheen = 0.06;
        gloss = 0.30; f0 = 0.04;
        height = (d.g - 0.5) * 0.00055 * fade; cavity = d.a;
    } else if (mat == MAT_CUP) {
        c = vec3(0.86, 0.84, 0.80); gloss = 0.2; rough = 0.34;
    }

    if (leather && player) {
        // Worn black-brown leather: lighter creases and edge abrasion, no painted seams.
        float wear = vnoise(vRest * 22.0) * 0.55 + vnoise(vRest * 110.0) * 0.45;
        c *= 0.84 + 0.30 * wear;
        c += vec3(0.035, 0.030, 0.024) * smoothstep(0.55, 0.85, wear);
    }

    // cloth fold normals are still driven by joint creases; the geometry supplies the rest
    if (cloth) {
        float fold = sin(vRest.z * 92.0 + vnoise(vRest * 17.0) * 7.0 + vRest.x * 30.0);
        height += 0.0024 * vCrease * fold * (1.0 - smoothstep(10.0, 40.0, dist));
    }

    // soaked clothes and skin in the rain
    float wet = uWetness;
    float wetMask = cloth ? 1.0 : 0.4;
    c *= 1.0 - 0.34 * wet * wetMask;
    gloss += (cloth ? 0.10 : 0.12) * wet;
    rough = mix(rough, max(0.07, rough * 0.55), wet * (cloth ? 0.5 : 0.8));

    vec3 Nb = N;
    if (height != 0.0) Nb = bumpNormal(N, vWorld, height, 1.0);

    vec3 albedo = reflectanceRange(linearColor(c), vec2(0.018, 0.80));
    float NdotLraw = dot(Nb, L);
    float NdotL = max(NdotLraw, 0.0);
    float NdotV = clamp(dot(Nb, V), 0.0, 1.0);

    float groundAO = mix(0.5, 1.0, smoothstep(0.0, 0.45, vLocalZ));
    float creaseAO = 1.0 - 0.30 * vCrease * (cloth ? 1.0 : 0.4);
    float ao = vAO * groundAO * creaseAO * mix(0.80, 1.0, Nb.z * 0.5 + 0.5);
    ao *= mix(0.84, 1.0, cavity);

    vec3 amb = uAmbColor * mix(0.50, 1.0, Nb.z * 0.5 + 0.5);
    vec3 bounce = uAmbColor * vec3(0.26, 0.22, 0.17) * max(-Nb.z, 0.0);
    float diffTerm = wrapAmt > 0.0 ? max((NdotLraw + wrapAmt) / (1.0 + wrapAmt), 0.0) : NdotL;
    vec3 diff = (amb + bounce) * 0.6 + uSunColor * diffTerm * (0.32 * sh);
    if (uNightFactor > 0.02) {
        vec3 diffE, specE;
        streetLights(vWorld, Nb, V, diffE, specE);
        diff += diffE * (uNightFactor / 3.14159265);
    }
    vec3 lit = albedo * diff * ao * (skinSurface ? 1.0 - f0 : 1.0);

    // subsurface glow: red light bleeding through thin skin at the terminator
    if (sssAmt > 0.0) {
        float term = smoothstep(-0.35, 0.25, NdotLraw) * (1.0 - smoothstep(0.25, 0.85, NdotLraw));
        float thin = 0.25 + 0.30 * (1.0 - NdotV);
        thin += 0.45 * facePatch(vec2(abs(vRest.y), vRest.z), vec2(0.073, vFaceInfo.x - 0.015), vec2(0.012, 0.033));
        lit += albedo * vec3(1.0, 0.42, 0.28) * uSunColor * term * thin * 0.045 * sssAmt * sh * ao;
        lit += albedo * vec3(0.9, 0.42, 0.32) * amb * 0.025 * sssAmt * pow(1.0 - NdotV, 2.0);
    }
    // fabric / leather sheen at grazing angles
    if (sheen > 0.0) lit += albedo * (amb + uSunColor * 0.15 * sh) * pow(1.0 - NdotV, 3.0) * sheen * ao;

    // specular
    if (gloss > 0.0) {
        vec3 H = normalize(L + V);
        float fres = f0 + (1.0 - f0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
        vec3 spec = uSunColor * humanGGX(Nb, V, L, clamp(rough, 0.07, 0.95), f0) * NdotL * sh;
        if (mat == MAT_HAIR) {
            vec3 T = normalize(cross(Nb, vec3(0.0, 0.0, 1.0)) + vec3(0.0001));
            float tH = dot(T, H);
            float kk = sqrt(max(1.0 - tH * tH, 0.0));
            spec = uSunColor * (pow(kk, 90.0) * 0.14 + pow(kk, 22.0) * 0.04) * (hair * 1.5 + 0.05) * sh * smoothstep(-0.1, 0.4, NdotLraw);
        } else if (skinSurface) {
            vec3 oil = uSunColor * humanGGX(Nb, V, L, 0.26, 0.028) * NdotL * sh;
            spec = mix(spec, oil, 0.18 * oilMask);
        }
        lit += spec * gloss * ao;
        if (mat == MAT_GLASS || mat == MAT_PHONE || mat == MAT_CANOPY || mat == MAT_EYE || mat == MAT_IRIS || mat == MAT_SCLERA || mat == MAT_REAL_SCLERA || mat == MAT_REAL_IRIS)
            lit += amb * fres * gloss * 1.2 * pow(1.0 - NdotV, 2.0);
    }
    if (mat == MAT_CANOPY && backface) lit += albedo * uSunColor * 0.20 * sh + albedo * amb * 0.5;
    if (mat == MAT_SCREEN) lit += emissionNits(vec3(0.70, 0.83, 1.0), 260.0);
    lit = max(lit, albedo * uAmbColor * 0.04 * ao);

    vec3 mapped = sceneOut(lit);
    float f = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
    FragColor = vec4(mix(mapped, displayToScene(uFogColor), f), 1.0);
}
"""
