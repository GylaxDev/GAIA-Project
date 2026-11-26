# --- shaders/streetlight.py ---
# Street lamps: the instanced pole + arm + luminaire mesh.  The luminaire glow
# sprite moved to shaders/lightfield.py, which draws every light source in the
# scene (lamps, vehicle lamps, facade emissives) as one long-range batch.
#
# Per-lamp instance buffer, shared with that field:
#   iLamp  = (x, y, mount height, arm angle)      -> location 3
#   iExtra = (arm length, r, g, b)                 -> location 4
#   iBaseZ = ground/deck height                   -> location 5
# arm length 0 => post-top lantern (lamp sits directly on the pole).
# Colour (0,0,0) => dead lamp (pole is drawn, nothing glows).

# ------------------------------------------------------------- pole geometry
STREETPOLE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;      // unit-frame vertex
layout(location=1) in vec3 aNormal;
layout(location=2) in float aPart;    // shaft/arm/housing/base/collar/lens/post-top
layout(location=3) in vec4 iLamp;     // x, y, height, angle
layout(location=4) in vec4 iExtra;    // arm length, r, g, b
layout(location=5) in float iBaseZ;
uniform mat4 uView, uProj;
uniform vec3 uOffset;
uniform vec2 uScale;
uniform float uGrow;
out vec3 vWorld;
out vec3 vNormal;
out float vPart;
out vec3 vCol;
out vec3 vLocal;
void main() {
    float H = max(iLamp.z - iBaseZ, .1);
    float A = iExtra.x;
    int part = int(aPart + .5);
    vec3 local = aPos, n = aNormal;
    if (part == 0) { local.z *= H; n.z /= H; }
    else if (part == 1) { local.x *= A; local.z += H; n.x /= max(A, .01); }
    else if (part == 2 || part == 5) { local.x += A; local.z += H; }
    else if (part != 3) local.z += H;  // base dimensions are metres, not a fraction of pole height
    float c = cos(iLamp.w), s = sin(iLamp.w);
    vec2 rot = vec2(local.x * c - local.y * s, local.x * s + local.y * c);
    vNormal = normalize(vec3((n.x * c - n.y * s) / uScale.x,
                             (n.x * s + n.y * c) / uScale.y, n.z / max(uGrow, .001)));
    vec3 p = vec3((iLamp.xy + rot) * uScale, iBaseZ + local.z * uGrow) + uOffset;
    vWorld = p; vPart = aPart; vCol = iExtra.yzw; vLocal = local;
    bool hidden = A > .01 ? part >= 6 : (part == 1 || part == 2 || part == 5);
    if (hidden) { gl_Position = vec4(2., 2., 2., 1.); return; }
    gl_Position = uProj * uView * vec4(p, 1.0);
}
"""

from .hdr import HDR_CONV, HDR_OUT
from .materials import MATERIAL_LIGHT
from .streetspill import STREET_SPILL

STREETPOLE_FS = ("""
#version 330 core
in vec3 vWorld; in vec3 vNormal; in float vPart; in vec3 vCol; in vec3 vLocal;
uniform vec3 uCamPos, uSunDir, uFogColor, uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uNightFactor, uExposure;
out vec4 FragColor;
""" + HDR_CONV + HDR_OUT + MATERIAL_LIGHT + STREET_SPILL + """
vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43; float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uSunDir);
    vec3 V = normalize(uCamPos - vWorld);
    float district = fract(sin(dot(floor(vWorld.xy / 80.), vec2(12.73, 41.19))) * 437.13);
    bool housing = abs(vPart-2.) < .1;
    bool lens = abs(vPart-5.) < .1 || abs(vPart-7.) < .1;
    vec3 albedo = housing ? vec3(.13, .145, .15) : mix(vec3(.035, .043, .041), vec3(.075, .08, .077), district);
    if (lens) albedo = vec3(.36, .39, .36);
    // Dirt at the cast base, subtle vertical weathering; no glossy plastic poles.
    float grime = (1.-smoothstep(.05, .45, vLocal.z)) * .23;
    albedo *= 1.-grime;
    vec3 lit = albedo * (uAmbColor * 0.8 + uSunColor * max(dot(N, L), 0.0) * 0.3);
    vec3 H = normalize(L+V);
    float spec = pow(max(dot(N, H), 0.), housing ? 48. : 24.);
    lit += uSunColor * spec * max(dot(N, L), 0.) * (housing ? .014 : .006);
    if (uNightFactor > 0.02) {
        vec3 diffE, specE;
        streetLights(vWorld, N, normalize(uCamPos - vWorld), diffE, specE);
        lit += albedo * diffE * (uNightFactor / 3.14159265);
    }
    // glowing lens on the underside of the luminaire
    bool lamp_on = dot(vCol, vec3(1.0)) > 0.01;
    if (lens && lamp_on) {
        lit += emissionNits(vCol, STREET_LAMP_NITS) * uNightFactor;
    }
    vec3 mapped = sceneOut(lit);
    float f = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
    FragColor = vec4(mix(mapped, displayToScene(uFogColor), f), 1.0);
}
""")
