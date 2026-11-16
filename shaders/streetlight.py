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
layout(location=2) in float aPart;    // 0 pole, 1 arm, 2 luminaire head
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
void main() {
    float H = iLamp.z - iBaseZ;
    float A = iExtra.x;
    vec3 local;
    if (aPart < 0.5)       local = vec3(aPos.x, aPos.y, aPos.z * H);                    // pole: z in [0,1] -> [0,H]
    else if (aPart < 1.5)  local = vec3(aPos.x * A, aPos.y, H + aPos.z);               // arm: x in [0,1] -> [0,A]
    else                   local = vec3(aPos.x + A, aPos.y, H + aPos.z);               // head hangs at arm tip
    float c = cos(iLamp.w), s = sin(iLamp.w);
    vec2 rot = vec2(local.x * c - local.y * s, local.x * s + local.y * c);
    vec3 n = aNormal;
    vNormal = vec3(n.x * c - n.y * s, n.x * s + n.y * c, n.z);
    vec3 p = vec3((iLamp.xy + rot) * uScale, iBaseZ + local.z * uGrow) + uOffset;
    vWorld = p; vPart = aPart; vCol = iExtra.yzw;
    gl_Position = uProj * uView * vec4(p, 1.0);
}
"""

from .hdr import HDR_CONV, HDR_OUT
from .materials import MATERIAL_LIGHT
from .streetspill import STREET_SPILL

STREETPOLE_FS = ("""
#version 330 core
in vec3 vWorld; in vec3 vNormal; in float vPart; in vec3 vCol;
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
    vec3 albedo = vPart < 1.5 ? vec3(0.07, 0.075, 0.08) : vec3(0.10, 0.10, 0.11);
    vec3 lit = albedo * (uAmbColor * 0.8 + uSunColor * max(dot(N, L), 0.0) * 0.3);
    if (uNightFactor > 0.02) {
        vec3 diffE, specE;
        streetLights(vWorld, N, normalize(uCamPos - vWorld), diffE, specE);
        lit += albedo * diffE * (uNightFactor / 3.14159265);
    }
    // glowing lens on the underside of the luminaire
    bool lamp_on = dot(vCol, vec3(1.0)) > 0.01;
    if (vPart > 1.5 && N.z < -0.5 && lamp_on) {
        lit += emissionNits(vCol, STREET_LAMP_NITS) * uNightFactor;
    }
    vec3 mapped = sceneOut(lit);
    float f = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
    FragColor = vec4(mix(mapped, displayToScene(uFogColor), f), 1.0);
}
""")
