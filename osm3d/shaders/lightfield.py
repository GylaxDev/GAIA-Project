# --- shaders/lightfield.py ---
# Long-range point lights (street lamps, head/tail lights, facade emissives) drawn
# as screen-facing instanced billboards.
#
# One light record is two vec4s so a single interleaved array feeds both backends:
#   (x, y, z, radius)  (r, g, b, intensity)
#
# Backends, chosen at runtime by the renderer:
#   ssbo=True   GL 4.3+  -> std430 block, indexed with gl_InstanceID
#   ssbo=False  GL 3.3   -> instanced vertex attributes (divisor 1)
#
# Every instance is rejected in the vertex shader when it is behind the eye or its
# projected centre lies outside the clip rect by more than its own radius.  A
# rejected instance pushes all of its vertices to depth 2, which fails the clip
# test, so no raster work is spent on it.  The CPU culls the same set first; the
# shader check is the safety net for sprites whose centre moved while the frame
# was being prepared.

_LIGHT_HEAD_SSBO = """#version 430 core
layout(location = 0) in vec2 aQuad;
struct Light { vec4 posRad; vec4 colInt; };
layout(std430, binding = 0) buffer LightBuffer { Light uLights[]; };
#define LIGHT_POS_RAD uLights[gl_InstanceID].posRad
#define LIGHT_COL_INT uLights[gl_InstanceID].colInt
"""

_LIGHT_HEAD_ATTR = """#version 330 core
layout(location = 0) in vec2 aQuad;
layout(location = 1) in vec4 iPosRad;
layout(location = 2) in vec4 iColInt;
#define LIGHT_POS_RAD iPosRad
#define LIGHT_COL_INT iColInt
"""

_LIGHT_VS_BODY = """
uniform mat4  uView, uProj;
uniform float uFarClip, uFarFadeStart;
uniform float uSizeGain, uSizeFloor, uMaxSprite;
out vec2  vQuad;
out vec3  vCol;
out float vDist;
out float vFade;
out float vIntensity;

const vec4 CULLED = vec4(0.0, 0.0, 2.0, 1.0);   // depth > 1: outside the clip volume

void main() {
    vec4 pr = LIGHT_POS_RAD;
    vec4 ci = LIGHT_COL_INT;
    vec4 vc = uView * vec4(pr.xyz, 1.0);
    float rad  = max(pr.w, 0.05);
    float dist = length(vc.xyz);

    // The view basis looks down -Z, so a light sphere is entirely behind the eye
    // once its nearest surface reaches Z = 0.
    bool behind = vc.z > -rad;

    // Billboards grow linearly with distance (a 0.6 m lamp is sub-pixel at 1 km),
    // so the on-screen half extent is derived from the same size the quad gets.
    float size = min(uMaxSprite, max(rad, uSizeFloor) + uSizeGain * dist);
    vec4  cc   = uProj * vc;
    float w    = max(abs(cc.w), 1e-4);
    float halfNdc = size * uProj[1][1] / w;

    bool offscreen = abs(cc.x) > w * (1.0 + halfNdc)
                  || abs(cc.y) > w * (1.0 + halfNdc)
                  || cc.z > cc.w;                 // past the far plane

    if (behind || offscreen || dist > uFarClip) {
        vQuad = vec2(0.0); vCol = vec3(0.0);
        vDist = 0.0; vFade = 0.0; vIntensity = 0.0;
        gl_Position = CULLED;
        return;
    }

    vc.xy += aQuad * size;
    vQuad      = aQuad;
    vCol       = ci.rgb;
    // A larger raster footprint distributes a fixed amount of source light.
    // Without this area correction a distant lamp gains energy as its sprite grows.
    vIntensity = ci.w * (rad * rad) / (size * size);
    vDist      = length(vc.xyz);
    // fade across the last stretch of the far plane: no popping at the skyline
    vFade      = 1.0 - smoothstep(uFarFadeStart, uFarClip, vDist);
    gl_Position = uProj * vc;
}
"""

from .hdr import HDR_CONV, HDR_OUT
from .materials import MATERIAL_LIGHT

LIGHTFIELD_FS = ("""#version 330 core
in vec2  vQuad;
in vec3  vCol;
in float vDist;
in float vFade;
in float vIntensity;
uniform float uNightFactor, uFogNear, uFogFar, uGain, uExposure;
out vec4 FragColor;
""" + HDR_CONV + HDR_OUT + MATERIAL_LIGHT + """
void main() {
    if (uNightFactor < 0.04 || vFade <= 0.0015) discard;
    float d2 = dot(vQuad, vQuad);
    if (d2 > 1.0) discard;                          // round the quad off
    float core = 0.98 * 28.0 * exp(-d2 * 28.0);
    float halo = 0.02 * 5.5 * exp(-d2 * 5.5);
    vec3 col = emissionNits(vCol, max(vIntensity, 0.0)) * (core + halo);
    float near = smoothstep(25.0, 90.0, vDist);    // resolved nearby lights use their actual lens mesh
    float fog = cityFog(vDist, uAirCameraHeight, uFogNear, uFogFar);
    col *= uGain * near * uNightFactor * (1.0 - fog);
    FragColor = vec4(sceneOut(col), vFade);
}
""")


def lightfield_shaders(ssbo):
    """Vertex + fragment source for the requested backend."""
    return _LIGHT_HEAD_SSBO + _LIGHT_VS_BODY if ssbo else _LIGHT_HEAD_ATTR + _LIGHT_VS_BODY, LIGHTFIELD_FS


LIGHT_RECORD_STRIDE = 8          # floats per light: pos.xyz, radius, rgb, intensity
