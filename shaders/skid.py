"""Tyre-mark ribbons: dark rubber on the road, depth-tested, alpha blended."""

SKID_VS = """#version 330 core
layout(location = 0) in vec3  aPos;
layout(location = 1) in float aU;
layout(location = 2) in float aLevel;
layout(location = 3) in float aBirth;
uniform mat4  uView, uProj;
uniform vec3  uCamPos;
uniform float uDepthQ;
uniform float uReverseZ;
out float vU;
out float vLevel;
out float vAge;
out float vDist;
uniform float uNow;
void main() {
    vec4 vp = uView * vec4(aPos, 1.0);
    vec4 clip = uProj * vp;
    // Pull the ribbon a little toward the camera so it never z-fights the road, whatever
    // the depth convention (same trick the glow quads use).
    float zd = max(-vp.z, 0.1);
    float pull = max(0.02, 4.0 * zd * zd * uDepthQ) / zd;
    vp.xyz *= (1.0 - min(pull, 0.5));
    clip = uProj * vp;
    gl_Position = clip;
    vU = aU;
    vLevel = aLevel;
    vAge = uNow - aBirth;
    vDist = length(aPos - uCamPos);
}
"""

SKID_FS = """#version 330 core
in float vU;
in float vLevel;
in float vAge;
in float vDist;
uniform float uWet;
layout(location = 0) out vec4 FragColor;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    // soft rubber edges plus a faint tread-block streak across the width
    float edge = smoothstep(0.0, 0.22, vU) * smoothstep(1.0, 0.78, vU);
    float streak = 0.80 + 0.20 * hash(vec2(floor(vU * 7.0), 3.0));
    float fresh = smoothstep(0.0, 0.5, vLevel);
    float fade = 1.0 - smoothstep(25.0, 110.0, vAge);       // marks wear off with traffic and time
    float far = 1.0 - smoothstep(120.0, 260.0, vDist);
    float a = edge * streak * fresh * fade * far * (0.78 - 0.25 * uWet);
    if (a < 0.01) discard;
    FragColor = vec4(vec3(0.004, 0.004, 0.005), a);
}
"""
