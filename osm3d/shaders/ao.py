"""Half-resolution, normal-aware screen-space contact occlusion (12 depth taps)."""
from .depth import SCENE_DEPTH

AO_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uDepth, uMat;
uniform mat4 uInvProj, uView, uProj;
uniform vec2 uResolution;
uniform float uRadius;
out vec4 FragColor;
""" + SCENE_DEPTH + """
vec3 viewPos(vec2 uv, float d) {
    vec4 p = uInvProj * vec4(uv * 2.0 - 1.0, depthNdc(d), 1.0);
    return p.xyz / p.w;
}
void main() {
    vec2 uv = (floor(vUV * uResolution) + 0.5) / uResolution;
    float d = texture(uDepth, uv).r;
    FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    if (depthSky(d)) return;
    vec3 P = viewPos(uv, d);
    float z = -P.z;
    FragColor.g = z;
    float fade = 1.0 - smoothstep(300.0, 650.0, z);
    if (fade <= 0.0) return;
    vec3 N = normalize(mat3(uView) * (texture(uMat, uv).rgb * 2.0 - 1.0));
    float radiusPx = min(64.0, uRadius * uProj[1][1] * uResolution.y * 0.5 / max(z, 0.1));
    if (radiusPx < 1.0) return;
    float rotation = fract(dot(floor(gl_FragCoord.xy), vec2(0.75487766, 0.56984029))) * 6.2831853;
    float occ = 0.0;
    for (int i = 0; i < 12; ++i) {
        float a = rotation + float(i) * 2.39996323;
        float r = sqrt((float(i) + 0.5) / 12.0);
        vec2 qUV = uv + vec2(cos(a), sin(a)) * radiusPx * r / uResolution;
        if (any(lessThan(qUV, vec2(0.0))) || any(greaterThan(qUV, vec2(1.0)))) continue;
        // Reconstruct at the depth texel's centre. Using the unsnapped ray with
        // nearest depth invents small ridges on otherwise perfectly flat walls.
        qUV = (floor(qUV * uResolution) + 0.5) / uResolution;
        float qd = texture(uDepth, qUV).r;
        if (depthSky(qd)) continue;
        vec3 delta = viewPos(qUV, qd) - P;
        float len2 = dot(delta, delta);
        float horizon = max(dot(N, delta) * inversesqrt(max(len2, 0.0001)) - 0.08, 0.0);
        occ += horizon * max(1.0 - len2 / (uRadius * uRadius), 0.0);
    }
    FragColor.r = clamp(occ * (2.8 / 12.0), 0.0, 0.75) * fade;
}
"""

AO_RESOLVE = """
uniform sampler2D uAO;
uniform float uAOOn, uAOStrength;
// Small spatial filter + plane-aware upsample. Comparing against the receiver's
// plane (rather than its centre depth) also smooths oblique facades and streets.
float contactAO(vec3 P, vec3 N) {
    vec2 size = vec2(textureSize(uAO, 0));
    vec2 p = vUV * size - 0.5;
    vec2 base = floor(p + 0.5);
    float sum = 0.0, weight = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 offset = vec2(float(i), float(j));
            vec2 uv = (base + offset + 0.5) / size;
            if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) continue;
            vec2 sampleAO = texture(uAO, uv).rg;
            vec2 depthUV = (floor(uv * uResolution) + 0.5) / uResolution;
            vec3 ray = vec3((depthUV * 2.0 - 1.0) / vec2(uProj[0][0], uProj[1][1]), -1.0);
            vec3 delta = ray * sampleAO.y - P;
            float planeError = abs(dot(N, delta));
            vec2 pixelDelta = base + offset - p;
            float w = exp(-dot(pixelDelta, pixelDelta) * 0.8
                          - planeError / max(0.08, -P.z * 0.002));
            w *= step(0.001, sampleAO.y);
            sum += sampleAO.x * w;
            weight += w;
        }
    }
    return weight > 1e-4 ? sum / weight * uAOStrength : 0.0;
}
"""
