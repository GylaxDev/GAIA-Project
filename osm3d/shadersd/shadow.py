SHADOW_PCF = """
uniform mat4 uLightSpaceMatrix;
uniform sampler2DShadow uShadowMap;
uniform float uShadowTexel;
uniform float uShadowTexelWorld;
uniform float uShadowBias;
uniform float uShadowStrength;
float shadowPCF(vec3 world, vec3 N, vec3 L) {
    if (uShadowStrength <= 0.0) return 1.0;
    vec4 lp = uLightSpaceMatrix * vec4(world + N * (uShadowTexelWorld * 1.6), 1.0);
    vec3 p = lp.xyz / lp.w * 0.5 + 0.5;
    vec2 e = min(p.xy, 1.0 - p.xy);
    float inside = smoothstep(0.0, uShadowTexel * 8.0, min(e.x, e.y));
    inside *= step(0.0, p.z) * step(p.z, 1.0);
    if (inside <= 0.0) return 1.0;
    float bias = uShadowBias * max(0.35, 1.0 - dot(N, L));
    float ref = p.z - bias;
    float sum = 0.0;
    for (int y = -2; y <= 2; ++y)
        for (int x = -2; x <= 2; ++x)
            sum += texture(uShadowMap, vec3(p.xy + vec2(x, y) * uShadowTexel, ref));
    return mix(1.0, sum * (1.0 / 25.0), uShadowStrength * inside);
}
"""

SHADOW_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uLightSpaceMatrix;
uniform vec3 uOffset;
uniform vec2 uScale;
uniform float uGrow;
void main() {
    vec3 p = vec3(aPos.xy * uScale, aPos.z * uGrow) + uOffset;
    gl_Position = uLightSpaceMatrix * vec4(p, 1.0);
}
"""

SHADOW_FS = """
#version 330 core
void main() {}
"""

SHADOW_TREE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=4) in vec3 iPos;
layout(location=5) in vec2 iScaleRot;
uniform mat4 uLightSpaceMatrix;
uniform vec3 uOffset;
uniform vec2 uScale;
uniform float uGrow;
void main() {
    float c = cos(iScaleRot.y), s = sin(iScaleRot.y);
    mat3 rot = mat3(c, s, 0, -s, c, 0, 0, 0, 1);
    vec3 local = rot * (aPos * iScaleRot.x) + iPos;
    vec3 p = vec3(local.xy * uScale, local.z * uGrow) + uOffset;
    gl_Position = uLightSpaceMatrix * vec4(p, 1.0);
}
"""