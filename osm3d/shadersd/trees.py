from .shadow import SHADOW_PCF

TREE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
layout(location=3) in float aPart;
layout(location=4) in vec3 iPos;
layout(location=5) in vec2 iScaleRot;
layout(location=6) in vec3 iTint;
uniform mat4 uView, uProj;
uniform vec3 uOffset;
uniform vec2 uScale;
uniform float uGrow;
out vec3 vWorld; out vec3 vNormal; out vec3 vColor; out float vPart;
void main() {
    float c = cos(iScaleRot.y), s = sin(iScaleRot.y);
    mat3 rot = mat3(c, s, 0, -s, c, 0, 0, 0, 1);
    vec3 local = rot * (aPos * iScaleRot.x) + iPos;
    vec3 p = vec3(local.xy * uScale, local.z * uGrow) + uOffset;
    vWorld = p; vNormal = rot * aNormal; vColor = aColor * iTint; vPart = aPart;
    gl_Position = uProj * uView * vec4(p, 1.0);
}
"""

TREE_FS = ("""
#version 330 core
in vec3 vWorld; in vec3 vNormal; in vec3 vColor; in float vPart;
uniform vec3 uCamPos, uSunDir, uFogColor;
uniform vec3 uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uNightFactor;
out vec4 FragColor;
""" + SHADOW_PCF + """
void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uSunDir);
    float sf = shadowPCF(vWorld, N, L);
    float wrap = vPart > 0.5 ? 0.35 : 0.0;
    float diff = max((dot(N, L) + wrap) / (1.0 + wrap), 0.0);
    float amb = mix(0.55, 1.0, N.z * 0.5 + 0.5);
    vec3 lit = vColor * (uAmbColor * amb * 0.85 + uSunColor * diff * sf * 1.05);
    lit += vColor * uAmbColor * 0.25 * uNightFactor;
    float f = smoothstep(uFogNear, uFogFar, length(uCamPos - vWorld));
    FragColor = vec4(mix(lit, uFogColor, f), 1.0);
}
""")