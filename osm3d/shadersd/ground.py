from .shadow import SHADOW_PCF

GROUND_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uView, uProj;
uniform vec2 uCenter;
uniform float uSize;
out vec3 vWorld;
void main() {
    vec3 p = vec3(aPos.xy * uSize + uCenter, aPos.z);
    vWorld = p;
    gl_Position = uProj * uView * vec4(p, 1.0);
}
"""

GROUND_FS = ("""
#version 330 core
in vec3 vWorld;
uniform vec3 uCamPos, uFogColor;
uniform vec3 uSunDir, uSunColor, uAmbColor;
uniform vec2 uCenter;
uniform float uFogNear, uFogFar, uRadius;
out vec4 FragColor;
""" + SHADOW_PCF + """
void main() {
    vec3 L = normalize(uSunDir);
    float sf = shadowPCF(vWorld, vec3(0.0, 0.0, 1.0), L);
    float sunUp = clamp(L.z, 0.0, 1.0);
    vec3 base_col = vec3(0.12, 0.14, 0.10);
    vec3 lit = base_col * (uAmbColor * 1.0 + uSunColor * sunUp * sf * 0.318);
    float f = smoothstep(uFogNear, uFogFar, length(uCamPos - vWorld));
    FragColor = vec4(mix(lit, uFogColor, f), 1.0);
}
""")

TERRAIN_FS = ("""
#version 330 core
in vec3 vWorld; in vec3 vNormal; in vec3 vColor; in vec2 vUV;
uniform vec3 uCamPos, uSunDir, uFogColor;
uniform vec3 uSunColor, uAmbColor;
uniform float uFogNear, uFogFar;
out vec4 FragColor;
""" + SHADOW_PCF + """
void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uSunDir);
    float sf = shadowPCF(vWorld, N, L);
    float NdotL = max(dot(N, L), 0.0);
    
    vec3 albedo = vColor;
    if (vColor.r < 0.1 && vColor.g < 0.1) {
        albedo = vec3(0.08, 0.08, 0.09);
        float isCenter = step(0.48, vUV.x) * step(vUV.x, 0.52);
        float isDash = step(0.5, fract(vUV.y / 10.0));
        albedo += vec3(0.8, 0.7, 0.1) * isCenter * isDash;
        if (vUV.x >= 0.0 && vUV.x <= 1.0) {
            float isEdge = step(vUV.x, 0.05) + step(0.95, vUV.x);
            albedo += vec3(0.8, 0.8, 0.8) * isEdge;
        }
    } else {
        albedo = mix(albedo, vec3(0.15, 0.25, 0.10), 0.3);
    }
    
    vec3 lit = albedo * (uAmbColor * 1.0 + uSunColor * NdotL * sf * 0.318);
    float f = smoothstep(uFogNear, uFogFar, length(uCamPos - vWorld));
    FragColor = vec4(mix(lit, uFogColor, f), 1.0);
}
""")

WATER_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=5) in vec2 aUV;
uniform mat4 uView, uProj;
uniform vec3 uOffset;
uniform vec2 uScale;
out vec3 vWorld;
void main() {
    vec3 p = vec3(aPos.xy * uScale, aPos.z) + uOffset;
    vWorld = p;
    gl_Position = uProj * uView * vec4(p, 1.0);
}
"""

WATER_FS = """
#version 330 core
in vec3 vWorld;
uniform vec3 uCamPos, uSunDir, uFogColor;
uniform vec3 uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uTime;
out vec4 FragColor;
void main() {
    vec3 N = normalize(vec3(
        sin(vWorld.x * 0.1 + uTime) * 0.02 + sin(vWorld.y * 0.15 - uTime * 1.2) * 0.02,
        cos(vWorld.y * 0.1 + uTime * 0.8) * 0.02 + cos(vWorld.x * 0.15 - uTime) * 0.02,
        1.0
    ));
    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    
    float ndv = max(dot(N, V), 0.0);
    float fresnel = 0.06 + 0.94 * pow(1.0 - ndv, 4.0);      // strong sky mirror at grazing angles
    float amb = clamp(dot(uAmbColor, vec3(0.333)) * 1.8, 0.12, 1.0);
    float diff = max(dot(N, normalize(uSunDir)), 0.0);
    vec3 deep = vec3(0.05, 0.22, 0.36);                        // visibly blue, unlike the dark ground
    vec3 diffuse = deep * (uSunColor * diff * 0.6 + uAmbColor * 1.6);
    vec3 skyReflection = mix(uFogColor, vec3(0.50, 0.68, 0.90), 0.45) * amb;
    vec3 H = normalize(normalize(uSunDir) + V);
    float spec = pow(max(dot(N, H), 0.0), 90.0);
    vec3 col = mix(diffuse, skyReflection, fresnel);
    col += uSunColor * spec * 0.6;

    float f = smoothstep(uFogNear, uFogFar, length(uCamPos - vWorld));
    FragColor = vec4(mix(col, uFogColor, f), 1.0);
}
"""

LINE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
layout(location=5) in vec2 aUV;
uniform mat4 uView, uProj;
uniform vec3 uOffset;
uniform vec2 uScale;
uniform float uGrow;
out vec3 vWorld; out vec3 vColor;
void main() {
    vec3 p = vec3(aPos.xy * uScale, aPos.z * uGrow) + uOffset;
    vWorld = p; vColor = aColor;
    gl_Position = uProj * uView * vec4(p, 1.0);
}
"""

LINE_FS = """
#version 330 core
in vec3 vWorld; in vec3 vColor;
uniform vec3 uCamPos, uFogColor;
uniform float uFogNear, uFogFar;
out vec4 FragColor;
void main() {
    float f = smoothstep(uFogNear, uFogFar, length(uCamPos - vWorld));
    FragColor = vec4(mix(vColor, uFogColor, f), 1.0);
}
"""