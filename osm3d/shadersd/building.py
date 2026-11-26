from .shadow import SHADOW_PCF

BUILDING_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
layout(location=3) in float aAO;
layout(location=4) in vec3 aStyle;
layout(location=5) in vec2 aUV;
uniform mat4 uView, uProj;
uniform vec3 uOffset;
uniform vec2 uScale;
uniform float uGrow;
out vec3 vWorld; out vec3 vNormal; out vec3 vColor; out float vAO; out vec3 vStyle; out float vLocalZ; out vec2 vUV;
void main() {
    vec3 p = vec3(aPos.xy * uScale, aPos.z * uGrow) + uOffset;
    vWorld = p; vNormal = aNormal; vColor = aColor; vAO = aAO; vStyle = aStyle; vLocalZ = aPos.z; vUV = aUV;
    gl_Position = uProj * uView * vec4(p, 1.0);
}
"""

BUILDING_FS = ("""
#version 330 core
in vec3 vWorld; in vec3 vNormal; in vec3 vColor; in float vAO;
in vec3 vStyle; in float vLocalZ; in vec2 vUV;
uniform vec3 uCamPos, uSunDir, uFogColor;
uniform vec3 uSunColor, uAmbColor;
uniform float uFogNear, uFogFar;
uniform float uNightFactor;
out vec4 FragColor;
""" + SHADOW_PCF + """
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
const float PI = 3.14159265;
vec3 spec_ggx(vec3 N, vec3 V, vec3 L, vec3 F0, float rough) {
    vec3 H = normalize(L + V);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    float a = rough * rough;
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * d * d);
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    float G = (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    return (D * G) * F / (4.0 * NdotV * NdotL + 0.001);
}
void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    float sf = shadowPCF(vWorld, N, L);
    float theme = vStyle.y;
    float seed  = vStyle.x;
    float bH    = vStyle.z;
    float localZ = vLocalZ;
    float u = vUV.x;
    float L_wall = vUV.y;
    
    vec3  albedo = vColor;
    float rough  = 0.8;
    float metal  = 0.0;
    float emisS  = 0.0;
    vec3  emisC  = vec3(0.0);
    bool is_wall = abs(N.z) < 0.55;
    bool is_roof = N.z > 0.55;
    
    if (is_roof) {
        albedo = vColor * 0.55 + vec3(0.05);
        rough = 0.9; metal = 0.0;
    } else if (is_wall) {
        if (theme < 0.5) { albedo = mix(vColor, vec3(0.2, 0.3, 0.4), 0.5); rough = 0.1; metal = 0.9; }
        else if (theme < 1.5) { rough = 0.9; metal = 0.0; }
        else if (theme < 2.5) { rough = 0.7; metal = 0.1; }
        else if (theme < 3.5) { rough = 0.5; metal = 0.6; }
        
        if (localZ > 0.3 && localZ < 4.2) {
            if (theme > 1.5 && theme < 2.5) {
                float store = step(0.35, fract(u / 4.5));
                albedo = mix(vec3(0.04, 0.04, 0.06), vec3(0.85, 0.78, 0.64), store);
                rough = mix(0.9, 0.3, store); metal = mix(0.0, 0.15, store);
                emisS = store * 0.75; emisC = vec3(1.0, 0.85, 0.6);
            } else {
                float door = step(0.72, fract(u / 5.0));
                albedo *= mix(1.0, 0.55, door);
                rough = mix(rough, 0.6, door);
            }
        } else if (localZ > 4.2) {
            float uStart = 1.5;
            float uEnd = max(uStart, L_wall - 1.5);
            float uLocal = u - uStart;
            
            vec2 cell = vec2(uLocal, localZ) / vec2(2.5, 3.5);
            vec2 frac = fract(cell);
            vec2 cid  = floor(cell);
            
            float wx = smoothstep(0.3, 0.35, frac.x) * smoothstep(0.7, 0.65, frac.x);
            float wy = smoothstep(0.2, 0.25, frac.y) * smoothstep(0.8, 0.75, frac.y);
            float windowMask = wx * wy * step(uStart, u) * step(u, uEnd);
            
            float line = smoothstep(0.04, 0.0, frac.y) + smoothstep(0.96, 1.0, frac.y);
            vec3 glass = (theme > 0.5 && theme < 1.5) ? vec3(0.22, 0.28, 0.35) : vec3(0.18, 0.22, 0.28);
            float inset = smoothstep(0.0, 0.1, frac.x) * smoothstep(1.0, 0.9, frac.x) * smoothstep(0.0, 0.1, frac.y) * smoothstep(1.0, 0.9, frac.y);
            
            albedo = mix(albedo, glass * mix(0.5, 1.0, inset), windowMask);
            rough  = mix(rough, 0.05, windowMask);
            metal  = mix(metal, 0.8, windowMask);
            
            float lit  = step(0.55, hash21(cid + vec2(seed * 137.0)));
            vec3 warm  = mix(vec3(1.0, 0.72, 0.42), vec3(0.9, 0.9, 0.82), hash21(cid + vec2(seed * 61.0 + 7.0)));
            emisS = windowMask * lit;
            emisC = warm;
            albedo *= 1.0 - line * 0.35;
        }
        if (localZ > bH - 0.7 && localZ < bH + 0.02) {
            albedo = mix(vColor, vec3(0.62, 0.60, 0.58), 0.45) * 1.05;
            rough = 0.72; metal = 0.0; emisS = 0.0;
        }
    }
    vec3 F0 = mix(vec3(0.04), albedo, metal);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    vec3 spec = spec_ggx(N, V, L, F0, rough);
    vec3 kD   = (1.0 - F0) * (1.0 - metal);
    vec3 diff = kD * albedo / PI;
    vec3 direct = (diff + spec) * NdotL * uSunColor * sf;
    float streetAO = mix(0.45, 1.0, smoothstep(0.0, 25.0, vWorld.z));
    float ao = vAO * mix(streetAO, 1.0, clamp(N.z, 0.0, 1.0));
    vec3 amb = uAmbColor * mix(0.55, 1.0, N.z * 0.5 + 0.5);
    vec3 lit = direct * mix(0.55, 1.0, ao) + albedo * amb * ao * 1.5;
    vec3 emissive = emisC * emisS * uNightFactor * 3.0;
    
    if (metal > 0.5) emissive += F0 * uAmbColor * 0.5 * (1.0 - uNightFactor);
        
    lit += emissive;
    lit += pow(1.0 - NdotV, 4.0) * uAmbColor * 0.35 * ao;
    float f = smoothstep(uFogNear, uFogFar, length(uCamPos - vWorld));
    FragColor = vec4(mix(lit, uFogColor, f), 1.0);
}
""")