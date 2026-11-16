# ------------------------------------------------------------------------------
# SHADOW COMMON
# ------------------------------------------------------------------------------
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
SHADOW_FS = "#version 330 core\nvoid main() {}"
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

# ------------------------------------------------------------------------------
# BUILDING (Architectural Pipeline)
# ------------------------------------------------------------------------------
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
    
    float theme = vStyle.y;    // 0=Glass, 1=Masonry, 2=Brick, 3=Industrial, 4=Roof, 5=Spire
    float seed  = vStyle.x;
    float bH    = vStyle.z;
    
    // Wall coordinates
    float u = vUV.x;
    float L_wall = max(0.001, vUV.y);
    
    vec3 albedo = vColor;
    float rough = 0.8;
    float metal = 0.0;
    float emisS = 0.0;
    vec3 emisC = vec3(0.0);
    
    bool is_wall = abs(N.z) < 0.55;
    bool is_roof = N.z > 0.55;
    
    // Default roof material
    if (is_roof) {
        albedo = vColor * 0.55 + vec3(0.05);
        rough = 0.9; metal = 0.0;
    } 
    // Spire & Asset material (Metal)
    else if (theme > 4.5) {
        albedo = vec3(0.6, 0.65, 0.7);
        rough = 0.2; metal = 1.0;
        emisS = uNightFactor * 1.5;
        emisC = vec3(0.8, 0.9, 1.0); // Aviation warning lights / spire glow
    }
    else if (is_wall) {
        // -------------------------------------------------------------
        // MATERIAL PALETTES
        // -------------------------------------------------------------
        if (theme < 0.5) {
            // Modern Glass Skyscraper: Dark blue, tight mullions, reflective
            albedo = mix(vColor, vec3(0.1, 0.15, 0.22), 0.7);
            rough = 0.1; metal = 0.9;
        } else if (theme < 1.5) {
            // Classic Masonry: Warm granite, vertical columns
            albedo = mix(vColor, vec3(0.75, 0.68, 0.6), 0.4);
            rough = 0.85; metal = 0.0;
        } else if (theme < 2.5) {
            // Brick: Terracotta, high roughness
            albedo = mix(vColor, vec3(0.55, 0.25, 0.2), 0.6);
            rough = 0.95; metal = 0.0;
        } else {
            // Industrial: Pale concrete / steel
            albedo = mix(vColor, vec3(0.5, 0.5, 0.5), 0.5);
            rough = 0.6; metal = 0.3;
        }

        // -------------------------------------------------------------
        // WINDOW & FACADE GEOMETRY (Procedural)
        // -------------------------------------------------------------
        if (vLocalZ > 0.3 && vLocalZ < 4.2) {
            // Ground floor retail
            float store = step(0.2, fract(u / 6.0));
            albedo = mix(vec3(0.04, 0.04, 0.06), vec3(0.85, 0.78, 0.64), store);
            rough = mix(0.9, 0.3, store); metal = mix(0.0, 0.15, store);
            emisS = store * 0.75; emisC = vec3(1.0, 0.85, 0.6);
        } else if (vLocalZ > 4.2) {
            // Upper floors
            float uStart = 1.5;
            float uEnd = max(uStart, L_wall - 1.5);
            
            // Window grid sizing depends on material
            vec2 grid_size = (theme < 0.5) ? vec2(1.5, 3.5) : vec2(2.5, 3.5);
            vec2 cell = vec2(u - uStart, vLocalZ) / grid_size;
            vec2 frac = fract(cell);
            vec2 cid  = floor(cell);
            
            // Window masks based on theme
            float wx, wy;
            if (theme < 0.5) {
                // Glass: Almost continuous horizontal bands
                wx = smoothstep(0.05, 0.1, frac.x) * smoothstep(0.95, 0.9, frac.x);
                wy = smoothstep(0.1, 0.15, frac.y) * smoothstep(0.9, 0.85, frac.y);
            } else {
                // Punched windows for masonry/brick
                wx = smoothstep(0.3, 0.35, frac.x) * smoothstep(0.7, 0.65, frac.x);
                wy = smoothstep(0.25, 0.3, frac.y) * smoothstep(0.75, 0.7, frac.y);
            }
            float windowMask = wx * wy * step(uStart, u) * step(u, uEnd);
            
            // Add 3D depth: darken edges, create horizontal floor ledges
            float is_ledge = smoothstep(0.06, 0.0, frac.y) + smoothstep(0.94, 1.0, frac.y);
            float is_mullion = smoothstep(0.04, 0.0, frac.x);
            
            vec3 glass = (theme < 1.5) ? vec3(0.22, 0.28, 0.35) : vec3(0.1, 0.12, 0.15);
            albedo = mix(albedo, glass, windowMask);
            rough  = mix(rough, 0.05, windowMask);
            metal  = mix(metal, 0.8, windowMask);
            
            // Fake AO / Depth for ledges
            albedo *= 1.0 - (is_ledge * 0.4 + is_mullion * 0.2);
            
            // -------------------------------------------------------------
            // DYNAMIC WINDOW LIGHTING (Non-uniform)
            // -------------------------------------------------------------
            float litRand = hash21(cid + vec2(seed * 137.0));
            float lit = step(0.45, litRand); // ~55% of windows lit
            
            // Mix of warm tungsten, bright LED, and cool office lights
            float typeRand = hash21(cid + vec2(seed * 61.0 + 7.0));
            vec3 lightColor = mix(vec3(1.0, 0.75, 0.4), vec3(0.9, 0.95, 1.0), typeRand);
            if (hash21(cid + vec2(seed * 99.0)) > 0.85) lightColor = vec3(0.4, 0.8, 1.0); // Cyan LED
            
            emisS = windowMask * lit;
            emisC = lightColor;
            
            // Randomly turn off some windows entirely for variation
            if (litRand < 0.1) emisS = 0.0;
        }
        
        // Crown Lighting for Skyscrapers
        if (vLocalZ > bH * 0.95 && bH > 40.0 && theme < 2.5) {
            float crownGlow = smoothstep(bH * 0.95, bH, vLocalZ);
            emisS = max(emisS, crownGlow);
            emisC = mix(vec3(1.0, 0.8, 0.5), vec3(0.8, 0.9, 1.0), step(0.5, hash21(vec2(seed))));
        }
    }
    
    // PBR Lighting
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
    lit += emisC * emisS * uNightFactor * 3.5;
    
    if (metal > 0.5) lit += F0 * uAmbColor * 0.5 * (1.0 - uNightFactor);
    lit += pow(1.0 - NdotV, 4.0) * uAmbColor * 0.35 * ao;
    
    float f = smoothstep(uFogNear, uFogFar, length(uCamPos - vWorld));
    FragColor = vec4(mix(lit, uFogColor, f), 1.0);
}
""")

# ------------------------------------------------------------------------------
# GROUND, WATER, TERRAIN, SKY, LINE, TREE, OVERLAY
# ------------------------------------------------------------------------------
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
uniform vec3 uCamPos, uFogColor, uSunDir, uSunColor, uAmbColor;
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
uniform vec3 uCamPos, uSunDir, uFogColor, uSunColor, uAmbColor;
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
uniform vec3 uCamPos, uSunDir, uFogColor, uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uTime;
out vec4 FragColor;
void main() {
    vec3 N = normalize(vec3(
        sin(vWorld.x * 0.08 + uTime * 0.7) * 0.035 + sin(vWorld.y * 0.13 - uTime * 1.1) * 0.025,
        cos(vWorld.y * 0.09 + uTime * 0.9) * 0.035 + cos(vWorld.x * 0.11 - uTime * 1.3) * 0.025,
        1.0
    ));
    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    vec3 deep = vec3(0.059, 0.169, 0.282);
    vec3 shallow = vec3(0.114, 0.325, 0.439);
    float ndv = max(dot(N, V), 0.0);
    float fresnel = 0.06 + 0.94 * pow(1.0 - ndv, 4.0);
    float amb = clamp(dot(uAmbColor, vec3(0.333)) * 1.8, 0.12, 1.0);
    float diff = max(dot(N, L), 0.0);
    vec3 body = mix(deep, shallow, clamp(1.0 - ndv, 0.0, 1.0) * 0.65);
    vec3 diffuse = body * (uSunColor * diff * 0.85 + uAmbColor * 2.1);
    vec3 skyReflection = mix(uFogColor, vec3(0.50, 0.68, 0.90), 0.45) * amb;
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 120.0);
    vec3 col = mix(diffuse, skyReflection, fresnel);
    col += uSunColor * spec * 0.85 + uSunColor * pow(max(dot(N, H), 0.0), 12.0) * 0.06;
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
uniform vec3 uCamPos, uSunDir, uFogColor, uSunColor, uAmbColor;
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
SKY_VS = """
#version 330 core
layout(location=0) in vec2 aPos;
uniform mat4 uInvViewProj;
uniform vec3 uCamPos;
out vec3 vRayDir;
void main() {
    gl_Position = vec4(aPos, 0.9999, 1.0);
    vec4 p = uInvViewProj * vec4(aPos, 1.0, 1.0);
    vRayDir = normalize(p.xyz / p.w - uCamPos);
}
"""
SKY_FS = """
#version 330 core
in vec3 vRayDir;
uniform vec3 uCamPos, uSunDir, uMoonDir, uFogColor;
uniform float uSunIntensity, uMoonIntensity, uMoonPhase, uNightFactor, uTime;
out vec4 FragColor;
const float PI = 3.14159265;
const float RE = 6371e3;
const float RA = 6471e3;
const float HR = 8000.0;
const float HM = 1200.0;
const vec3  BETA_R = vec3(5.8e-6, 13.5e-6, 33.1e-6);
const float BETA_M = 21e-6;
float rayleighPhase(float mu) { return 3.0 / (16.0 * PI) * (1.0 + mu * mu); }
float miePhase(float mu, float g) {
    float g2 = g * g;
    return (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + mu * mu)) / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}
vec3 atmosphere(vec3 viewDir, vec3 sunDir, float sunIntensity) {
    vec3 camPos = vec3(0.0, 0.0, RE + 100.0);
    float b = dot(camPos, viewDir);
    float c = dot(camPos, camPos) - RA * RA;
    float disc = b * b - c;
    if (disc < 0.0) return vec3(0.0);
    float tMax = -b + sqrt(disc);
    if (viewDir.z < 0.0) {
        float tEarth = -b - sqrt(max(0.0, b * b - (dot(camPos, camPos) - RE * RE)));
        tMax = min(tMax, tEarth);
    }
    const int N_VIEW = 12; const int N_SUN = 6;
    float ds = tMax / float(N_VIEW);
    vec3 sumR = vec3(0.0); vec3 sumM = vec3(0.0);
    float opticalR = 0.0; float opticalM = 0.0;
    for (int i = 0; i < N_VIEW; i++) {
        vec3 p = camPos + viewDir * (float(i) + 0.5) * ds;
        float height = length(p) - RE;
        float hr = exp(-height / HR) * ds; float hm = exp(-height / HM) * ds;
        opticalR += hr; opticalM += hm;
        float sunB = dot(p, sunDir); float sunC = dot(p, p) - RE * RE;
        float sunDisc = sunB * sunB - sunC;
        if (sunDisc > 0.0 && sunB < 0.0) continue;
        sunC = dot(p, p) - RA * RA;
        sunDisc = sunB * sunB - sunC;
        float sunTMax = -sunB + sqrt(max(0.0, sunDisc));
        float sunDs = sunTMax / float(N_SUN);
        float sunOptR = 0.0; float sunOptM = 0.0;
        for (int j = 0; j < N_SUN; j++) {
            vec3 ps = p + sunDir * (float(j) + 0.5) * sunDs;
            float hs = max(0.0, length(ps) - RE);
            sunOptR += exp(-hs / HR) * sunDs; sunOptM += exp(-hs / HM) * sunDs;
        }
        vec3 tau = BETA_R * (opticalR + sunOptR) + BETA_M * 1.1 * (opticalM + sunOptM);
        vec3 attn = exp(-tau);
        sumR += hr * attn; sumM += hm * attn;
    }
    float mu = dot(viewDir, sunDir);
    float phaseR = rayleighPhase(mu); float phaseM = miePhase(mu, 0.65);
    vec3 color = sumR * BETA_R * phaseR + sumM * BETA_M * phaseM;
    return sunIntensity * color * 22.0;
}
float starField(vec3 dir) {
    vec3 scaled = dir * 350.0; vec3 cell = floor(scaled); vec3 f = fract(scaled) - 0.5;
    float h = fract(sin(dot(cell, vec3(127.1, 311.7, 74.7))) * 43758.5453);
    return step(0.9965, h) * smoothstep(0.42, 0.0, length(f));
}
float milkyWay(vec3 dir) {
    vec3 axis = normalize(vec3(0.35, 0.45, 0.82));
    float band = exp(-pow(dot(dir, axis), 2.0) * 45.0);
    float texture = 0.4 + 0.6 * sin(atan(dir.y, dir.x) * 6.0 + 1.2);
    return band * texture;
}
void main() {
    vec3 dir = normalize(vRayDir);
    vec3 sunDir = normalize(uSunDir); vec3 moonDir = normalize(uMoonDir);
    vec3 sky = atmosphere(dir, sunDir, uSunIntensity);
    if (uMoonIntensity > 0.0) {
        float moonMu = dot(dir, moonDir);
        sky += vec3(0.35, 0.40, 0.55) * uMoonIntensity * 0.05 * pow(max(moonMu, 0.0), 4.0);
    }
    float sunDot = max(dot(dir, sunDir), 0.0);
    float sunDiscMask = smoothstep(0.99990, 0.99997, sunDot);
    float sunGlow = pow(sunDot, 180.0); float sunHalo = pow(sunDot, 14.0) * 0.35;
    float sunUp = smoothstep(-0.06, 0.04, sunDir.z);
    sky += vec3(1.6, 1.35, 1.0) * (sunDiscMask * 25.0 + sunGlow * 2.0 + sunHalo) * sunUp;
    float moonDot = max(dot(dir, moonDir), 0.0);
    float moonDiscMask = smoothstep(0.99968, 0.99985, moonDot);
    float moonGlow = pow(moonDot, 220.0) * 0.6;
    float moonUp = smoothstep(-0.06, 0.04, moonDir.z);
    sky += vec3(0.92, 0.94, 1.05) * (moonDiscMask * uMoonPhase * 3.0 + moonGlow * uMoonPhase) * moonUp;
    if (uNightFactor > 0.001) {
        float twinkle = 0.85 + 0.15 * sin(uTime * 2.7 + dot(dir, sunDir) * 40.0);
        sky += vec3(0.9, 0.95, 1.05) * starField(dir) * twinkle * uNightFactor * 3.0;
        sky += vec3(0.10, 0.12, 0.20) * milkyWay(dir) * uNightFactor * 0.25;
    }
    float horizonFade = smoothstep(-0.05, 0.05, dir.z);
    sky = mix(uFogColor, sky, horizonFade);
    FragColor = vec4(sky, 1.0);
}
"""