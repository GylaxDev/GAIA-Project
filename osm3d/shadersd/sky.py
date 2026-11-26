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
uniform vec3  uCamPos;
uniform vec3  uSunDir;
uniform vec3  uMoonDir;
uniform vec3  uFogColor;
uniform float uSunIntensity;
uniform float uMoonIntensity;
uniform float uMoonPhase;
uniform float uNightFactor;
uniform float uTime;
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
    return (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + mu * mu))
         / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * mu, 1.5));
}

vec3 atmosphere(vec3 viewDir, vec3 sunDir, float sunIntensity) {
    vec3 camPos = vec3(0.0, 0.0, RE + 100.0); // Z-up
    float b = dot(camPos, viewDir);
    float c = dot(camPos, camPos) - RA * RA;
    float disc = b * b - c;
    if (disc < 0.0) return vec3(0.0);
    float tMax = -b + sqrt(disc);
    
    if (viewDir.z < 0.0) {
        float tEarth = -b - sqrt(max(0.0, b * b - (dot(camPos, camPos) - RE * RE)));
        tMax = min(tMax, tEarth);
    }

    const int N_VIEW = 12;
    const int N_SUN  = 6;
    float ds = tMax / float(N_VIEW);
    vec3 sumR = vec3(0.0); vec3 sumM = vec3(0.0);
    float opticalR = 0.0; float opticalM = 0.0;

    for (int i = 0; i < N_VIEW; i++) {
        vec3 p = camPos + viewDir * (float(i) + 0.5) * ds;
        float height = length(p) - RE;
        float hr = exp(-height / HR) * ds;
        float hm = exp(-height / HM) * ds;
        opticalR += hr; opticalM += hm;

        float sunB = dot(p, sunDir);
        float sunC = dot(p, p) - RE * RE;
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
            sunOptR += exp(-hs / HR) * sunDs;
            sunOptM += exp(-hs / HM) * sunDs;
        }

        vec3 tau = BETA_R * (opticalR + sunOptR) + BETA_M * 1.1 * (opticalM + sunOptM);
        vec3 attn = exp(-tau);
        sumR += hr * attn; sumM += hm * attn;
    }

    float mu = dot(viewDir, sunDir);
    float phaseR = rayleighPhase(mu);
    float phaseM = miePhase(mu, 0.65);

    vec3 color = sumR * BETA_R * phaseR + sumM * BETA_M * phaseM;
    return sunIntensity * color * 22.0;
}

float starField(vec3 dir) {
    vec3 scaled = dir * 350.0;
    vec3 cell = floor(scaled);
    vec3 f = fract(scaled) - 0.5;
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
    vec3 sunDir = normalize(uSunDir);
    vec3 moonDir = normalize(uMoonDir);
    vec3 sky = atmosphere(dir, sunDir, uSunIntensity);
    
    if (uMoonIntensity > 0.0) {
        float moonMu = dot(dir, moonDir);
        sky += vec3(0.35, 0.40, 0.55) * uMoonIntensity * 0.05 * pow(max(moonMu, 0.0), 4.0);
    }
    
    float sunDot = max(dot(dir, sunDir), 0.0);
    float sunDiscMask = smoothstep(0.99990, 0.99997, sunDot);
    float sunGlow     = pow(sunDot, 180.0);
    float sunHalo     = pow(sunDot, 14.0) * 0.35;
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