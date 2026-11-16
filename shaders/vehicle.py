# --- shaders/vehicle.py ---
from .hdr import HDR_CONV, HDR_OUT
from .streetspill import STREET_SPILL
from .materials import MATERIAL_LIGHT
from .shadow import SHADOW_PCF
# Single-pass instanced car shader.  The per-vertex `aPart` is the material id (see
# vehicle_models.py): 0 paint | 1 glass | 2 tyre | 3 warm headlight | 4 taillight | 5 chrome/alloy
# | 6 trim | 7 plate | 8 fixed paint | 10 matte plastic | 11 LED signature | 12 piano black
# | 13 panoramic roof glass | 14 cool LED projector | 15 ground-glow quad | 16/17 indicators.
# Flags: lamps=1, brake=2, left=4, right=8; bits 4..11 hold the flash phase.
# Signal furniture uses lens materials 18..21 and state=flags&3, left arrow=16.
VEHICLE_INPUTS = """
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
layout(location=3) in float aPart;

layout(location=4) in vec4 iModel0;
layout(location=5) in vec4 iModel1;
layout(location=6) in vec4 iModel2;
layout(location=7) in vec4 iModel3;
layout(location=8) in vec3 iColor;
layout(location=9) in float iFlags;
layout(location=10) in vec4 aWheel; // pivot xyz, role: static/rear/front/front-caliper
layout(location=11) in vec4 iMotion; // wheel radians, steering radians, speed, visibility

uniform vec3 uOffset;
uniform vec2 uScale;
"""

# Both the colour/depth/SSR pass and the dedicated caster pass use these exact
# wheel, steering and affine transforms. Motion-channel w values are never xyz.
VEHICLE_ANIMATION = """
vec3 spinWheel(vec3 p, float angle) {
    float c = cos(angle), s = sin(angle);
    return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}
vec3 steerWheel(vec3 p, float angle) {
    float c = cos(angle), s = sin(angle);
    return vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

void animateVehicle(inout vec3 position, inout vec3 normal) {
    int role = int(aWheel.w + .5);
    if (role > 0) {
        position -= aWheel.xyz;
        // The caliper steers with the front assembly but never spins with the rim.
        if (role == 1 || role == 2) {
            position = spinWheel(position, iMotion.x);
            normal = spinWheel(normal, iMotion.x);
        }
        if (role == 2 || role == 3) {
            position = steerWheel(position, iMotion.y);
            normal = steerWheel(normal, iMotion.y);
        }
        position += aWheel.xyz;
        // Player suspension: the chassis already carries the sprung body pose in its
        // matrix. Each wheel assembly slides vertically in the body frame by its own
        // travel (FL, FR, RL, RR stored in the unused w entries of the model matrix).
        if ((int(iFlags + .5) & 4096) != 0) {
            float drop = aWheel.x > 0.0 ? (aWheel.y > 0.0 ? iModel0.w : iModel1.w)
                                        : (aWheel.y > 0.0 ? iModel2.w : iModel3.w);
            position.z += drop;
        }
    }
}
vec3 vehicleWorld(vec3 position) {
    mat3 model = mat3(iModel0.xyz, iModel1.xyz, iModel2.xyz);
    vec3 local_p = model * position + iModel3.xyz;
    return vec3(local_p.xy * uScale, local_p.z) + uOffset;
}
"""

# Automotive paint is layered: pigment base, a flake/pearl scatter layer, then a
# clear coat carrying a faint factory "orange peel".  These helpers are analytic
# and single-sample; they approximate, rather than measure, a real paint system.
VEHICLE_PAINT = """
const float COAT_IOR = 1.50;              // clear coat, refractive index
const float COAT_F0  = 0.04;              // ((n - 1) / (n + 1))^2 for n = 1.5

float paintHash(vec3 p) {
    return fract(sin(dot(floor(p), vec3(12.9898, 78.233, 37.719))) * 43758.5453);
}
vec3 paintHash3(vec3 p) {
    vec3 c = floor(p);
    return fract(sin(vec3(dot(c, vec3(12.9898, 78.233, 37.719)),
                          dot(c, vec3(39.3468, 11.1351, 83.1552)),
                          dot(c, vec3(73.1563, 52.2351, 9.1514)))) * 43758.5453);
}

// Slope detail for micro-surfaces (orange peel, tyre grain), in the tangent plane
// of N.  Amplitudes are small angles: the result only tilts shading, never geometry.
vec3 microNormal(vec3 P, vec3 N, float wavelength, float amp) {
    float w = 6.2831853 / max(wavelength, 1e-4);
    vec3 slope = vec3(cos(P.x * w * 0.80 + 1.3) * sin(P.y * w * 0.63 + 2.1),
                      cos(P.y * w * 0.71 + 0.4) * sin(P.z * w * 0.90 + 3.3),
                      cos(P.z * w * 0.66 + 2.7) * sin(P.x * w * 0.57 + 1.1)) * amp;
    return normalize(N - slope + N * dot(slope, N));
}

// Every flake cell gets its own tilt, so only the cells aligned with the
// view/sun pair glint.  Cell size is the representative metallic-flake pitch.
vec3 flakeNormal(vec3 P, vec3 N, float size, float tilt) {
    vec3 h = paintHash3(P / max(size, 1e-4));
    vec3 axis = normalize(h * 2.0 - 1.0 + vec3(1e-3, 2e-3, 3e-3));
    float a = (h.x * 2.0 - 1.0) * tilt;
    float c = cos(a), s = sin(a);
    return normalize(N * c + cross(axis, N) * s + axis * dot(axis, N) * (1.0 - c));
}

// Two-beam interference across the clear-coat/pigment boundary.  Returns a 0..1
// per-channel reflectance modulation; the caller normalizes it into a hue shift
// so the flip-flop changes tint rather than overall exposure.
vec3 thinFilmTint(float cosV, float thicknessNm) {
    float sinT2 = (1.0 - cosV * cosV) / (COAT_IOR * COAT_IOR);
    float cosT = sqrt(max(0.0, 1.0 - sinT2));
    float opd = 2.0 * COAT_IOR * thicknessNm * cosT;               // nanometres
    vec3 phase = 6.2831853 * opd / vec3(680.0, 550.0, 440.0) + 3.14159265;
    return 0.5 + 0.5 * cos(phase);                                  // pi reversal at the denser layer
}

// A stable brush direction: horizontal on faces, radial on horizontal surfaces.
vec3 brushTangent(vec3 N) {
    vec3 up = abs(N.z) > 0.95 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
    return normalize(cross(up, N));
}

// Anisotropic (brushed / machined) highlight: tight across the brush, wide along it.
float brushedHighlight(vec3 N, vec3 V, vec3 L, vec3 T, float streak, float sharp) {
    vec3 H = normalize(V + L);
    float TdH = dot(T, H), BdH = dot(cross(N, T), H);
    float aT = mix(0.62, 0.16, sharp), aB = mix(0.14, 0.025, sharp);
    float d = exp(-(TdH * TdH) / (aT * aT) - (BdH * BdH) / (aB * aB));
    return d * (0.65 + 0.35 * streak);
}
"""

# Lamp optics.  Vehicle lamps used to be one constant emission per material, which at night
# exposure (+10 stops) clipped the whole lamp to a flat white shape.  A real lamp is a stack:
# a tiny, extremely bright light-guide / projector image, a metallised reflector a few hundred
# times dimmer, a dark sealing rim, and a tinted cover over the lot.  vehicle_lamps.py bakes
# each lamp's own coordinates into the vertex colour slot, emitters as (u, v, 2 + depth) and
# lens covers as (emitter proximity near, wide, 2 + depth); b > 1.5 marks a baked vertex.
VEHICLE_LAMPS = """
const vec3 TAIL_AMBER = vec3(1.00, 0.34, 0.10);
const vec3 TAIL_RED   = vec3(1.00, 0.025, 0.010);   // deep red cover: little green, so ACES keeps it red when bright

float lampHash(vec2 c) { return fract(sin(dot(c, vec2(127.1, 311.7))) * 43758.5453); }
float lampNight() { return smoothstep(0.10, 0.60, uNightFactor); }

// Dim, diffuse reflectance of what the lamp is made of (linear).
vec3 lampAlbedo(int part) {
    if (part == 4)  return vec3(0.30, 0.010, 0.010);          // red-metallised tail reflector
    if (part == 16 || part == 17) return vec3(0.62, 0.15, 0.010);   // amber bulb cup
    if (part == 11) return vec3(0.80, 0.84, 0.90);            // LED light guide
    return vec3(0.70, 0.70, 0.67);                            // chromed headlamp reflector / projector
}

// LED chips on a light guide: a hex lattice that only resolves when it spans pixels.
float chipDots(vec2 cell, float fw) {
    vec2 fq = fract(cell + vec2(0.5 * mod(floor(cell.y), 2.0), 0.0)) - 0.5;
    float dotv = 1.0 - smoothstep(0.22, 0.48, length(fq));
    float resolved = 1.0 - smoothstep(0.15, 0.60, fw);
    return mix(0.82, 0.55 + 0.55 * dotv, resolved);
}

// Every pattern returns (hot, halo, fill):
//   hot  - the chips / filament image.  Clips at night exposure and feeds the bloom.
//   halo - light bleeding around it through the cover.
//   fill - the reflector's own faint facets, the only part still readable at night exposure.
// c = (u, v, 2 + depth), u 0 inboard .. 1 outboard, v 0 bottom .. 1 top.
vec3 tailPattern(vec3 c, float chips) {
    float u = c.x, v = c.y, w = clamp(c.z - 2.0, 0.0, 1.0);
    float rim = smoothstep(0.0, 0.07, u) * smoothstep(0.0, 0.07, 1.0 - u)
              * smoothstep(0.0, 0.09, v) * smoothstep(0.0, 0.09, 1.0 - v);
    float bars = 0.0, bleed = 0.0;
    for (int k = 0; k < 3; ++k) {                               // three horizontal light guides
        float d = v - (0.20 + 0.30 * float(k));
        bars  = max(bars,  exp(-d * d / 0.0025));
        bleed = max(bleed, exp(-d * d / 0.0150));
    }
    float taper = mix(1.0, 0.45, smoothstep(0.20, 1.00, u));    // hottest at the inboard feed end
    float ends  = smoothstep(0.02, 0.12, u) * smoothstep(0.0, 0.08, 1.0 - u);
    float hot   = bars * taper * ends * chips * rim;
    float halo  = bleed * taper * rim;
    vec2 fc = vec2(u * 14.0, v * 9.0);                          // ribbed, faceted reflector
    float facet = (0.35 + 0.65 * lampHash(floor(fc))) * (0.72 + 0.28 * cos(6.2831853 * fc.y));
    float fill  = facet * mix(0.35, 1.0, w) * (0.30 + 0.70 * halo) * rim;
    return vec3(hot, halo, fill);
}

vec3 projectorPattern(vec3 c) {                                 // LED / halogen projector module
    vec2 q = (c.xy - 0.5) * 2.0;
    float r = length(q);
    float core = pow(clamp(1.0 - r * r, 0.0, 1.0), 2.0);   // broad lens image, not a pin-prick
    float halo = exp(-r * r * 2.2);
    float rings = 0.78 + 0.22 * cos(r * 34.0);                  // condenser / fresnel rings
    float fill = (0.45 + 0.55 * rings) * mix(0.35, 1.0, halo) * smoothstep(1.05, 0.80, r)
               * mix(0.45, 1.0, clamp(c.z - 2.0, 0.0, 1.0));
    return vec3(core, halo, fill);
}

vec3 guidePattern(vec3 c, float chips) {                        // edge-lit LED signature / DRL guide
    float across = 1.0 - abs(c.y * 2.0 - 1.0);
    float ends = smoothstep(0.0, 0.05, c.x) * smoothstep(0.0, 0.05, 1.0 - c.x);
    float hot = across * across * mix(0.55, 1.0, smoothstep(0.0, 0.35, c.x)) * ends * chips;
    return vec3(hot, pow(across, 0.7) * ends, 0.5 + 0.5 * across);
}

vec3 bulbPattern(vec3 c) {                                      // indicator bulb behind an amber cover
    vec2 q = (c.xy - 0.5) * 2.0;
    float r2 = dot(q, q);
    return vec3(exp(-r2 * 5.0), exp(-r2 * 1.6), 0.55 + 0.45 * lampHash(floor(c.xy * 7.0)));
}

vec3 lampPattern(int part, vec3 c, float chips) {
    if (part == 4)  return tailPattern(c, chips);
    if (part == 11) return guidePattern(c, chips);
    if (part == 16 || part == 17) return bulbPattern(c);
    return projectorPattern(c);                                 // 3, 14
}
"""

VEHICLE_DISSOLVE = """
void vehicleDissolve(float visibility) {
    if (visibility <= 0.0) discard;
    float threshold = fract(dot(floor(gl_FragCoord.xy), vec2(.754877666, .569840296)));
    if (visibility <= threshold) discard;
}
"""

VEHICLE_VS = ("#version 330 core\n" + VEHICLE_INPUTS + VEHICLE_ANIMATION + """
uniform mat4 uView, uProj;
uniform vec3 uCamPos;
uniform float uDepthQ;       // 1 / (near * 2^24)
out vec3 vWorld;
out vec3 vShadowRelative;
out vec3 vNormal;
out vec3 vColor;
out vec2 vLocalXY;
out float vLocalZ;
out float vPart;
out float vHeadOn;
out vec3 vWheelLocal;        // un-animated position about the wheel pivot (tread / rim patterns)
out vec3 vWheelNormal;       // un-animated normal: tread face vs sidewall
flat out int vWheelRole;     // 0 body | 1 rear wheel | 2 front wheel | 3 caliper
flat out int vFlags;
flat out float vVisibility;
void main() {
    vec3 position = aPos, normal = aNormal;
    animateVehicle(position, normal);
    mat3 model = mat3(iModel0.xyz, iModel1.xyz, iModel2.xyz);
    vWheelLocal = aPos - aWheel.xyz;
    vWheelNormal = aNormal;
    vWheelRole = int(aWheel.w + .5);
    vec3 wp = vehicleWorld(position);
    
    vWorld   = wp;
    vShadowRelative = wp - uCamPos;
    vNormal  = model * normal;
    vColor   = (aPart < 0.5) ? iColor : aColor;
    vLocalXY = position.xy;
    vLocalZ  = position.z;
    vPart    = aPart;
    vHeadOn  = mod(iFlags, 2.0);
    vFlags = int(iFlags + .5);
    vVisibility = clamp(iMotion.w, 0.0, 1.0);
    vec4 pv = uView * vec4(wp, 1.0);
    if (int(aPart + .5) == 15) {
        // under-car glow quad lies on the road surface: lift it a few depth quanta toward
        // the eye along its view ray so it cannot z-fight with the road decals
        float zd = max(length(pv.xyz), 1.0);
        pv.xyz *= 1.0 - max(0.015, 4.0 * zd * zd * uDepthQ) / zd;
    }
    gl_Position = uProj * pv;
}
""")

VEHICLE_SHADOW_VS = ("#version 330 core\n" + VEHICLE_INPUTS + VEHICLE_ANIMATION + """
uniform mat4 uLightSpaceMatrix;
out vec3 vWorld;
out vec3 vNormal;
out float vPart;
flat out float vVisibility;
void main() {
    vec3 position = aPos, normal = aNormal;
    animateVehicle(position, normal);
    vWorld = vehicleWorld(position);
    vNormal = mat3(iModel0.xyz, iModel1.xyz, iModel2.xyz) * normal;
    vPart = aPart;
    vVisibility = clamp(iMotion.w, 0.0, 1.0);
    gl_Position = uLightSpaceMatrix * vec4(vWorld, 1.0);
}
""")

VEHICLE_SHADOW_FS = ("""
#version 330 core
in float vPart;
flat in float vVisibility;
""" + VEHICLE_DISSOLVE + """
void main() {
    if (int(vPart + .5) == 15) discard; // light pools are never opaque casters
    vehicleDissolve(vVisibility);
}
""")

VEHICLE_FS = ("""
#version 330 core
in vec3 vWorld;
in vec3 vNormal;
in vec3 vColor;
in vec2 vLocalXY;
in float vLocalZ;
in float vPart;
in float vHeadOn;
in vec3 vWheelLocal;
in vec3 vWheelNormal;
flat in int vWheelRole;
flat in int vFlags;
flat in float vVisibility;
uniform sampler2D uSkyLUT;     // atmosphere table shared with the sky pass (reflections)
uniform float uSkyLutOn;
uniform vec3  uSkySunDir;
uniform vec3  uCamPos, uSunDir, uFogColor;
uniform vec3  uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uNightFactor, uExposure;
uniform vec3  uHeadlightColor;
uniform vec3  uTaillightColor;
uniform vec3  uGlowColor;
uniform vec3  uLedColor;
uniform float uTrafficClock;
uniform int   uLensPass;   // 0 = transmittance multiply, 1 = additive reflection (parts 22/23 only)
// Layered paint controls: x coat glossiness (.95-.98), y orange-peel amplitude,
// z metallic-flake density, w pearl / thin-film strength.  See MATERIAL_LIGHTING.md.
uniform vec4  uPaintCoat;
out vec4 FragColor;
""" + HDR_CONV + HDR_OUT + MATERIAL_LIGHT + STREET_SPILL + SHADOW_PCF + VEHICLE_PAINT + VEHICLE_LAMPS + VEHICLE_DISSOLVE + """
vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43; float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Cheap environment: bright sky above the horizon, dark road below it.
vec3 envColor(vec3 R) {
    vec3 haze = acesInv(pow(max(uFogColor, vec3(0.0)), vec3(2.2))) / max(uExposure, 1e-4);
    vec3 sky  = haze + uAmbColor * 0.55;
    vec3 gnd  = haze * 0.15 + uAmbColor * 0.10;
    float h = smoothstep(-0.12, 0.45, R.z);
    vec3 e = mix(gnd, sky, h);
    return e;
}

// Reflection environment: the real atmosphere table the sky pass uses (cheap: one
// texture fetch, no cloud noise), darkened toward the horizon by the city canyon
// and falling to a dark road below it.  Falls back to the analytic gradient.
vec3 skyEnv(vec3 R) {
    if (uSkyLutOn < 0.5) return envColor(R);
    vec3 gnd    = envColor(vec3(0.0, 0.0, -1.0));
    vec3 facade = envColor(vec3(0.0, 0.0, 1.0)) * 0.30;
    vec2 sxy = (length(uSkySunDir.xy) > 1e-3) ? normalize(uSkySunDir.xy) : vec2(1.0, 0.0);
    vec2 rxy = (length(R.xy) > 1e-4) ? normalize(R.xy) : sxy;
    float relAz = acos(clamp(dot(rxy, sxy), -1.0, 1.0));
    float v = sqrt(clamp((R.z + 0.1) / 1.1, 0.0, 1.0));
    vec3 atm = texture(uSkyLUT, vec2(relAz * 0.31830989, v)).rgb;
    float canyon = 0.5 * (1.0 - smoothstep(0.03, 0.50, R.z));
    return mix(gnd, mix(atm, facade, canyon), smoothstep(0.0, 0.035, R.z));
}

// Rough reflection: blend the sharp lookup toward the half-way (cosine-ish) lookup.
vec3 skyEnvRough(vec3 R, vec3 N, float rough) {
    vec3 sharp = skyEnv(R);
    if (rough < 0.05) return sharp;
    return mix(sharp, skyEnv(normalize(R + N)), smoothstep(0.0, 0.7, rough));
}

// Sun highlight of a microfacet-like lobe.  Exponent from roughness; the peak is
// scaled to the solar disc's reflected energy so wide lobes stay dim.
float sunLobe(vec3 R, vec3 L, float rough) {
    float n = clamp(2.0 / max(rough * rough, 1e-4) - 2.0, 8.0, 16384.0);
    return pow(max(dot(R, L), 0.0), n) * 0.8 * min(1.0, sqrt(n / 1024.0));
}

void main() {
    int part = int(vPart + 0.5);
    // Signal-lens LED lattice footprint (local-space metres / 10.5 mm pitch). Derivatives are
    // taken here, in uniform control flow, before any discard or material branch.
    vec2 lensCell = vec2(vLocalXY.y, vLocalZ) / 0.0105;
    float lensFw = max(fwidth(lensCell.x), fwidth(lensCell.y));
    vec2 chipCell = vec2(vLocalXY.y, vLocalZ) / 0.0075;          // 7.5 mm LED chip pitch, car-fixed
    float chipFw = max(fwidth(chipCell.x), fwidth(chipCell.y));
    bool baked = vColor.b > 1.5;                                  // lamp coordinates present (vehicle_lamps.py)
    if (vVisibility <= 0.0) discard;
    if (part == 15) {                        // translucent ground-glow quads (night)
        float d = length(vec2(vLocalXY.x * 0.32, vLocalXY.y * 0.65));
        float fade = max(0.0, 1.0 - d);
        fade *= fade;
        float dist = length(uCamPos - vWorld);
        float fogFade = 1.0 - cityFog(dist, vWorld.z, uFogNear, uFogFar);
        float intensity = fade * uNightFactor * vHeadOn * fogFade * vVisibility;
        FragColor = vec4(sceneOut(emissionNits(uGlowColor, 2.0)), intensity);
        return;
    }
    // Screen-space dissolve keeps the opaque depth/SSR pass correct during the
    // worker's smooth spawn/recycle fade, without sorting translucent car bodies.
    vehicleDissolve(vVisibility);
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    float sf = shadowPCF(vWorld, N, L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    float fres  = pow(1.0 - NdotV, 5.0);
    vec3  R     = reflect(-V, N);
    float RdotL = max(dot(R, L), 0.0);
    vec3 albedo = reflectanceRange(linearColor(vColor), PAINT_REFLECTANCE);
    if (part == 2) albedo = reflectanceRange(albedo, vec2(0.015, 0.035));
    if (part == 5) albedo = vec3(0.55, 0.556, 0.554);   // chromium reflectance
    if (baked && part < 22) albedo = lampAlbedo(part);   // the colour slot holds lamp coordinates, not a tint
    float camD  = length(uCamPos - vWorld);
    // lamps follow the day/night cycle smoothly instead of snapping on at one threshold
    float nightOn = smoothstep(0.25, 0.55, uNightFactor) * vHeadOn;

    // contact darkening: undersides, wheel wells and bumpers sit in shadow
    float ao = mix(0.30, 1.0, smoothstep(0.05, 0.70, vLocalZ));

    vec3 amb = uAmbColor * mix(0.55, 1.0, N.z * 0.5 + 0.5);
    // Only direct sunlight is shadowed; sky fill, street lamps and emission remain.
    vec3 diff = amb * 0.6 + uSunColor * NdotL * (0.32 * sf);
    if (uNightFactor > 0.02) {
        vec3 diffE, specE;
        streetLights(vWorld, N, V, diffE, specE);
        diff += diffE * (uNightFactor / 3.14159265);
    }
    vec3 sunDisc = uSunColor * (pow(RdotL, 1024.0) * 0.8 + pow(RdotL, 64.0) * 0.015) * sf;
    vec3 env = skyEnv(R);
    if (part == 22 || part == 23) {           // translucent lamp lens: clear headlamp cover / red tail cover
        vec3 Nl = dot(N, V) < 0.0 ? -N : N;   // thin shells are seen from both sides
        float cv = clamp(dot(Nl, V), 0.0, 1.0);
        float fl = pow(1.0 - cv, 5.0);
        vec3 Rl = reflect(-V, Nl);
        float F1 = COAT_F0 + (1.0 - COAT_F0) * fl;
        float F = 2.0 * F1 / (1.0 + F1);
        float fogL = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
        if (uLensPass == 0) {
            vec3 T = (part == 23) ? vec3(0.66, 0.17, 0.15) : vec3(0.985, 0.99, 0.99);
            FragColor = vec4(mix(T * (1.0 - F), vec3(1.0), fogL), 1.0);
        } else {
            float RdL = max(dot(Rl, L), 0.0);
            vec3 refl = (skyEnv(Rl) * (0.55 + 0.45 * ao) + uSunColor * pow(RdL, 900.0) * 0.9 * sf) * F;
            vec3 litL = refl;
            // The cover only lights up near the emitters behind it; away from them it stays dark glass.
            float nearP = baked ? vColor.r : 1.0, wideP = baked ? vColor.g : 0.0;
            float nm = lampNight();
            float lensK = baked ? mix(8.0, 1.0, nm) : 1.0;
            if (part == 23) {
                litL += vec3(0.50, 0.015, 0.012) * diff * 0.10 * ao;
                float veil = baked ? (0.0040 * nearP + 0.0008 * wideP) * lensK : 0.07;
                litL += emissionNits((baked ? TAIL_RED : uTaillightColor), TAILLIGHT_NITS) * nightOn * veil;
                if ((vFlags & 2) != 0)
                    litL += emissionNits((baked ? TAIL_RED : uTaillightColor), BRAKELIGHT_NITS) * veil * (baked ? 0.30 : 1.0);
            } else {
                float veil = baked ? (0.00025 * nearP + 0.00004 * wideP) * lensK : 0.03;
                litL += emissionNits(uLedColor, HEADLIGHT_NITS) * nightOn * veil;
            }
            FragColor = vec4(sceneOut(litL) * (1.0 - fogL), 0.0);
        }
        return;
    }
    bool wheelPart = vWheelRole > 0;

    // SSR uses this finish weight with the smooth world normal. Rubber, matte
    // trim and emitting lenses remain occluders rather than reflection receivers.
    float reflectivity = 0.0;
    vec3 lit;
    if (part == 1) {                          // glazing: laminated glass, IOR 1.5, tinted, dim cabin behind
        reflectivity = 0.92;
        float F1 = COAT_F0 + (1.0 - COAT_F0) * fres;              // Schlick dielectric, n = 1.5
        float F  = 2.0 * F1 / (1.0 + F1);                         // front + back surface of the pane
        float priv = smoothstep(-0.20, -0.70, vLocalXY.x);        // rear glass sits behind the B pillar
        vec3 tint = mix(vec3(0.78, 0.90, 0.92), vec3(0.22, 0.26, 0.28), priv);
        float sheen = smoothstep(-0.2, 0.9, R.z);                 // sky shows in the upper part of the glass
        // what is transmitted: a dim cabin lit from above (headliner bright, footwells dark)
        vec3 cabin = diff * vec3(0.016, 0.018, 0.021) * mix(0.55, 1.35, smoothstep(0.55, 1.30, vLocalZ));
        lit = cabin * tint * (1.0 - F)
              + env * F * (0.60 + 0.40 * sheen)
              + sunDisc * (0.04 + 0.96 * fres);
        lit *= mix(0.55, 1.0, ao);
    } else if (part == 2) {                   // tyre rubber: roughness ~0.9, tread + sidewall relief
        vec3 w = vWheelLocal;                                      // wheel-fixed: pattern spins with the tyre
        float tfade = smoothstep(40.0, 5.0, camD);
        float rad = length(w.xz);
        float ang = atan(w.z, w.x);
        float side = smoothstep(0.45, 0.85, abs(vWheelNormal.y));  // 1 on the sidewall, 0 on the tread face
        float tcell = abs(fract(w.y / 0.042) - 0.5) * 2.0;
        float groove = smoothstep(0.78, 0.95, tcell);              // circumferential grooves
        float lug = smoothstep(0.80, 0.95, abs(fract(ang * (30.0 / 6.2831853) + w.y * 4.0) - 0.5) * 2.0);
        float tread = max(groove, lug * 0.6) * tfade * (1.0 - side);
        float rings = (0.5 + 0.5 * sin(rad * 1100.0)) * tfade * side;   // moulded sidewall ribbing
        vec3 rubber = albedo * (1.0 - 0.55 * tread) * (1.0 - 0.20 * rings);
        vec3 Ng = microNormal(w, N, 0.0035, 0.04 * tfade);
        float grain = 0.55 + 0.45 * max(dot(Ng, L), 0.0);          // shading grain, not gloss
        lit = rubber * diff * ao * grain
              + skyEnvRough(R, N, 0.90) * 0.018 * (0.4 + fres) * ao
              + uSunColor * sunLobe(reflect(-V, Ng), L, 0.85) * 0.05 * sf * ao;
    } else if (part == 5) {                   // metal: chrome trim (mirror) or satin alloy rim
        reflectivity = 1.0;
        if (wheelPart) {                      // alloy: metalness 1, roughness ~0.28, machined concentric lathe lines
            vec3 F0 = vec3(0.80, 0.81, 0.82);
            float lathe = 1.0 - 0.07 * (0.5 + 0.5 * sin(length(vWheelLocal.xz) * 1400.0)) * smoothstep(30.0, 4.0, camD);
            lit = skyEnvRough(R, N, 0.28) * mix(F0, vec3(1.0), fres) * lathe * ao
                + uSunColor * sunLobe(R, L, 0.28) * F0 * 0.6 * sf * ao;
        } else {                              // chrome / mirror: metalness 1, roughness ~0.03
            vec3 F0 = vec3(0.62, 0.63, 0.62);
            lit = (env + sunDisc) * mix(F0, vec3(1.0), fres) * ao;
        }
    } else if (part == 6) {                   // black trim / plastic, or brake disc behind the spokes
        if (wheelPart) {                      // dark cast iron, machined rings, deep inside the wheel
            float rr = length(vWheelLocal.xz);
            float rings = 0.5 + 0.5 * sin(rr * 1600.0) * smoothstep(30.0, 4.0, camD);
            vec3 iron = vec3(0.075, 0.070, 0.066) * (0.80 + 0.20 * rings);
            lit = iron * diff * ao * 0.45
                + skyEnvRough(R, N, 0.50) * 0.045 * ao * 0.6
                + uSunColor * sunLobe(R, L, 0.50) * 0.05 * sf * ao;
        } else {
            lit = albedo * diff * ao + skyEnvRough(R, N, 0.55) * 0.025 * (0.4 + fres) * ao + sunDisc * 0.04;
        }
    } else if (part == 7) {                   // licence plate
        lit = albedo * diff;
    } else if (part == 10) {                  // matte plastic: arches, cladding, skirts, mirror bases
        lit = albedo * diff * ao * 0.95 + env * 0.012;
    } else if (part == 11) {                  // LED signature (emission applied below); passive = glossy white lens
        float k = 0.08 + 0.55 * fres;
        lit = mix(albedo * diff, env * ao, k) + sunDisc * k;
    } else if (part >= 18 && part <= 21) {    // signal lens: dark tinted polycarbonate behind a hood
        int lensState = vFlags & 3;
        bool lensOn = (part == 18 && lensState == 2) || (part == 19 && lensState == 1)
                   || (part == 20 && lensState == 0) || (part == 21 && (vFlags & 16) != 0);
        float k = 0.06 + 0.58 * fres;
        // Unlit lenses are nearly black glass with a faint tint; they pick up a sky reflection
        // and a sun glint instead of glowing like painted plastic.
        lit = mix(albedo * diff * ao * (lensOn ? 0.30 : 0.05), env * ao * 0.9, k) + sunDisc * (k + 0.08);
    } else if (part == 16 || part == 17) {    // passive indicator lenses, never glow quads
        lit = albedo * diff * ao + env * .04 + sunDisc * .06;
    } else if (part == 12) {                  // piano black: pillars, grille mesh, lamp housings
        reflectivity = 0.72;
        float k = 0.05 + 0.70 * fres;
        lit = mix(albedo * diff * ao, env * ao, k) + sunDisc * k * ao;
    } else if (part == 13) {                  // panoramic roof glass: darker, more mirror-like, privacy tinted
        reflectivity = 0.96;
        float F1 = COAT_F0 + (1.0 - COAT_F0) * fres;
        float k = 0.10 + 0.80 * fres;
        float sheen = smoothstep(-0.1, 0.9, R.z);
        lit = mix(vec3(0.006, 0.008, 0.012) * diff, env * (0.22 + 0.40 * sheen), max(k, 2.0 * F1 / (1.0 + F1))) + sunDisc * k;
        lit *= mix(0.6, 1.0, ao);
    } else if (part == 3 || part == 4 || part == 14) {   // lamp interior: metallised reflector / projector optics
        float facet = baked ? 0.60 + 0.40 * lampHash(floor(vColor.xy * vec2(14.0, 9.0))) : 1.0;
        float shine = (part == 4) ? 0.05 : 0.14;                // chrome headlamp bowl vs red-metallised tail bowl
        float matte = (part == 4) ? 0.55 : 0.22;                // a chrome bowl is a mirror: little diffuse, mostly reflection
        lit = albedo * diff * ao * matte * facet
            + skyEnvRough(R, N, 0.30) * shine * (0.35 + 0.65 * fres) * ao * facet
            + sunDisc * shine * 0.9 * facet;
    } else {                                  // layered automotive paint (0 = instance colour, 8 = fixed)
        bool caliper = vWheelRole == 3;
        if (part == 0 || part == 8) reflectivity = 0.60;
        float gloss  = clamp(uPaintCoat.x, 0.0, 1.0);
        float lum    = dot(albedo, PHOTO_LUMA);
        float variant = paintHash(vec3(dot(vColor, vec3(97.0, 51.0, 29.0)), 7.3, 3.1));
        // Finish per colour: ~20 % single-stage solid, ~58 % two-coat metallic, ~22 % tri-coat pearl.
        float solid = (part == 0 && variant < 0.20) ? 1.0 : 0.0;
        float tri   = (part == 0 && variant > 0.78) ? 1.0 : 0.0;
        float met   = (part == 0 ? 1.0 : 0.6) * clamp(uPaintCoat.z, 0.0, 2.0)
                    * mix(0.55, 1.0, smoothstep(0.03, 0.22, lum)) * (1.0 - solid);
        float pearl = (part == 0 ? 1.0 : 0.0) * clamp(uPaintCoat.w, 0.0, 1.0)
                    * max(smoothstep(0.20, 0.55, lum), tri);
        if (caliper) { met = 0.0; pearl = 0.0; }
        float flakeFade = smoothstep(80.0, 12.0, camD);           // micro-normals resolve up close only
        float peelFade  = smoothstep(60.0, 6.0, camD);
        vec3 Pm = vec3(vLocalXY, vLocalZ);                         // car-fixed coordinates: texture never swims
        // Clear-coat surface: two octaves of 7-16 mm factory "orange peel"; single-stage paint is rougher.
        float peelAmp = clamp(uPaintCoat.y, 0.0, 3.0) * mix(1.0, 1.5, solid);
        vec3 Nc = microNormal(Pm, N, 0.016, peelAmp * 0.0022 * peelFade);
        Nc = microNormal(Pm + 3.7, Nc, 0.0065, peelAmp * 0.0010 * peelFade);
        // Flakes sit under the coat: fine ones average to a sheen, a sparse coarse set glints.
        vec3 Ng = flakeNormal(Pm, Nc, 0.0009, 0.35 * met * flakeFade);
        float glit = step(0.86, paintHash(Pm / 0.0035 + 11.0));
        vec3 Ngl = flakeNormal(Pm + 5.1, Nc, 0.0035, 0.60 * met * flakeFade * glit);
        vec3 paint = (part == 0) ? albedo * mix(0.62, 1.0, smoothstep(0.30, 0.55, vLocalZ)) : albedo;
        float cosV = clamp(dot(Nc, V), 0.0, 1.0);
        if (pearl > 0.0) {                                        // flip-flop: Fresnel-driven hue shift + thin film
            vec3 film = thinFilmTint(cosV, mix(320.0, 540.0, variant));
            film = clamp(film / max(dot(film, PHOTO_LUMA), 0.30), vec3(0.0), vec3(1.6));
            paint *= mix(vec3(1.0), film, pearl * 0.35);
            vec3 alt = paint.brg;
            alt *= dot(paint, PHOTO_LUMA) / max(dot(alt, PHOTO_LUMA), 1e-3);
            paint = mix(paint, alt, pearl * 0.40 * smoothstep(0.15, 0.95, 1.0 - cosV));
        }
        float coatF = COAT_F0 + (1.0 - COAT_F0) * pow(1.0 - cosV, 5.0);
        vec3 Rc = reflect(-V, Nc);
        float coatPow = mix(256.0, 4096.0, gloss * gloss);        // 0.95-0.98: sharp automotive highlight (roughness ~0.02)
        // Both lobes are scaled to the solar disc's reflected energy (sunDisc is a 0.8
        // peak), not to the direct irradiance, so a clear coat glint stays a highlight.
        float coatSun = pow(max(dot(Rc, L), 0.0), coatPow) * 0.10 * sf;
        float flakeSun = pow(max(dot(reflect(-V, Ng), L), 0.0), 1200.0) * sf * met
                       * 0.05 * mix(0.25, 1.0, flakeFade);
        float glitterSun = pow(max(dot(reflect(-V, Ngl), L), 0.0), 500.0) * sf * met * glit
                         * 0.12 * flakeFade;
        float farSheen = sf * (1.0 - flakeFade) * met * 0.004;
        // Base layer: pigment diffuse plus metallic flakes mirroring the sky through the coat.
        vec3 baseLit = paint * diff * ao;
        if (met > 0.0) {
            vec3 Rf = reflect(-V, Ngl);
            vec3 flakeEnv = skyEnvRough(Rf, Ngl, 0.25);
            baseLit = mix(baseLit, paint * flakeEnv * 1.1 * ao, 0.20 * met);
        }
        // Clear coat is colourless: it reflects the (orange-peel perturbed) world, over the base.
        vec3 coatEnv = skyEnv(Rc) * ao;
        lit = mix(baseLit, coatEnv, coatF)
              + sunDisc * coatF * 0.35 * ao
              + uSunColor * (coatSun + flakeSun + glitterSun + farSheen) * ao;
    }

    vec3 pat = vec3(1.0, 0.0, 0.0);           // unbaked lamps keep the old flat emission
    float lampView = 1.0;
    if (baked && part < 22) {
        pat = lampPattern(part, vColor, chipDots(chipCell, chipFw));
        lampView = mix(0.30, 1.0, smoothstep(0.0, 0.75, NdotV));   // lamps are directional, not Lambertian discs
    }
    float nmix = lampNight();
    float hotTint = baked ? pat.x : 0.0;
    if (part == 3 || part == 14) {            // warm halogen / cool-white LED projectors
        vec3 lampColor = (part == 14) ? uLedColor : uHeadlightColor;
        float level = pat.x + mix(0.40, 0.020, nmix) * pat.y + mix(0.30, 0.0020, nmix) * pat.z;
        lit += emissionNits(mix(lampColor, vec3(1.0), 0.45 * hotTint), HEADLIGHT_NITS) * level * lampView * nightOn;
    } else if (part == 4) {                   // tail-lamp: LED light guides behind a red cover
        vec3 tint = mix(baked ? TAIL_RED : uTaillightColor, TAIL_AMBER, 0.75 * hotTint);
        float lvTail  = pat.x + mix(0.35, 0.010, nmix) * pat.y + mix(0.25, 0.0040, nmix) * pat.z;
        float lvBrake = pat.x + mix(0.35, 0.006, nmix) * pat.y + mix(0.25, 0.0040, nmix) * pat.z;
        lit += emissionNits(tint, TAILLIGHT_NITS) * lvTail * lampView * nightOn;
        if ((vFlags & 2) != 0) lit += emissionNits(tint, BRAKELIGHT_NITS) * lvBrake * lampView;
    } else if (part == 11) {                  // LED guide: daytime-running-light glow by day, dimmer at night
        float ledOn = vHeadOn * mix(1.0, 0.2, nightOn);
        float level = pat.x + mix(0.55, 0.020, nmix) * pat.y + mix(0.45, 0.0040, nmix) * pat.z;
        lit += emissionNits(uLedColor, DRL_NITS) * level * ledOn * lampView;
    } else if (part == 16 || part == 17) {
        int flag = part == 16 ? 4 : 8;
        float phase = fract(uTrafficClock / .8 + float((vFlags >> 4) & 255) / 256.0);
        float flash = min(clamp(phase / .06, 0.0, 1.0), clamp((.5 - phase) / .06, 0.0, 1.0));
        float level = pat.x + mix(0.40, 0.010, nmix) * pat.y + mix(0.30, 0.0020, nmix) * pat.z;
        if ((vFlags & flag) != 0) lit += emissionNits(vec3(1.0, .32, .025), INDICATOR_NITS) * level * flash * lampView;
    } else if (part >= 18 && part <= 21) {
        int state = vFlags & 3;
        bool signalLit = (part == 18 && state == 2) || (part == 19 && state == 1)
                   || (part == 20 && state == 0) || (part == 21 && (vFlags & 16) != 0);
        if (signalLit) {
            // Real LED modules: a lattice of tiny emitters, hottest at the centre where the
            // colour washes toward white, with a slightly dimmer rim and a stronger daytime drive.
            vec2 lc = (part == 21) ? vec2(3.05, 4.72)
                    : vec2(2.4, part == 18 ? 5.10 : (part == 19 ? 4.72 : 4.34));
            float rr = length(vec2(vLocalXY.y, vLocalZ) - lc) / 0.145;
            vec2 cell = lensCell;
            vec2 fq = fract(cell + vec2(0.5 * mod(floor(cell.y), 2.0), 0.0)) - 0.5;
            float ledDot = 1.0 - smoothstep(0.26, 0.50, length(fq));
            float resolved = 1.0 - smoothstep(0.12, 0.55, lensFw);   // dots only when they span pixels
            float led = mix(0.80, 0.42 + 1.0 * ledDot, resolved);
            float edge = (part == 21) ? 1.0 : 1.0 - 0.42 * smoothstep(0.40, 1.0, rr);
            float core = (part == 21) ? 0.35 : exp(-rr * rr * 3.2);
            vec3 tint = mix(vColor, vec3(1.0), core * 0.24);
            lit += emissionNits(tint, TRAFFIC_SIGNAL_NITS) * mix(1.8, 1.0, uNightFactor)
                 * led * edge * (0.85 + 0.40 * core) * smoothstep(0.0, .25, NdotV);
        }
    }
    lit = max(lit, albedo * uAmbColor * 0.04 * ao);

    vec3 mapped = sceneOut(lit);

    float f = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
    FragColor = vec4(mix(mapped, displayToScene(uFogColor), f), 1.0);
}
""")
