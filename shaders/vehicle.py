# --- shaders/vehicle.py ---
from .hdr import HDR_CONV, HDR_OUT
from .streetspill import STREET_SPILL
from .materials import MATERIAL_LIGHT
from .env import ENV_SKY
from .shadow import SHADOW_PCF
from .sun import SUN_LIGHT
# Single-pass instanced car shader.  The per-vertex `aPart` is the material id (see
# vehicle_models.py): 0 paint | 1 glass | 2 tyre | 3 warm headlight | 4 taillight | 5 chrome
# | 6 trim | 7 plate | 8 fixed paint | 9 satin metal | 10 matte plastic | 11 LED signature | 12 piano black
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

// Reflected solar disc: the source of the clear-coat / chrome / glass glare.
// The sun has a 0.27-degree angular radius and is roughly 34,000x brighter than a sunlit white
// body, so a polished panel returns a white-hot point.  Roughness widens the
// lobe while `discArea / spread` conserves its energy. Only polished coatings,
// chrome and glass use this tight core; satin metal uses a broad irradiance lobe.
// The result is scene-linear HDR;
// the bloom, anamorphic streak and starburst passes downstream do the spreading.
// F is the interface reflectance and gain trims the effect per material.
vec3 solarGlare(vec3 R, vec3 L, float rough, float F, float gain) {
    float rl = max(dot(R, L), 0.0);
    float a2 = 2.0 * (1.0 - rl);                                   // 1 - cos(angle)
    float spread = SUN_RADIUS * SUN_RADIUS + rough * rough * 0.010;
    float core = exp(-a2 / spread) * (SUN_RADIUS * SUN_RADIUS) / spread;
    // Two halos: the tight coat/clearance glare and the wide bloom source.
    float halo = exp(-a2 / 0.0012) * 0.55 + exp(-a2 / 0.0200) * 0.16;
    vec3 solar = uSunColor * (SUN_RADIANCE / SOLAR_IRRADIANCE);
    return (solar * core + uSunColor * halo) * F * gain;
}

// A stable brush direction: horizontal on faces, radial on horizontal surfaces.
vec3 brushTangent(vec3 N) {
    vec3 up = abs(N.z) > 0.95 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
    return normalize(cross(up, N));
}

// Anisotropic (brushed / machined) highlight: tight across the brush, wide along it.
float brushedHighlight(vec3 N, vec3 V, vec3 L, vec3 T, float streak, float sharp) {
    vec3 H = (V + L) / max(length(V + L), 1e-4);
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

# Cabin windows and lamp covers use the blended transmission draw, which returns before
# mrt_fs can store a finish weight, so screen-space reflections skipped them entirely.
# This companion target stores just the pane surface (depth is written by the same draw).
VEHICLE_GLASS_MAT_FS = ("""
#version 330 core
in vec3 vWorld;
in vec3 vNormal;
flat in float vVisibility;
uniform vec3 uCamPos;
layout(location = 0) out vec4 oMat;
""" + VEHICLE_DISSOLVE + """
void main() {
    vehicleDissolve(vVisibility);
    // Thin shells are seen from both sides: store the eye-facing normal so the
    // reflection ray leaves the pane outward, exactly as the colour pass reflects it.
    vec3 N = normalize(vNormal);
    N = dot(N, normalize(uCamPos - vWorld)) < 0.0 ? -N : N;
    oMat = vec4(N * 0.5 + 0.5, 0.92);   // same finish weight as the opaque glazing (part 1)
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
// uSkyLUT / uSkyLutOn / uSkySunDir, the cloud deck and the atmosphere uniforms
// arrive with ENV_SKY, which supplies envSky() exactly as the sky pass draws it.
uniform vec3  uCamPos, uSunDir, uFogColor;
uniform vec3  uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uNightFactor, uExposure;
uniform vec3  uHeadlightColor;
uniform vec3  uTaillightColor;
uniform vec3  uGlowColor;
uniform vec3  uLedColor;
uniform float uWheelAO;      // 0..1 recess occlusion inside the wheel (brake disc, hub)
uniform float uTrafficClock;
uniform int   uLensPass;   // 0 = transmittance multiply, 1 = additive reflection (parts 22/23/24)
// Layered paint controls: x coat glossiness (.95-.98), y orange-peel amplitude,
// z metallic-flake density, w pearl / thin-film strength.  See MATERIAL_LIGHTING.md.
uniform vec4  uPaintCoat;
out vec4 FragColor;
""" + HDR_CONV + HDR_OUT + MATERIAL_LIGHT + ENV_SKY + SUN_LIGHT + STREET_SPILL + SHADOW_PCF + VEHICLE_PAINT + VEHICLE_LAMPS + VEHICLE_DISSOLVE + """
vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43; float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Fallback environment when the atmosphere table is unavailable (8-bit / no post
// path): bright sky above the horizon, dark road below it.
vec3 envColor(vec3 R) {
    vec3 haze = acesInv(pow(max(uFogColor, vec3(0.0)), vec3(2.2))) / max(uExposure, 1e-4);
    vec3 sky  = haze + uAmbColor * 0.55;
    vec3 gnd  = haze * 0.15 + uAmbColor * 0.10;
    float h = smoothstep(-0.12, 0.45, R.z);
    vec3 e = mix(gnd, sky, h);
    return e;
}

// Reflection environment: the shared envSky() the sky pass, building glazing and
// water already reflect.  It carries the same advected cloud deck as the sky, so
// an overcast car shows cloud colour instead of a clear blue dome, and it returns
// dark street/city radiance for every direction at or below the horizon -- a car's
// lower silhouette and shaded flank no longer pick up a bright sky rim.
vec3 skyEnv(vec3 R) {
    if (uSkyLutOn < 0.5) return weatherSky(envColor(R), R, normalize(uSunDir));
    return envSky(R, normalize(uSunDir));
}

// Rough reflection: blend the sharp lookup toward the half-way (cosine-ish) lookup.
vec3 skyEnvRough(vec3 R, vec3 N, float rough) {
    vec3 sharp = skyEnv(R);
    if (rough < 0.05) return sharp;
    return mix(sharp, skyEnv(normalize(R + N)), smoothstep(0.0, 0.7, rough));
}

// Satin alloy/unplated metal integrates a broad cone of the environment. Two
// sharp lookups are not a rough reflection: they still carry a crisp horizon.
// Sharp scene-geometry SSR is deliberately excluded for this finish below.
vec3 satinEnv(vec3 R, vec3 N) {
    vec3 axis = normalize(R + N * 0.30);
    vec3 T = brushTangent(axis), B = cross(axis, T);
    return (skyEnv(axis) * 2.0
          + skyEnv(normalize(axis + T * 0.65)) + skyEnv(normalize(axis - T * 0.65))
          + skyEnv(normalize(axis + B * 0.65)) + skyEnv(normalize(axis - B * 0.65))) / 6.0;
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
    // Filter subpixel lathe lines in uniform control flow, before material branches.
    float wheelRadius = length(vWheelLocal.xz), wheelFw = fwidth(wheelRadius);
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
    // Reflected solar disc / sparkle needs an *unobstructed* sun AND one the cloud
    // deck has not already swallowed, exactly like the building glazing glint.  The
    // mirror image of a 0.5-degree disc cannot be half there: a soft PCF penumbra,
    // a settling/receiver-biased map or any partial shadow value would otherwise
    // still paint a white-hot glint inside the shade.  Diffuse light keeps the
    // smooth `sf`; sky and cloud reflections are ambient and are never gated.
    float sunVis = smoothstep(0.55, 0.92, sf) * (1.0 - 0.95 * clamp(uCloudCover, 0.0, 1.0));
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
    // Wheel recess: the brake disc and dark backing sit ~4 cm behind the spokes, inside the
    // tyre bead and the arch. Screen-space AO (3 m radius) cannot resolve that, so it is
    // analytic: deepest at the hub, easing toward the braking surface at the disc's rim.
    bool wheelRecess = vWheelRole > 0 && (part == 8 || part == 6);
    bool brakeDisc = wheelRecess && part == 8 && vWheelRole != 3;
    if (wheelRecess) {
        float rrW = length(vWheelLocal.xz);
        float well = mix(0.06, 0.40, smoothstep(0.04, 0.18, rrW));
        ao *= mix(1.0, well, clamp(uWheelAO, 0.0, 1.0));
        if (brakeDisc) albedo *= mix(1.0, 0.55, clamp(uWheelAO, 0.0, 1.0));
    }

    vec3 amb = uAmbColor * mix(0.55, 1.0, N.z * 0.5 + 0.5);
    // Only direct sunlight is shadowed; sky fill, street lamps and emission remain.
    vec3 diff = amb * 0.6 + uSunColor * NdotL * (0.32 * sf);
    // Street lamps light the cars themselves, not just the road under them: the same
    // cell grid the ground samples for its spill, plus the compact luminaire highlight a
    // polished body returns.  Both carry uNightFactor, so daylight costs nothing.
    vec3 lampSharp = vec3(0.0), lampSoft = vec3(0.0);
    if (uNightFactor > 0.02) {
        vec3 diffE, specE;
        streetLights(vWorld, N, V, diffE, specE);
        diff += diffE * (uNightFactor / 3.14159265);
        streetLampGlints(vWorld, N, V, lampSharp, lampSoft);
        lampSharp *= uNightFactor;
        lampSoft *= uNightFactor;
    }
    // Small glossy glint for trim, lens and lamp parts.  The paint, chrome,
    // piano-black and glazing branches compute their own Fresnel-aware glare.
    vec3 sunDisc = uSunColor * (pow(RdotL, 1024.0) * 0.8 + pow(RdotL, 64.0) * 0.015) * sunVis;
    vec3 env = skyEnv(R);
    if (part == 22 || part == 23 || part == 24) { // real transmission: lamp covers and cabin windows
        vec3 Nl = dot(N, V) < 0.0 ? -N : N;   // thin shells are seen from both sides
        float cv = clamp(dot(Nl, V), 0.0, 1.0);
        float fl = pow(1.0 - cv, 5.0);
        vec3 Rl = reflect(-V, Nl);
        float F1 = COAT_F0 + (1.0 - COAT_F0) * fl;
        float F = 2.0 * F1 / (1.0 + F1);
        float fogL = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
        if (uLensPass == 0) {
            // Clear front glazing, cool blue-green; progressively darker privacy
            // glass behind the B pillar.  Lamp covers keep their signal tints.
            float privacy = smoothstep(-0.20, -0.70, vLocalXY.x);
            vec3 clearT = mix(vec3(0.90, 0.94, 0.955), vec3(0.38, 0.42, 0.44), privacy);
            vec3 T = (part == 23) ? vec3(0.66, 0.17, 0.15)
                     : (part == 24) ? clearT : vec3(0.985, 0.99, 0.99);
            FragColor = vec4(mix(T * (1.0 - F), vec3(1.0), fogL), 1.0);
        } else {
            // Sky reflection plus the real solar disc, so a windscreen can flare
            // white-hot exactly where the sun is mirrored instead of washing out.
            vec3 refl = skyEnv(Rl) * (0.55 + 0.45 * ao) * F
                      + solarGlare(Rl, L, 0.012, F, part == 24 ? 1.0 : 0.35) * sunVis;
            vec3 litL = refl + lampSharp * F;
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
            } else if (part == 22) {
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
        vec3 Nl = dot(N, V) < 0.0 ? -N : N;                       // two-sided thin shell
        float cv = clamp(dot(Nl, V), 0.0, 1.0);
        float fresG = pow(1.0 - cv, 5.0);
        float F1 = COAT_F0 + (1.0 - COAT_F0) * fresG;             // Schlick dielectric, n = 1.5
        float F  = 2.0 * F1 / (1.0 + F1);                         // front + back surface of the pane
        vec3 Rg = reflect(-V, Nl);
        float priv = smoothstep(-0.20, -0.70, vLocalXY.x);        // rear glass sits behind the B pillar
        // Cool blue-green front glazing, privacy tint behind the B pillar.  This
        // is the transmission tint applied to the dim cabin, not the reflection.
        vec3 tint = mix(vec3(0.56, 0.68, 0.71), vec3(0.15, 0.19, 0.21), priv);
        float sheen = smoothstep(-0.15, 0.90, Rg.z);              // sky shows in the upper part of the glass
        vec3 cabin = diff * vec3(0.022, 0.025, 0.029)
                   * mix(0.45, 1.30, smoothstep(0.55, 1.30, vLocalZ));
        lit = cabin * tint * (1.0 - F)
              + skyEnv(Rg) * (0.55 + 0.45 * sheen) * F
              + solarGlare(Rg, L, 0.012, F, 1.0) * sunVis
              + lampSharp * F;
        // The pane's glassy horizon is envSky() itself: the old additive sun-tinted
        // rim was never shadowed or cloud-gated, so it drew a bright edge on glass in
        // shade.  Below the horizon envSky already returns dark street radiance.
        lit *= mix(0.60, 1.0, ao);
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
              + uSunColor * sunLobe(reflect(-V, Ng), L, 0.85) * 0.05 * sunVis * ao
              + lampSoft * 0.05 * ao;
    } else if (part == 9 || (part == 5 && wheelPart)) { // satin metal; legacy chrome-tagged rims are alloy
        vec3 tint = mix(vec3(0.35), vec3(1.0), clamp(linearColor(vColor) * 3.0, vec3(0.0), vec3(1.0)));
        vec3 F0 = vec3(0.80, 0.81, 0.82) * tint;
        float resolved = 1.0 - smoothstep(0.8, 3.14159265, wheelFw * 1400.0);
        float lathe = wheelPart ? 1.0 - 0.035 * (0.5 + 0.5 * sin(wheelRadius * 1400.0) * resolved)
                                      * (1.0 - smoothstep(4.0, 30.0, camD)) : 1.0;
        // Roughness ~0.42: a broad, low-energy sheen, not a reflected white-hot solar disc.
        float sheen = sunLobe(R, L, 0.42) * 3.0
                    + brushedHighlight(N, V, L, brushTangent(N), 0.4, 0.0) * 0.045;
        lit = satinEnv(R, N) * mix(F0, vec3(1.0), fres) * lathe * ao
            + uSunColor * sheen * F0 * NdotL * sunVis * ao
            + lampSoft * F0 * ao;
    } else if (part == 5) {                   // genuine chrome / mirror: roughness ~0.03, sharp SSR
        reflectivity = 1.0;
        vec3 F0 = vec3(0.62, 0.63, 0.62);
        lit = env * mix(F0, vec3(1.0), fres) * ao
            + solarGlare(R, L, 0.03, 1.0, 0.80) * mix(F0, vec3(1.0), fres) * sunVis * ao
            + (lampSharp + lampSoft * 0.12) * mix(F0, vec3(1.0), fres) * ao;
    } else if (part == 6 || brakeDisc) {      // black trim / plastic, or bare cast-iron brake disc
        if (wheelPart) {                      // dark cast iron, machined rings, deep inside the wheel
            float resolved = 1.0 - smoothstep(0.8, 3.14159265, wheelFw * 1600.0);
            float rings = 0.5 + 0.5 * sin(wheelRadius * 1600.0) * resolved * (1.0 - smoothstep(4.0, 30.0, camD));
            vec3 iron = vec3(0.075, 0.070, 0.066) * (0.80 + 0.20 * rings);
            lit = iron * diff * ao * 0.45
                + skyEnvRough(R, N, 0.50) * 0.045 * ao * 0.6
                + uSunColor * sunLobe(R, L, 0.50) * 0.05 * sunVis * ao;
        } else {
            lit = albedo * diff * ao + skyEnvRough(R, N, 0.55) * 0.025 * (0.4 + fres) * ao
                + sunDisc * 0.04 + lampSoft * 0.30 * ao;
        }
    } else if (part == 7) {                   // licence plate
        lit = albedo * diff + lampSoft * 0.15;
    } else if (part == 10) {                  // matte plastic: arches, cladding, skirts, mirror bases
        lit = albedo * diff * ao * 0.95 + env * 0.012 + lampSoft * 0.10 * ao;
    } else if (part == 11) {                  // LED signature (emission applied below); passive = glossy white lens
        float k = 0.08 + 0.55 * fres;
        lit = mix(albedo * diff, env * ao, k) + sunDisc * k + lampSharp * k;
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
        lit = mix(albedo * diff * ao, env * ao, k)
            + solarGlare(R, L, 0.05, max(k, 0.04), 0.65) * sunVis * ao
            + lampSharp * max(k, 0.03) * ao;
    } else if (part == 13) {                  // panoramic roof glass: darker, more mirror-like, privacy tinted
        reflectivity = 0.96;
        vec3 Nl = dot(N, V) < 0.0 ? -N : N;
        float cv = clamp(dot(Nl, V), 0.0, 1.0);
        float F1 = COAT_F0 + (1.0 - COAT_F0) * pow(1.0 - cv, 5.0);
        float F  = 2.0 * F1 / (1.0 + F1);
        vec3 Rg = reflect(-V, Nl);
        float sheen = smoothstep(-0.10, 0.90, Rg.z);
        lit = vec3(0.008, 0.011, 0.015) * diff * (1.0 - F)
              + skyEnv(Rg) * (0.24 + 0.55 * sheen) * F
              + solarGlare(Rg, L, 0.010, F, 1.0) * sunVis
              + lampSharp * F;
        lit *= mix(0.60, 1.0, ao);
    } else if (part == 3 || part == 4 || part == 14) {   // lamp interior: metallised reflector / projector optics
        float facet = baked ? 0.60 + 0.40 * lampHash(floor(vColor.xy * vec2(14.0, 9.0))) : 1.0;
        float shine = (part == 4) ? 0.05 : 0.14;                // chrome headlamp bowl vs red-metallised tail bowl
        float matte = (part == 4) ? 0.55 : 0.22;                // a chrome bowl is a mirror: little diffuse, mostly reflection
        lit = albedo * diff * ao * matte * facet
            + skyEnvRough(R, N, 0.30) * shine * (0.35 + 0.65 * fres) * ao * facet
            + sunDisc * shine * 0.9 * facet
            + lampSharp * shine * 0.35 * facet;
    } else {                                  // layered automotive paint (0 = instance colour, 8 = fixed)
        bool caliper = vWheelRole == 3;
        if (part == 0 || part == 8) reflectivity = 0.60;
        float gloss  = clamp(uPaintCoat.x, 0.0, 1.0);
        float lum    = dot(albedo, PHOTO_LUMA);
        float chroma = max(albedo.r, max(albedo.g, albedo.b))
                     - min(albedo.r, min(albedo.g, albedo.b));
        float variant = paintHash(vec3(dot(vColor, vec3(97.0, 51.0, 29.0)), 7.3, 3.1));
        // Neutral bright paint (white, light silver) is a plain solid finish.  A
        // pearl flip-flop needs coloured pigment to shift; on white it only reads
        // as cream/pearl rather than the intended showroom white.
        float neutralLight = (1.0 - smoothstep(0.025, 0.080, chroma))
                           * smoothstep(0.45, 0.65, lum);
        // Finish per colour: ~20 % single-stage solid, ~58 % two-coat metallic, ~22 % tri-coat pearl.
        float solid = ((part == 0 && variant < 0.20) || neutralLight > 0.5) ? 1.0 : 0.0;
        float tri   = (part == 0 && variant > 0.78 && neutralLight < 0.5) ? 1.0 : 0.0;
        float met   = (part == 0 ? 1.0 : 0.6) * clamp(uPaintCoat.z, 0.0, 2.0)
                    * mix(0.55, 1.0, smoothstep(0.03, 0.22, lum)) * (1.0 - solid);
        float pearl = (part == 0 ? 1.0 : 0.0) * clamp(uPaintCoat.w, 0.0, 1.0)
                    * max(smoothstep(0.20, 0.55, lum), tri) * (1.0 - neutralLight);
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
        // Broad coat lobe (a soft sheen around the sun) plus the actual solar
        // disc: the latter is scene-linear HDR, so bloom, anamorphic streak and
        // starburst spread it into the FH6-style glare.  The clear coat is
        // colourless, so white paint keeps a neutral white glare.
        float coatSun = pow(max(dot(Rc, L), 0.0), coatPow) * 0.10 * sunVis;
        float coatRough = mix(0.045, 0.14, 1.0 - gloss);
        vec3 coatGlare = solarGlare(Rc, L, coatRough, coatF, 1.0) * sunVis;
        float flakeSun = pow(max(dot(reflect(-V, Ng), L), 0.0), 1200.0) * sunVis * met
                       * 0.05 * mix(0.25, 1.0, flakeFade);
        float glitterSun = pow(max(dot(reflect(-V, Ngl), L), 0.0), 500.0) * sunVis * met * glit
                         * 0.12 * flakeFade;
        float farSheen = sunVis * (1.0 - flakeFade) * met * 0.004;
        // Base layer: pigment diffuse plus metallic flakes mirroring the sky through the coat.
        vec3 baseLit = paint * diff * ao;
        if (met > 0.0) {
            // Flakes mirror a *blurred* sky, not a second sharp reflection.  Sampling the
            // environment through each flake's own +-0.35 rad tilt turned the sun's aureole
            // in the sky table into scattered white sparkle on a panel that is merely
            // reflecting the sky -- including the shaded flank.  The real glint stays in
            // flakeSun / glitterSun, which are shadow- and cloud-gated.
            vec3 flakeEnv = skyEnvRough(Rc, Nc, 0.35);
            baseLit = mix(baseLit, paint * flakeEnv * 1.1 * ao, 0.20 * met);
        }
        // Clear coat is colourless: it reflects the (orange-peel perturbed) world, over the base.
        vec3 coatEnv = skyEnv(Rc) * ao;
        lit = mix(baseLit, coatEnv, coatF)
              + coatGlare * mix(0.55, 1.0, ao)
              + uSunColor * (coatSun + flakeSun + glitterSun + farSheen) * ao
              + lampSharp * coatF * mix(0.35, 1.0, ao)
              + lampSoft * 0.05 * ao;
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
