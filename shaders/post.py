# --- shaders/post.py ---
# Post-processing for the HDR scene buffer:
#
#   AE_METER_FS       scene -> 128x128 map of (weighted log2 luminance, weight)
#   AE_ADAPT_FS       mip-averaged meter + previous state -> new exposure (EV) in a 1x1 texture
#   BLOOM_PREFILTER_FS  soft-knee threshold in *exposed* HDR units (reads the adapted EV)
#   BLOOM_DOWN_FS     4x4 box reduce for the wide glow level
#   BLOOM_COMPOSITE_FS  exposure, glow, anamorphic streak, ACES, film look, grain, vignette
#
# The look stage is display-referred and has three presets (uLook):
#   0 neutral   1 Colorful (golden / teal)   2 Cinematic (green-teal city, warm practicals)

from .hdr import HDR_CONV
from .depth import SCENE_DEPTH

AE_METER_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uScene;
out vec4 FragColor;
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
void main() {
    // Each 128x128 texel stands for a block of the frame; nine taps spread over the block.
    float acc = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec3 c = texture(uScene, vUV + vec2(float(i), float(j)) * (1.0 / 384.0)).rgb;
            // Bright outliers (sun, sky seen from a shaded street, lamps) are capped well below their
            // true value: a camera exposing for a dark interior lets the window blow out.
            float l = clamp(dot(c, LUMA), 0.0, 4.0);
            acc += log2(max(l, 1e-3));
        }
    }
    float lg = acc / 9.0;
    // strongly centre-weighted: exposure follows the subject, not the frame
    vec2 q = vUV - 0.5;
    float w = 0.10 + 0.90 * exp(-dot(q, q) * 9.0);
    FragColor = vec4(lg * w, w, 0.0, 1.0);
}
"""

AE_ADAPT_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uMeter;     // 128x128, mip chain generated: level 7 is 1x1
uniform sampler2D uPrev;      // previous 1x1 state: r = EV, g = measured mean log2 luminance
uniform float uDt;
uniform float uKey;           // luminance the scene should average after exposure
uniform float uStrength;      // 0 = off, 1 = full correction
uniform float uMinEV, uMaxEV;
uniform float uSpeedBrighter; // 1/s when exposure must fall (walking into daylight: fast)
uniform float uSpeedDarker;   // 1/s when exposure must rise (walking into shade: slow)
uniform float uReset;
uniform float uDead;          // EV dead-band: small metering changes are ignored (stable, filmic exposure)
uniform float uFastK;         // share of a sudden change the "pupil" reflex answers at once
uniform float uFastRate;      // 1/s of that reflex
out vec4 FragColor;
void main() {
    vec2 m = textureLod(uMeter, vec2(0.5), 7.0).rg;
    float lg = m.r / max(m.g, 1e-4);
    float target = clamp((log2(max(uKey, 1e-4)) - lg) * uStrength, uMinEV, uMaxEV);
    vec4 pv = texelFetch(uPrev, ivec2(0), 0);
    // Two stages, like an eye or a film camera operator: a quick partial reflex (b: fast) and a
    // slow settle (g: slow).  Stepping into shade is first met with a dark frame that opens up;
    // stepping out of it blows the highlights and then closes down.
    float d = target - pv.b;
    float tgt = pv.b + sign(d) * max(abs(d) - uDead, 0.0);
    float rate = tgt < pv.b ? uSpeedBrighter : uSpeedDarker;
    float slow = uReset > 0.5 ? target : mix(pv.b, tgt, 1.0 - exp(-max(uDt, 0.0) * rate));
    float fast = uReset > 0.5 ? 0.0 : mix(pv.a, uFastK * (target - slow), 1.0 - exp(-max(uDt, 0.0) * uFastRate));
    FragColor = vec4(slow + fast, lg, slow, fast);
}
"""

BLOOM_PREFILTER_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uScene;
uniform sampler2D uAE;
uniform vec2 uTexel;
uniform float uThreshold;
uniform float uHdrIn;
uniform float uEVComp;
uniform float uAEOn;
out vec4 FragColor;
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
vec3 tapSafe(vec2 uv, float expo) {
    vec3 t = texture(uScene, uv).rgb;
    if (any(isnan(t)) || any(isinf(t))) return vec3(0.0);   // a single bad pixel must not blow up the blur
    return min(t * expo, vec3(24.0));
}
void main() {
    float ev = uEVComp + (uAEOn > 0.5 ? texelFetch(uAE, ivec2(0), 0).r : 0.0);
    float expo = uHdrIn > 0.5 ? exp2(ev) : 1.0;
    vec3 a = tapSafe(vUV + uTexel * vec2(-1.0, -1.0), expo);
    vec3 b = tapSafe(vUV + uTexel * vec2( 1.0, -1.0), expo);
    vec3 c = tapSafe(vUV + uTexel * vec2(-1.0,  1.0), expo);
    vec3 d = tapSafe(vUV + uTexel * vec2( 1.0,  1.0), expo);
    // Karis average: bright single pixels must not flicker through the blur
    float wa = 1.0 / (1.0 + dot(a, LUMA)), wb = 1.0 / (1.0 + dot(b, LUMA));
    float wc = 1.0 / (1.0 + dot(c, LUMA)), wd = 1.0 / (1.0 + dot(d, LUMA));
    vec3 col = (a * wa + b * wb + c * wc + d * wd) / (wa + wb + wc + wd);
    float lum = max(dot(col, LUMA), 1e-4);
    // Keep a smooth onset without admitting ordinary lights at half the cutoff.
    // The 20% knee starts at 80% of the threshold, in exposed HDR units.
    float knee = max(uThreshold * 0.2, 1e-3);
    float soft = clamp(lum - uThreshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    float w = max(soft, lum - uThreshold) / lum;
    FragColor = vec4(min(col * w, vec3(24.0)), 1.0);
}
"""

BLOOM_DOWN_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uImage;
uniform vec2 uTexel;
out vec4 FragColor;
void main() {
    vec3 s = texture(uImage, vUV + uTexel * vec2(-1.0, -1.0)).rgb
           + texture(uImage, vUV + uTexel * vec2( 1.0, -1.0)).rgb
           + texture(uImage, vUV + uTexel * vec2(-1.0,  1.0)).rgb
           + texture(uImage, vUV + uTexel * vec2( 1.0,  1.0)).rgb;
    FragColor = vec4(s * 0.25, 1.0);
}
"""

BLOOM_COMPOSITE_FS = ("""
#version 330 core
in vec2 vUV;
uniform sampler2D uScene;
uniform sampler2D uBloom;       // 1/4 res, blurred
uniform sampler2D uBloomWide;   // 1/16 res, blurred twice
uniform sampler2D uAE;
uniform float uStrength;        // near glow gain
uniform float uWide;            // wide glow gain
uniform float uStreak;          // anamorphic streak gain
uniform vec2 uResolution;
uniform float uNight;
uniform float uTime;
uniform float uGrade;           // look strength 0..1
uniform int   uLook;            // 0 neutral, 1 Colorful, 2 Cinematic, 3 Cyberpunk
uniform float uHdrIn;           // 1: scene is linear HDR, tone-map here
uniform float uEVComp;
uniform float uAEOn;
uniform float uContrast;        // >1: deeper darks, brighter lights, pivoting on the scene average
uniform float uPivot;           // exposed luminance that stays put (the eye-adaptation key)
uniform float uLetterbox;       // 0 = off, else bar height as a fraction of the frame
uniform sampler2D uFlareTex;    // screen-space star + ghosts (1/4 res), see STAR_FS
uniform float uFlareTexOn;
uniform float uFlareMaxBrightness;  // exposed-HDR luminance ceiling
uniform float uVigStrength;     // 0 = off; darkening at the frame corners (0..1)
uniform float uVigStart;        // radius (0 centre .. 1 corner) where the falloff begins
uniform sampler2D uDepth;       // scene depth: tells the glare how much open sky surrounds the sun
uniform vec2  uSunUV;           // sun position in the frame (may lie just outside 0..1)
uniform float uSunGlare;        // 0 = off; already scaled by sun height and overcast
uniform vec3  uSunTint;
uniform mat4  uInvViewProj;     // clip (uv, depth) -> world, this frame
uniform mat4  uPrevViewProj;    // the same world point in the previous frame's clip space
uniform float uMotionBlur;      // 0 = off; shutter length as a fraction of the frame interval
uniform float uMotionMax;       // longest smear, as a fraction of the frame
uniform float uCarOn;           // 1 = the player car has a rigid-motion exemption
uniform vec4  uCarCur;          // x, y, z (world), heading: this frame
uniform vec4  uCarPrev;         // the same, previous frame
uniform vec3  uCarBox;          // half length, half width, height (metres, incl. margin)
out vec4 FragColor;

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
""" + HDR_CONV + SCENE_DEPTH + """
float skyAt(vec2 uv) {
    uv = clamp(uv, vec2(0.001), vec2(0.999));
    return depthSky(texture(uDepth, uv).r) ? 1.0 : 0.0;
}

// Soft circular sun bloom, as a lens/sensor does with a very bright disc in hazy air: a hot,
// rounded core, a smooth exponential halo and a long faint tail. It is radially symmetric (no
// spikes), added in exposed HDR light before tone mapping so the core clips to white and the
// rest rolls off like film. Open sky around the sun scales it; buildings in front of the sun
// still pick up the glow across their edges instead of cutting it off.
vec3 sunGlare() {
    if (uSunGlare <= 0.0) return vec3(0.0);
    float asp = uResolution.x / uResolution.y;
    vec2 d = (vUV - uSunUV) * vec2(asp, 1.0);
    float r = length(d);
    if (r > 1.8) return vec3(0.0);
    vec2 beyond = max(max(-uSunUV, uSunUV - 1.0), vec2(0.0));
    float outside = max(beyond.x, beyond.y);
    float vis;
    if (outside <= 0.0) {
        float s = 3.0 * skyAt(uSunUV);
        for (int i = 0; i < 8; ++i) {
            float a = 0.7853982 * float(i);
            vec2 o = vec2(cos(a) / asp, sin(a));
            s += skyAt(uSunUV + o * 0.012) + 0.75 * skyAt(uSunUV + o * 0.035);
        }
        vis = s / 17.0;
        vis = vis * vis * (3.0 - 2.0 * vis);
    } else {
        vis = 1.0 - smoothstep(0.0, 0.35, outside);     // sun just out of frame: veiling glare only
    }
    float core  = exp(-r * r / (2.0 * 0.011 * 0.011));
    float inner = exp(-r / 0.040);
    float mid   = exp(-r / 0.130);
    float wide  = 1.0 / (1.0 + (r / 0.32) * (r / 0.32));
    float g = 3.0 * core + 1.5 * inner + 0.40 * mid + 0.09 * wide;
    return uSunTint * (g * uSunGlare * vis);
}

float gradeHash(vec2 p) {
    p = fract(p * vec2(443.897, 441.423));
    p += dot(p, p.yx + 19.19);
    return fract((p.x + p.y) * p.x);
}

// Camera motion blur: reproject each pixel through its depth into the previous frame and
// gather along that screen-space path.  Static geometry smears along its true image motion;
// the sky sits on the far plane, so it follows the camera's rotation.
vec2 motionPath(vec2 uv) {
    float d = texture(uDepth, uv).r;
    vec4 world = uInvViewProj * vec4(uv * 2.0 - 1.0, depthNdc(d), 1.0);
    if (abs(world.w) < 1e-6) return vec2(0.0);
    vec3 wp = world.xyz / world.w;
    if (uCarOn > 0.5) {
        // Pixels on the player's car follow the car, not the static world: take their
        // previous position through the car's own frame-to-frame motion.  Without this the
        // chase-locked car is smeared as if the whole scene were swept past it.
        vec2 dxy = wp.xy - uCarCur.xy;
        float cc = cos(uCarCur.w), sc = sin(uCarCur.w);
        vec2 lp = vec2(cc * dxy.x + sc * dxy.y, -sc * dxy.x + cc * dxy.y);
        float lz = wp.z - uCarCur.z;
        float out_ = max(max(abs(lp.x) - uCarBox.x, abs(lp.y) - uCarBox.y),
                         max(0.06 - lz, lz - uCarBox.z));
        float m = 1.0 - smoothstep(0.0, 0.12, out_);
        if (m > 0.0) {
            float cp = cos(uCarPrev.w), sp = sin(uCarPrev.w);
            vec3 pw = vec3(uCarPrev.xy + vec2(cp * lp.x - sp * lp.y, sp * lp.x + cp * lp.y),
                           uCarPrev.z + lz);
            wp = mix(wp, pw, m);
        }
    }
    vec4 prev = uPrevViewProj * vec4(wp, 1.0);
    if (prev.w <= 1e-5) return vec2(0.0);
    vec2 vel = uv - (prev.xy / prev.w * 0.5 + 0.5);
    float len = length(vel);
    if (len > uMotionMax) vel *= uMotionMax / max(len, 1e-6);
    return vel;
}

// Exposed HDR light, gathered along the motion path with a centre-weighted shutter.
vec3 motionScene(vec2 uv, float expo) {
    vec3 centre = texture(uScene, uv).rgb;
    if (uMotionBlur <= 0.0) return centre * expo;
    vec2 vel = motionPath(uv) * uMotionBlur;
    if (dot(vel, vel) < 1e-9) return centre * expo;
    const int TAPS = 12;
    float jitter = gradeHash(gl_FragCoord.xy);   // fixed per-pixel dither: no frame-to-frame flicker
    vec3 sum = centre * 0.5;                 // keep the pixel itself anchored while it moves
    float wsum = 0.5;
    for (int i = 0; i < TAPS; ++i) {
        float t = ((float(i) + jitter) / float(TAPS)) * 2.0 - 1.0;
        float w = 1.0 - 0.55 * abs(t);
        sum += texture(uScene, clamp(uv + vel * t, vec2(0.0005), vec2(0.9995))).rgb * w;
        wsum += w;
    }
    return sum / wsum * expo;
}

// Filmic contrast in linear light, rotating around the scene's own average so a bright day and a
// dark interior both keep their mid-tones.  Scaling by luminance keeps every hue.
vec3 filmContrast(vec3 c) {
    float l = dot(c, LUMA);
    if (l < 1e-6 || uContrast == 1.0) return c;
    return c * pow(l / uPivot, uContrast - 1.0);
}

vec3 toneMap(vec3 c) {
    return uHdrIn > 0.5 ? pow(acesFwd(filmContrast(c)), vec3(1.0 / 2.2)) : clamp(c, 0.0, 1.0);
}

// ------------------------------------------------------------------ Colorful
// golden-warm highlights, teal-blue shadows, punchy contrast, rich saturation.
vec3 gtaGrade(vec3 c, float night) {
    float l  = dot(c, LUMA);
    float sh = 1.0 - smoothstep(0.0, 0.55, l);
    float hi = smoothstep(0.40, 1.00, l);
    vec3 shadowTint = mix(vec3(0.88, 1.01, 1.10), vec3(0.94, 0.90, 1.14), night);
    vec3 highTint   = mix(vec3(1.08, 1.00, 0.86), vec3(1.12, 0.94, 0.98), night);
    c *= mix(vec3(1.0), shadowTint, sh * 0.60);
    c *= mix(vec3(1.0), highTint,   hi * 0.70);
    c = clamp(c, 0.0, 1.0);
    c = mix(c, c * c * (3.0 - 2.0 * c), 0.42);
    float l2  = dot(c, LUMA);
    float chroma = max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
    float sat = 1.20 * (1.0 + 0.32 * (1.0 - clamp(chroma * 1.6, 0.0, 1.0)));
    c = mix(vec3(l2), c, sat);
    float g = clamp((c.g - max(c.r, c.b)) * 4.0, 0.0, 1.0);
    c = mix(c, c * vec3(1.06, 1.05, 0.88), g * 0.45);
    float b = clamp((c.b - c.r) * 2.2, 0.0, 1.0);
    c = mix(c, c * vec3(0.92, 1.02, 1.07), b * 0.40);
    c = c * 0.965 + vec3(0.010, 0.014, 0.024);
    return clamp(c, 0.0, 1.0);
}

// ------------------------------------------------------- Cinematic
// Reference look: green-cyan shadows, olive mids and pale yellow-green daylight.
// Warm practicals keep their amber color, while the brightest whites stay clean.
vec3 matrixGrade(vec3 c, float night) {
    c = clamp(c, 0.0, 1.0);
    float l0 = dot(c, LUMA);
    float warm = smoothstep(0.06, 0.30, c.r - c.b) * smoothstep(0.18, 0.55, l0);
    // Turn blue sky/fill toward the reference's desaturated cyan-green.
    float blue = smoothstep(0.015, 0.20, c.b - c.r);
    c *= mix(vec3(1.0), vec3(0.92, 1.12, 0.78), blue * 0.70);
    c *= l0 / max(dot(c, LUMA), 1e-5);
    c = mix(vec3(l0), c, 1.0 - blue * 0.42);      // milkier, less saturated sky like the reference
    c = mix(vec3(l0), c, mix(mix(0.80, 0.90, night), 1.0, warm));
    float sh = 1.0 - smoothstep(0.08, 0.50, l0);
    float hi = smoothstep(0.42, 0.85, l0);
    float md = max(0.0, 1.0 - sh - hi);
    vec3 tint = mix(vec3(0.80, 1.04, 0.94), vec3(.66, 1.07, 1.03), night) * sh
              + mix(vec3(0.91, 1.035, 0.89), vec3(.91, 1.05, .80), night) * md
              + vec3(0.99, 1.025, 0.88) * hi;
    tint /= dot(tint, LUMA);  // the grade should not change exposure
    float protect = max(warm * 0.90, smoothstep(0.88, 1.0, l0));
    c *= mix(tint, vec3(1.0), protect);
    c = mix(c, c * c * (3.0 - 2.0 * c), mix(.25, .14, night));
    // Slightly lifted, green-tinted blacks: the matte, hazy toe of the demo footage.
    c = c * 0.975 + vec3(0.009, 0.015, 0.012) * mix(1.0, 0.35, night);
    return clamp(c, 0.0, 1.0);
}

// ------------------------------------------------------- Cyberpunk
// Night City: cold teal-cyan shadows, hot magenta / amber highlights, lifted tinted blacks,
// strong contrast.  Saturated pixels (neon, signage) are protected from the split-toning so
// their own hue survives; vibrance lifts the muted ones.
vec3 cyberGrade(vec3 c, float night) {
    c = clamp(c, 0.0, 1.0);
    float l0 = dot(c, LUMA);
    float chroma = max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
    float sh = 1.0 - smoothstep(0.02, 0.50, l0);
    float hi = smoothstep(0.30, 0.95, l0);
    vec3 shadowTint = mix(vec3(0.84, 1.03, 1.10), vec3(0.74, 1.00, 1.20), night);   // teal-cyan
    vec3 highTint   = mix(vec3(1.10, 1.02, 0.86), vec3(1.16, 0.90, 1.04), night);   // amber -> magenta
    shadowTint /= dot(shadowTint, LUMA);
    highTint   /= dot(highTint, LUMA);
    float protect = smoothstep(0.22, 0.60, chroma) * 0.75;
    float dayK = mix(0.35, 1.0, night);   // the split-toning is a night-time effect; daylight stays natural
    vec3 tint = mix(vec3(1.0), shadowTint, sh * 0.90 * dayK) * mix(vec3(1.0), highTint, hi * 0.65 * dayK);
    c *= mix(tint, vec3(1.0), protect);
    c = mix(c, c * c * (3.0 - 2.0 * c), mix(0.40, 0.46, night));
    float l2 = dot(c, LUMA);
    float chroma2 = max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
    float vib = 1.0 + (0.14 + 0.34 * (1.0 - clamp(chroma2 * 2.2, 0.0, 1.0))) * mix(0.30, 1.0, night);
    c = mix(vec3(l2), c, vib);
    // matte, teal-violet toe: the blacks of a hazy city are never empty
    c = c * 0.97 + vec3(0.006, 0.015, 0.024) * mix(0.35, 1.5, night);
    return clamp(c, 0.0, 1.0);
}

vec3 look(vec3 c) {
    if (uGrade <= 0.0 || uLook == 0) return c;
    vec3 g = (uLook == 3) ? cyberGrade(c, uNight) : (uLook == 2) ? matrixGrade(c, uNight) : gtaGrade(c, uNight);
    return mix(c, g, uGrade);
}

vec3 sceneAt(vec2 uv, float expo) {
    return texture(uScene, uv).rgb * expo;
}

void main() {
    vec2 px = 1.0 / uResolution;
    vec2 q = vUV - 0.5;
    float r2 = dot(q, q);
    float expo = uHdrIn > 0.5 ? exp2(uEVComp + (uAEOn > 0.5 ? texelFetch(uAE, ivec2(0), 0).r : 0.0)) : 1.0;

    // lens fringing toward the corners (stronger for the filmic Matrix look)
    // Off: splitting R/G/B turns every thin bright line (fins, mullions, roof edges) into a
    // magenta + green pair of stripes.  Raise it again for a lens-fringe look.
    float caAmt = 0.0;
    vec2 ca = q * r2 * caAmt * uGrade;
    vec3 hdr = motionScene(vUV, expo);
    if (caAmt > 0.0) hdr = vec3(motionScene(vUV + ca, expo).r, hdr.g, motionScene(vUV - ca, expo).b);

    // glow: near + wide, added in exposed HDR light before the tone-map
    vec3 glow = vec3(0.0);
    if (uStrength > 0.0) glow += texture(uBloom, vUV).rgb * uStrength;
    if (uWide > 0.0)     glow += texture(uBloomWide, vUV).rgb * uWide;
    // anamorphic horizontal streak taken from the blurred near glow
    if (uStreak > 0.0) {
        vec3 st = vec3(0.0);
        float wsum = 0.0;
        for (int i = -7; i <= 7; ++i) {
            float w = exp(-float(i * i) * 0.045);
            st += texture(uBloom, vUV + vec2(float(i) * 7.0 * px.x, 0.0)).rgb * w;
            wsum += w;
        }
        vec3 streakTint = (uLook == 3) ? vec3(0.60, 0.85, 1.00) : (uLook == 2) ? vec3(0.55, 1.00, 0.85) : vec3(0.80, 0.92, 1.00);
        glow += st / wsum * uStreak * streakTint;
    }
    glow += sunGlare();
    if (uFlareTexOn > 0.5 && uFlareMaxBrightness > 0.0) {
        vec3 flare = texture(uFlareTex, vUV).rgb;
        // Roll off overlapping bright objects without clipping their colour or dimming small glints.
        float brightness = dot(flare, LUMA);
        hdr += flare / (1.0 + brightness / uFlareMaxBrightness);
    }
    vec3 col = toneMap(hdr + glow);

    // light unsharp mask on the tone-mapped image: keeps facade detail crisp after SSR
    if (uGrade > 0.0) {
        vec3 blur = 0.25 * (toneMap(sceneAt(vUV + vec2( px.x, 0.0), expo))
                          + toneMap(sceneAt(vUV + vec2(-px.x, 0.0), expo))
                          + toneMap(sceneAt(vUV + vec2(0.0,  px.y), expo))
                          + toneMap(sceneAt(vUV + vec2(0.0, -px.y), expo)));
        col += (col - blur) * 0.12 * uGrade;     // was 0.30: it rang into bright halos on thin edges
    }

    col = look(col);

    // 35 mm grain, strongest in the mid-tones
    float gr = gradeHash(gl_FragCoord.xy + fract(uTime) * 61.0) - 0.5;
    float l = dot(col, LUMA);
    float grain = (uLook >= 2) ? 0.010 : 0.022;
    col += gr * grain * uGrade * (1.0 - abs(l * 2.0 - 1.0) * 0.6);

    // cinematic vignette: aspect-corrected oval (0 at the centre, 1 at the corners), an
    // artistic falloff plus a touch of natural cos^4 light loss.  The old one used the raw
    // uv radius, which peaks at 0.5 in the corners, so it never darkened more than ~9%.
    if (uVigStrength > 0.0) {
        float asp = uResolution.x / uResolution.y;
        vec2 vq = (vUV - 0.5) * vec2(asp, 1.0) * 2.0;
        float vr = length(vq) / length(vec2(asp, 1.0));
        float art = 1.0 - uVigStrength * pow(smoothstep(uVigStart, 1.05, vr), 1.4);
        float nat = 1.0 / pow(1.0 + 0.35 * vr * vr, 2.0);
        float vig = art * mix(1.0, nat, 0.35);
        col *= vig;
        // lenses also lose a little saturation and warmth toward the edge
        float lv = dot(col, LUMA);
        col = mix(vec3(lv), col, mix(1.0, 0.90, 1.0 - vig));
    }

    // 1 LSB triangular dither against banding in the dark gradients
    col += (gradeHash(gl_FragCoord.xy * 1.37 + 7.0) - 0.5) / 255.0;

    // optional cinemascope bars
    if (uLetterbox > 0.0 && (vUV.y < uLetterbox || vUV.y > 1.0 - uLetterbox)) col = vec3(0.0);
    FragColor = vec4(col, 1.0);
}
""")


# Screen-space lens flare, computed from the image itself at 1/4 resolution.  Any bright
# pixel -- the sun, a glint on a window, a lamp -- is a light source:
#   * starburst: the aperture-blade diffraction spikes, gathered along 2*blades (odd) or blades
#     (even) directions: sharp near the source from the 1/4-res highlights, soft and long from
#     the 1/16-res wide blur.  f/22 reaches further than f/14.
#   * ghosts: the classic flipped-through-the-centre reflections with chromatic dispersion
#   * a rainbow comet: one elongated, prism-smeared ghost (the coloured streak in film shots)
#   * a faint halo ring
STAR_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uNear;      // 1/4 res blurred highlights (exposed HDR, knee already applied)
uniform sampler2D uFar;       // 1/16 res wide blur of the same
uniform float uAspect;        // width / height
uniform float uThr;           // extra threshold on top of the bloom knee
uniform float uFStop;
uniform int   uBlades;
uniform float uStarGain;
uniform float uGhostGain;
out vec4 FragColor;

const float TAU = 6.2831853;
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

vec3 hi(vec3 c, float thr) {
    float l = dot(c, LUMA);
    return c * (max(l - thr, 0.0) / max(l, 1e-4));
}
vec3 nearHi(vec2 uv, float thr) { return hi(texture(uNear, uv).rgb, thr); }
vec3 farHi(vec2 uv)  { return hi(texture(uFar, uv).rgb * 3.0, uThr); }
vec3 hue(float t) { return clamp(abs(fract(vec3(t) + vec3(0.0, 0.6667, 0.3333)) * 6.0 - 3.0) - 1.0, 0.0, 1.0); }

void main() {
    float fn = clamp((uFStop - 8.0) / 14.0, 0.0, 1.0);          // 1 at f/22, ~0.43 at f/14
    vec2 toUV = vec2(1.0 / uAspect, 1.0);                       // height units -> uv
    vec3 col = vec3(0.0);

    // ---------------- ghosts, comet, halo
    if (uGhostGain > 0.0) {
        float gthr = uThr * 2.0;                                // only strong lights make ghosts
        vec2 c = vec2(0.5);
        vec2 toC = c - vUV;
        vec2 gv = toC * 0.38;
        vec2 gdir = normalize(gv + 1e-5);
        vec3 gh = vec3(0.0);
        for (int i = 1; i <= 4; ++i) {
            vec2 o = fract(vUV + gv * float(i));
            float w = pow(clamp(1.0 - length((c - o) * vec2(uAspect, 1.0)) / 0.80, 0.0, 1.0), 4.0);
            vec3 s = vec3(nearHi(o + gdir * 0.010, gthr).r, nearHi(o, gthr).g, nearHi(o - gdir * 0.010, gthr).b);
            gh += s * w * (i == 1 ? vec3(1.0, 0.55, 0.30) : (i == 2 ? vec3(0.45, 0.85, 1.0) : (i == 3 ? vec3(0.85, 0.60, 1.0) : vec3(1.0, 0.85, 0.5))));
        }
        // rainbow comet: a radial smear of flipped samples, one hue per sample
        vec3 comet = vec3(0.0);
        for (int k = 0; k < 8; ++k) {
            float t = float(k) / 7.0;
            vec2 o = c + (vUV - c) * (-0.52 - 0.34 * t);
            float w = pow(clamp(1.0 - length((o - c) * vec2(uAspect, 1.0)) / 0.75, 0.0, 1.0), 2.0);
            comet += nearHi(o, gthr) * hue(0.02 + 0.80 * t) * w;
        }
        comet *= 0.55;
        // halo ring
        vec2 hv = gdir * 0.34;
        vec2 ho = fract(vUV + hv);
        float hw = pow(clamp(1.0 - length((c - ho) * vec2(uAspect, 1.0)) / 0.80, 0.0, 1.0), 5.0);
        vec3 halo = vec3(nearHi(ho + gdir * 0.012, gthr).r, nearHi(ho, gthr).g, nearHi(ho - gdir * 0.012, gthr).b) * hw;
        col += (gh * 1.0 + comet * 1.0 + halo * 0.8) * uGhostGain;
    }
    FragColor = vec4(max(col, vec3(0.0)), 1.0);
}
"""


# ---------------------------------------------------------------- starburst (streak passes)
# The aperture-blade diffraction star.  Every bright pixel -- the sun, a window glint, a lamp --
# is smeared along four axes (two-sided, so 4 axes = 8 spikes for an 8-blade iris).  A streak
# of 128 texels at the default half-length costs only 4 passes of 7 taps per axis: pass k
# gathers at multiples of length * 4^k texels.  Each pass is normalized so overlapping
# highlights spread their brightness instead of multiplying it at every step; the renderer
# shares the star gain across the four axes.
# The source is squared, so only the hot core of a highlight survives and the spikes taper
# from a thin line instead of being as wide as the glow.
STAR_SRC_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uScene;     // full-res HDR scene: the source must be sharp, not the blurred glow
uniform sampler2D uAE;
uniform vec2 uTexel;          // one full-res texel
uniform float uThr;           // exposed-HDR level where a highlight starts to make spikes
uniform float uClamp;         // cap, so the sun does not drown every other glint
uniform float uHdrIn;
uniform float uEVComp;
uniform float uAEOn;
out vec4 FragColor;
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
vec3 tapSafe(vec2 uv, float expo) {
    vec3 t = texture(uScene, uv).rgb;
    if (any(isnan(t)) || any(isinf(t))) return vec3(0.0);
    return t * expo;
}
void main() {
    float ev = uEVComp + (uAEOn > 0.5 ? texelFetch(uAE, ivec2(0), 0).r : 0.0);
    float expo = uHdrIn > 0.5 ? exp2(ev) : 1.0;
    // max-pool the 4x4 block: a few-pixel core must survive the 4x downsample
    vec3 a = tapSafe(vUV + uTexel * vec2(-1.0, -1.0), expo);
    vec3 b = tapSafe(vUV + uTexel * vec2( 1.0, -1.0), expo);
    vec3 c = tapSafe(vUV + uTexel * vec2(-1.0,  1.0), expo);
    vec3 d = tapSafe(vUV + uTexel * vec2( 1.0,  1.0), expo);
    vec3 m = max(max(a, b), max(c, d));
    float l = dot(m, LUMA);
    float v = min(max(l - uThr, 0.0), uClamp);
    v = v * v / uClamp;                           // spike strength ~ brightness^2: only the hottest glints star
    FragColor = vec4(m * (v / max(l, 1e-4)), 1.0);
}
"""

STAR_STREAK_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uImage;
uniform vec2  uStep;          // uv step for i = 1 (already x length x 4^pass)
uniform float uDecay;         // per-tap decay for this pass
uniform float uGain;          // 1 on the intermediate passes, the star gain on the last
uniform float uSharp;         // >0 on the last pass: contrast curve that thins the lines
out vec4 FragColor;
vec3 streakSample(vec2 uv) {
    // The shared bloom textures clamp to their edges.  Off-screen taps must be black,
    // otherwise a bright edge pixel is repeated along the whole diffraction spike.
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return vec3(0.0);
    return texture(uImage, uv).rgb;
}
void main() {
    vec3 acc = texture(uImage, vUV).rgb;
    float w = 1.0;
    float wsum = 1.0;
    for (int i = 1; i <= 3; ++i) {
        w *= uDecay;
        acc += w * (streakSample(vUV + uStep * float(i))
                  + streakSample(vUV - uStep * float(i)));
        wsum += 2.0 * w;
    }
    acc /= wsum;
    if (uSharp > 0.0) acc = acc * acc / (acc + uSharp);
    FragColor = vec4(acc * uGain, 1.0);
}
"""
