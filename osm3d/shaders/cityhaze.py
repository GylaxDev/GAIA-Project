"""Volumetric city haze: the coloured, light-catching air that gives Night City its soul.

The per-surface ``cityFog`` in hdr.py is a plain distance fade towards one sky colour, which
is why night streets read as empty and flat.  This pass runs on the finished HDR frame
(after reflections and eye adaptation, before bloom) and ray-marches a real participating
medium from the camera to every pixel's depth:

  * height fog   dense in the street canyon, thinning out above the rooftops, so towers
                 climb out of the haze instead of fading uniformly
  * drifting     low-frequency noise advected by the weather's wind, so the air has
                 patches and slowly moves; plus ground-hugging wisps (steam, vents)
  * coloured     ambient light is tinted by teal / magenta / amber city zones that vary
                 across the map, the way different districts glow different colours
  * lit by the   nearby neon signs scatter into the air analytically (a halo around each
    city        sign, tinted by its own colour) and oncoming headlights throw visible
                shafts; both come from the same light tables the surfaces already use
  * physical     daylight haze is still ambient + sun, in the scene's calibrated units;
                 the stylised glow terms are defined in *exposed* units (what the viewer
                 sees), so the look stays stable across exposure changes

Additive emitters (lamp sprites, the moon, the sun) write no depth, so the depth under them
is the background's.  Attenuating them by the background's transmittance would dim a street
lamp against the sky by the whole length of the view ray, so very bright pixels are treated
as emitters and keep most of their light.

Everything is uniform-driven (see Renderer.city_haze_* settings) and costs one full-screen
pass: ``uHazeSteps`` samples of the medium plus one cheap loop over the neon table.
"""

from .depth import SCENE_DEPTH
from .weather import HEADLIGHT_BEAMS, HEADLIGHT_REACH

HAZE_MAX_STEPS = 28

def _tint(rgb):
    """Normalise a colour to luminance 1, so a district tint changes hue, not brightness."""
    y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return tuple(c / y for c in rgb)

# Colour of the city's glow in the air at night: the warm tint of the engine's own urban sky
# horizon (sodium / LED light scattered by smog), normalised to luminance 1.  There is no
# artificial palette: by day the haze is lit only by the sky ambient and the sun.
HAZE_GLOW_TINT = _tint((0.75, 0.64, 0.48))

# Tuned on a night canyon street against the real post chain (eye adaptation + bloom + grade).
# Keys map to ``uHaze<Key>`` uniforms, except AmbientDay / AmbientNight (blended by night).
HAZE_DEFAULTS = dict(
    Density=0.0020,
    DayDensity=0.45,       # fraction of the density kept by day (storm / rain fog is already thick)        # extinction at street level, 1/m (~2 km visibility at ground level)
    Height=70.0,           # scale height of the layer, m
    Noise=0.65,            # patchiness
    NoiseScale=85.0,       # metres per noise cell
    Aniso=0.5,             # forward scattering
    Ground=0.7,            # ground-hugging wisps
    MaxDist=2500.0,
    AmbientDay=0.9,        # physical ambient scattering in daylight
    AmbientNight=0.05,     # ... kept low after dark so the air stays dark and the colour comes from
    Glow=0.011,            #     the city glow, in exposed units
    Neon=0.0,              # neon sign halos in the air: off (they read as saturated pink / cyan blobs)
    Beam=1.5,              # headlight shaft gain
    Lamp=12.0,             # street-lamp halo gain
    Emitter=1.0,
    Steps=12,
)
HAZE_MAX_LAMPS = 32
HAZE_NEON_REACH = 150.0      # metres: a sign lights the air no further than this from the ray

HAZE_FS = (f"""
#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uScene;      // finished HDR frame (scene-referred, before eye adaptation)
uniform sampler2D uDepth;
uniform sampler2D uAE;         // 1x1: r = exposure (EV) the composite will apply this frame
uniform float uAEOn, uEVComp;
uniform mat4  uInvViewProj;
uniform vec3  uCamPos;
uniform float uTime;
uniform vec2  uWindOffset;     // integrated wind displacement, metres

// Lighting, in the scene's own units (the same uniforms every surface shader reads).
uniform float uExposure;       // sky.exposure: scene light -> HDR buffer
uniform float uNight;          // 0 day .. 1 night
uniform vec3  uSunDir;         // towards the dominant light (sun, or moon at night)
uniform vec3  uSunColor;       // its irradiance
uniform vec3  uAmbColor;       // diffuse sky / street bounce
uniform float uWeather;        // density multiplier from rain / storm / fog

// Look.
uniform float uHazeDensity;    // extinction at street level, 1/m
uniform float uDensityScale;   // 1 at night, lower by day (the weather's own fog is already thick)
uniform float uHazeHeight;     // scale height of the haze layer, m
uniform float uHazeFloor;      // world z of the street
uniform float uHazeNoise;      // 0 uniform .. 1 strongly patchy
uniform float uHazeNoiseScale; // metres per noise cell
uniform float uHazeAniso;      // Henyey-Greenstein g
uniform float uHazeGround;     // ground-hugging wisps (steam / mist)
uniform float uHazeMaxDist;    // march limit, m
uniform float uHazeAmbient;    // gain on the physical ambient scattering
uniform float uHazeGlow;       // night glow of the city in the air (exposed units)
uniform float uHazeNeon;       // neon halo gain (exposed units * m^2)
uniform float uHazeBeam;       // headlight shaft gain
uniform vec3  uHazeGlowTint;   // colour of the city glow in the air (luma 1)
uniform float uHazeEmitter;    // 0..1 how much very bright pixels dodge the fog
uniform int   uHazeSteps;

#define MAX_NEON 48
uniform vec3 uNeonPos[MAX_NEON];
uniform vec3 uNeonCol[MAX_NEON];
uniform int  uNumNeon;

#define MAX_LAMPS {HAZE_MAX_LAMPS}
uniform vec3 uLampPos[MAX_LAMPS];   // nearest street lamps
uniform vec3 uLampCol[MAX_LAMPS];
uniform int  uNumLamps;
uniform float uHazeLamp;            // street-lamp halo gain

#define MAX_BEAMS {HEADLIGHT_BEAMS}
uniform vec4 uBeamPos[MAX_BEAMS];   // xyz lamp position
uniform vec4 uBeamDir[MAX_BEAMS];   // xyz unit axis, w = power
uniform int  uBeamCount;

#define HAZE_MAX_STEPS {HAZE_MAX_STEPS}
#define NEON_REACH {HAZE_NEON_REACH:.1f}
#define BEAM_REACH {HEADLIGHT_REACH:.1f}
""" + SCENE_DEPTH + """
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}
float vnoise(vec3 x) {
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i),                    hash13(i + vec3(1, 0, 0)), f.x),
                   mix(hash13(i + vec3(0, 1, 0)),    hash13(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(hash13(i + vec3(0, 0, 1)),    hash13(i + vec3(1, 0, 1)), f.x),
                   mix(hash13(i + vec3(0, 1, 1)),    hash13(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

// Interleaved gradient noise: fixed per pixel, so the sampling pattern never crawls.
float ign(vec2 px) { return fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715)))); }

// Henyey-Greenstein, normalised over the sphere.
float hgPhase(float mu, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (12.566371 * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-3), 1.5));
}

// Extinction (1/m) at a world position.
float hazeSigma(vec3 p, float aloft) {
    float h = p.z - uHazeFloor;
    float base = exp(-max(h, 0.0) / uHazeHeight);
    if (h < 0.0) base *= exp(h / 25.0);
    vec3 q = (p + vec3(uWindOffset * 2.0, 0.0)) / uHazeNoiseScale;
    float n = vnoise(q + vec3(0.0, 0.0, uTime * 0.004));
    float patchy = max(1.0 + uHazeNoise * (n - 0.5) * 3.0, 0.06);
    float wisps = 0.0;
    if (uHazeGround > 0.0 && h < 30.0) {
        wisps = uHazeGround * exp(-max(h, 0.0) / 7.0)
              * smoothstep(0.52, 0.80, vnoise(q * 2.1 + 9.0));
    }
    return uHazeDensity * uDensityScale * uWeather * (base * patchy + wisps) * aloft;
}

// Closed-form single scattering of a point source along the ray [0, dist]: the inverse-square
// falloff integrates to an atan difference, so a ray passing a lamp at 3 m still gets its full
// halo with no step noise.  `size` softens the source (a sign is bigger than a lamp).
vec3 lightHalo(vec3 pos, vec3 col, vec3 ro, vec3 rd, float dist, float tau, float size) {
    vec3 wv = pos - ro;
    float tc = dot(wv, rd);
    if (tc < -NEON_REACH || tc > dist + NEON_REACH) return vec3(0.0);
    float h2 = max(dot(wv, wv) - tc * tc, 0.0);
    if (h2 > NEON_REACH * NEON_REACH) return vec3(0.0);
    float hh = sqrt(h2 + size);
    float integ = (atan((dist - tc) / hh) - atan(-tc / hh)) / hh;
    float lh = pos.z - uHazeFloor;
    float sigmaL = uHazeDensity * uDensityScale * uWeather * exp(-max(lh, 0.0) / uHazeHeight);
    float reach = 1.0 - smoothstep(NEON_REACH * 0.5, NEON_REACH, sqrt(h2));
    return col * (integ * sigmaL * reach * exp(-tau * clamp(tc / dist, 0.0, 1.0)));
}

void main() {
    vec4 scene = texture(uScene, vUV);
    float d = texture(uDepth, vUV).r;
    vec2 ndc = vUV * 2.0 - 1.0;

    vec4 np = uInvViewProj * vec4(ndc, uReverseZ > 0.5 ? 1.0 : -1.0, 1.0);
    vec3 ro = uCamPos;
    vec3 rd = normalize(np.xyz / np.w - ro);

    float dist = uHazeMaxDist;
    if (!depthSky(d)) {
        vec4 wp = uInvViewProj * vec4(ndc, depthNdc(d), 1.0);
        if (abs(wp.w) > 1e-9) dist = min(length(wp.xyz / wp.w - ro), uHazeMaxDist);
    }

    float ev = uEVComp + (uAEOn > 0.5 ? texelFetch(uAE, ivec2(0), 0).r : 0.0);
    float invExpo = exp2(-ev);

    float night = clamp(uNight, 0.0, 1.0);
    float mu = dot(rd, uSunDir);
    float sunPhase = hgPhase(mu, uHazeAniso);

    // Samples are packed towards the camera (t ~ u^1.4): the street in front of the viewer
    // matters most, and the far haze is smooth.
    int steps = clamp(uHazeSteps, 4, HAZE_MAX_STEPS);
    float N = float(steps);
    float jitter = ign(gl_FragCoord.xy);
    float tPrev = 0.0, T = 1.0;
    vec3 accScene = vec3(0.0);      // scene units * exposure: the HDR buffer's own domain
    vec3 accShown = vec3(0.0);      // exposed units, converted once at the end

    // Lights further than a beam's reach never matter: skip the loop entirely.
    bool beams = uBeamCount > 0 && uHazeBeam > 0.0 && night > 0.02;
    for (int i = 0; i < HAZE_MAX_STEPS; ++i) {
        if (i >= steps) break;
        float t1 = dist * pow(float(i + 1) / N, 1.4);
        float seg = t1 - tPrev;
        float tm = tPrev + seg * jitter;
        vec3 p = ro + rd * tm;
        float h = p.z - uHazeFloor;

        float sigma = hazeSigma(p, 1.0);
        float Tseg = exp(-sigma * seg);
        float w = T * (1.0 - Tseg);          // light picked up in this segment, already extincted

        float canyon = exp(-max(h, 0.0) / 60.0);

        // Physical part: ambient (a uniformly bright sky has radiance E / pi) and the sun,
        // which the street canyon shades.
        vec3 sun = min(uSunColor * sunPhase * (1.0 - 0.65 * canyon), vec3(3.0));
        accScene += w * (uAmbColor * (0.318 * uHazeAmbient) + sun) * uExposure;

        // Stylised part: the city's own glow in the air, brightest near the street.
        float glowH = 0.30 + 0.70 * exp(-max(h, 0.0) / 110.0);
        accShown += w * uHazeGlowTint * (uHazeGlow * night * glowH);

        if (beams) {
            vec3 b = vec3(0.0);
            for (int k = 0; k < MAX_BEAMS; ++k) {
                if (k >= uBeamCount) break;
                vec3 dl = p - uBeamPos[k].xyz;          // lamp -> sample
                float r2 = dot(dl, dl);
                if (r2 > BEAM_REACH * BEAM_REACH) continue;
                float r = sqrt(max(r2, 0.01));
                vec3 L = dl / r;
                float cone = smoothstep(0.90, 0.985, dot(L, uBeamDir[k].xyz));
                if (cone <= 0.0) continue;
                float fade = 1.0 - smoothstep(BEAM_REACH * 0.35, BEAM_REACH, r);
                // Looking into oncoming lamps (L opposes the view ray) is forward scattering.
                b += (cone * fade * uBeamDir[k].w * hgPhase(dot(L, -rd), 0.55)) / (r2 + 9.0);
            }
            accShown += T * seg * sigma * uHazeBeam * b * vec3(1.0, 0.90, 0.68);
        }

        T *= Tseg;
        tPrev = t1;
        if (T < 0.02) break;
    }

    // Neon halos: each sign is a point source; the air along the ray integrates its
    // inverse-square falloff in closed form (atan), so even a thin ray that passes a
    // sign at 3 m picks up the full halo, with no step noise.
    if (uNumNeon > 0 && uHazeNeon > 0.0 && night > 0.02) {
        float tau = -log(max(T, 1e-4));
        float nightK = smoothstep(0.02, 0.5, night);
        vec3 neon = vec3(0.0);
        for (int i = 0; i < MAX_NEON; ++i) {
            if (i >= uNumNeon) break;
            neon += lightHalo(uNeonPos[i], uNeonCol[i], ro, rd, dist, tau, 6.0);
        }
        // A long street lines dozens of signs up along one ray; soft-limit the sum so the far
        // end glows rather than clipping.
        neon *= uHazeNeon * nightK;
        accShown += neon / (1.0 + dot(neon, LUMA) * 6.0);
    }

    // Street lamps: a warm halo around every lamp along the street, brightest in damp air.
    if (uNumLamps > 0 && uHazeLamp > 0.0 && night > 0.02) {
        float tau = -log(max(T, 1e-4));
        vec3 lamps = vec3(0.0);
        for (int i = 0; i < MAX_LAMPS; ++i) {
            if (i >= uNumLamps) break;
            lamps += lightHalo(uLampPos[i], uLampCol[i], ro, rd, dist, tau, 2.0);
        }
        lamps *= uHazeLamp * smoothstep(0.02, 0.5, night);
        accShown += lamps / (1.0 + dot(lamps, LUMA) * 6.0);
    }

    // Half-resolution result: light picked up along the ray (HDR buffer units) and transmittance.
    FragColor = vec4(min(accScene + accShown * invExpo, vec3(60000.0)), T);
}
""")


# Full-resolution finish: depth-aware upsample of the half-resolution medium, then the emitter
# exemption (which needs the full-resolution scene colour).  Four half-res taps are weighted by how
# close their depth is to this pixel's, so fog does not bleed across building silhouettes.
HAZE_APPLY_FS = ("""
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uScene;
uniform sampler2D uHalf;       // rgb = in-scattered light, a = transmittance
uniform sampler2D uDepth;
uniform sampler2D uAE;
uniform float uAEOn, uEVComp;
uniform vec2  uProjZ;          // projection [2][2], [2][3]
uniform float uHazeEmitter;
uniform vec2  uFull;           // full resolution in pixels
""" + SCENE_DEPTH + """
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

float linearDepth(float d) {
    if (depthSky(d)) return 1.0e5;
    return uProjZ.y / (depthNdc(d) + uProjZ.x);      // view-space distance for standard and reverse-Z
}

void main() {
    vec4 scene = texture(uScene, vUV);
    float dz = linearDepth(texture(uDepth, vUV).r);
    vec2 hs = vec2(textureSize(uHalf, 0));
    vec2 f = vUV * hs - 0.5;
    ivec2 i0 = ivec2(floor(f));
    vec2 fr = f - vec2(i0);
    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            ivec2 c = clamp(i0 + ivec2(i, j), ivec2(0), ivec2(hs) - 1);
            // the half-res texel was marched with the depth under its centre
            vec2 cuv = (vec2(c) + 0.5) / hs;
            float tz = linearDepth(texture(uDepth, cuv).r);
            float bil = (i == 0 ? 1.0 - fr.x : fr.x) * (j == 0 ? 1.0 - fr.y : fr.y);
            float w = bil / (0.02 + abs(tz - dz) / max(dz, 1.0) * 8.0);
            sum += texelFetch(uHalf, c, 0) * w;
            wsum += w;
        }
    }
    vec4 h = sum / max(wsum, 1e-6);

    float ev = uEVComp + (uAEOn > 0.5 ? texelFetch(uAE, ivec2(0), 0).r : 0.0);
    float lum = dot(scene.rgb, LUMA) * exp2(ev);
    float emitter = smoothstep(1.2, 4.0, lum) * uHazeEmitter;
    float T = mix(h.a, 1.0, emitter);
    FragColor = vec4(min(scene.rgb * T + h.rgb, vec3(60000.0)), scene.a);
}
""")
