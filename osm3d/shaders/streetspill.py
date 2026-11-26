# --- shaders/streetspill.py ---
# Shared GLSL: light spilled by the nearest street lamps onto any surface.
#
# Lamps are bucketed on the CPU into a coarse world-space grid (osm3d/spill.py)
# and looked up from a texture by the fragment's own cell, so a fragment only
# loops the lamps that can actually reach it.  That is what lets the spill reach
# the whole resident city instead of the lamps nearest the camera.
#
# The grid stores one record per (cell, lamp) pair -- position plus 8-bit colour
# packed into the fourth channel of the same texel -- so each lamp in the loop
# costs a single dependent texel fetch.
#
# Two entry points: the vertex stage pre-evaluates the diffuse sum for every
# vertex, and the fragment stage only runs the full per-pixel loop close to the
# eye.  Distant streets are lit from the interpolated vertex values, which is
# what makes the whole city affordable.
#
# The numbers below are shared with spill.py (which sizes the grid from them), so
# the CPU buckets and the shader cut-off can never disagree.

from ..core.photometry import NITS_PER_SCENE_UNIT, STREET_LAMP_CANDELA

LAMP_REACH = 100.0            # metres: a lamp contributes nothing beyond this
LAMP_FADE_START = 60.0        # metres: smooth roll-off begins
MAX_CELL_LIGHTS = 48          # candidates kept per grid cell, nearest first
VERTEX_LAMP_LIMIT = 12        # per-vertex pass keeps only the nearest few lamps
RAIN_LAMP_LIMIT = 8           # scattering pass: rain only needs the closest lamps
SPILL_NEAR_START = 200.0      # metres: per-pixel shading starts to replace the vertex sum
SPILL_NEAR_END = 400.0        # metres: per-pixel shading has fully replaced it
# Lamp glint gains on cars.  The lobe is normalized like a luminaire seen in a
# polished surface, so a lamp at 6 m reads as a compact white highlight the way the
# sun's reflected disc does in daylight, scaled by each material's Fresnel term.
GLINT_SHARP_GAIN = 6.0
GLINT_SOFT_GAIN = 1.0

STREET_SPILL = f"""
#define MAX_CELL_LIGHTS {MAX_CELL_LIGHTS}
#define LAMP_REACH2 {LAMP_REACH * LAMP_REACH:.1f}
#define LAMP_REACH {LAMP_REACH:.1f}
#define LAMP_FADE_START {LAMP_FADE_START:.1f}
#define STREET_SPECULAR 0         // 1 = lamp reflection streak on glossy surfaces (costly)
uniform sampler2D uLampCell;      // per cell: (first record, count, highest lamp z, unused)
uniform sampler2D uLampRec;       // xyz luminaire position, w = packed rgb
uniform vec2  uLampOrigin;        // world xy of the (0,0) cell corner
uniform float uLampCellSize;

// 0 = use the interpolated per-vertex sum, 1 = shade this pixel exactly.
float spillNearWeight(float eyeDist) {{
    return 1.0 - smoothstep({SPILL_NEAR_START:.1f}, {SPILL_NEAR_END:.1f}, eyeDist);
}}

void streetLamps(vec3 P, vec3 N, vec3 V, int limit, out vec3 diffE, out vec3 specE) {{
    diffE = vec3(0.0);
    specE = vec3(0.0);

    ivec2 grid = textureSize(uLampCell, 0);
    ivec2 cell = ivec2(floor((P.xy - uLampOrigin) / uLampCellSize));
    if (cell.x < 0 || cell.y < 0 || cell.x >= grid.x || cell.y >= grid.y) return;

    vec4 head  = texelFetch(uLampCell, cell, 0);
    if (head.y < 0.5 || P.z >= head.z) return;
    int first  = int(head.x + 0.5);
    int count  = min(int(head.y + 0.5), limit);
    int width  = textureSize(uLampRec, 0).x;

    for (int k = 0; k < MAX_CELL_LIGHTS; k++) {{
        if (k >= count) break;
        int slot = first + k;
        int row  = slot / width;
        vec4 rec = texelFetch(uLampRec, ivec2(slot - row * width, row), 0);

        vec3  d  = rec.xyz - P;
        float r2 = dot(d, d);
        if (r2 > LAMP_REACH2 || d.z <= 0.0) continue;

        float r   = sqrt(max(r2, 0.001));
        vec3  Ld  = d / r;                           // surface -> lamp
        float nl  = max(dot(N, Ld), 0.0);
        if (nl <= 0.0) continue;                     // facing away: no light, skip the rest

        int rgb8 = int(rec.w);                       // exact rgb888 integer in float32
        vec3 lc = vec3(float(rgb8 >> 16), float((rgb8 >> 8) & 255),
                       float(rgb8 & 255)) * (1.0 / 255.0);
        // Match the photopic normalization used by the visible lamp lens.
        lc /= max(dot(lc, vec3(0.2126, 0.7152, 0.0722)), 1e-5);

        float cosNadir = max(Ld.z, 0.0);             // lamp points straight down
        float c2 = cosNadir * cosNadir;
        // Broad batwing approximation: peak throw at 45 degrees, 60% at nadir,
        // and a cutoff near the horizon. Light reaches the lanes between poles.
        float I = (0.60 + 0.40 * 4.0 * c2 * (1.0 - c2))
                * smoothstep(0.05, 0.20, cosNadir);
        float fade = 1.0 - smoothstep(LAMP_FADE_START, LAMP_REACH, r);
        float inv  = 1.0 / max(r2, 0.25);

        // Candela / m² = lux; convert to the sun's normalized irradiance units.
        diffE += lc * ({STREET_LAMP_CANDELA / NITS_PER_SCENE_UNIT:.9f} * I * nl * fade * inv);

#if STREET_SPECULAR
        if (r2 < 3600.0 && limit > 16) {{
            // narrow streak of reflection along the road surface; h^48 by squaring
            vec3  H = normalize(Ld + V);
            float h  = max(dot(N, H), 0.0);
            float h2 = h * h, h4 = h2 * h2, h8 = h4 * h4, h16 = h8 * h8;
            specE += lc * (h16 * h16 * h16 * (I * fade * {90.0 / NITS_PER_SCENE_UNIT:.9f} * inv));
        }}
#endif
    }}
}}

void streetLights(vec3 P, vec3 N, vec3 V, out vec3 diffE, out vec3 specE) {{
    streetLamps(P, N, V, MAX_CELL_LIGHTS, diffE, specE);
}}

// Compact luminaire highlights for small glossy objects (cars).  The cell list is
// nearest-first, so a short cap keeps the lamps that can actually put a highlight on
// this fragment; the full 48-lamp loop with a pow() chain per lamp is what made the
// shared STREET_SPECULAR path too costly to enable everywhere.
#define VEHICLE_GLINT_LAMPS 6
void streetLampGlints(vec3 P, vec3 N, vec3 V, out vec3 sharp, out vec3 soft) {{
    sharp = vec3(0.0);
    soft = vec3(0.0);

    ivec2 grid = textureSize(uLampCell, 0);
    ivec2 cell = ivec2(floor((P.xy - uLampOrigin) / uLampCellSize));
    if (cell.x < 0 || cell.y < 0 || cell.x >= grid.x || cell.y >= grid.y) return;

    vec4 head = texelFetch(uLampCell, cell, 0);
    if (head.y < 0.5 || P.z >= head.z) return;
    int first = int(head.x + 0.5);
    int count = min(int(head.y + 0.5), VEHICLE_GLINT_LAMPS);
    int width = textureSize(uLampRec, 0).x;

    for (int k = 0; k < VEHICLE_GLINT_LAMPS; k++) {{
        if (k >= count) break;
        int slot = first + k;
        int row = slot / width;
        vec4 rec = texelFetch(uLampRec, ivec2(slot - row * width, row), 0);

        vec3 d = rec.xyz - P;
        float r2 = dot(d, d);
        if (r2 > 3600.0 || d.z <= 0.0) continue;    // beyond 60 m a glint is a pixel at most
        vec3 Ld = d * inversesqrt(max(r2, 0.001));
        float nl = max(dot(N, Ld), 0.0);
        if (nl <= 0.0) continue;

        int rgb8 = int(rec.w);
        vec3 lc = vec3(float(rgb8 >> 16), float((rgb8 >> 8) & 255),
                       float(rgb8 & 255)) * (1.0 / 255.0);
        lc /= max(dot(lc, vec3(0.2126, 0.7152, 0.0722)), 1e-5);

        float cosNadir = max(Ld.z, 0.0);
        float c2 = cosNadir * cosNadir;
        float I = (0.60 + 0.40 * 4.0 * c2 * (1.0 - c2))
                * smoothstep(0.05, 0.20, cosNadir);
        float fade = 1.0 - smoothstep(LAMP_FADE_START, LAMP_REACH, sqrt(max(r2, 0.001)));
        float inv = 1.0 / max(r2, 0.25);
        vec3 H = normalize(Ld + V);
        float h = max(dot(N, H), 0.0);

        // Glossy coat/glass/paint lobe and a broad one for satin trim, rubber and
        // brushed metal, both in irradiance units like the diffuse sum above.
        sharp += lc * ({GLINT_SHARP_GAIN:.3f} * pow(h, 1400.0) * I * fade * inv);
        soft  += lc * ({GLINT_SOFT_GAIN:.3f} * pow(h, 90.0) * I * fade * inv);
    }}
}}

// Diffuse-only sum for the vertex stage: the cell list is nearest-first, so a
// small cap keeps the lamps that carry nearly all of the energy.
vec3 streetLampDiffuse(vec3 P, vec3 N) {{
    vec3 d, s;
    streetLamps(P, N, vec3(0.0, 0.0, 1.0), {VERTEX_LAMP_LIMIT}, d, s);
    return d;
}}

// Irradiance on a tiny scattering volume (a rain streak), which has no surface
// normal: every lamp that can see the point contributes, and the scattered
// radiance keeps a mild forward lobe for drops backlit by the luminaire.
vec3 streetLampScatter(vec3 P, vec3 V) {{
    vec3 E = vec3(0.0);

    ivec2 grid = textureSize(uLampCell, 0);
    ivec2 cell = ivec2(floor((P.xy - uLampOrigin) / uLampCellSize));
    if (cell.x < 0 || cell.y < 0 || cell.x >= grid.x || cell.y >= grid.y) return E;

    vec4 head = texelFetch(uLampCell, cell, 0);
    if (head.y < 0.5 || P.z >= head.z) return E;
    int first = int(head.x + 0.5);
    int count = min(int(head.y + 0.5), {RAIN_LAMP_LIMIT});
    int width = textureSize(uLampRec, 0).x;

    for (int k = 0; k < MAX_CELL_LIGHTS; k++) {{
        if (k >= count) break;
        int slot = first + k;
        int row = slot / width;
        vec4 rec = texelFetch(uLampRec, ivec2(slot - row * width, row), 0);

        vec3 d = rec.xyz - P;
        float r2 = dot(d, d);
        if (r2 > LAMP_REACH2 || d.z <= 0.0) continue;
        vec3 Ld = d * inversesqrt(max(r2, 0.001));

        int rgb8 = int(rec.w);
        vec3 lc = vec3(float(rgb8 >> 16), float((rgb8 >> 8) & 255),
                       float(rgb8 & 255)) * (1.0 / 255.0);
        lc /= max(dot(lc, vec3(0.2126, 0.7152, 0.0722)), 1e-5);

        float cosNadir = max(Ld.z, 0.0);
        float c2 = cosNadir * cosNadir;
        float I = (0.60 + 0.40 * 4.0 * c2 * (1.0 - c2))
                * smoothstep(0.05, 0.20, cosNadir);
        float fwd = max(dot(-Ld, V), 0.0);
        float phase = 0.35 + 0.65 * fwd * fwd;
        float fade = 1.0 - smoothstep(LAMP_FADE_START, LAMP_REACH, sqrt(max(r2, 0.001)));

        E += lc * ({STREET_LAMP_CANDELA / NITS_PER_SCENE_UNIT:.9f} * I * phase * fade
                / max(r2, 0.25));
    }}
    return E;
}}
"""
