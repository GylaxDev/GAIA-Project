"""Flat, texture-filtered ground materials plus shared analytic bark/noise helpers.

Microdetail is baked once into one RGBA8 texture array, not evaluated as nearby
blade/Voronoi/fbm loops. Explicit gradients, taken before material branching,
select stable mips even at triangle/material boundaries. Joints, lane wear and
large stains remain world/ribbon aligned; all relief changes normals, not depth.
"""

from .materials import MATERIAL_LIGHT
from .vegetation import VEGETATION_LIGHT
from ..world.ground_textures import GROUND_METRES, GROUND_SLOPE_LIMIT

SURFACE_LIB = (MATERIAL_LIGHT + VEGETATION_LIGHT
               + f"const float GROUND_METRES = {GROUND_METRES:.6f};\n"
               + f"const float GROUND_SLOPE_LIMIT = {GROUND_SLOPE_LIMIT:.6f};\n" + """
uniform float uSurfDetail;
uniform sampler2DArray uGroundTex;
uniform float uGroundTextureOn;

vec4 sf_texture(vec2 wp, float layer, mat2 dW) {
    if (uGroundTextureOn < .5) return vec4(.5, 128. / 255., 128. / 255., .035);
    float scale = 1. / (GROUND_METRES * clamp(uSurfDetail, .05, 2.));
    return textureGrad(uGroundTex, vec3(wp / GROUND_METRES, layer), dW[0] * scale, dW[1] * scale);
}
vec2 sf_texture_gradient(vec4 tile) {
    return (tile.gb * 255. - 128.) * (GROUND_SLOPE_LIMIT / 127.);
}

float sf_hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec2 sf_hash2(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Value noise with analytic derivative (quintic fade).  x = value 0..1, yz = d/dp.
vec3 sf_noised(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    float a = sf_hash(i);
    float b = sf_hash(i + vec2(1.0, 0.0));
    float c = sf_hash(i + vec2(0.0, 1.0));
    float d = sf_hash(i + vec2(1.0, 1.0));
    float k1 = b - a, k2 = c - a, k4 = a - b - c + d;
    return vec3(a + k1 * u.x + k2 * u.y + k4 * u.x * u.y,
                du * vec2(k1 + k4 * u.y, k2 + k4 * u.x));
}

// Footprint-limited fbm.  p in metres, f0 in cycles per metre.  Returns (height centred on 0,
// d height / d p).  Octaves are rotated 53 degrees apart so the lattice never lines up.
vec3 sf_fbm(vec2 p, float f0, int oct, float fw, float gain) {
    float h = 0.0;
    vec2 d = vec2(0.0);
    float a = 0.5;
    float freq = f0;
    mat2 M = mat2(f0, 0.0, 0.0, f0);
    const mat2 R = mat2(1.2, 1.6, -1.6, 1.2);       // x2 scale, ~53 degree turn
    for (int i = 0; i < oct; i++) {
        float fade = 1.0 - smoothstep(0.045, 0.17, fw * freq);
        if (fade <= 0.001) break;
        vec3 n = sf_noised(M * p);
        h += (n.x - 0.5) * a * fade;
        d += (transpose(M) * n.yz) * (a * fade);
        M = R * M;
        freq *= 2.0;
        a *= gain;
    }
    return vec3(h, d);
}

// ------------------------------------------------------------------ asphalt
// roadW  carriageway width in metres (0 = car park / generic asphalt: no lane wear)
// ruv    x across 0..1, y along in metres (the road ribbon's own UV)
// endD   distance to the end of the ribbon segment (junction mouth)
void sf_asphalt(vec2 wp, mat2 dW, float roadW, vec2 ruv, float endD, float dist, float fw,
                out vec3 albedo, out float rough, out vec2 grad, out float wear, out float grit) {
    // sun-bleached / fresh tar patches over tens of metres
    vec4 tile = sf_texture(wp, 1., dW);
    vec3 mac = sf_fbm(wp, 0.03, 2, fw, 0.5);
    float tone = clamp(0.5 + mac.x * 2.4, 0.0, 1.0);
    albedo = mix(vec3(0.026, 0.027, 0.029), vec3(0.075, 0.073, 0.069), tone);
    rough = .87 - tile.a * .035;
    wear = 0.0;
    grad = sf_texture_gradient(tile) * .80 + mac.yz * .01;

    float oil = 0.0, edgeDust = 0.0;
    if (roadW >= 5.0) {
        float xm = (ruv.x - 0.5) * roadW;                       // metres from the centre line
        float lanes = max(1.0, floor(roadW / 6.8 + 0.5));
        float laneW = roadW / (2.0 * lanes);
        float laneId = floor(abs(xm) / laneW);
        float lu = (fract(abs(xm) / laneW) - 0.5) * laneW;      // metres from this lane's centre
        float wander = (sf_noised(wp * .12).x - .5) * .32;
        float tracks  = exp(-pow((abs(lu + wander) - 0.82) / 0.40, 2.0));
        float between = exp(-pow(lu / 0.24, 2.0));               // oil drip strip between them
        float patchy  = sf_noised(vec2(ruv.y * 0.045, laneId * 7.3)).x;
        float junction = 1.0 - smoothstep(2.0, 14.0, min(ruv.y, endD));   // braking zones wear more
        wear = tracks * clamp(0.12 + 0.35 * patchy + 0.20 * junction, 0.0, 1.0);
        float drips = smoothstep(0.58, 0.78, sf_noised(vec2(ruv.y * 0.16, laneId * 3.1 + 11.0)).x);
        oil = between * drips * 0.65;
        float edgeM = min(ruv.x, 1.0 - ruv.x) * roadW;           // metres from the kerb
        edgeDust = 1.0 - smoothstep(0.15, 0.95, edgeM);
    }
    albedo *= 1.0 - 0.18 * wear;
    albedo = mix(albedo, albedo * 0.52, oil);
    albedo = mix(albedo, vec3(0.140, 0.132, 0.120) * (0.8 + 0.4 * tone), edgeDust * 0.50);
    rough = mix(rough, 0.76, wear);
    rough = mix(rough, 0.68, oil);
    rough = mix(rough, 0.88, edgeDust * 0.7);

    // Baked millimetre aggregate, filtered in hardware rather than six noise octaves.
    grit = (tile.r - .5) * .65 + tile.a * .06;
    albedo *= 1. + (tile.r - .5) * .80;
    albedo += tile.a * .020 * (1. - .6 * wear);
    grad *= 1. - .35 * wear;

    // repair patches with a raised seam
    if (dist < 240.0) {
        vec2 cs = vec2(8.5, 13.0);
        vec2 cid = floor(wp / cs);
        if (sf_hash(cid + 17.0) > 0.90) {
            vec2 q = (fract(wp / cs) - 0.5) * cs;
            vec2 hb = cs * mix(vec2(0.16), vec2(0.40), sf_hash2(cid + 3.1));
            vec2 dq = abs(q) - hb;
            float sdf = max(dq.x, dq.y);
            float inside = 1.0 - smoothstep(-max(.03, fw), max(.03, fw), sdf);
            float w = 0.05 + fw;
            float t = clamp(abs(sdf) / w, 0.0, 1.0);
            float seam = 1.0 - t * t * (3.0 - 2.0 * t);
            albedo = mix(albedo, albedo * vec3(0.78, 0.78, 0.81), inside * 0.85);
            rough = mix(rough, 0.86, inside);
            albedo *= 1.0 - 0.40 * seam * (0.05 / w);
            vec2 g = (dq.x > dq.y) ? vec2(sign(q.x), 0.0) : vec2(0.0, sign(q.y));
            grad += g * sign(sdf) * (0.010 * 6.0 * t * (1.0 - t) / w) * (0.05 / w);
        }
    }

    // Sparse sealed hairline cracks, not a per-pixel neighbouring-cell search.
    if (dist < 95.0) {
        vec2 id = floor(wp / vec2(5.5, 8.0));
        if (sf_hash(id + 36.1) > .88) {
            vec2 q = (fract(wp / vec2(5.5, 8.0)) - .5) * vec2(5.5, 8.0);
            float wave = (sf_noised(vec2(q.y * 1.3, sf_hash(id) * 7.)).x - .5) * .14;
            float e = abs(q.x - q.y * .13 + wave);
            float w = .009 + fw;
            float crack = (1. - smoothstep(0., w, e)) * (.009 / w)
                        * (1. - smoothstep(2.8, 3.8, abs(q.y))) * (1. - smoothstep(55., 95., dist));
            albedo *= 1. - .48 * crack;
            rough = mix(rough, .79, crack);
        }
    }

    // oil / fuel stains
    if (dist < 70.0) {
        vec2 sc = wp * 0.30;
        vec2 si = floor(sc);
        if (sf_hash(si + 9.7) > 0.84) {
            vec2 ctr = 0.25 + 0.5 * sf_hash2(si + 2.9);
            float rad = 0.05 + 0.08 * sf_hash(si + 4.4);
            float sd = length(fract(sc) - ctr);
            float sp = 1.0 - smoothstep(rad * 0.5, rad, sd);
            albedo *= 1.0 - 0.45 * sp;
            rough = mix(rough, 0.62, sp);
        }
    }
    albedo = reflectanceRange(max(albedo, vec3(0.02)), ASPHALT_REFLECTANCE);
}

// -------------------------------------------------------------------- grass
void sf_grass(vec2 wp, mat2 dW, float fw,
              out vec3 albedo, out float ao, out vec2 grad) {
    vec4 tile = sf_texture(wp, 0., dW);
    vec3 macro = sf_noised(wp * .026);
    // Lawn is never one flat tone: broad lush/thin patches (tens of metres) plus
    // mid-scale mowing and wear variation (a few metres), both filtered by footprint.
    float mid = sf_noised(wp * .21 + 4.7).x;
    float fine = sf_noised(wp * .83 + 9.1).x;
    float lush = clamp(.5 + (macro.x - .5) * .8 + (mid - .5) * .5 + (fine - .5) * .30 * (1. - smoothstep(.2, 1.2, fw)), 0., 1.);
    float dry = smoothstep(.82, .97, macro.x) * .10 + tile.a * .30;
    vec3 dk = mix(GRASS_DARK, GRASS_DRY_DARK, dry);
    vec3 lt = mix(GRASS_LIGHT, GRASS_DRY_LIGHT, dry);
    float t = clamp(tile.r * .55 + lush * .45 + (macro.x - .5) * .08, 0., 1.);
    albedo = mix(dk, lt, t) * vec3(.92, 1.0, .88);   // deep, cool lawn green
    ao = mix(.86, .96, t) * (1. - tile.a * .025);
    grad = sf_texture_gradient(tile) * .55;
}

// ----------------------------------------------------------------- concrete
// sp   slab-space metres (world xy for plazas, ribbon (across, along) for footpaths)
// J    d(sp)/d(world xy), so a joint's slope can be turned back into a world-space gradient
void sf_concrete(vec2 sp, mat2 J, vec2 wp, mat2 dW, vec2 slab, float dist, float fw,
                 out vec3 albedo, out float rough, out vec2 grad) {
    vec2 cell = floor(sp / slab);
    vec2 f = fract(sp / slab);
    float tone = sf_hash(cell + 5.3);
    vec4 tile = sf_texture(wp, 2., dW);
    vec3 mac = sf_fbm(wp, 0.11, 2, fw, 0.5);
    // Separate pours age differently; occasional replacement slabs are lighter.
    // The palette stays neutral and matte, not a bright beige checkerboard.
    float age = clamp(.46 + mac.x * .85 + (tone-.5)*.26, 0., 1.);
    albedo = mix(vec3(.248, .258, .256), vec3(.332, .336, .326), age);
    float repair = smoothstep(.89, .94, tone);
    albedo = mix(albedo, vec3(.380, .378, .358), repair*.45);
    albedo *= 1. + (tile.r-.5)*.40 - tile.a*.08;
    rough = .90 + tile.a*.045;
    grad = sf_texture_gradient(tile)*.90 + mac.yz*.004;

    vec2 e = min(f, 1.0 - f) * slab;                               // metres to the nearest joint
    float jd = min(e.x, e.y);
    float w = .011 + fw * .65;
    float ic = clamp(.011 / w, 0., 1.);
    float t = clamp(jd / w, 0.0, 1.0);
    float joint = (1.0 - t * t * (3.0 - 2.0 * t)) * ic;
    albedo *= 1. - .58 * joint;
    float grime = (1.-smoothstep(.012, .11+fw, jd)) * (.11/(.11+fw));
    float damp = smoothstep(.45, .75, sf_noised(wp*.34 + 8.1).x) * .6;
    albedo = mix(albedo, albedo*vec3(.73, .72, .67), grime*(.22+damp*.30));
    albedo *= 1.-damp*.09*(1.-repair);
    rough = mix(rough, .95, joint);
    // slope of the groove, in slab space, mapped to world space
    vec2 gs = (e.x < e.y) ? vec2(sign(0.5 - f.x), 0.0) : vec2(0.0, sign(0.5 - f.y));
    vec2 gj = gs * (.0025 * 6. * t * (1. - t) / w) * ic;
    grad += transpose(J) * gj;

    // A sparse, broken fracture in old pours. Footprint integration makes it
    // disappear into coverage at distance instead of sparkling like black wire.
    if (tone < .23 && dist < 100.) {
        vec2 q = (f-.5)*slab;
        float wiggle = (sf_noised(vec2(q.y*3.3, tone*31.)).x-.5)*.075;
        float cd = abs(q.x-q.y*.31-wiggle);
        float cw = .006 + fw;
        float crack = (1.-smoothstep(0., cw, cd))*(.006/cw)
                    * (1.-smoothstep(slab.y*.31, slab.y*.50, abs(q.y)))
                    * (1.-smoothstep(60., 100., dist));
        albedo *= 1.-.68*crack;
    }
    if (jd < .10 && dist < 90.) {
        float chips = smoothstep(.59, .79, sf_noised(sp*17.+tone*9.).x)
                    * (1.-smoothstep(.006, .045+fw, jd)) * (.045/(.045+fw));
        albedo *= 1.-chips*.24;
    }
}

// Stone/concrete curb faces have their own vertical grain and a dirty foot.
void sf_curb(vec2 uv, mat2 dUV, float fw, out vec3 albedo, out vec2 grad) {
    vec4 tile = sf_texture(uv, 2., dUV);
    albedo = mix(vec3(.235, .238, .229), vec3(.335, .329, .303), tile.r);
    float foot = 1.-smoothstep(.015, .115, uv.y);
    albedo *= 1.-foot*.32-tile.a*.08;
    float edge = min(fract(uv.x/1.25), 1.-fract(uv.x/1.25))*1.25;
    float w = .007+fw;
    albedo *= 1.-.35*(1.-smoothstep(0., w, edge))*(.007/w);
    grad = sf_texture_gradient(tile)*.65;
}

// --------------------------------------------------------------------- dirt
void sf_dirt(vec2 wp, mat2 dW, float fw, out vec3 albedo, out vec2 grad) {
    vec3 a = sf_fbm(wp, 0.05, 2, fw, 0.5);
    vec4 tile = sf_texture(wp, 3., dW);
    albedo = mix(vec3(0.100, 0.077, 0.050), vec3(0.190, 0.146, 0.098), clamp(0.5 + a.x * 1.8, 0.0, 1.0));
    albedo *= 1. + (tile.r - .5) * .65;
    grad = sf_texture_gradient(tile) * .80 + a.yz * .02;
}

// Generic urban infill is worn mineral ground, not an untextured green-grey sheet.
void sf_urban(vec2 wp, mat2 dW, float fw, out vec3 albedo, out vec2 grad) {
    vec3 macro = sf_fbm(wp, .065, 2, fw, .5);
    vec4 tile = sf_texture(wp, 2., dW);
    albedo = mix(vec3(.145, .151, .146), vec3(.225, .221, .203), clamp(.5 + macro.x * 1.7, 0., 1.));
    albedo *= 1.+(tile.r-.5)*.34-tile.a*.06;
    grad = sf_texture_gradient(tile)*.60;
}
""")
