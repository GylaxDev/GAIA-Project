"""Vegetation-only reflectances and identical scene/caster tree transforms."""
from ..world.foliage import LEAF_ALPHA_CUTOFF

TREE_ALPHA_TEST = f"const float LEAF_ALPHA_CUTOFF = {LEAF_ALPHA_CUTOFF:.6f};\n"

TREE_INSTANCE = """
float treeSeed(vec3 pos, vec2 scaleRot) {
    vec3 h = fract(vec3(pos.xy * .017, scaleRot.y + .13) * vec3(.1031, .1030, .0973));
    h += dot(h, h.yzx + 33.33);
    return fract((h.x + h.y) * h.z);
}
vec3 treeShape(float seed) {
    return vec3(mix(.90, 1.10, seed), mix(.89, 1.11, fract(seed * 7.13)),
                mix(.95, 1.05, fract(seed * 13.71)));
}
vec2 treeLean(float seed) {
    return vec2(cos(seed * 31.7), sin(seed * 31.7)) * .28;
}
vec3 treeDeform(vec3 p, float seed) {
    float t = clamp(p.z / 9.5, 0.0, 1.0);
    vec3 q = p * treeShape(seed);
    q.xy += treeLean(seed) * t * t;
    return q;
}
vec3 treeInstancePosition(vec3 point, vec3 pos, vec2 scaleRot, vec2 tileScale,
                          float grow, vec3 offset) {
    float seed = treeSeed(pos, scaleRot);
    float c = cos(scaleRot.y), s = sin(scaleRot.y);
    mat3 rot = mat3(c, s, 0, -s, c, 0, 0, 0, 1);
    vec3 local = rot * (treeDeform(point, seed) * scaleRot.x) + pos;
    return vec3(local.xy * tileScale, local.z * grow) + offset;
}
vec3 treeInstanceNormal(vec3 point, vec3 normal, vec3 pos, vec2 scaleRot,
                        vec2 tileScale, float grow) {
    float seed = treeSeed(pos, scaleRot);
    vec3 shape = treeShape(seed);
    float derivative = (point.z > 0.0 && point.z < 9.5) ? 2.0 * point.z / (9.5 * 9.5) : 0.0;
    vec3 n = vec3(normal.xy / shape.xy, 0.0);
    n.z = (normal.z - dot(treeLean(seed), n.xy) * derivative) / shape.z;
    float c = cos(scaleRot.y), s = sin(scaleRot.y);
    n = mat3(c, s, 0, -s, c, 0, 0, 0, 1) * n;
    return normalize(n / max(vec3(tileScale, grow), vec3(.001)));
}
vec2 treeAtlasUV(vec2 uv, vec3 pos, vec2 scaleRot, float cardSeed) {
    float variant = floor(fract(treeSeed(pos, scaleRot) + cardSeed) * 4.0);
    return (uv + vec2(mod(variant, 2.0), floor(variant * .5))) * .5;
}
"""

VEGETATION_LIGHT = """
#ifndef GAIA_VEGETATION_LIGHT
#define GAIA_VEGETATION_LIGHT
// Scene-linear greens with restrained blue/yellow separation, not a global grade.
const vec3 GRASS_DARK = vec3(.061, .097, .045);
const vec3 GRASS_LIGHT = vec3(.113, .168, .083);
const vec3 GRASS_DRY_DARK = vec3(.100, .100, .069);
const vec3 GRASS_DRY_LIGHT = vec3(.171, .157, .107);
const vec3 GRASS_MEAN = vec3(.087, .133, .064);
float turfDiffuse(vec3 supportNormal, vec3 lightDir) {
    // Unresolved turf represents a blade canopy, not a flat green Lambertian
    // sheet. Average tilted tips and randomly oriented upright ribbons; this
    // keeps its solar response close to the resolved field at the LOD boundary.
    float c = clamp(dot(supportNormal, lightDir), 0., 1.);
    float upright = sqrt(max(0., 1. - c * c)) * .3183099;
    return (.46 * c + .54 * upright + .08) / 1.08;
}
vec3 foliageReflectance(vec3 tint) {
    tint = max(tint, vec3(.001));
    float lum = dot(tint, PHOTO_LUMA);
    // Old caches stored dark olive RGB that was incorrectly treated as sRGB.
    // Correct it here as linear tint, and bound the BASE before leaf/cavity detail.
    tint = mix(vec3(lum), tint, .76);
    return tint * clamp(lum, .065, .145) / max(lum, .001);
}
#endif
"""
