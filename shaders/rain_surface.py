"""Depth-backed rain sheltering and footprint-filtered, radial impact ripples."""

RAIN_EXPOSURE = """
uniform sampler2D uRainRoof;
uniform mat4 uRainViewProj, uRainInvViewProj;
uniform float uRainRoofOn, uRainRoofBias;
uniform vec2 uRainRoofTexel;
float rainExposure(vec3 world) {
    if (uRainRoofOn < .5) return 1.;
    vec3 p = (uRainViewProj * vec4(world, 1.)).xyz * .5 + .5;
    if (any(lessThan(p.xy, vec2(0.))) || any(greaterThan(p.xy, vec2(1.)))) return 1.;
    float roof = texture(uRainRoof, p.xy).r;
    // Conventional, top-down depth, independent of the scene's reversed depth.
    return 1. - smoothstep(roof + uRainRoofBias, roof + uRainRoofBias * 2., p.z);
}
"""

RAIN_RIPPLES = """
float rainHash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
vec2 rainRipples(vec2 world, float time, float footprint) {
    float resolved = 1. - smoothstep(.025, .11, footprint);
    if (resolved < .001) return vec2(0.);
    const float spacing = .62;
    vec2 cell = floor(world / spacing);
    vec2 gradient = vec2(0.);
    // Neighbour cells keep expanding rings continuous across cell boundaries.
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        vec2 c = cell + vec2(x, y);
        float seed = rainHash(c);
        float clock = time * (1.1 + .5 * seed) + seed * 7.;
        float life = fract(clock);
        vec2 jitter = vec2(rainHash(c + floor(clock) + 3.1), rainHash(c - floor(clock) + 7.7));
        vec2 delta = world - (c + .22 + .56 * jitter) * spacing;
        float r = length(delta);
        float band = r - (.015 + life * .26);
        float envelope = exp(-band * band * 1100.) * pow(1. - life, 2.) * smoothstep(0., .06, life);
        float slope = cos(band * 95.) * envelope * .055;
        gradient += delta / max(r, .001) * slope;
    }
    return gradient * resolved;
}
"""
