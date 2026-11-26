"""Filtered, advected spectral microstructure used only by the water material."""
from ..world.water_ripples import WATER_RIPPLE_SPANS

WATER_RIPPLES = (f'const vec2 rippleSpans = vec2({WATER_RIPPLE_SPANS[0]:.9f}, {WATER_RIPPLE_SPANS[1]:.9f});\n' + """
uniform sampler2DArray uWaterRippleTex;
uniform float uWaterRippleOn;

void waterRipples(vec2 p, float time, vec2 dx, vec2 dy, float scale,
                  inout vec2 slope, inout float variance, out float grain) {
    grain = .5;
    if (uWaterRippleOn < .5 || scale < .001) return;
    // Fixed world-space travel, not currentWind * elapsedTime: changing gusts
    // cannot teleport or rotate the ripple field. Two speeds decorrelate each pair.
    vec2 direction = vec2(.83646265, .54802394);
    float strength = waterSea() * scale * (.65 + .35 * waterAlignment(direction));
    for (int cascade = 0; cascade < 2; ++cascade) {
        float span = rippleSpans[cascade];
        float speed = cascade == 0 ? 2.05 : .72;
        vec2 aUV = (p - direction * (time * speed)) / span;
        vec2 bUV = (p - vec2(.943, .333) * (time * speed * .73)) / span;
        vec4 a = textureGrad(uWaterRippleTex, vec3(aUV, float(cascade * 2)), dx / span, dy / span);
        vec4 b = textureGrad(uWaterRippleTex, vec3(bUV, float(cascade * 2 + 1)), dx / span, dy / span);
        slope += (a.xy + b.xy) * (.70710678 * strength);
        variance += max(a.w - dot(a.xy, a.xy) + b.w - dot(b.xy, b.xy), 0.) * (.5 * strength * strength);
        if (cascade == 0) grain = (a.z + b.z) * .5;
        else grain = mix(grain, (a.z + b.z) * .5, .65);
    }
}
""")
