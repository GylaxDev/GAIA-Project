"""Street-level opening schedules and varied solid room selection."""

STOREFRONT = """
float shopOpen(float usage, vec4 h) {
    float clock = mod(uHours + 24.0, 24.0);
    float opens = usage < .5 ? 6.5 : (usage < 1.5 ? 9.0 : 7.0);
    float closes = usage < .5 ? 22.0 : (usage < 1.5 ? 21.0 : 23.0);
    opens += h.x * .8; closes -= h.y * .8;
    float openAmount = smoothstep(opens, opens + .18, clock)
                 * (1.0 - smoothstep(closes, closes + .18, clock));
    return usage > 3.5 ? .75 : mix(.025, 1.0, openAmount);
}

vec3 shopRadiance(vec3 rd, vec2 uv, vec2 size, float usage, vec4 h, vec3 amb, vec2 footprint) {
    float D = usage < .5 ? 7.5 : (usage < 1.5 ? 8.0 : (usage < 2.5 ? 10.0 : (usage < 3.5 ? 16.0 : 9.0)));
    D *= mix(.88, 1.12, h.w);
    vec3 lamp = usage < .5 ? vec3(1.0, .86, .68) : vec3(1.0, .95, .85);
    return furnishedRoom(rd, uv, size, D, int(clamp(floor(usage + .5), 0., 4.)),
                         h, shopOpen(usage, h), lamp, amb, footprint);
}
"""
