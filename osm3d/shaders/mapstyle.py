"""Shared colour ramp for OSM raster tiles.

Both the 3D globe and the 2D flat map shade tiles with ``map_shade`` so the two
views of the same place agree on their land, water and ink colours.

The source tiles are OpenStreetMap Carto: off-white paper, pale blue water and
hills in green. ``map_shade`` re-inks them into an Apple-Maps-like palette:
green countryside, grey built-up areas, blue water, white roads, amber and
yellow highways and dark labels.
"""

MAP_SHADE_GLSL = """
vec3 map_shade(vec3 c) {
    float lum = dot(c, vec3(0.299, 0.587, 0.114));
    float ink = 1.0 - lum;
    // Carto sea and rivers are blue with almost no red (b - r > 0.2); land is
    // warm off-white or green, so this separates cleanly.
    float water = smoothstep(0.10, 0.17, c.b - c.r);
    // Vegetation (park, wood, grass) is green above its red channel.
    float green = smoothstep(0.02, 0.09, c.g - c.r);
    // Built-up areas are neutral grey, darker than the off-white land paper.
    float urban = (1.0 - green) * (1.0 - smoothstep(0.865, 0.925, lum));
    // Road fills are yellow (secondary) through amber (primary, motorway).
    float warm = smoothstep(0.14, 0.32, c.r - c.b) * smoothstep(0.60, 0.80, lum);
    vec3 warm_col = mix(vec3(0.985, 0.940, 0.740), vec3(0.985, 0.820, 0.560), warm);
    // Road fill and label halos are pure white paper.
    float pale = smoothstep(0.955, 0.990, min(c.r, min(c.g, c.b))) * (1.0 - water);

    vec3 land = mix(vec3(0.545, 0.645, 0.455), vec3(0.395, 0.510, 0.355), green);
    land = mix(land, vec3(0.735, 0.735, 0.705), urban * 0.90);
    land = mix(land, vec3(0.985, 0.985, 0.965), pale * 0.85);
    land = mix(land, warm_col, warm * 0.85);
    // Ink (labels, borders, coastlines, road casing) turns dark instead of light.
    land = mix(land, vec3(0.120, 0.140, 0.160), smoothstep(0.35, 0.72, ink));

    vec3 sea = mix(vec3(0.205, 0.425, 0.690), vec3(0.130, 0.300, 0.520),
                   smoothstep(0.35, 0.75, ink));

    return mix(land, sea, water);
}
"""
