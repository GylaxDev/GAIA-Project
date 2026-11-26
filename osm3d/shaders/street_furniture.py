"""Cheap, metric-UV wear for the batched opaque street-furniture material."""

STREET_FURNITURE = """
void streetPropSurface(float material, float seed, vec2 uv, vec2 footprint, vec3 normal,
                       inout vec3 albedo, inout float rough, inout float metal) {
    float fw = max(footprint.x, footprint.y);
    float detail = 1.-smoothstep(.012, .09, fw);
    if (abs(material-3.) < .1) {
        // Grain runs along the actual slat, independent of street heading.
        float timber = vnoise(vec2(uv.x*.85, uv.y*15.) + seed*13.);
        float fibres = sin(uv.y*410. + timber*8. + sin(uv.x*3.+seed*19.)*.7);
        albedo = mix(albedo, vec3(.54, .52, .46), .20+timber*.13);
        albedo *= .91+timber*.15-fibres*detail*.045;
        rough = .88;
    } else if (abs(material-4.) < .1) {
        // Desaturated powder coat with sparse chips, not polished plastic.
        float age = vnoise(uv*2.7+seed*23.);
        float grain = vnoise(uv*37.+seed*11.);
        float chip = smoothstep(.73, .89, grain)*detail*(.15+age*.35);
        albedo *= .91+age*.13;
        albedo = mix(albedo, vec3(.33, .245, .16), chip*.30);
        if (abs(normal.z)<.5) albedo *= 1.-.12*(1.-smoothstep(.02, .22, uv.y));
        rough = .76+age*.08;
    } else if (abs(material-5.) < .1) {
        float mineral = vnoise(uv*43.+seed*7.);
        albedo *= 1.+(mineral-.5)*detail*.24;
        rough = .96;
    } else if (abs(material-6.) < .1) {
        rough = .57;
        metal = .68;  // restrained bare fasteners, not the painted end frames
    } else if (abs(material-7.) < .1) {
        rough = .98;
        albedo *= .94+.08*seed;
    }
}
"""
