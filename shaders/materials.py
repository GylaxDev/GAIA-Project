"""Shared photometric conversion and linear material reflectance limits."""

from ..core.photometry import (SUN_IRRADIANCE, NITS_PER_SCENE_UNIT, LUMINANCE_NITS,
                          REFLECTANCE, OFFICE_LUX, INTERIOR_DAY_GAIN, INTERIOR_NIGHT_GAIN,
                          INTERIOR_FIXTURE_DAY_GAIN, INTERIOR_FIXTURE_NIGHT_GAIN,
                          DAY_EXPOSURE, NIGHT_EXPOSURE_STOPS)

MATERIAL_LIGHT = (
    "#ifndef GAIA_MATERIAL_LIGHT\n#define GAIA_MATERIAL_LIGHT\n"
    + f"const float SOLAR_IRRADIANCE = {SUN_IRRADIANCE:.6f};\n"
    + f"const float NITS_TO_SCENE = {1.0 / NITS_PER_SCENE_UNIT:.12f};\n"
    + f"const float DAY_EXPOSURE = {DAY_EXPOSURE:.4f};\n"
    + f"const float NIGHT_EXPOSURE_STOPS = {NIGHT_EXPOSURE_STOPS:.2f};\n"
    + f"const float OFFICE_ILLUMINANCE = {OFFICE_LUX:.1f};\n"
    + f"const float INTERIOR_DAY_GAIN = {INTERIOR_DAY_GAIN:.1f};\n"
    + f"const float INTERIOR_NIGHT_GAIN = {INTERIOR_NIGHT_GAIN:.2f};\n"
    + f"const float INTERIOR_FIXTURE_DAY_GAIN = {INTERIOR_FIXTURE_DAY_GAIN:.1f};\n"
    + f"const float INTERIOR_FIXTURE_NIGHT_GAIN = {INTERIOR_FIXTURE_NIGHT_GAIN:.4f};\n"
    + "".join(f"const float {name}_NITS = {value:.1f};\n"
              for name, value in LUMINANCE_NITS.items())
    + "".join(f"const vec2 {name}_REFLECTANCE = vec2({lo:.3f}, {hi:.3f});\n"
              for name, (lo, hi) in REFLECTANCE.items())
    + """
const vec3 PHOTO_LUMA = vec3(0.2126, 0.7152, 0.0722);
float nits(float value) { return value * NITS_TO_SCENE; }
vec3 linearColor(vec3 srgb) {
    srgb = clamp(srgb, 0.0, 1.0);
    return mix(srgb / 12.92, pow((srgb + 0.055) / 1.055, vec3(2.4)),
               step(vec3(0.04045), srgb));
}
vec3 reflectanceRange(vec3 linear, vec2 range) {
    float y = dot(linear, PHOTO_LUMA);
    if (y <= 1e-6) return vec3(range.x);
    return clamp(linear * (clamp(y, range.x, range.y) / y), 0.0, 0.95);
}
vec3 emissionNits(vec3 tint, float luminance) {
    vec3 c = max(tint, vec3(0.0));
    return c * (nits(luminance) / max(dot(c, PHOTO_LUMA), 1e-5));
}
#endif
"""
)
