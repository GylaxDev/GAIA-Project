"""Shared scene-linear solar highlight for the sky and reflected window glints."""

import math
from ..core.photometry import SUN_IRRADIANCE

SUN_LIGHT = f"const float SUN_RADIANCE = {SUN_IRRADIANCE / (math.pi * 0.00465 ** 2):.6f};\n" + """
const float SUN_RADIUS = 0.00465;     // radians: the sun's apparent angular radius (~0.27 degrees)
float sunRadiance(float cosAngle, float angularBlur) {
    float angle2 = 2.0 * (1.0 - clamp(cosAngle, 0.0, 1.0));
    float disc2 = SUN_RADIUS * SUN_RADIUS;
    float blur2 = angularBlur * angularBlur;
    float spread2 = disc2 + blur2;
    // Rough glass spreads the highlight while conserving its integrated brightness.
    float core = exp(-angle2 / spread2) * (disc2 / spread2);
    float halo2 = 0.012 * 0.012;
    float halo = exp(-angle2 / (halo2 + blur2)) * (halo2 / (halo2 + blur2));
    return SUN_RADIANCE * core + 24.0 * halo;
}
"""
