"""Cheap analytic sky environment shared by glass facades and the sea.

``envSky(R, L)`` returns the radiance a mirror would reflect in direction ``R``
(world space, +Z up).  It is a deep cobalt zenith fading to the physical horizon
haze colour (``uFogColor``), with warm light around the sun and a patch of
fair-weather cumulus - the same cloud deck the sky pass draws, so reflections
agree with what is overhead. Directions on or below the horizon return a dark
ground/city fallback before either atmosphere path is evaluated.

Output is scene-linear HDR (same units as the lit colour *before* the exposure /
ACES step), so it can be added straight onto ``lit``.

The including shader must declare: uFogColor, uExposure, uNightFactor, uTime,
uSunColor. The shared cloud deck declares the weather uniforms itself.
"""

from .materials import MATERIAL_LIGHT
from .clouds import CLOUD_DECK
from .atmosphere import ATMOSPHERE
from .celestial import CELESTIAL_SKY

ENV_SKY = MATERIAL_LIGHT + CLOUD_DECK + ATMOSPHERE + CELESTIAL_SKY + """
uniform sampler2D uSkyLUT;     // atmosphere table from the sky shader (see SKY_LUT_FS)
uniform float uSkyLutOn;       // 0 -> LUT unavailable, use the analytic gradient below
uniform vec3  uSkySunDir;      // true sun direction (uSunDir may be the moon at night)
uniform float uAtmosphereIntensity, uSkyGlow;

vec3 envSky(vec3 R, vec3 L) {
    float day = 1.0 - clamp(uNightFactor, 0.0, 1.0);
    // The lower hemisphere is ground/occluded city, NOT the horizon sky LUT.
    // Return before atmosphere/cloud/sun evaluation so neither path leaks sky
    // into downward reflections. SSR can still supply actual visible street detail.
    vec3 ground = mix(vec3(0.8, 0.75, 0.68) * nits(0.015),
                      vec3(0.006, 0.0058, 0.0054), day);
    if (R.z <= 0.0) return ground;
    // display-referred haze -> approximate scene radiance
    vec3 y = pow(max(uFogColor, vec3(0.0)), vec3(2.2));
    vec3 a = 2.51 - 2.43 * y, b = 0.03 - 0.59 * y;
    vec3 horizon = (-b + sqrt(max(b * b + 0.56 * a * y, vec3(0.0)))) / (2.0 * a * max(uExposure, 1e-4));
    vec3 env;
    if (uSkyLutOn > 0.5) {
        // Exactly what the sky pass draws in this direction (same atmosphere()), then the
        // same fade into the horizon haze it applies around the horizon line.
        vec2 sxy = (length(uSkySunDir.xy) > 1e-3) ? normalize(uSkySunDir.xy) : vec2(1.0, 0.0);
        vec2 rxy = (length(R.xy) > 1e-4) ? normalize(R.xy) : sxy;
        float relAz = acos(clamp(dot(rxy, sxy), -1.0, 1.0));
        float v = sqrt(clamp((R.z + 0.1) / 1.1, 0.0, 1.0));
        vec3 atm = texture(uSkyLUT, vec2(relAz * 0.31830989, v)).rgb;
        env = mix(horizon, atm, smoothstep(-0.05, 0.05, R.z));
    } else {
        env = analyticAtmosphere(R, normalize(uSkySunDir), horizon, uAtmosphereIntensity);
        env += urbanSky(R, uNightFactor, uSkyGlow);
    }

    // Reflected sky: keep the galaxy and the moon's smooth halo, but never its disc.
    // A per-pixel normal makes the disc alias into sparkle on paint and glazing.
    env += celestialSkyReflected(R, normalize(uSkySunDir));
    env = weatherSky(env, R, normalize(uSkySunDir));
    // A narrow ABOVE-horizon transition avoids a hard reflection seam without
    // showing any sky below zero elevation (smoothstep edges are ordered).
    return mix(ground, env, smoothstep(0.0, 0.035, R.z));
}
"""
