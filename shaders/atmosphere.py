"""Shared spherical scattering / twilight model. CPU companion: osm3d.core.atmosphere."""

ATMOSPHERE = """
#ifndef GAIA_ATMOSPHERE
#define GAIA_ATMOSPHERE
uniform float uAtmosphereHeight, uAerosolDensity;
const float ATM_PI = 3.14159265359;
const float ATM_RE = 6371e3;
const float ATM_RA = 6471e3;
const vec3 ATM_BETA_R = vec3(5.802e-6, 13.558e-6, 33.1e-6);
const vec3 ATM_BETA_M = vec3(3.996e-6);
const vec3 ATM_BETA_O = vec3(.650e-6, 1.881e-6, .085e-6);

vec3 atmTint(vec3 color) { return color / max(dot(color, PHOTO_LUMA), 1e-8); }
float atmOzone(float h) { return max(0., 1. - abs(h - 25000.) / 15000.); }
vec3 atmDensity(float h) {
    h = max(h, 0.);
    return vec3(exp(-h / 8000.), exp(-h / 1200.), atmOzone(h));
}
vec3 twilightRadiance(vec3 dir, vec3 sunDir) {
    float degrees = asin(clamp(sunDir.z, -1., 1.)) * (180. / ATM_PI);
    float elevation = max(dir.z, 0.);
    vec2 rayXY = dir.xy / max(length(dir.xy), 1e-6);
    vec2 sunXY = sunDir.xy / max(length(sunDir.xy), 1e-6);
    float toward = clamp(dot(rayXY, sunXY), -1., 1.);
    vec3 blue = atmTint(vec3(.24, .46, 1.));
    vec3 purple = atmTint(vec3(.47, .32, .82));
    vec3 warm = atmTint(vec3(1., .32, .09));
    float violet = exp(-pow((elevation - .16) / .17, 2.)) * pow(max(toward, 0.), 2.);
    vec3 color = mix(blue, purple, violet * .32);
    float rim = exp(-elevation * 14.) * pow(max(toward, 0.), 4.)
              * smoothstep(-10., -1., degrees) * (1. - smoothstep(0., 6., degrees));
    color = mix(color, warm, rim * .68);
    float beltHeight = radians(4. + max(-degrees, 0.) * .65);
    float belt = exp(-pow((elevation - beltHeight) / .065, 2.)) * pow(max(-toward, 0.), 3.)
               * smoothstep(-8., -2., degrees) * (1. - smoothstep(0., 4., degrees));
    color += atmTint(vec3(.85, .38, .48)) * belt * .22;
    float shadow = exp(-elevation * 28.) * pow(max(-toward, 0.), 2.)
                 * (1. - smoothstep(-6., -.5, degrees));
    color *= 1. - shadow * .48;
    float gain = exp2(min(degrees, 0.) * .86) * (1. - smoothstep(0., 7., degrees))
               * smoothstep(-20., -17., degrees);
    return color * nits(450.) * gain;
}

vec3 atmosphere(vec3 viewDir, vec3 sunDir, float sunIrradiance) {
    vec3 camPos = vec3(0., 0., ATM_RE + clamp(uAtmosphereHeight, 2., 80000.));
    float b = dot(camPos, viewDir);
    // Factored radius difference avoids subtracting two near-equal large squares.
    float outer = (ATM_RA - camPos.z) * (ATM_RA + camPos.z);
    float tMax = -b + sqrt(max(0., b * b + outer));
    float earthDisc = b * b + (ATM_RE - camPos.z) * (ATM_RE + camPos.z);
    if (b < 0. && earthDisc >= 0.) tMax = min(tMax, -b - sqrt(earthDisc));
    tMax = clamp(tMax, 0., @PATHCAP@);
    float aerosol = clamp(uAerosolDensity, .1, 8.);
    vec3 optical = vec3(0.), sumR = vec3(0.), sumM = vec3(0.);
    const int N_VIEW = 20, N_SUN = 6;
    for (int i = 0; i < N_VIEW; ++i) {
        float a = float(i) / float(N_VIEW), z = float(i + 1) / float(N_VIEW);
        float ds = tMax * (z * z - a * a);
        vec3 p = camPos + viewDir * (tMax * (a * a + z * z) * .5);
        vec3 rho = atmDensity(length(p) - ATM_RE);
        vec3 segment = rho * ds;
        float sunB = dot(p, sunDir), radius = length(p);
        float sunEarth = sunB * sunB + (ATM_RE - radius) * (ATM_RE + radius);
        if (!(sunB < 0. && sunEarth > 0.)) {
            float sunT = -sunB + sqrt(max(0., sunB * sunB + (ATM_RA - radius) * (ATM_RA + radius)));
            vec3 sunOptical = vec3(0.);
            for (int j = 0; j < N_SUN; ++j) {
                float sa = float(j) / float(N_SUN), sz = float(j + 1) / float(N_SUN);
                vec3 ps = p + sunDir * (sunT * (sa * sa + sz * sz) * .5);
                sunOptical += atmDensity(length(ps) - ATM_RE) * (sunT * (sz * sz - sa * sa));
            }
            vec3 total = optical + segment * .5 + sunOptical;
            vec3 tau = ATM_BETA_R * total.x + ATM_BETA_M * (aerosol * 1.1 * total.y) + ATM_BETA_O * total.z;
            vec3 attenuation = exp(-tau);
            sumR += segment.x * attenuation;
            sumM += segment.y * attenuation;
        }
        optical += segment;
    }
    float mu = clamp(dot(viewDir, sunDir), -1., 1.), g = .76;
    float phaseR = 3. / (16. * ATM_PI) * (1. + mu * mu);
    float phaseM = (1. - g * g) / (4. * ATM_PI * pow(max(1. + g * g - 2. * g * mu, 1e-4), 1.5));
    vec3 color = (sumR * ATM_BETA_R * phaseR + sumM * ATM_BETA_M * (aerosol * phaseM)) * sunIrradiance;
    color += twilightRadiance(viewDir, sunDir) * (sunIrradiance / SOLAR_IRRADIANCE);
    return max(color, vec3(0.));
}

vec3 urbanSky(vec3 dir, float night, float glow) {
    float h = exp(-max(dir.z, 0.) * 4.);
    vec3 tint = mix(vec3(@ZENITH@), vec3(@HORIZON@), h);
    return (emissionNits(tint, (.11 + .22 * h) * (@GAIN@ * glow / .3))
            + emissionNits(vec3(.36, .52, .50), .00025)) * night;
}

// Allocation-failure path: same twilight / airglow as the LUT, inexpensive daylight.
vec3 analyticAtmosphere(vec3 dir, vec3 sunDir, vec3 horizon, float intensity) {
    float up = clamp(dir.z, 0., 1.);
    float day = smoothstep(-.035, .16, sunDir.z);
    vec3 zenith = vec3(.020, .047, .105) * intensity * day;
    vec3 base = mix(horizon * day, zenith, pow(up, .42));
    float mu = max(dot(dir, sunDir), 0.);
    base += vec3(.055, .045, .028) * pow(mu, 32.) * intensity * day;
    return base + twilightRadiance(dir, sunDir) * intensity;
}
#endif
"""

from ..core.atmosphere import URBAN_ZENITH, URBAN_HORIZON, URBAN_GLOW_GAIN, HORIZON_PATH_CAP
ATMOSPHERE = (ATMOSPHERE
              .replace("@ZENITH@", ", ".join(f"{c:.3f}" for c in URBAN_ZENITH))
              .replace("@HORIZON@", ", ".join(f"{c:.3f}" for c in URBAN_HORIZON))
              .replace("@GAIN@", f"{URBAN_GLOW_GAIN:.3f}")
              .replace("@PATHCAP@", f"{HORIZON_PATH_CAP:.1f}"))
