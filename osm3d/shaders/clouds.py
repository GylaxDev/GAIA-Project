"""Lightweight 2D cloud layers shared by sky, stars and material reflections.

Two curved sheets use filtered planar noise and approximate optical-depth
lighting. There is no volume marching, 3D noise or cloud texture allocation;
both layers still follow the weather system's integrated world-space wind.
"""
from .materials import MATERIAL_LIGHT

CLOUD_DECK = MATERIAL_LIGHT + """
#ifndef GAIA_CLOUD_DECK
#define GAIA_CLOUD_DECK
uniform float uCloudCover, uStorm, uLightning;
uniform vec2 uWind, uWindOffset;
uniform vec3 uCloudLight, uCloudSunLight, uCloudOrigin, uCloudSunDir;

float cloudHash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float cloudNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * f * (f * (f * 6. - 15.) + 10.);
    return mix(mix(cloudHash(i), cloudHash(i + vec2(1., 0.)), u.x),
               mix(cloudHash(i + vec2(0., 1.)), cloudHash(i + vec2(1., 1.)), u.x), u.y);
}
float cloudFilteredNoise(vec2 p, float rayFootprint) {
    return mix(cloudNoise(p), .5, smoothstep(.35, 1.4, rayFootprint));
}
float cloudFbm(vec2 p, float rayFootprint) {
    // Fixed three-octave 2D budget, rather than noise at every volume step.
    return .57 * cloudFilteredNoise(p, rayFootprint)
         + .28 * cloudFilteredNoise(p * 2.03 + vec2(17.2, 9.1), rayFootprint * 2.03)
         + .15 * cloudFilteredNoise(p * 4.11 + vec2(3.7, 21.3), rayFootprint * 4.11);
}
float cloudShellDistance(vec3 dir, float layerHeight) {
    float h = clamp(uCloudOrigin.z, 0., 80000.);
    if (h >= layerHeight && dir.z >= 0.) return -1.;
    float r = 6371e3 + h, b = r * dir.z;
    float delta = (layerHeight - h) * (2. * 6371e3 + layerHeight + h);
    float root = sqrt(max(b * b + delta, 0.));
    // Stable quadratic for upward rays: no flat-deck division / horizon stretching.
    return b >= 0. ? delta / max(root + b, 1.) : -b + root;
}
float cloudRayFootprint(vec3 dir, float distance, float rayCone) {
    // Account for the shell intersection stretching at grazing angles, without
    // taking noise derivatives inside a material/layer branch.
    float incidence = abs(dir.z + distance / (6371e3 + clamp(uCloudOrigin.z, 0., 80000.)));
    return max(distance, 0.) * rayCone / max(incidence, .075);
}
vec4 cloudColumn(vec3 dir) {
    // Capture the pixel cone before any early return or layer divergence.
    float rayCone = max(length(dFdx(dir)), length(dFdy(dir)));
    float cover = clamp(uCloudCover, 0., 1.);
    if (cover <= .005 || dir.z < -.015) return vec4(0.);
    float lowHeight = mix(1400., 750., clamp(uStorm, 0., 1.));
    float lowDistance = cloudShellDistance(dir, lowHeight);
    float highDistance = cloudShellDistance(dir, 7600.);
    float lowTau = 0., highTau = 0., illumination = .5;
    if (lowDistance > 0. && cover > .07) {
        vec2 point = (uCloudOrigin.xy + dir.xy * min(lowDistance, 120000.) - uWindOffset) * .00045;
        float footprint = cloudRayFootprint(dir, lowDistance, rayCone) * .00045;
        vec2 warp = vec2(cloudFilteredNoise(point * .55 + 2.7, footprint * .55),
                         cloudFilteredNoise(point * .55 + 17.3, footprint * .55)) - .5;
        float body = cloudFbm(mat2(.8, -.6, .6, .8) * point + warp * .75, footprint * 1.4);
        float detail = cloudFilteredNoise(point * 6.1 + 7.4, footprint * 6.1);
        float threshold = mix(.74, .28, cover);
        float edgeFootprint = .055 + min(footprint, .5) * .18;
        float shape = smoothstep(threshold - edgeFootprint, threshold + .14 + edgeFootprint,
                                 body + (detail - .5) * .075);
        float core = smoothstep(threshold + .02, threshold + .24, body);
        // Cheap self-shadowing: compare the cloud body with the same field shifted towards the sun.
        // Thinner towards the sun means a lit edge; thicker means this side is in shade.  Gives
        // cumulus a lit crown and a grey belly instead of a uniformly shaded sheet.
        float sunLen = length(uCloudSunDir.xy);
        vec2 sunStep = sunLen > 1e-3 ? uCloudSunDir.xy / sunLen : vec2(0.);
        float toward = cloudFbm(mat2(.8, -.6, .6, .8) * (point + sunStep * .22) + warp * .75, footprint * 1.4);
        float slope = (body - toward) * smoothstep(.08, .55, sunLen);
        float patchTau = shape * (1.8 + 3.2 * core) * smoothstep(.07, .25, cover)
                       * (1. - smoothstep(30000., 110000., lowDistance));
        float deck = smoothstep(.68, .95, cover);
        // Dense fronts close every gap, including the horizon; a thin sheet's
        // optical depth replaces volumetric integration and shades its cores.
        lowTau = mix(patchTau, 7. + body * 3., deck) * mix(2.4, 1., clamp(dir.z, 0., 1.));
        illumination = clamp(.66 - .34 * core + slope * 1.9 + (detail - .5) * .10, .20, .96);
    }
    if (highDistance > 0.) {
        vec2 wind = length(uWind) > .01 ? normalize(uWind) : vec2(1., 0.);
        vec2 point = (uCloudOrigin.xy + dir.xy * min(highDistance, 300000.) - uWindOffset * 1.7) * .00018;
        vec2 p = vec2(dot(point, wind) * .85, dot(point, vec2(-wind.y, wind.x)) * 2.3);
        float footprint = cloudRayFootprint(dir, highDistance, rayCone) * .00018 * 2.3;
        p.y += (cloudFbm(p * .35 + 8.3, footprint * .35) - .5) * 2.8;
        float fibre = cloudFbm(p + 7.3, footprint * 1.4);
        float fringe = cloudFilteredNoise(p * vec2(2.3, 6.7) + 14.8, footprint * 6.7);
        float veil = smoothstep(.55 - cover * .06, .80 - cover * .04, fibre)
                   * (.28 + .72 * fringe);
        highTau = veil * (.055 + .20 * cover) * mix(1., 1.8, 1. - max(dir.z, 0.))
                * (1. - smoothstep(90000., 280000., highDistance)) * smoothstep(0., .04, dir.z);
    }
    return vec4(lowTau, highTau, illumination, max(lowDistance, 0.));
}
float cloudDensity(vec3 dir) {
    vec4 column = cloudColumn(dir);
    return 1. - exp(-column.x - column.y);
}
float cloudSunVisibility() {
    return 1. - smoothstep(.55, .90, clamp(uCloudCover, 0., 1.));
}
vec3 cloudIncident() {
    vec3 incident = dot(uCloudLight, PHOTO_LUMA) > 1e-8 ? uCloudLight
                  : mix(vec3(.14, .15, .16), emissionNits(vec3(.65, .72, .82), .30), uNightFactor);
    return incident;
}
vec3 cloudLayerRadiance(vec3 dir, vec3 sunDir, vec4 column, bool high) {
    float tau = high ? column.y : column.x;
    // Higher cirrus can still catch sunlight after low clouds enter Earth's shadow.
    float sunUp = high ? smoothstep(-.049, .015, sunDir.z) : smoothstep(-.026, .025, sunDir.z);
    float forward = pow(max(dot(dir, sunDir), 0.), 18.);
    float transmit = exp(-tau * (high ? .32 : .30));
    // Diffuse fill does not brighten every thin boundary into a white outline.
    // Only forward-scattered direct sunlight supplies the silver lining below.
    float bodyShade = high ? .88 : .36 + .66 * column.z;
    vec3 stormTint = mix(vec3(1.), vec3(.80, .87, 1.12), uStorm);
    vec3 ambient = cloudIncident() * bodyShade * (1. - .52 * uStorm) * stormTint;
    // Optical depth darkens thick cores; thin sunward rims give silver linings.
    vec3 direct = uCloudSunLight * sunUp * (transmit * (.12 + .42 * forward) + (high ? .035 : .022));
    vec3 flash = mix(vec3(.26, .29, .34), vec3(1., 1.08, 1.2) * nits(45.), uNightFactor) * uLightning;
    return ambient + direct + flash;
}
vec3 cloudRadiance(vec3 dir, vec3 sunDir) {
    vec4 column = cloudColumn(dir);
    return cloudLayerRadiance(dir, sunDir, column, false);
}
vec3 weatherSky(vec3 base, vec3 dir, vec3 sunDir) {
    vec4 column = cloudColumn(dir);
    if (column.x + column.y <= .001) return base;
    vec3 high = mix(base, cloudLayerRadiance(dir, sunDir, column, true), 1. - exp(-column.y));
    // Atmospheric in-scattering veils distant cloud structure without bright,
    // aliased rims. This is deliberately a cheap aerial-perspective approximation.
    float air = 1. - exp(-column.w * .000075);
    // Distant cloud fades into the haze *under the deck*, not into the clear-sky horizon: with a
    // storm or overcast above, the clear atmosphere's yellow / orange horizon must not show through.
    vec3 deckHaze = cloudIncident() * .80 * (1. - .52 * uStorm) * mix(vec3(1.), vec3(.80, .87, 1.12), uStorm);
    vec3 hazeTo = mix(base, deckHaze, smoothstep(.12, .80, uCloudCover));
    vec3 low = mix(cloudLayerRadiance(dir, sunDir, column, false), hazeTo, air);
    return mix(high, low, 1. - exp(-column.x));
}
#endif
"""
