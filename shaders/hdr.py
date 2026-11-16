"""Shared GLSL for the HDR scene pipeline.

Every scene shader used to tone-map (ACES) and gamma-encode on its own, which
made the 16-bit scene buffer display-referred: nothing could adapt exposure,
and bloom only had the few additive passes to work with.  Now the scene is
written *scene-referred* (linear light x ``uExposure``), and a single post pass
applies eye adaptation, bloom, tone mapping, the film grade and gamma.

``uHdrOut`` keeps the old behaviour available (wireframe, no float targets):
0 = tone-map in the shader as before, 1 = write linear HDR.

Two snippets, because some passes (rain, light sprites) do not carry ``uExposure``:

HDR_CONV  needs nothing.  acesInv / displayToScene / addScene.
HDR_OUT   needs ``uniform float uExposure`` declared above it.  sceneOut / sceneFog.
"""

HDR_CONV = """
uniform float uHdrOut;
uniform float uAirExtinction;     // inverse metres, 3.912 / meteorological visibility
uniform float uAirCameraHeight;
float cityFog(float dist, float height, float nearD, float farD) {
    // Aerosol density falls with altitude; retain molecular scattering above the city.
    float meanHeight = max(0.0, (height + uAirCameraHeight) * 0.5);
    float density = 0.12 + 0.88 * exp(-meanHeight / 1200.0);
    float air = 1.0 - exp(-max(uAirExtinction, 0.0) * max(dist, 0.0) * density);
    // Only the last part of the resident city fades out to conceal the streaming boundary.
    return max(air, smoothstep(nearD, max(farD, nearD + 1.0), dist));
}
// Narkowicz ACES fit and its analytic inverse (solve a x^2 + b x = y (c x^2 + d x + e)).
vec3 acesFwd(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 acesInv(vec3 y) {
    y = clamp(y, 0.0, 0.995);
    vec3 A = 2.51 - 2.43 * y;
    vec3 B = 0.03 - 0.59 * y;
    vec3 C = -0.14 * y;
    return (-B + sqrt(max(B * B - 4.0 * A * C, 0.0))) / (2.0 * A);
}
// A display-referred colour (fog, haze, glow tints) expressed as scene light, so that
// the post tone-map reproduces exactly that colour.  Identity when tone-mapping in place.
vec3 displayToScene(vec3 c) {
    return uHdrOut > 0.5 ? acesInv(pow(max(c, vec3(0.0)), vec3(2.2))) : c;
}
// Overlays that used to be added on top of the *gamma-encoded* frame (neon, lamp
// sprites).  Squaring below 1.0 keeps their apparent strength in linear light, values
// above 1.0 keep their HDR headroom for bloom.
vec3 addScene(vec3 c) {
    if (uHdrOut < 0.5) return c;
    c = max(c, vec3(0.0));
    return mix(c * c, c, step(vec3(1.0), c));
}
"""

HDR_OUT = """
vec3 sceneOut(vec3 lit) {
    vec3 e = lit * uExposure;
    if (uHdrOut > 0.5) return e;
    return pow(acesFwd(e), vec3(1.0 / 2.2));
}
"""
