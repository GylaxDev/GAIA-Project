"""Static catalog vertices, transformed by sidereal time; not procedural stars."""
from .hdr import HDR_CONV, HDR_OUT
from .materials import MATERIAL_LIGHT
from .clouds import CLOUD_DECK

STARFIELD_VS = """
#version 330 core
layout(location=0) in vec3 aDirection;
layout(location=1) in vec3 aMotion;
layout(location=2) in vec3 aColor;
layout(location=3) in float aMagnitude;
layout(location=4) in float aSeed;
uniform mat4 uView, uProj;
uniform mat3 uEquatorialToWorld;
uniform float uEpochYears, uPixelScale;
out vec3 vDirection, vColor;
out float vMagnitude, vSeed, vSigma;
void main() {
    vec3 eq = normalize(aDirection + aMotion * uEpochYears);
    vec3 dir = normalize(uEquatorialToWorld * eq);
    // Rotation only: the sky is infinitely distant and never translates with the camera.
    vec4 clip = uProj * vec4(mat3(uView) * dir, 0.);
    gl_Position = vec4(clip.xy, .9999 * clip.w, clip.w);
    vSigma = max(.65, uPixelScale * .00017);
    gl_PointSize = 6. * vSigma + 2.;
    vDirection = dir;
    vColor = aColor;
    vMagnitude = aMagnitude;
    vSeed = aSeed;
}
"""

STARFIELD_FS = """
#version 330 core
in vec3 vDirection, vColor;
in float vMagnitude, vSeed, vSigma;
uniform float uExposure, uNightFactor, uStarVisibility, uStarLimit, uStarExtinction, uTime;
out vec4 FragColor;
""" + HDR_CONV + HDR_OUT + MATERIAL_LIGHT + CLOUD_DECK + """
void main() {
    vec3 dir = normalize(vDirection);
    if (dir.z <= 0. || uStarVisibility <= .001 || vMagnitude > uStarLimit + .3) discard;
    float pointSize = 6. * vSigma + 2.;
    vec2 pixel = (gl_PointCoord - .5) * pointSize;
    float psf = exp(-dot(pixel, pixel) / (2. * vSigma * vSigma)) / (vSigma * vSigma);
    if (psf < .002) discard;
    float altitude = degrees(asin(clamp(dir.z, 0., 1.)));
    float airMass = 1. / max(dir.z + .50572 * pow(altitude + 6.07995, -1.6364), .025);
    vec3 transmission = exp(-vec3(.10, .16, .28) * max(airMass - 1., 0.) * uStarExtinction);
    float amplitude = .035 + .12 * pow(1. - dir.z, 2.);
    float twinkle = 1. + amplitude * sin(uTime * (2.1 + vSeed * 3.7) + vSeed * 237.);
    float limit = 1. - smoothstep(uStarLimit - .4, uStarLimit + .3, vMagnitude);
    float visibility = uStarVisibility * limit * (1. - cloudDensity(dir));
    vec3 light = vColor * nits(18.) * pow(10., -.4 * vMagnitude) * transmission
               * (psf * twinkle * visibility);
    FragColor = vec4(sceneOut(light), 0.);
}
"""

CONSTELLATION_FS = """
#version 330 core
in vec3 vDirection;
uniform float uExposure, uNightFactor, uStarVisibility;
out vec4 FragColor;
""" + HDR_CONV + HDR_OUT + MATERIAL_LIGHT + CLOUD_DECK + """
void main() {
    vec3 dir = normalize(vDirection);
    if (dir.z <= .01) discard;
    float visibility = uStarVisibility * (1. - cloudDensity(dir)) * smoothstep(.01, .12, dir.z);
    FragColor = vec4(sceneOut(emissionNits(vec3(.26, .43, .65), .45) * visibility), 0.);
}
"""
