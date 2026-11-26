from .hdr import HDR_CONV, HDR_OUT
from .sun import SUN_LIGHT
from .materials import MATERIAL_LIGHT
from .clouds import CLOUD_DECK
from .atmosphere import ATMOSPHERE
from .celestial import CELESTIAL_SKY

SKY_VS = """
#version 330 core
layout(location=0) in vec2 aPos;
uniform mat4 uInvProjection;
uniform mat3 uInvViewRotation;
out vec3 vRayDir;
void main() {
    gl_Position = vec4(aPos, 0.9999, 1.0);
    vec4 p = uInvProjection * vec4(aPos, 1.0, 1.0);
    vRayDir = uInvViewRotation * normalize(p.xyz / p.w);
}
"""

SKY_FS = ("""
#version 330 core
in vec3 vRayDir;
uniform vec3  uCamPos;
uniform vec3  uSunDir;
uniform vec3  uFogColor;
uniform float uSunIntensity;
uniform float uSunDiscIntensity;
uniform float uNightFactor;
uniform float uSkyGlow;
uniform float uTime;
uniform float uExposure;
uniform sampler2D uSkyLUT;
uniform float uSkyLutOn;
out vec4 FragColor;
""" + HDR_CONV + HDR_OUT + SUN_LIGHT + MATERIAL_LIGHT + CLOUD_DECK + """
""" + ATMOSPHERE + CELESTIAL_SKY + """
void main() {
    vec3 dir = normalize(vRayDir);
    vec3 sunDir = normalize(uSunDir);

    vec3 sky;
    if (uSkyLutOn > 0.5) {
        vec2 sxy = normalize(sunDir.xy + vec2(1e-6));
        vec2 rxy = normalize(dir.xy + vec2(1e-6));
        float az = acos(clamp(dot(rxy, sxy), -1.0, 1.0));
        float elev = sqrt(clamp((dir.z + 0.1) / 1.1, 0.0, 1.0));
        sky = texture(uSkyLUT, vec2(az / ATM_PI, elev)).rgb;
    } else {
        vec3 horizon = acesInv(pow(max(uFogColor, vec3(0.)), vec3(2.2))) / max(uExposure, 1e-4);
        sky = analyticAtmosphere(dir, sunDir, horizon, uSunIntensity);
        sky += urbanSky(dir, uNightFactor, uSkyGlow);
    }

    float sunDot = max(dot(dir, sunDir), 0.0);
    // A hot solar disc with the same radiance profile used by reflective window glass.
    sky += vec3(1.0, 0.95, 0.9) * sunRadiance(sunDot, 0.0) * uSunDiscIntensity
         * cloudSunVisibility();

    sky += celestialSky(dir, sunDir);
    sky = weatherSky(sky, dir, sunDir);
    // Under a cloud deck the horizon IS the fog the buildings fade into: blend to exactly that
    // colour so there is no seam between the far skyline and the sky behind it.
    float deckAmt = smoothstep(.12, .80, clamp(uCloudCover, 0., 1.));
    vec3 fogLight = acesInv(pow(max(uFogColor, vec3(0.)), vec3(2.2))) / max(uExposure, 1e-4);
    sky = mix(sky, fogLight, deckAmt * pow(1. - smoothstep(0., .30, dir.z), 1.6));

    vec3 color = sceneOut(sky);

    float horizonFade = smoothstep(-0.05, 0.05, dir.z);
    color = mix(displayToScene(uFogColor), color, horizonFade);

    FragColor = vec4(color, 1.0);
}
""")

# ---------------------------------------------------------------------------------------
# Sky look-up table.  Glass facades and the sea reflect the sky, and used to approximate it
# with a hand-tuned gradient that did not match the real atmosphere pass (different hue, a
# cream/green haze where the sky is teal).  Instead the *same* atmosphere() function renders
# a small (azimuth relative to the sun, elevation) table once per sky change and envSky()
# samples it, so reflections are the sky by construction.  The sky is symmetric about the
# sun's vertical plane, hence relative azimuth 0..pi is enough.  v = sqrt((z + 0.1) / 1.1).
SKY_LUT_VS = """
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main() {
    vUV = aPos * 0.5 + 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

SKY_LUT_FS = ("""
#version 330 core
in vec2 vUV;
uniform vec3  uSunDir;
uniform float uSunIntensity;
uniform float uNightFactor;
uniform float uSkyGlow;
out vec4 FragColor;
""" + MATERIAL_LIGHT + ATMOSPHERE + """
void main() {
    float relAz = vUV.x * ATM_PI;
    float z = vUV.y * vUV.y * 1.1 - 0.1;
    float r = sqrt(max(1.0 - z * z, 0.0));
    vec3 sd = normalize(uSunDir);
    vec2 sxy = (length(sd.xy) > 1e-3) ? normalize(sd.xy) : vec2(1.0, 0.0);
    vec2 perp = vec2(-sxy.y, sxy.x);
    vec3 dir = vec3(r * (cos(relAz) * sxy + sin(relAz) * perp), z);
    vec3 sky = atmosphere(dir, sd, uSunIntensity * SOLAR_IRRADIANCE);
    sky += urbanSky(dir, uNightFactor, uSkyGlow);
    FragColor = vec4(min(sky, vec3(64.0)), 1.0);
}
""")
