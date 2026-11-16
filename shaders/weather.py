from ..core.photometry import HEADLIGHT_CANDELA, NITS_PER_SCENE_UNIT
from .hdr import HDR_CONV, HDR_OUT
from .rain_surface import RAIN_EXPOSURE
from .streetspill import STREET_SPILL

# Headlight-lit rain: the nearest cars' projectors, scattered by the drops ahead
# of them.  lights.py caps and orders the same list, so the shader loops only
# beams that can actually reach the camera's neighbourhood.
HEADLIGHT_BEAMS = 8
HEADLIGHT_REACH = 90.0

HEADLIGHT_SCATTER = f"""
#define RAIN_HEADLIGHT_BEAMS {HEADLIGHT_BEAMS}
uniform vec4 uBeamPos[RAIN_HEADLIGHT_BEAMS];   // xyz luminaire position, w unused
uniform vec4 uBeamDir[RAIN_HEADLIGHT_BEAMS];   // xyz beam axis, w = lamp power
uniform int uBeamCount;

vec3 headlightScatter(vec3 P) {{
    vec3 E = vec3(0.0);
    for (int k = 0; k < RAIN_HEADLIGHT_BEAMS; k++) {{
        if (k >= uBeamCount) break;
        vec3 d = uBeamPos[k].xyz - P;                  // drop -> lamp
        float r2 = dot(d, d);
        if (r2 > {HEADLIGHT_REACH * HEADLIGHT_REACH:.1f}) continue;
        vec3 L = d * inversesqrt(max(r2, 0.001));
        // A projector throws a cone ahead of the car, aimed slightly down: drops
        // outside it stay dark, so passing traffic reveals the rain it faces.
        float cone = smoothstep(.74, .93, dot(-L, normalize(uBeamDir[k].xyz)));
        float fade = 1.0 - smoothstep({HEADLIGHT_REACH * .45:.1f}, {HEADLIGHT_REACH:.1f},
                                      sqrt(max(r2, .001)));
        E += vec3(1.0, .90, .68) * (uBeamDir[k].w * cone * fade / max(r2, .25));
    }}
    return E * {HEADLIGHT_CANDELA / NITS_PER_SCENE_UNIT:.9f};
}}
"""

# Precipitation is the largest overdraw pass in the frame: thousands of thin
# billboards.  Lighting, aerial fog and roof sheltering therefore run once per
# vertex, and the fragment stage keeps only the drop's soft profile.  Sampling
# the shelter depth map per fragment (as this pass used to) dominated its cost.
PRECIP_VS = "".join(("""
#version 330 core
layout(location=0) in vec2 aQuad;
layout(location=1) in vec4 aSeed;

uniform mat4 uView, uProj;
uniform vec3 uCamPos, uCamRight, uCamUp;
uniform vec2 uWind, uWindOffset;
uniform float uTime, uSnowMix, uAmount, uPixelScale;
uniform vec3 uFogColor, uSunColor, uAmbColor;
uniform float uExposure, uFogNear, uFogFar, uNightFactor;

out vec2 vQuad;
out float vAlpha;
out float vType;
out vec3 vColor;
""", HDR_CONV, HDR_OUT, RAIN_EXPOSURE, STREET_SPILL, HEADLIGHT_SCATTER, """
void main() {
    float layer = step(.55, fract(aSeed.w * 17.9));
    vec3 box = mix(vec3(64., 64., 56.), vec3(192., 192., 96.), layer);
    vec3 anchor = uCamPos + vec3(0., 0., 4.);

    // World-space, camera-centred wrapping box.  The mesh never changes:
    // particles that leave one side re-enter on the other side.
    float fall = mix(11.0 + aSeed.w * 6., .65 + aSeed.w * 1.15, uSnowMix);
    float drift = mix(.70 + .25 * aSeed.w, .42, uSnowMix);
    vec3 velocity = vec3(uWind * drift, -fall);
    vec3 p = aSeed.xyz * box + vec3(uWindOffset * drift, -fall * uTime);
    p = anchor + mod(p - anchor + box * 0.5, box) - box * 0.5;

    float phase = aSeed.w * 6.2831853;
    p.xy += vec2(sin(uTime * 1.15 + phase),
                 cos(uTime * .87 + phase * 1.7)) * .40 * uSnowMix;
    p.z += sin(uTime * .93 + phase * 1.3) * .22 * uSnowMix;

    float viewDepth = max(-(uView * vec4(p, 1.)).z, .2);

    // A millimetre-scale drop, never a widened ribbon.  Sub-pixel particles are
    // expanded to about half a pixel and compensated by coverage so they keep
    // their light contribution without turning into shimmering bars.
    float physicalWidth = mix(.0042 + .0034 * aSeed.w, .022 + .030 * aSeed.w, uSnowMix);
    float rasterWidth = max(physicalWidth, .50 * viewDepth / uPixelScale);
    float coverage = min(1., physicalWidth / rasterWidth);

    // Photographic shutter-length trail: a fast drop smears over a few tens of
    // centimetres, and the streak length varies from drop to drop.
    float shutter = (.0210 + .0170 * aSeed.w) * (1. + .35 * fract(aSeed.y * 29.7));
    float halfLength = mix(fall * shutter * .5, physicalWidth, uSnowMix);

    // The streak leans along the real falling trajectory, so wind slants it,
    // and its ribbon stays edge-on to the view.
    vec3 up = normalize(-velocity);
    vec3 side = cross(up, normalize(uCamPos - p));
    side = length(side) > .001 ? normalize(side) : uCamRight;

    // Narrow the trailing end so the streak tapers instead of ending in a block.
    float taper = mix(1., .34, (aQuad.y + 1.) * .5);
    taper = mix(taper, 1., uSnowMix);
    vec3 world = p + mix(side, uCamRight, uSnowMix) * aQuad.x * rasterWidth * taper
                   + mix(up, uCamUp, uSnowMix) * aQuad.y * halfLength;

    float dist = length(world - uCamPos);
    float nearLayer = 1. - smoothstep(27., 46., dist);
    float farLayer = smoothstep(26., 42., dist) * (1. - smoothstep(105., 135., dist));
    float alpha = mix(nearLayer * .50, farLayer * .34, layer)
                * smoothstep(1.2, 2.6, dist) * coverage;
    // Amount controls the instance count; do not square it again into opacity.
    alpha *= (.70 + .30 * sqrt(uAmount)) * smoothstep(.005, .025, uAmount);
    // Dense far rain reads as a veil, not as bright specks: let it thin out.
    alpha *= 1. - .25 * smoothstep(25., 100., dist);
    // Shelter the whole drop from its centre: a per-corner test would clip single
    // vertices of a quad and stretch the remaining triangle into a stray sliver.
    alpha *= rainExposure(p);

    vAlpha = alpha * mix(.72, .90, uSnowMix);
    vQuad = aQuad;
    vType = uSnowMix;
    if (vAlpha < .0025) {
        // Sheltered or fully faded: emit a clipped vertex so no fragments run.
        gl_Position = vec4(0., 0., 2., 1.);
        vColor = vec3(0.);
        return;
    }

    // A droplet is a tiny scattering volume, so the street lamps around it light
    // it from every side: rain glows inside a lamp's cone instead of vanishing
    // into the ambient sky term.
    vec3 lampE = vec3(0.0);
    if (uNightFactor > .02) {
        lampE = streetLampScatter(p, normalize(uCamPos - p)) * (.12 * uNightFactor);
        // Headlights carry further than their pool on the road: the beam lights
        // the streaks between the car and whatever it is driving towards.
        lampE += headlightScatter(p) * (.16 * uNightFactor);
    }

    // Drops refract and reflect the bright sky; modelling them as dark diffuse
    // paint made rain disappear.  Per-drop brightness breaks the uniform sheet.
    float sparkle = fract(aSeed.x * 37.1 + aSeed.y * 11.7);
    vec3 rainLight = uAmbColor * (2.10 + 1.00 * sparkle) + uSunColor * .03 + lampE;
    vec3 snowLight = vec3(.78, .80, .81) * (uAmbColor * .90 + uSunColor * .318) + lampE * .80;
    float fog = cityFog(dist, world.z, uFogNear, uFogFar);
    vColor = mix(sceneOut(mix(rainLight, snowLight, uSnowMix)), displayToScene(uFogColor), fog * .65);

    gl_Position = uProj * uView * vec4(world, 1.0);
}
"""))

PRECIP_FS = """
#version 330 core
in vec2 vQuad;
in float vAlpha;
in float vType;
in vec3 vColor;
out vec4 FragColor;

void main() {
    // Thin bright core, soft ends; snow keeps the round flake profile.
    float across = 1. - abs(vQuad.x);
    float along = 1. - abs(vQuad.y);
    float streak = pow(across, 1.25) * pow(along, .60);
    float soft = 1. - smoothstep(.2, 1., length(vQuad));
    float alpha = mix(streak, soft, vType) * vAlpha;
    if (alpha < .0035) discard;
    FragColor = vec4(vColor, alpha);
}
"""

SPLASH_VS = "".join(("""
#version 330 core
layout(location=0) in vec2 aQuad;
layout(location=1) in vec4 aSeed;
uniform mat4 uView, uProj;
uniform vec3 uCamPos, uCamRight, uCamUp;
uniform vec3 uFogColor, uSunColor, uAmbColor;
uniform float uTime, uPixelScale, uExposure, uFogNear, uFogFar, uNightFactor;
out vec2 vQuad;
out float vAlpha;
out vec3 vColor;
""", HDR_CONV, HDR_OUT, RAIN_EXPOSURE, STREET_SPILL, """
void main() {
    vec2 box = vec2(72.);
    vec2 xy = uCamPos.xy + mod(aSeed.xy * box - uCamPos.xy + box * .5, box) - box * .5;
    vec2 uv = (uRainViewProj * vec4(xy, 0., 1.)).xy * .5 + .5;
    uv = (floor(uv / uRainRoofTexel) + .5) * uRainRoofTexel;
    float depth = texture(uRainRoof, uv).r;
    vec4 support = uRainInvViewProj * vec4(uv * 2. - 1., depth * 2. - 1., 1.);
    vec3 surface = support.xyz / support.w;
    float clock = uTime * (1.3 + .7 * aSeed.w) + aSeed.z * 37.;
    float life = fract(clock) / .38;
    float rainLive = step(life, 1.) * step(depth, .99999) * step(.5, uRainRoofOn);
    rainLive *= step(0., uv.x) * step(uv.x, 1.) * step(0., uv.y) * step(uv.y, 1.);
    // Reject silhouette cells and steep walls rather than splashing in mid-air.
    float dx = abs(texture(uRainRoof, uv + vec2(uRainRoofTexel.x, 0.)).r - depth);
    float dy = abs(texture(uRainRoof, uv + vec2(0., uRainRoofTexel.y)).r - depth);
    rainLive *= 1. - smoothstep(uRainRoofBias * 4., uRainRoofBias * 12., max(dx, dy));
    life = clamp(life, 0., 1.);
    float angle = aSeed.w * 100.;
    vec3 center = surface + vec3(vec2(cos(angle), sin(angle)) * life * .10,
                                  .015 + sin(life * 3.14159265) * (.055 + .085 * aSeed.w));
    float dist = length(center - uCamPos);
    float viewDepth = max(-(uView * vec4(center, 1.)).z, .2);
    // A crown a few millimetres wide, not a hovering dot.
    float size = .0055 + .0045 * aSeed.z;
    float rasterSize = max(size, .55 * viewDepth / uPixelScale);
    vec3 world = center + uCamRight * aQuad.x * rasterSize + uCamUp * aQuad.y * rasterSize * 1.25;
    gl_Position = uProj * uView * vec4(world, 1.);
    vQuad = aQuad;
    vAlpha = rainLive * sin(life * 3.14159265) * (1. - smoothstep(14., 34., dist))
           * smoothstep(1., 2., dist) * min(1., size / rasterSize) * .38;
    vec3 splashLamp = vec3(0.0);
    if (uNightFactor > .02) {
        vec3 ld, ls;
        streetLamps(world, vec3(0., 0., 1.), normalize(uCamPos - world), 8, ld, ls);
        splashLamp = ld * (.55 * uNightFactor);
    }
    vColor = mix(sceneOut(uAmbColor * 2.1 + uSunColor * .04 + splashLamp), displayToScene(uFogColor),
                 cityFog(dist, world.z, uFogNear, uFogFar));
}
"""))

SPLASH_FS = """
#version 330 core
in vec2 vQuad;
in float vAlpha;
in vec3 vColor;
out vec4 FragColor;

void main() {
    float alpha = (1. - smoothstep(.15, 1., length(vQuad))) * vAlpha;
    if (alpha < .002) discard;
    FragColor = vec4(vColor, alpha);
}
"""
