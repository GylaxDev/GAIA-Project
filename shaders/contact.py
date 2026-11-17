"""GTA IV-style contact shadow under cars.

One instanced, oriented quad per visible car is laid on the road just before the
cars are drawn. It *multiplies* the scene colour (blend ZERO, SRC_COLOR), so it
darkens whatever lighting the road already has, including streetlight pools and
wet-road reflections, instead of replacing it.

Footprint = rounded box (the car's length x width) with a soft falloff, plus an
elliptical dark patch at every tyre contact. The darkness is *clipped by the
dynamic shadow map*: where the sun still reaches the ground (outside the car's
cast shadow) the contact shadow fades out, so it reads as the darkest core of the
cast shadow instead of a second, unrelated blob. With shadows off or at night
(shadow strength 0) there is nothing to clip against and the blob stands alone.
"""
from .shadow import SHADOW_PCF

CONTACT_VS = """#version 330 core
layout(location = 0) in vec2 aCorner;     // unit quad, -1..1
layout(location = 1) in vec3 aOrigin;     // car origin on the road
layout(location = 2) in vec3 aAxisX;      // car forward (unit)
layout(location = 3) in vec3 aAxisY;      // car left (unit)
layout(location = 4) in vec4 aSize;       // half length, half width, wheelbase, visibility
uniform mat4  uView, uProj;
uniform vec3  uCamPos;
uniform float uDepthQ;
uniform float uMargin;                    // soft falloff outside the body, metres
out vec3 vWorld;
out vec3 vShadowRelative;
out vec2 vLocal;
flat out vec4 vSize;
void main() {
    vec2 local = aCorner * (aSize.xy + uMargin);
    vec3 wp = aOrigin + aAxisX * local.x + aAxisY * local.y;
    // Float a few cm above the road: roads crown, slope and carry layered decals, and a
    // quad that is exactly coplanar with the car origin clips through wherever they rise.
    wp += normalize(cross(aAxisX, aAxisY)) * 0.04;
    vWorld = wp;
    vShadowRelative = wp - uCamPos;
    vLocal = local;
    vSize = aSize;
    vec4 vp = uView * vec4(wp, 1.0);
    // Same trick as the tyre marks: nudge toward the eye along the view ray so the
    // quad never z-fights the road, whatever the depth convention.
    float zd = max(-vp.z, 0.1);
    float pull = max(0.06, 8.0 * zd * zd * uDepthQ) / zd;
    vp.xyz *= (1.0 - min(pull, 0.5));
    gl_Position = uProj * vp;
}
"""

CONTACT_FS = ("""#version 330 core
in vec3 vWorld;
in vec2 vLocal;   // vShadowRelative comes from SHADOW_PCF
flat in vec4 vSize;
uniform vec3  uCamPos, uSunDir;
uniform float uMargin;
uniform float uContactStrength;           // 0..1 darkness under the car
uniform float uContactClip;               // 0 = ignore shadow map, 1 = fully clipped by it
uniform float uContactFadeNear, uContactFadeFar;
layout(location = 0) out vec4 FragColor;
""" + SHADOW_PCF + """
float sdRoundBox(vec2 p, vec2 b, float r) {
    vec2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}
void main() {
    float hl = vSize.x, hw = vSize.y, wb = vSize.z;
    // Derivatives must be taken outside any branch.
    vec3 sdx = dFdx(vShadowRelative), sdy = dFdy(vShadowRelative);
    // Body footprint, slightly wider than the car so it overlaps the tyre patches (no gap).
    float r = min(hw * 0.6, 0.8);
    float d = sdRoundBox(vLocal, vec2(hl * 1.0, hw * 1.07) - r, r);
    // Darkest in the enclosed middle, mostly dark at the silhouette, soft outside.
    // Analytic per pixel, so it stays crisp at any distance (no texture resolution).
    float occ = smoothstep(-uMargin, 0.15, -d);
    // Tyre contact patch, tighter and sharper than before, merged into the body shadow.
    float dist = length(uCamPos - vWorld);
    if (dist < 70.0) {
        float track = 0.874 * hw;
        vec2 q = abs(vec2(abs(vLocal.x) - 0.5 * wb, abs(vLocal.y) - track)) / vec2(0.62, 0.30);
        occ = max(occ, 0.97 * exp(-dot(q, q) * 1.8));
    }
    float fade = 1.0 - smoothstep(uContactFadeNear, uContactFadeFar, dist);
    float k0 = occ * vSize.w * uContactStrength * fade;
    if (k0 < 0.004) discard;

    // Under the body the car itself blocks the sun, so no lookup is needed there. The dynamic
    // shadow only clips the soft margin outside the body, where it matters, and the (relatively
    // costly, texel-limited) shadow-map read runs only for those pixels.
    float outside = smoothstep(-0.10, 0.25, d) * uContactClip;
    if (outside > 0.001 && max(uShadowStrength, max(uNearShadowStrength, uVehicleShadowStrength)) > 0.0001) {
        float vis = shadowPCF(vWorld, vec3(0.0, 0.0, 1.0), normalize(uSunDir), sdx, sdy);
        occ *= mix(1.0, 1.0 - smoothstep(0.10, 0.75, vis), outside);
    }
    float k = occ * vSize.w * uContactStrength * fade;
    // Dither the quantisation so the multiply stays smooth on 8-bit targets.
    k -= (fract(dot(floor(gl_FragCoord.xy), vec2(0.754877666, 0.569840296))) - 0.5) * (1.0 / 255.0);
    if (k < 0.002) discard;
    FragColor = vec4(vec3(1.0 - k), 1.0);
}
""")
