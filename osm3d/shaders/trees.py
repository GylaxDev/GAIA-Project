# --- shaders/trees.py ---
from .shadow import SHADOW_PCF
from .hdr import HDR_CONV, HDR_OUT
from .materials import MATERIAL_LIGHT
from .surface import SURFACE_LIB
from .vegetation import TREE_INSTANCE, TREE_ALPHA_TEST, VEGETATION_LIGHT

TREE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
layout(location=3) in float aPart;
layout(location=4) in vec3 iPos;
layout(location=5) in vec2 iScaleRot;
layout(location=6) in vec3 iTint;
layout(location=7) in vec2 aUV;
layout(location=8) in float aCardSeed;
uniform mat4 uView, uProj;
uniform vec3 uOffset;
uniform vec3 uCamPos;
uniform vec2 uScale;
uniform float uGrow;
out vec3 vWorld; out vec3 vNormal; out vec3 vColor; out float vPart;
out vec3 vShadowRelative;
out vec2 vUV; out vec2 vTexUV; out float vSeed; out float vCrown;
""" + MATERIAL_LIGHT + VEGETATION_LIGHT + TREE_INSTANCE + """
void main() {
    vec3 p;
    if (aPart > 1.5) {
        p = treeBillboardPosition(aPos, iPos, iScaleRot, uScale, uGrow, uOffset, uCamPos);
        vNormal = vec3(0.0, 0.0, 1.0);
        vTexUV = treeBillboardUV(aUV, iPos, iScaleRot, uCamPos, uScale, uOffset);
        vCrown = 1.0;
    } else {
        p = treeInstancePosition(aPos, iPos, iScaleRot, uScale, uGrow, uOffset);
        vNormal = treeInstanceNormal(aPos, aNormal, iPos, iScaleRot, uScale, uGrow);
        vTexUV = treeAtlasUV(aUV, iPos, iScaleRot, aCardSeed);
        // Crown depth in template space: leaves near the centre and base are shaded by the foliage
        // around them, the outer upper shell sees the sky.  Gives the crown a rounded, dense volume
        // instead of a flat scatter of equally bright cards.
        float rad = length(aPos.xy) / 3.4;
        float up = clamp((aPos.z - 3.6) / 6.4, 0.0, 1.0);
        vCrown = aPart > 0.5 ? mix(.42, 1.0, smoothstep(.10, 1.05, rad * .75 + up * .55)) : 1.0;
    }
    vWorld = p;
    vShadowRelative = p - uCamPos;
    // Every tree gets its own hue and lightness within the species: some yellower and
    // sunnier, some bluer and darker, so a row of trees stops reading as one stamped model.
    float ts = treeSeed(iPos, iScaleRot);
    float hue = ts - .5, lightness = fract(ts * 5.31) - .5;
    vec3 varied = vec3(1.0 + .07 * hue, 1.0 + .05 * lightness, 1.0 - .08 * hue) * (1.0 + .20 * lightness);
    if (aPart > 1.5) {
        // The impostor atlas already carries leaf/bark albedo, so only the
        // per-instance reflectance variation is applied here.
        vColor = foliageReflectance(iTint) * varied;
    } else if (aPart > 0.5) {
        vColor = aColor * foliageReflectance(iTint) * varied;
    } else vColor = aColor;
    vPart = aPart;
    vUV = aUV;
    vSeed = aCardSeed;
    gl_Position = uProj * uView * vec4(p, 1.0);
}
"""

TREE_FS = ("""
#version 330 core
in vec3 vWorld; in vec3 vNormal; in vec3 vColor; in float vPart;
in vec2 vUV; in vec2 vTexUV; in float vSeed; in float vCrown;
uniform sampler2D uLeafTex, uLeafNormalTex, uTreeImpostorTex;
uniform vec3 uCamPos, uSunDir, uFogColor;
uniform vec3 uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uNightFactor, uExposure;
out vec4 FragColor;
""" + SHADOW_PCF + HDR_CONV + HDR_OUT + SURFACE_LIB + TREE_ALPHA_TEST + """
const float PI = 3.14159265;
vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43; float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
vec3 barkLayer(vec2 uv, vec2 frequency, vec2 uvDx, vec2 uvDy) {
    // Filter in this layer's anisotropic lattice, not max(UV) times its
    // highest frequency. The latter erased resolved grooves on tapered limbs.
    float footprint = max(length(uvDx * frequency), length(uvDy * frequency));
    float resolved = 1.0 - smoothstep(.18, .58, footprint);
    if (resolved <= .001) return vec3(0.0);
    vec3 n = sf_noised(uv * frequency);
    return vec3(n.x - .5, n.yz * frequency) * resolved;
}
void main() {
    // Texture/tangent derivatives are captured before any alpha-cut divergence.
    vec2 texDx = dFdx(vTexUV), texDy = dFdy(vTexUV);
    vec2 uvDx = dFdx(vUV), uvDy = dFdy(vUV);
    vec3 worldDx = dFdx(vShadowRelative), worldDy = dFdy(vShadowRelative);
    vec4 leaf = vec4(1.0), detail = vec4(.5, .5, 1.0, .92);
    vec4 impostor = vec4(0.0);
    if (vPart > 1.5) {
        // The atlas is baked leaf/bark albedo, already carrying crown occlusion.
        impostor = textureGrad(uTreeImpostorTex, vTexUV, texDx, texDy);
        if (impostor.a < LEAF_ALPHA_CUTOFF) discard;
    } else if (vPart > 0.5) {
        leaf = textureGrad(uLeafTex, vTexUV, texDx, texDy);
        if (leaf.a < LEAF_ALPHA_CUTOFF) discard;
        detail = textureGrad(uLeafNormalTex, vTexUV, texDx, texDy);
    }
    vec3 N = normalize(vNormal);
    // The crown's outward volume normals must not flip at card intersections.
    if (vPart < .5 && !gl_FrontFacing) N = -N;
    vec3 fallback = normalize(cross(abs(N.z) < .9 ? vec3(0,0,1) : vec3(0,1,0), N));
    float det = uvDx.x * uvDy.y - uvDx.y * uvDy.x;
    vec3 tangent = abs(det) > 1e-10 ? (worldDx * uvDy.y - worldDy * uvDx.y) / det : fallback;
    tangent -= N * dot(N, tangent);
    vec3 T = dot(tangent, tangent) > 1e-10 ? normalize(tangent) : fallback;
    vec3 B = cross(N, T);
    if (abs(det) > 1e-10 && dot(B, (worldDy * uvDx.x - worldDx * uvDy.x) / det) < 0.0) B = -B;
    vec3 L = normalize(uSunDir);
    vec3 V = normalize(uCamPos - vWorld);
    float dist = length(uCamPos - vWorld);
    vec3 albedo = vColor;
    vec3 Nd = N;
    float cavity = 1.0;
    if (vPart > 1.5) {
        // Baked albedo is multiplied by the instance reflectance on foliage only;
        // the brown trunk keeps its own colour.  The leaf atlas is nearly neutral,
        // so separate the bright crown from the dark bark by baked luminance.
        float leafness = smoothstep(.20, .38,
                                    dot(impostor.rgb, vec3(.2126, .7152, .0722)));
        albedo = impostor.rgb * mix(vec3(1.0), vColor, leafness);
        vec2 toCam = uCamPos.xy - vWorld.xy;
        float len = length(toCam);
        toCam = len > 1e-4 ? toCam / len : vec2(0.0, 1.0);
        // A crown is lit as a rounded volume, not as a flat card facing the eye.
        N = normalize(vec3(toCam * .35, .93));
        Nd = N;
    } else if (vPart > 0.5) {
        albedo = vColor * leaf.rgb;
        vec2 tilt = detail.rg * 2.0 - 1.0;
        Nd = normalize(T * tilt.x + B * tilt.y + N * sqrt(max(.05, 1.0 - dot(tilt, tilt))));
        cavity = detail.b;
    } else {
        // Metre-scale branch UVs keep bark fissures running ALONG every limb.
        vec2 dx = uvDx / clamp(uSurfDetail, .05, 2.0);
        vec2 dy = uvDy / clamp(uSurfDetail, .05, 2.0);
        vec2 bp = vUV + vec2(vSeed * 13.7, vSeed * 7.3);
        vec3 grooves = barkLayer(bp, vec2(18.0, 2.6), dx, dy);
        vec3 fine = barkLayer(bp + vec2(5.7, .2), vec2(43.0, 8.0), dx, dy);
        vec3 grain = barkLayer(bp + vec2(9.3, 1.7), vec2(58.0), dx, dy);
        float fissure = 1.0 - smoothstep(-.23, -.03, grooves.x);
        albedo *= mix(.57, 1.27, smoothstep(-.23, .22, grooves.x)) * (1. + fine.x * .42 + grain.x * .35);
        vec2 grad = grooves.yz * .018 + fine.yz * .0045 + grain.yz * .0015;
        Nd = normalize(N - T * grad.x - B * grad.y);
        cavity = 1.0 - .16 * fissure;
        float fw = max(length(dx), length(dy));
        float lichen = smoothstep(.67, .83, sf_noised(bp * 6.0).x) * (1.0 - smoothstep(.04, .10, fw));
        albedo = mix(albedo, vec3(.165, .166, .143), lichen * .18);
    }
    float sf = shadowPCF(vWorld, N, L, worldDx, worldDy);
    float NoL = max(dot(Nd, L), 0.0);
    float diff = vPart > .5 ? max((dot(Nd, L) + .16) / 1.16, 0.0) : NoL;
    float crownLight = vPart > .5 ? mix(vCrown, 1.0, .35 * sf) : 1.0;   // sunlit leaves are less occluded
    vec3 direct = albedo * uSunColor * diff * sf / PI * crownLight;
    if (vPart > 0.5) {
        // Restrained thin-leaf transmission; no yellow/neon-green fill boost.
        float back = pow(max(dot(V, -L), 0.0), 3.0);
        direct += albedo * vec3(1.04, 1.0, .88) * uSunColor * sf / PI
                * back * (.08 + .20 * (1.0 - NoL));
        vec3 H = normalize(L + V + vec3(1e-7));
        float NoV = max(dot(Nd, V), .02), NoH = max(dot(Nd, H), 0.0);
        float rough = max(detail.a, .62), a = rough * rough;
        float denominator = NoH * NoH * (a * a - 1.0) + 1.0;
        float D = a * a / (PI * denominator * denominator);
        float k = (rough + 1.0) * (rough + 1.0) / 8.0;
        float G = NoV / (NoV * (1.0 - k) + k) * NoL / (NoL * (1.0 - k) + k + 1e-4);
        float F = .028 + .972 * pow(1.0 - max(dot(V, H), 0.0), 5.0);
        direct += uSunColor * sf * D * G * min(F, .25) / (4.0 * NoV) * .16;
    }
    float skyVisibility = vPart > .5 ? .48 + .27 * max(N.z, 0.0) : .52;
    vec3 ambient = uAmbColor * albedo * skyVisibility * cavity * (vPart > .5 ? vCrown : 1.0);
    vec3 lit = direct + ambient;
    lit = max(lit, albedo * uAmbColor * 0.04);

    vec3 mapped = sceneOut(lit);

    float f = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
    FragColor = vec4(mix(mapped, displayToScene(uFogColor), f), 1.0);
}
""")

SHADOW_TREE_FS = """
#version 330 core
in vec2 vTexUV; in float vPart;
uniform sampler2D uLeafTex;
""" + TREE_ALPHA_TEST + """
void main() {
    vec2 texDx = dFdx(vTexUV), texDy = dFdy(vTexUV);
    if (vPart > 0.5 && textureGrad(uLeafTex, vTexUV, texDx, texDy).a < LEAF_ALPHA_CUTOFF) discard;
}
"""


# Startup bake for the tree impostor atlas.  The production tree template is
# rasterised from four horizontal directions into a 2x2 RGBA8 atlas; the scene
# shader then expands one camera-facing quad and samples the view that faces the
# eye.  Lighting stays dynamic, so only albedo, alpha and crown occlusion are
# baked here.
TREE_IMPOSTOR_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=2) in vec3 aColor;
layout(location=3) in float aPart;
layout(location=7) in vec2 aUV;
layout(location=8) in float aCardSeed;
uniform mat4 uBakeMatrix;
out vec3 vColor; out float vPart; out vec2 vUV; out float vCard; out float vCrown;
void main() {
    gl_Position = uBakeMatrix * vec4(aPos, 1.0);
    vColor = aColor;
    vPart = aPart;
    vUV = aUV;
    vCard = aCardSeed;
    float rad = length(aPos.xy) / 3.4;
    float up = clamp((aPos.z - 3.6) / 6.4, 0.0, 1.0);
    vCrown = aPart > 0.5 ? mix(.42, 1.0, smoothstep(.10, 1.05, rad * .75 + up * .55)) : 1.0;
}
"""

TREE_IMPOSTOR_FS = """
#version 330 core
in vec3 vColor; in float vPart; in vec2 vUV; in float vCard; in float vCrown;
uniform sampler2D uLeafTex;
out vec4 FragColor;
""" + TREE_ALPHA_TEST + """
void main() {
    vec3 albedo = vColor;
    if (vPart > 0.5) {
        float variant = floor(fract(0.37 + vCard) * 4.0);
        vec2 uv = (vUV + vec2(mod(variant, 2.0), floor(variant * .5))) * .5;
        vec4 leaf = textureLod(uLeafTex, uv, 0.0);
        if (leaf.a < LEAF_ALPHA_CUTOFF) discard;
        // Crown occlusion is baked into the albedo; dynamic sun/shadow lighting
        // still runs per frame in TREE_FS.
        albedo = vColor * leaf.rgb * mix(.70, 1.0, vCrown);
    }
    FragColor = vec4(albedo, 1.0);
}
"""
