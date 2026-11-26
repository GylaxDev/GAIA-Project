"""Procedural curved grass: shared colour/depth geometry and thin-blade lighting."""
from .shadow import SHADOW_PCF
from .hdr import HDR_CONV, HDR_OUT
from .materials import MATERIAL_LIGHT
from .vegetation import VEGETATION_LIGHT


GRASS_INSTANCE = """
uniform float uGrassTime, uSnow;
uniform vec2 uWind;
float grassSeed(float p) {
    vec3 h = fract(vec3(p) * vec3(.1031, .1030, .0973));
    h += dot(h, h.yzx + 33.33);
    return fract((h.x + h.y) * h.z);
}
void grassBlade(vec2 blade, float index, vec3 root, vec2 slope, vec3 data,
                out vec3 position, out vec3 normal, out vec2 info) {
    float seed = grassSeed(data.x * 93.71 + index * 7.137);
    float r1 = grassSeed(seed * 71.3 + 1.7);
    float r2 = grassSeed(seed * 53.1 + 8.2);
    float angle = seed * 6.2831853;
    float t = blade.y;
    vec2 side0 = vec2(cos(angle), sin(angle));
    vec2 lean = vec2(-side0.y, side0.x);
    float twist = (r2 - .5) * .65;
    vec2 side = vec2(cos(angle + twist * t), sin(angle + twist * t));
    float dry = smoothstep(.86, .98, r1);
    float height = mix(.115, .255, r2) * data.z * mix(1., .58, dry);
    height *= data.y * mix(1., .48, clamp(uSnow, 0., 1.));
    float width = mix(.0035, .0085, grassSeed(seed * 33.7 + 2.1));
    width *= sqrt(max(data.y, 0.)) * data.z;
    float taper = pow(max(0., 1. - t), .72) * mix(.65, 1., smoothstep(0., .22, t));
    vec2 w = uWind + vec2(.8, .45);
    vec2 windDir = dot(w, w) > 1e-8 ? normalize(w) : vec2(1., 0.);
    float windGain = min(.065, .016 + length(uWind) * .003);
    float wave = sin(uGrassTime * 1.8 + root.x * .19 + root.y * .13 + seed * 9.);
    // Overlapping bowed blades and a few short, flatter dry strands, rather
    // than eight parallel spikes. The triangle/instance budget is unchanged.
    vec2 bend = lean * height * mix(.35, 1.18, r1)
              + windDir * windGain * (.60 + .40 * wave) * data.y;
    vec2 offset = side0 * (mix(.018, .065, grassSeed(data.x * 31.7 + index * 4.3)));
    vec2 xy = offset + bend * t * t + side * (blade.x * width * taper);
    // All blade roots follow the supporting triangle, not an arbitrary flat XY plane.
    position = root + vec3(xy, dot(slope, xy) + height * (t - .18 * t * t) - .008);
    vec3 across = vec3(side, dot(slope, side));
    vec3 tangent = vec3(2. * bend * t, dot(slope, 2. * bend * t) + max(height, .0001) * (1. - .36 * t));
    normal = normalize(cross(across, tangent));
    // A subtle longitudinal fold, without doubling the ribbon's triangle count.
    normal = normalize(normal + normalize(across) * blade.x * .18);
    info = vec2(seed, dry);
}
"""


GRASS_SHADOW_PCF = """
uniform mat4 uGrassLightSpaceMatrix;
uniform sampler2DShadow uGrassShadowMap;
uniform float uGrassShadowTexel, uGrassShadowBias, uGrassShadowStrength;
float grassShadowTap(vec2 uv, vec3 p, vec2 slope) {
    return texture(uGrassShadowMap, vec3(uv, p.z + dot(slope, uv - p.xy) - uGrassShadowBias));
}
float grassShadowPCF(vec3 world, vec3 worldDx, vec3 worldDy, float receiverSpan) {
    if (uGrassShadowStrength <= 0.) return 1.;
    vec3 p = (uGrassLightSpaceMatrix * vec4(world, 1.)).xyz * .5 + .5;
    vec2 edge = min(p.xy, 1. - p.xy);
    float inside = smoothstep(.02, .10, min(edge.x, edge.y));
    inside *= step(0., p.z) * step(p.z, 1.);
    if (inside <= 0.) return 1.;
    vec2 slope = receiverPlaneSlope(uGrassLightSpaceMatrix, worldDx, worldDy);
    if (receiverSpan > 0.) {
        // A finite ribbon cannot be extrapolated arbitrarily far past its
        // edge-on light projection. Limit that extrapolation, not the bias.
        float depthScale = .5 * length(vec3(uGrassLightSpaceMatrix[0][2],
                uGrassLightSpaceMatrix[1][2], uGrassLightSpaceMatrix[2][2]));
        slope *= min(1., receiverSpan * depthScale / max(dot(abs(slope), vec2(uGrassShadowTexel)), 1e-8));
    }
    vec2 texel = p.xy / uGrassShadowTexel - .5;
    vec2 base = (floor(texel) + .5) * uGrassShadowTexel;
    vec2 f = fract(texel);
    float a = grassShadowTap(base, p, slope);
    float b = grassShadowTap(base + vec2(uGrassShadowTexel, 0.), p, slope);
    float c = grassShadowTap(base + vec2(0., uGrassShadowTexel), p, slope);
    float d = grassShadowTap(base + vec2(uGrassShadowTexel), p, slope);
    return mix(1., mix(mix(a, b, f.x), mix(c, d, f.x), f.y), inside * uGrassShadowStrength);
}
float grassShadowPCF(vec3 world, vec3 worldDx, vec3 worldDy) {
    return grassShadowPCF(world, worldDx, worldDy, 0.);
}
"""


_GRASS_ATTRIBUTES = """
layout(location=0) in vec2 aBlade;
layout(location=1) in float aBladeIndex;
layout(location=2) in vec3 iRoot;
layout(location=3) in vec2 iSlope;
layout(location=4) in vec3 iData;
"""

GRASS_VS = """
#version 330 core
""" + _GRASS_ATTRIBUTES + """
uniform mat4 uView, uProj;
uniform vec3 uCamPos;
out vec3 vWorld, vNormal, vShadowRelative;
out vec2 vBlade;
flat out vec2 vInfo;
flat out vec2 vSlope;
""" + GRASS_INSTANCE + """
void main() {
    grassBlade(aBlade, aBladeIndex, iRoot, iSlope, iData, vWorld, vNormal, vInfo);
    vBlade = aBlade;
    vSlope = iSlope;
    vShadowRelative = vWorld - uCamPos;
    gl_Position = uProj * uView * vec4(vWorld, 1.);
}
"""

GRASS_SHADOW_VS = """
#version 330 core
""" + _GRASS_ATTRIBUTES + """
uniform mat4 uLightSpaceMatrix;
""" + GRASS_INSTANCE + """
void main() {
    vec3 p, normal;
    vec2 info;
    grassBlade(aBlade, aBladeIndex, iRoot, iSlope, iData, p, normal, info);
    gl_Position = uLightSpaceMatrix * vec4(p, 1.);
}
"""

GRASS_FS = """
#version 330 core
in vec3 vWorld, vNormal;
in vec2 vBlade;
flat in vec2 vInfo;
flat in vec2 vSlope;
uniform vec3 uCamPos, uSunDir, uSunColor, uAmbColor, uFogColor;
uniform float uFogNear, uFogFar, uExposure, uSunIntensity;
uniform float uGrassSSS, uWetness, uSnow;
out vec4 FragColor;
""" + SHADOW_PCF + GRASS_SHADOW_PCF + HDR_CONV + HDR_OUT + MATERIAL_LIGHT + VEGETATION_LIGHT + """
void main() {
    vec3 worldDx = dFdx(vShadowRelative), worldDy = dFdy(vShadowRelative);
    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;
    vec3 L = normalize(uSunDir);
    vec3 V = normalize(uCamPos - vWorld);
    float t = clamp(vBlade.y, 0., 1.);
    float seed = vInfo.x, dry = vInfo.y;
    vec3 albedo = mix(GRASS_DARK, GRASS_LIGHT, .25 + .70 * seed);
    albedo = mix(albedo, mix(GRASS_DRY_DARK, GRASS_DRY_LIGHT, seed), dry);
    // Dark thatch at the root, fine parallel fibres, sun-bleached tapered tips.
    albedo *= mix(.62, 1.38, smoothstep(.02, .85, t));
    float fibre = .95 + .05 * cos(vBlade.x * 12. + seed * 9.);
    albedo *= fibre * mix(1., .76, clamp(uWetness, 0., 1.));
    albedo = mix(albedo, vec3(.78, .80, .81), clamp(uSnow, 0., 1.) * smoothstep(.15, .70, t));
    // The 12.5 cm city footprint is larger than a blade. Its receiver-plane
    // filter uses the support slope at the blade's actual position; extending
    // an edge-on blade plane by entire city texels caused isolated light leaks.
    float sf = shadowPCF(vWorld, N, L, vec3(1., 0., vSlope.x), vec3(0., 1., vSlope.y));
    // Union with micro shadows, not multiplication of two penumbras.
    if (sf > .0001) sf = min(sf, grassShadowPCF(vWorld, worldDx, worldDy, .04));
    float NoL = max(dot(N, L), 0.);
    float wrapped = max((dot(N, L) + .12) / 1.12, 0.);
    float rootAO = mix(.65, 1., smoothstep(.03, .78, t));
    vec3 ambientLit = albedo * uAmbColor * (.98 + .08 * abs(N.z)) * rootAO;
    vec3 direct = albedo * uSunColor * wrapped * sf * .3183099;
    // Thin-sheet subsurface approximation: broader transmission through the
    // blade plus a forward-scattering lobe when looking towards the sun.
    // It is direct sunlight, never an unshadowed/emissive green rim.
    float backSurface = max(dot(-N, L), 0.);
    vec3 scatterDir = normalize(L + N * .28);
    float forward = pow(max(dot(V, -scatterDir), 0.), 4.);
    float thickness = mix(.18, .90, smoothstep(.08, .90, t)) * (1. - .72 * dry);
    float transmission = (backSurface * .56 + forward * 1.05) * thickness;
    direct += albedo * vec3(1.03, 1., .88) * uSunColor * sf * .3183099
            * transmission * uGrassSSS * clamp(uSunIntensity, 0., 1.) * (1. - uSnow);
    vec3 H = normalize(V + L + vec3(1e-7));
    float glint = pow(max(dot(N, H), 0.), 28.) * NoL;
    direct += uSunColor * sf * glint * .006 * mix(1., 1.35, uWetness);
    vec3 lit = max(ambientLit + direct, albedo * uAmbColor * .025);
    float f = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
    FragColor = vec4(mix(sceneOut(lit), displayToScene(uFogColor), f), 1.);
}
"""
