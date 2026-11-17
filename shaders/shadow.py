from .vegetation import TREE_INSTANCE

SHADOW_PCF = """
in vec3 vShadowRelative;
uniform mat4 uLightSpaceMatrix;
uniform sampler2DShadow uShadowMap;
uniform float uShadowTexel;
uniform float uShadowTexelWorld;
uniform float uShadowBias;
uniform float uShadowStrength;
uniform mat4 uNearLightSpaceMatrix;
uniform sampler2DShadow uNearShadowMap;
uniform float uNearShadowTexel;
uniform float uNearShadowBias;
uniform float uNearShadowStrength;
uniform mat4 uVehicleLightSpaceMatrix;
uniform sampler2DShadow uVehicleShadowMap;
uniform float uVehicleShadowTexel;
uniform float uVehicleShadowTexelWorld;
uniform float uVehicleShadowBias;
uniform float uVehicleShadowStrength;
vec2 receiverPlaneSlope(mat4 matrix, vec3 worldDx, vec3 worldDy) {
    vec3 dx = (matrix * vec4(worldDx, 0.0)).xyz * .5;
    vec3 dy = (matrix * vec4(worldDy, 0.0)).xyz * .5;
    float det = dx.x * dy.y - dx.y * dy.x;
    if (abs(det) < 1e-12) return vec2(0.0);
    return vec2(dy.y * dx.z - dx.y * dy.z,
                dx.x * dy.z - dy.x * dx.z) / det;
}
float cityShadowPCF(vec3 world, vec3 N, vec3 L, vec3 worldDx, vec3 worldDy) {
    if (uShadowStrength <= 0.0) return 1.0;
    vec4 lp = uLightSpaceMatrix * vec4(world, 1.0);
    vec3 p = lp.xyz / lp.w * 0.5 + 0.5;
    vec2 e = min(p.xy, 1.0 - p.xy);
    float inside = smoothstep(0.0, uShadowTexel * 8.0, min(e.x, e.y));
    inside *= step(0.0, p.z) * step(p.z, 1.0);
    if (inside <= 0.0) return 1.0;
    vec2 slope = receiverPlaneSlope(uLightSpaceMatrix, worldDx, worldDy);
    // Bound the residual hardware 2x2 footprint, not the whole PCF kernel.
    // Integer tap spacing keeps this fractional footprint identical per tap.
    vec2 f = fract(p.xy / uShadowTexel - .5);
    float bias = uShadowBias * max(.2, abs(dot(N, L)))
               + dot(abs(slope), max(f, 1.0 - f)) * uShadowTexel;
    // Distant map only: preserve inexpensive soft city silhouettes.
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            vec2 offset = vec2(x, y) * uShadowTexel;
            sum += texture(uShadowMap, vec3(p.xy + offset, p.z + dot(slope, offset) - bias));
        }
    return mix(1.0, sum * (1.0 / 9.0), uShadowStrength * inside);
}
float nearShadowCompare(vec2 uv, vec3 p, vec2 slope, float bias) {
    return texture(uNearShadowMap, vec3(uv, p.z + dot(slope, uv - p.xy) - bias));
}
float nearShadowPCF(vec3 p, vec3 N, vec3 L, vec3 worldDx, vec3 worldDy) {
    vec2 slope = receiverPlaneSlope(uNearLightSpaceMatrix, worldDx, worldDy);
    // A small world-unit guard covers depth quantisation and projection rounding.
    float bias = uNearShadowBias;
    // Four comparisons at actual texel centres, each against the receiver
    // plane. This is bilinear PCF without a multi-metre normal/slope offset
    // or a broad blur that washes small rooftop units off their supporting roof.
    vec2 t = p.xy / uNearShadowTexel - .5;
    vec2 base = (floor(t) + .5) * uNearShadowTexel;
    vec2 f = fract(t);
    float a = nearShadowCompare(base, p, slope, bias);
    float b = nearShadowCompare(base + vec2(uNearShadowTexel, 0.0), p, slope, bias);
    float c = nearShadowCompare(base + vec2(0.0, uNearShadowTexel), p, slope, bias);
    float d = nearShadowCompare(base + vec2(uNearShadowTexel), p, slope, bias);
    float visibility = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    return mix(1.0, visibility, uNearShadowStrength);
}
float vehicleShadowTap(vec2 uv, float ref) {
    return textureLod(uVehicleShadowMap, vec3(uv, ref), 0.0);
}
float vehicleShadowPCF(vec3 world, vec3 N, vec3 L) {
    if (uVehicleShadowStrength <= 0.0) return 1.0;
    // Retain the dedicated car map's bias/filtering for tyre/road contact.
    vec4 lp = uVehicleLightSpaceMatrix * vec4(world + N * (uVehicleShadowTexelWorld * .4), 1.0);
    vec3 p = lp.xyz / lp.w * .5 + .5;
    vec2 e = min(p.xy, 1.0 - p.xy);
    float inside = smoothstep(0.0, uVehicleShadowTexel * 8.0, min(e.x, e.y));
    inside *= step(0.0, p.z) * step(p.z, 1.0);
    if (inside <= 0.0) return 1.0;
    float ref = p.z - uVehicleShadowBias * max(.4, 1.0 - dot(N, L));
    // Kernel radius in texels. Wide enough to dissolve the map's stair-stepped silhouettes,
    // narrow enough to keep car edges crisp.
    float radius = uVehicleShadowTexel * 1.9;
    // Cheap rejection: centre + four diagonals at 85 % radius (each a hardware-bilinear tap).
    // Pixels that are fully lit or fully shadowed stop here, which is most of the screen.
    float sum = vehicleShadowTap(p.xy, ref);
    for (int i = 0; i < 4; ++i) {
        vec2 o = vec2((i & 1) == 0 ? -.6 : .6, (i & 2) == 0 ? -.6 : .6) * radius;
        sum += vehicleShadowTap(p.xy + o, ref);
    }
    float vis = sum * .2;
    if (sum > .001 && sum < 4.999) {
        // Penumbra / silhouette: 12 Vogel-disk taps, rotated per pixel (interleaved gradient
        // noise) so the residual banding becomes fine, stable dither rather than stair-steps.
        float rot = fract(52.9829189 * fract(dot(floor(gl_FragCoord.xy), vec2(.06711056, .00583715)))) * 6.2831853;
        float acc = 0.0;
        for (int i = 0; i < 12; ++i) {
            float a = rot + float(i) * 2.39996323;
            float r = sqrt((float(i) + .5) / 12.0);
            acc += vehicleShadowTap(p.xy + vec2(cos(a), sin(a)) * (r * radius), ref);
        }
        vis = acc * (1.0 / 12.0);
    }
    return mix(1.0, vis, uVehicleShadowStrength * inside);
}
float shadowPCF(vec3 world, vec3 N, vec3 L, vec3 worldDx, vec3 worldDy) {
    float city = 1.0;
    if (max(uShadowStrength, uNearShadowStrength) > 0.0) {
        vec3 nearP = (uNearLightSpaceMatrix * vec4(world, 1.0)).xyz * .5 + .5;
        vec2 edge = min(nearP.xy, 1.0 - nearP.xy);
        float nearWeight = smoothstep(uNearShadowTexel * 2.0, max(.04, uNearShadowTexel * 16.0),
                                      min(edge.x, edge.y));
        nearWeight *= step(0.0, nearP.z) * step(nearP.z, 1.0) * step(.0001, uNearShadowStrength);
        // Close depth REPLACES coarse depth; min() here would retain the blurry
        // city shadow around every crisp contact. Blend only at the map border.
        if (nearWeight < .9999)
            city = cityShadowPCF(world, N, L, worldDx, worldDy);
        if (nearWeight > 0.0)
            city = mix(city, nearShadowPCF(nearP, N, L, worldDx, worldDy), nearWeight);
    }
    // Already fully shadowed city pixels cannot be made darker by cars. Take
    // union, not a product that double-darkens overlapping penumbras.
    if (city <= .0001) return city;
    return min(city, vehicleShadowPCF(world, N, L));
}
float shadowPCF(vec3 world, vec3 N, vec3 L) {
    // Subtract the camera BEFORE interpolation to preserve grazing/high-roof
    // precision. Cutout materials pass derivatives captured before discard.
    return shadowPCF(world, N, L, dFdx(vShadowRelative), dFdy(vShadowRelative));
}
"""

SHADOW_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uLightSpaceMatrix;
uniform vec3 uOffset;
uniform vec2 uScale;
uniform float uGrow;
void main() {
    vec3 p = vec3(aPos.xy * uScale, aPos.z * uGrow) + uOffset;
    gl_Position = uLightSpaceMatrix * vec4(p, 1.0);
}
"""

SHADOW_FS = """
#version 330 core
void main() {}
"""

SHADOW_TREE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=3) in float aPart;
layout(location=4) in vec3 iPos;
layout(location=5) in vec2 iScaleRot;
layout(location=7) in vec2 aUV;
layout(location=8) in float aCardSeed;
uniform mat4 uLightSpaceMatrix;
uniform vec3 uOffset;
uniform vec2 uScale;
uniform float uGrow;
out vec2 vTexUV; out float vPart;
""" + TREE_INSTANCE + """
void main() {
    vec3 p = treeInstancePosition(aPos, iPos, iScaleRot, uScale, uGrow, uOffset);
    vTexUV = treeAtlasUV(aUV, iPos, iScaleRot, aCardSeed);
    vPart = aPart;
    gl_Position = uLightSpaceMatrix * vec4(p, 1.0);
}
"""
