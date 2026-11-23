from .shadow import SHADOW_PCF
from .streetspill import STREET_SPILL
from .neon import NEON_SPILL
from .env import ENV_SKY
from .hdr import HDR_CONV, HDR_OUT
from .surface import SURFACE_LIB
from .materials import MATERIAL_LIGHT
from .rain_surface import RAIN_EXPOSURE, RAIN_RIPPLES
from .water import WATER_VS, WATER_FS

GROUND_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uView, uProj;
uniform vec3 uCamPos;
uniform vec2 uCenter;
uniform float uSize;
out vec3 vWorld;
out vec3 vShadowRelative;
void main() {
    vec3 p = vec3(aPos.xy * uSize + uCenter, aPos.z);
    vWorld = p;
    vShadowRelative = p - uCamPos;
    gl_Position = uProj * uView * vec4(p, 1.0);
}
"""

GROUND_FS = ("""
#version 330 core
in vec3 vWorld;
uniform vec3 uCamPos, uFogColor;
uniform vec3 uSunDir, uSunColor, uAmbColor;
uniform vec2 uCenter;
uniform float uFogNear, uFogFar, uRadius, uNightFactor, uExposure;
out vec4 FragColor;
""" + SHADOW_PCF + HDR_CONV + HDR_OUT + SURFACE_LIB + """
vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43; float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
void main() {
    vec3 L = normalize(uSunDir);
    float sf = shadowPCF(vWorld, vec3(0.0, 0.0, 1.0), L);
    float sunUp = clamp(L.z, 0.0, 1.0);
    mat2 dW = mat2(dFdx(vShadowRelative.xy), dFdy(vShadowRelative.xy));
    float fw = max(length(dW[0]), length(dW[1])) / clamp(uSurfDetail, .05, 2.);
    vec3 base_col; float ao; vec2 grad;
    sf_grass(vWorld.xy, dW, fw, base_col, ao, grad);
    vec3 lit = base_col * (uAmbColor * .88 * ao + uSunColor * turfDiffuse(vec3(0., 0., 1.), L) * sf * .318);
    lit = max(lit, base_col * uAmbColor * 0.04);
    
    vec3 mapped = sceneOut(lit);
    
    float f = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
    FragColor = vec4(mix(mapped, displayToScene(uFogColor), f), 1.0);
}
""")

TERRAIN_FS = ("""
#version 330 core
in vec3 vWorld; in vec3 vNormal; in vec3 vColor; in vec2 vUV; in vec3 vStyle;
in vec3 vSpill;
uniform vec3 uCamPos, uSunDir, uFogColor;
uniform vec3 uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uNightFactor, uTime, uExposure;
uniform float uWetness, uRain, uSnow, uWeatherTime;
out vec4 FragColor;
""" + SHADOW_PCF + STREET_SPILL + NEON_SPILL + HDR_CONV + HDR_OUT + SURFACE_LIB + ENV_SKY + RAIN_EXPOSURE + RAIN_RIPPLES + """
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    float a = hash(i); float b = hash(i + vec2(1,0));
    float c = hash(i + vec2(0,1)); float d = hash(i + vec2(1,1));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0; float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * noise(p); p *= 2.0; a *= 0.5;
    }
    return v;
}
vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43; float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
// Box-filtered stripe coverage, including subpixel lines (no distant brightening).
float groundStripe(float distance, float halfWidth, float footprint) {
    float lo = max(-halfWidth, distance - footprint);
    float hi = min(halfWidth, distance + footprint);
    return clamp((hi - lo) / (2. * footprint), 0., 1.);
}
void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    float sf = shadowPCF(vWorld, N, L);
    float sunUp = clamp(L.z, 0.0, 1.0);
    float dist = length(uCamPos - vWorld);

    // Capture derivatives before any material divergence. TextureGrad selects
    // hardware-filtered mips; metric footprint also filters joints and lane paint.
    vec2 wp = vWorld.xy;
    float det = clamp(uSurfDetail, 0.05, 2.0);
    mat2 dW = mat2(dFdx(vShadowRelative.xy), dFdy(vShadowRelative.xy));
    mat2 dUV = mat2(dFdx(vUV), dFdy(vUV));
    float fw = max(length(dW[0]), length(dW[1])) / det;

    bool is_road = (vColor.r < 0.05 && vColor.g > 0.1);
    bool is_footpath = (vColor.r >= 0.05 && vColor.r < 0.5);
    vec3 albedo = vColor;
    float glossK = 0.03;
    float paving = 0.0;
    float rough = 0.85;        // microfacet roughness for the sun highlight
    float ao = 1.0;            // baked thatch / pore occlusion
    float isGrass = 0.0;
    vec2 grad = vec2(0.0);     // surface height gradient (the normal map)

    if (is_road) {
        float road_width = max(vColor.g, 1.0);

        // The mesh carries exclusive forward/backward allocations in vStyle.yz.
        // A shared single lane carries zero allocations and has no divider.
        float forwardLanes = max(0., floor(vStyle.y + .5));
        float backwardLanes = max(0., floor(vStyle.z + .5));
        float laneCount = forwardLanes + backwardLanes;
        float divider = forwardLanes / max(laneCount, 1.);
        float paintClass = vColor.b;                 // 0 unmarked, .5 street, 1 highway
        float line_u = 0.06 / road_width;            // 12 cm lines
        float gap_u = 0.14 / road_width;             // double-yellow spacing
        float edge_u = 0.06 / road_width;

        float acrossAA = max(1e-5, (abs(dUV[0].x) + abs(dUV[1].x)) * .5);
        // Two-way streets: solid double yellow.
        float isCenter = 0.;
        if (forwardLanes > 0. && backwardLanes > 0.)
            isCenter = max(groundStripe(abs(vUV.x - divider + gap_u), line_u, acrossAA),
                           groundStripe(abs(vUV.x - divider - gap_u), line_u, acrossAA));
        float isLane = 0.;
        for (int i = 1; i < 16; ++i) {
            if (float(i) >= laneCount) break;
            if (backwardLanes > 0. && forwardLanes > 0. && abs(float(i) - forwardLanes) < .5) continue;
            isLane += groundStripe(abs(vUV.x - float(i) / laneCount), line_u, acrossAA);
        }
        // City streets have no edge lines; only highways do.
        float isEdge = groundStripe(min(vUV.x, 1. - vUV.x) - 0.25 / road_width, edge_u, acrossAA)
                     * smoothstep(.75, .95, paintClass);

        // 3 m dash, 9 m gap (12 m period). vUV.y is arc-length in meters.
        float alongAA = max(.001, (abs(dUV[0].y) + abs(dUV[1].y)) * .5);
        float dashDistance = abs(fract(vUV.y / 12. + .125) - .5) * 12.;
        float isDash = groundStripe(dashDistance, 1.5, alongAA);
        isDash = mix(isDash, .25, smoothstep(1.5, 3., alongAA));

        // Keep junction boxes clear (stop line + zebra live there).
        float intersection_mask = smoothstep(7.0, 11.0, min(vUV.y, vStyle.x));
        float marked = step(.25, paintClass);

        float wear, grit;
        sf_asphalt(wp, dW, road_width, vUV, vStyle.x, dist, fw, albedo, rough, grad, wear, grit);

        // thermoplastic paint: slightly off-white / ochre, chipped where the grit pokes through
        vec3 yellow_line = vec3(0.60, 0.44, 0.05);
        vec3 white_line = vec3(0.62, 0.62, 0.60);
        float chip = clamp(0.88 + grit * 2.2, 0.45, 1.0) * (1.0 - 0.25 * wear);
        float paintY = isCenter * intersection_mask * chip * marked;
        float paintW = clamp(isEdge + isLane * isDash, 0.0, 1.0) * intersection_mask * chip * marked;
        // Crossing decals use this SAME asphalt/lighting path, not a dark prop
        // rectangle. Their gaps preserve road wear, grain, normals and wetness;
        // the negative paint class also keeps lane lines out of the band.
        if (paintClass < -.5) {
            float coverage = 1.;                        // -2: solid stop line
            if (paintClass > -1.5) {                    // -1: zebra band
                float pitch = max(vStyle.y, .2);
                float stripe = clamp(vStyle.z, 0., pitch);
                float position = vUV.x * road_width - .06;
                float distance = abs(fract(position / pitch) - .5) * pitch;
                float footprint = max(acrossAA * road_width, .001);
                coverage = groundStripe(distance, stripe * .5, footprint);
                // Unresolved stripes converge to their true paint fraction,
                // never a solid white slab or a flickering black/white pattern.
                coverage = mix(coverage, stripe / pitch, smoothstep(pitch * .5, pitch, footprint));
            }
            paintW = coverage * chip;
        }
        albedo = mix(albedo, yellow_line, paintY);
        albedo = mix(albedo, white_line, paintW);
        rough = mix(rough, 0.55, max(paintY, paintW));

        glossK = 0.004;    // rough aggregate has no resolved mirror image
        paving = 1.0;
    } else if (is_footpath) {
        // Poured concrete slabs with expansion joints
        float fpW = max(vColor.g, 1.0);
        vec2 sp = vec2(vUV.x * fpW, vUV.y);
        mat2 dS = mat2(dUV[0] * vec2(fpW, 1.), dUV[1] * vec2(fpW, 1.));
        float detW = dW[0][0] * dW[1][1] - dW[1][0] * dW[0][1];
        mat2 J = abs(detW) > 1e-9 ? dS * inverse(dW) : mat2(1.0);
        sf_concrete(sp, J, wp, dW, vec2(min(fpW, 1.4), 1.75), dist, fw, albedo, rough, grad);

        // Restrained edge grime, not a bright white outline around every slab.
        float edgeM = min(vUV.x, 1. - vUV.x) * fpW;
        albedo *= 1.-.15*(1.-smoothstep(.035, .32+fw, edgeM));

        glossK = 0.006;
        paving = 1.0;
    } else {
        float mat = vColor.r;  // 1 grass, 2 asphalt, 3 concrete, 4 dirt, 5 urban, 6 seabed
        if (mat < 1.5) {
            sf_grass(wp, dW, fw, albedo, ao, grad);
            isGrass = 1.0;
            rough = 0.95;
            glossK = 0.03;
        } else if (mat < 2.5) {
            float wear, grit;
            sf_asphalt(wp, dW, 0.0, vec2(0.0), 1000.0, dist, fw, albedo, rough, grad, wear, grit);
            glossK = 0.004;
            paving = 1.0;
        } else if (mat < 3.5) {
            sf_concrete(wp, mat2(1.0), wp, dW, vec2(1.8, 2.1), dist, fw, albedo, rough, grad);
            glossK = 0.006;
            paving = 1.0;
        } else if (mat < 4.5) {
            sf_dirt(wp, dW, fw, albedo, grad);
            rough = 0.95;
            glossK = 0.03;
        } else if (mat > 6.5 && mat < 7.5) {
            // raised concrete footway (explicit 3D slab from urban.py)
            vec2 sp = vStyle.y > .5 ? vUV : wp;
            mat2 dS = vStyle.y > .5 ? dUV : dW;
            float detW = dW[0][0]*dW[1][1]-dW[1][0]*dW[0][1];
            mat2 J = abs(detW)>1e-9 ? dS*inverse(dW) : mat2(1.);
            sf_concrete(sp, J, wp, dW, vec2(1.075, 1.75), dist, fw, albedo, rough, grad);
            if (vStyle.y > .5) albedo *= 1.-.17*(1.-smoothstep(.04, .36+fw, sp.x));
            glossK = 0.006;
            paving = 1.0;
        } else if (mat >= 7.5 && mat < 7.8) {
            // Explicit curb faces must not fall through to the seabed palette.
            // vUV is already metric (arc, height), so its derivative is the footprint.
            float curbFW = max(length(dUV[0]), length(dUV[1]));
            sf_curb(vUV, dUV, curbFW, albedo, grad);
            rough = .94;
            glossK = .003;
            paving = 1.;
        } else if (mat < 5.5) {
            sf_urban(wp, dW, fw, albedo, grad);
            rough = .90;
            glossK = .004;
        } else {
            albedo = vec3(0.10, 0.12, 0.11);
        }
    }

    // Ground palettes contain bounded linear reflectances, not display RGB.
    vec2 albRange = SOIL_REFLECTANCE;
    if (is_road || (!is_footpath && vColor.r >= 1.5 && vColor.r < 2.5)) albRange = ASPHALT_REFLECTANCE;
    else if (is_footpath || paving > 0.5) albRange = CONCRETE_REFLECTANCE;
    else if (isGrass > 0.5) albRange = FOLIAGE_REFLECTANCE;
    // Road markings have their own higher reflectance, already applied above.
    if (!is_road) albedo = reflectanceRange(albedo, albRange);

    // Standing water: low-frequency puddle pockets over asphalt and concrete.
    // Wet surfaces lose diffuse energy and become far more specular.
    // The three puddle fbm() calls (12 noise taps) are skipped entirely in dry
    // weather, which is almost always.
    float wet = clamp(uWetness, 0.0, 1.0);
    if (wet > .01) wet *= rainExposure(vWorld);
    float puddle = 0.0;
    if (paving > 0.5 && wet > 0.01) {
        float pn = 0.58 * fbm(vWorld.xy * 0.035)
                 + 0.28 * fbm(vWorld.xy * 0.11 + vec2(7.3, 2.1))
                 + 0.14 * fbm(vWorld.xy * 0.31 + vec2(1.2, 5.0));
        // Drainage leaves isolated shallow pockets, not a continuous water sheet.
        puddle = smoothstep(.57, .70, pn) * smoothstep(.32, .92, wet) * smoothstep(.75, .98, N.z);
        albedo *= mix(1.0, 0.66, wet);
        glossK = mix(glossK, 0.035, wet);
        rough = mix(rough, max(.44, rough * .68), wet);
        rough = mix(rough, .15, puddle);

        // Expanding rain rings perturb the puddle normal without any CPU work.
        if (uRain > 0.01 && puddle > 0.005) {
            vec2 rings = rainRipples(vWorld.xy, uWeatherTime, fw) * puddle * uRain;
            N = normalize(N + vec3(-rings, 0.));
        }
    }

    // ---- normal map: tilt the (smooth) geometric normal by the material's height gradient.
    // `N` stays the smooth normal: it feeds the SSR material buffer, where pixel-scale bump
    // would only make reflections noisy.  Puddles are flat water, so they erase the bump.
    float bumpFade = (1.0 - smoothstep(30.0, 160.0, dist)) * min(det, 1.0) * (1.0 - puddle);
    vec3 axis = abs(N.x) < .9 ? vec3(1., 0., 0.) : vec3(0., 1., 0.);
    vec3 Tg = normalize(axis-N*dot(axis, N));
    vec3 Bg = cross(N, Tg);
    vec3 Nd = normalize(N - (Tg * grad.x + Bg * grad.y) * bumpFade);

    float NdotL = max(dot(Nd, L), 0.0);
    float diffL = mix(NdotL, turfDiffuse(Nd, L), isGrass);

    vec3 amb = uAmbColor * mix(0.55, 1.0, 1.0 - abs(N.z));
    vec3 ambientLit = albedo * amb * 0.65 * ao;
    if (isGrass > .5) ambientLit = albedo * uAmbColor * .88 * ao;
    vec3 lit = ambientLit + albedo * uSunColor * diffL * sf * 0.318;

    if (isGrass < .5) {
        // sun highlight (GGX, dielectric): sparkles on the aggregate, a broad glare at low sun
        float NoV = max(dot(Nd, V), 1e-3);
        vec3 H = normalize(L + V);
        float NoH = max(dot(Nd, H), 0.0);
        float VoH = max(dot(V, H), 0.0);
        float a = max(rough * rough, 0.02);
        float a2 = a * a;
        float dd = NoH * NoH * (a2 - 1.0) + 1.0;
        float D = a2 / (3.14159265 * dd * dd);
        float kk = (rough + 1.0) * (rough + 1.0) / 8.0;
        float Gv = (NoV / (NoV * (1.0 - kk) + kk)) * (NdotL / (NdotL * (1.0 - kk) + kk + 1e-4));
        float F = 0.04 + 0.96 * pow(1.0 - VoH, 5.0);
        float spec = min(D * Gv * F / (4.0 * NoV), 4.0);
        float specFade = 1.0 - smoothstep(200.0, 800.0, dist);
        lit += uSunColor * spec * sf * sunUp * specFade * (1.0 - 0.8 * puddle);

        // dry-surface sky reflection: roads turn silvery at grazing angles
        float fr = pow(1.0 - max(dot(N, V), 0.0), 5.0);
        lit += uAmbColor * (0.04 + 0.96 * fr) * (1.0 - rough) * 0.08 * (1.0 - puddle);
    }

    // Lamp contribution is scaled by uNightFactor below: skip the whole loop in
    // daylight.  The grid lookup narrows it to this fragment's own lamps, and
    // only the near field pays for per-pixel shading -- the rest of the city is
    // lit from the per-vertex sum, so there is no camera-distance cutoff.
    vec3 sDiff = vec3(0.0), sSpec = vec3(0.0);
    if (uNightFactor > 0.02) {
        float nearW = spillNearWeight(length(uCamPos - vWorld));
        sDiff = vSpill;
        if (nearW > 0.002) {
            vec3 pd = vec3(0.0), ps = vec3(0.0);
            streetLights(vWorld, Nd, V, pd, ps);
            sDiff = mix(sDiff, pd, nearW);
            sSpec = ps * nearW;
        }
    }
    lit += (albedo / 3.14159265 * sDiff + sSpec * glossK) * uNightFactor;

    // wet-asphalt reflection of billboards / neon signage
    vec3 neon = neonSpill(vWorld, Nd, V);
    lit += neon * (albedo * 0.55 + glossK * 0.35) * uNightFactor;

    // Water pockets reflect the same cloud deck as open water, while ordinary
    // wet aggregate keeps a rough sheen instead of turning the whole street into a mirror.
    float fresnel = .0204 + .9796 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    if (puddle > .005) {
        vec3 wetRefl = envSky(reflect(-V, N), L);
        lit = mix(lit, wetRefl, puddle * fresnel);
    }
    lit += uAmbColor * wet * paving * (1. - puddle) * fresnel * .10;
    lit += neon * puddle * 0.06 * uNightFactor;

    lit = max(lit, albedo * uAmbColor * 0.04);

    vec3 mapped = sceneOut(lit);

    float f = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
    FragColor = vec4(mix(mapped, displayToScene(uFogColor), f), 1.0);
}
""")

LINE_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
layout(location=5) in vec2 aUV;
uniform mat4 uView, uProj;
uniform vec3 uOffset;
uniform vec2 uScale;
uniform float uGrow;
out vec3 vWorld; out vec3 vColor;
void main() {
    vec3 p = vec3(aPos.xy * uScale, aPos.z * uGrow) + uOffset;
    vWorld = p; vColor = aColor;
    gl_Position = uProj * uView * vec4(p, 1.0);
}
"""

LINE_FS = ("""
#version 330 core
in vec3 vWorld; in vec3 vColor;
uniform vec3 uCamPos, uFogColor;
uniform vec3 uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uExposure;
out vec4 FragColor;
""" + HDR_CONV + HDR_OUT + MATERIAL_LIGHT + """
vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43; float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
void main() {
    vec3 mapped = sceneOut(vec3(0.025) * (uAmbColor * 0.55 + uSunColor * 0.1));
    float f = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
    FragColor = vec4(mix(mapped, displayToScene(uFogColor), f), 1.0);
}
""")
