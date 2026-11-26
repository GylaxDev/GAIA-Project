from .shadow import SHADOW_PCF
from .streetspill import STREET_SPILL
from .neon import NEON_SPILL
from .env import ENV_SKY
from .hdr import HDR_CONV, HDR_OUT
from .street_furniture import STREET_FURNITURE
from .sun import SUN_LIGHT
from .interiors import INTERIORS
from .storefront import STOREFRONT

BUILDING_VS = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec3 aColor;
layout(location=3) in float aAO;
layout(location=4) in vec3 aStyle;
layout(location=5) in vec2 aUV;
layout(location=6) in float aEmissive;
layout(location=7) in float aPartKind;
uniform mat4 uView, uProj;
uniform vec3 uOffset;
uniform vec3 uCamPos;
uniform vec2 uScale;
uniform float uGrow;
uniform float uNightFactor;
// Depth layering.  A 24-bit depth buffer resolves z^2 / (near * 2^24) metres at distance z,
// which at 1 km is 10+ cm: thinner than the lift between terrain, road and paint.  Decal
// layers are nudged toward the eye along their own view ray (so they do not move on screen)
// by at least that quantum.
uniform float uDepthQ;        // 1 / (near * 2^24)
uniform float uGroundPass;    // 1 while drawing the terrain/road ribbon mesh
float layerRank() {
    if (uGroundPass > 0.5) {
        // terrain 0 < road 1 < footpath / kerb / marking decal 2.
        // Terrain staging preserves road width and the negative marking class,
        // not aPartKind, so decal rank must come from the road-material inputs.
        return aColor.y > 0.5 ? (aColor.x > 0.05 || aColor.z < -0.5 ? 2.0 : 1.0) : 0.0;
    }
    if (aPartKind > 10.5) return 2.0;                       // road paint
    if (abs(aPartKind - 10.0) < .1) return 1.0;             // storefront portal
    if (aPartKind > 6.5 && aPartKind < 8.5) return 1.0;     // street furniture / signs
    // Building shells carry their z-fight group in style.x (rank + jitter).  OSM
    // overlaps duplicate walls and roof caps a few millimetres apart; they tie in the
    // depth buffer and shimmer in a stipple, so _conflict_ranks (mesh_extruder.py)
    // colours the conflict graph and each rank here is nudged by >= 2 quanta - far
    // more than any such gap at any distance.  style.x < 1 means rank 0, no shift.
    return floor(aStyle.x + 0.001);
}
out vec3 vWorld;
out vec3 vShadowRelative;
out vec3 vNormal;
out vec3 vColor;
out float vAO;
out vec3 vStyle;
out float vLocalZ;
out vec2 vUV;
out float vEmissive;
out float vPartKind;
out vec3 vSpill;
""" + STREET_SPILL + """
void main() {
    vec3 p = vec3(aPos.xy * uScale, aPos.z * uGrow) + uOffset;
    vWorld  = p;
    vShadowRelative = p - uCamPos;
    vNormal = aNormal;
    vColor  = aColor;
    vAO     = aAO;
    vStyle  = aStyle;
    vLocalZ = aPos.z;
    vUV     = aUV;
    vEmissive = aEmissive;
    vPartKind = aPartKind;
    // Street-lamp spill, evaluated per vertex.  The fragment stage keeps this
    // for everything beyond 400 m and shades near surfaces exactly instead.
    vSpill = (uNightFactor > 0.02)
           ? streetLampDiffuse(p, aNormal) : vec3(0.0);
    vec4 pv = uView * vec4(p, 1.0);
    float rank = layerRank();
    if (rank > 0.0) {
        float zd = max(length(pv.xyz), 1.0);
        float shift = rank * max(0.012, 2.0 * zd * zd * uDepthQ);
        pv.xyz *= 1.0 - shift / zd;
    }
    gl_Position = uProj * pv;
}
"""

BUILDING_FS = ("""
#version 330 core
in vec3 vWorld;
in vec3 vNormal;
in vec3 vColor;
in float vAO;
in vec3 vStyle;
in float vLocalZ;
in vec2 vUV;
in float vEmissive;
in float vPartKind;
in vec3 vSpill;
uniform vec3 uCamPos, uSunDir, uFogColor;
uniform vec3 uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uNightFactor, uTime, uExposure, uSunIntensity;
uniform float uHours;
uniform float uWetness, uRain, uSnow, uWeatherTime;
out vec4 FragColor;
""" + SHADOW_PCF + STREET_SPILL + NEON_SPILL + ENV_SKY + HDR_CONV + HDR_OUT + SUN_LIGHT + """
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}
const float PI = 3.14159265;
vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43; float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
vec3 spec_ggx(vec3 N, vec3 V, vec3 L, vec3 F0, float rough) {
    vec3 H = normalize(L + V);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    float a  = rough * rough;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D  = a2 / (PI * d * d);
    float k  = (rough + 1.0) * (rough + 1.0) / 8.0;
    float G  = (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));
    vec3  F  = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    return (D * G) * F / (4.0 * NdotV * NdotL + 0.001);
}

// Restrained architectural sRGB: limestone, sandstone, weathered concrete and granite.
vec3 stuccoPalette(float h) {
    if (h < 0.24) return vec3(0.72, 0.70, 0.65);
    if (h < 0.44) return vec3(0.63, 0.60, 0.54);
    if (h < 0.62) return vec3(0.66, 0.65, 0.62);
    if (h < 0.78) return vec3(0.53, 0.54, 0.53);
    if (h < 0.92) return vec3(0.79, 0.78, 0.74);
    return vec3(0.46, 0.45, 0.43);
}

// Exact box-filter of a repeating interval. Window bays survive at skyline distances
// until their *pixel footprint* becomes subpixel, then converge to their actual area.
float bandIntegral(float x, float lo, float hi) {
    return floor(x) * (hi - lo) + clamp(fract(x) - lo, 0.0, hi - lo);
}
float band(float x, float lo, float hi, float footprint) {
    float w = max(footprint, 0.001);
    return clamp((bandIntegral(x + w * 0.5, lo, hi)
                - bandIntegral(x - w * 0.5, lo, hi)) / w, 0.0, 1.0);
}

// Roof cladding.  A warm, saturated roof colour is clay pan tile laid in courses
// down the pitch; a neutral one stays a weathered membrane / gravel deck.  Both
// keep the per-building roof colour as their mean so a tiled district still reads
// terracotta from the air instead of dissolving into uniform concrete.
vec3 roofSurface(vec3 wp, vec3 base, vec3 N, float seed) {
    float dist = length(uCamPos - wp);
    float near = 1.0 - smoothstep(70.0, 260.0, dist);
    // A pitched face is clad in tiles whatever its colour (grey tile is a real roof);
    // only a level cap falls back to the membrane / gravel deck.
    float warm = clamp((base.r - base.b) * 4.5, 0.0, 1.0);
    if (warm < 0.18 && N.z > 0.985) {
        vec3 mean = mix(vec3(0.38, 0.375, 0.36), base, 0.30);
        if (near <= 0.001) return mean;
        float n = 0.7 * vnoise(wp.xy * 0.7) + 0.3 * vnoise(wp.xy * 2.9 + seed * 17.0);
        vec3 col = mean * (0.78 + 0.44 * mix(0.5, n, near));
        // faint panel seams
        vec2 g = abs(fract(wp.xy / 3.2 + seed * 7.0) - 0.5);
        col *= 1.0 - 0.10 * near * smoothstep(0.46, 0.5, max(g.x, g.y));
        return mix(mean, col, near);
    }
    // Courses run down the pitch: one axis across the slope, one up it.
    vec3 mean = base * 0.82;
    if (near <= 0.001) return mean;
    vec2 dn = N.xy;
    float fl = length(dn);
    dn = fl > 1e-4 ? dn / fl : vec2(0.0, 1.0);
    vec2 tg = vec2(-dn.y, dn.x);
    float sinPitch = max(sqrt(max(1.0 - N.z * N.z, 0.0)), 0.22);
    float u = dot(wp.xy, tg) / 0.28;
    float v = dot(wp.xy, dn) / (0.30 / sinPitch);
    u += 0.5 * mod(floor(v), 2.0);          // staggered joints between courses
    vec2 f = fract(vec2(u, v));
    float pan    = 0.5 + 0.5 * cos((f.x - 0.5) * 6.2831853);  // rounded pan barrel
    float gap    = smoothstep(0.0, 0.10, min(f.x, 1.0 - f.x));
    float course = smoothstep(0.0, 0.12, f.y);                // shadow under each course
    float tile   = hash21(floor(vec2(u, v)) + floor(seed * 97.0));
    vec3 col = base * (0.68 + 0.40 * pan) * (0.88 + 0.24 * tile);
    col *= 0.68 + 0.32 * gap * (0.72 + 0.28 * course);
    // rain streaks running down the pitch
    float streak = vnoise(vec2(dot(wp.xy, tg) * 1.9, dot(wp.xy, dn) * 0.35));
    col *= 1.0 - 0.14 * smoothstep(0.55, 0.95, streak);
    return mix(mean, col, near);
}


// ---------------------------------------------------------------------------------
// Stable suite identity for the shared finite 3D room material.
float h11(float x) { return fract(sin(x * 127.1 + 311.7) * 43758.5453); }
vec4 roomHash(vec2 id, float seed) {
    // `seed` is an interpolated per-building constant; sin() amplifies its last-bit noise
    // into per-pixel randomness unless it is snapped first.
    seed = floor(seed * 1024.0 + 0.5) / 1024.0;
    id = floor(id + 0.5);
    return vec4(h11(dot(id, vec2(12.9, 78.2)) + seed * 7.1),
                h11(dot(id, vec2(26.6, 41.3)) + seed * 13.7),
                h11(dot(id, vec2(93.1, 17.9)) + seed * 3.3),
                h11(dot(id, vec2(51.7, 63.4)) + seed * 19.9));
}
""" + INTERIORS + STOREFRONT + STREET_FURNITURE + """
void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 L = normalize(uSunDir);
    // wall tangent along increasing u (derivatives must run in uniform control flow)
    vec3 T0 = normalize(cross(vec3(0.0, 0.0, 1.0), N) + vec3(1e-5, 0.0, 0.0));
    float sgnU = dot(T0, dFdx(vWorld)) * dFdx(vUV.x) + dot(T0, dFdy(vWorld)) * dFdy(vUV.x);
    vec3 Tu = T0 * (sgnU < 0.0 ? -1.0 : 1.0);
    vec3 Rv = -V;
    float sf = shadowPCF(vWorld, N, L);
    float dayK = 1.0 - clamp(uNightFactor, 0.0, 1.0);

    float theme  = vStyle.y;
    // vStyle.x is a per-building constant, but interpolation leaves last-bit noise that the
    // hashes below amplify (x131 ...) into per-pixel flicker.  Snap it to its exact value.
    float seed   = floor(vStyle.x * 1024.0 + 0.5) / 1024.0;
    float bH     = vStyle.z;
    float localZ = vLocalZ;
    float u      = vUV.x;
    float L_wall = vUV.y;
    vec2 facadeFootprint = max(fwidth(vec2(u, localZ)), vec2(0.001));
    vec2 propFootprint = max(fwidth(vUV), vec2(.001));

    vec3  albedo = vColor;
    float rough  = 0.8;
    float metal  = 0.0;
    float emisS  = 0.0;
    vec3  emisC  = vec3(0.0);
    // 1 = near facade, falls toward the dim unresolved level once a window bay is only a
    // few pixels across or the facade has receded into the skyline.
    float windowLod = 1.0;
    float glassMask = 0.0;
    float glassF0   = 0.04;   // uncoated glass, IOR ~1.5; coated facades use 0.08 below
    float glassRough = 0.06;  // pane roughness, independent of distance-averaged opaque frames
    vec3 glassTransmission = vec3(0.92, 0.95, 0.97);
    vec3  interiorRad = vec3(0.0);
    float interiorW   = 0.0;
    vec3  inAmb = uAmbColor * 0.11;   // daylight reaching a room is a small fraction of the sky fill outside
    float bounceAO  = 1.0;

    if (abs(vPartKind - 10.0) < 0.1) {
        // ================= METRIC STOREFRONT PORTAL =================
        glassMask = 1.0; glassF0 = .04; rough = .06;
        albedo = vec3(.055,.065,.065);
        glassTransmission = vec3(.93,.95,.93);
        vec4 hh = roomHash(vec2(0.,0.), seed);
        vec3 rd = vec3(dot(Rv,Tu), Rv.z, dot(Rv,-N));
        // Metric powder-coated jambs, sill, transom and door frame: these are
        // planar facade detail and do not need independent six-sided mesh boxes.
        float edgeFrame = max(step(min(vUV.x,vStyle.z-vUV.x),.07),
                              step(min(vUV.y,vEmissive-vUV.y),.075));
        float doorWidth = min(1.4,vStyle.z*.32);
        float doorX = abs(vUV.x-vStyle.z*.5);
        float jamb = step(abs(doorX-doorWidth*.5),.027)*step(vUV.y,vEmissive-.30);
        float head = step(abs(vUV.y-(vEmissive-.31)),.03)*step(doorX,doorWidth*.5);
        float bar = step(abs(vUV.y-1.06),.025)*step(abs(vUV.x-vStyle.z*.5),.52);
        if (max(edgeFrame,max(jamb,head))>.5) {
            glassMask=0.; albedo=vec3(.16,.17,.17); rough=.48;
        } else if (bar>.5) {
            glassMask=0.; albedo=vec3(.52,.53,.50); rough=.32;
        } else {
            interiorRad = shopRadiance(rd, vUV, vec2(vStyle.z,vEmissive), theme, hh,
                                      uAmbColor*.10+uSunColor*sf*dayK*.008, facadeFootprint);
            interiorW = 1.0;
        }
    } else if (abs(vPartKind - 6.0) < 0.1) {
        // ================= BASE (podium / shopfront) =================
        albedo = mix(vec3(0.29, 0.285, 0.27), stuccoPalette(fract(seed * 7.3)), 0.45);
        float joint = 1.0 - band(localZ / 0.6, 0.012, 0.988, facadeFootprint.y / 0.6);
        albedo *= 1.0 - 0.16 * joint;
        rough = 0.72; metal = 0.0;

        float bay   = u / 3.0;
        float fx    = fract(bay);
        float inBay = step(0.05, fx) * step(fx, 0.95);
        float glassZ = step(0.35, localZ) * step(localZ, 3.75);
        float isGlass = inBay * glassZ;
        // slim aluminium mullion through each bay and a signage fascia above
        float mull = glassZ * (1.0 - smoothstep(0.012, 0.03, abs(fx - 0.5)));
        float fascia = step(3.75, localZ);

        if (fascia > 0.5) {
            albedo = mix(vec3(0.10, 0.105, 0.115), stuccoPalette(fract(seed * 7.3 + floor(bay) * 0.17)), 0.20);
            rough = 0.55;
        }
        if (isGlass > 0.5) {
            albedo = vec3(0.05, 0.065, 0.085);
            rough = 0.06; metal = 0.0;
            glassMask = 1.0; glassF0 = 0.04;
            float shopId = floor(bay);
            float shopOn = step(hash21(vec2(shopId, seed * 3.0 + 1.0)), 0.45);
            emisS = shopOn * 0.55 * uNightFactor
                  + shopOn * 0.012 * dayK * (0.5 + 0.5 * localZ / 3.75);   // lit interiors are barely visible in daylight
            emisC = vec3(1.0, 0.78, 0.50);
            float sd = length(uCamPos - vWorld);
            // No distance fade: interiorRadiance() blends to its flat mean colour by itself
            // (interior LOD), so the window never switches between "room" and "no room".
            // Beyond the fog end the pixel is pure fog colour anyway.
            float iw = isGlass;
            if (iw > 0.0 && sd < uFogFar) {
                float grp = floor(bay / 2.0);
                vec4 hh = roomHash(vec2(grp, 3.0), seed);
                vec2 ro = vec2((mod(floor(bay), 2.0) + fx) * 3.0, localZ);
                vec3 rd = vec3(dot(Rv, Tu), Rv.z, dot(Rv, -N));
                float usage = floor(hash21(vec2(grp + 41.7, seed * 17.9)) * 4.99);
                interiorRad = shopRadiance(rd, ro, vec2(4.5,3.0), usage, hh,
                                          uAmbColor*.10+uSunColor*sf*dayK*.008, facadeFootprint);
                interiorW = iw;
            }
        }
        if (mull > 0.5 && isGlass > 0.5) {
            albedo = vec3(0.24, 0.25, 0.25); rough = 0.35; metal = 0.0; glassMask = 0.0; emisS = 0.0; interiorW = 0.0;
        }

    } else if (abs(vPartKind) < 0.1) {
        // ================= MID-SECTION (facade) =================
        if (N.z > 0.55) {
            albedo = roofSurface(vWorld, vColor, N, seed);
            rough = 0.9; metal = 0.0;
        } else {
            float hs  = hash21(vec2(seed * 91.7 + 3.1, seed * 13.3 + 7.7));
            float hs2 = hash21(vec2(seed * 37.1 + 1.3, seed * 5.9 + 2.2));
            float fz   = localZ - 4.5;          // floors start on top of the podium
            float topD = bH - localZ;           // distance below the wall top

            float floorH, colW, margin;
            if (theme < 0.5)      { floorH = 3.6; colW = 1.35; margin = 0.40; }   // curtain wall
            else if (theme < 1.5) { floorH = 3.0; colW = 2.5; margin = 1.10; }   // brick
            else if (theme < 2.5) { floorH = 3.4; colW = 2.3; margin = 0.55; }   // stone
            else if (theme < 3.5) { floorH = 4.8; colW = 2.6; margin = 1.50; }   // metal / industrial
            else if (theme < 4.5) { floorH = 3.2; colW = 2.4; margin = 0.55; }   // concrete
            else if (theme < 5.5) { floorH = 2.9; colW = 1.95; margin = 0.70; } // pre-war brick walk-up
            else if (theme < 6.5) { floorH = 3.3; colW = 2.4; margin = 0.85; } // beaux-arts limestone
            else if (theme < 7.5) { floorH = 3.4; colW = 1.6; margin = 0.50; } // art-deco pier strips
            else                  { floorH = 3.8; colW = 2.05; margin = 0.60; } // cast-iron loft
            colW *= mix(0.90, 1.16, hs2);

            // Fit a whole number of window bays to this wall so nothing is cut off at corners.
            float usable = max(L_wall - 2.0 * margin, 0.0);
            float nCols  = floor(usable / colW);
            float hasWin = step(1.0, nCols) * step(0.0, fz) * step(1.1, topD)
                         * step(margin, u) * step(u, L_wall - margin);
            nCols = max(nCols, 1.0);
            // Narrow walls / trim can have no usable window area.  Hashing an infinite
            // cell coordinate made emission NaN even when hasWin was zero (0 * NaN).
            float cw = max(usable / nCols, 0.1);
            vec2 cell  = vec2((u - margin) / cw, fz / floorH);
            vec2 ffrac = fract(cell);
            vec2 cid   = floor(cell);
            vec2 pixel = facadeFootprint / vec2(cw, floorH);

            float winMask = 0.0;
            float dist = length(uCamPos - vWorld);
            float detail = 1.0 - smoothstep(0.025, 0.14, max(facadeFootprint.x, facadeFootprint.y));
            float far = smoothstep(0.4, 1.2, max(pixel.x, pixel.y));
            // Distant lit panes are small, and once a bay approaches subpixel size the
            // suite-wide lighting is no longer resolved: full room radiance then spreads
            // over whole pixels and, after bloom, reads as big glowing slabs.  Settle the
            // skyline toward the subdued unresolved level instead of keeping every pane at
            // close-range brightness.  Near facades keep their full interior detail.
            // ~12 px per bay is still crisp; below ~3 px the pane is no longer resolved.
            // The unresolved level already carries the room's furniture/curtain occlusion
            // (furnishedRoom far average), so this only trims the remaining excess.
            float subpixel = smoothstep(0.08, 0.30, max(pixel.x, pixel.y));
            float receded  = smoothstep(120.0, 650.0, dist);
            windowLod = mix(1.0, 0.55, max(subpixel, receded));

            vec4 windowRect = vec4(.035, .965, .24, .90);
            if (theme < 0.5) {
                // ---------- glass curtain wall: spandrels + mullions + fins ----------
                vec3 tint;
                if (hs < 0.35)      tint = vec3(0.17, 0.20, 0.22);   // blue-grey
                else if (hs < 0.65) tint = vec3(0.18, 0.21, 0.20);   // low-iron glazing
                else if (hs < 0.85) tint = vec3(0.14, 0.15, 0.16);   // graphite
                else               tint = vec3(0.24, 0.22, 0.19);   // bronze
                vec3 frame = hs2 < 0.6 ? vec3(0.28, 0.29, 0.29) : vec3(0.48, 0.49, 0.48);
                albedo = tint * 0.80;  // opaque insulated spandrel
                rough = 0.55; metal = 0.0;     // powder-coated frames are dielectric paint

                float gx = band(cell.x, 0.035, 0.965, pixel.x);
                float gy = band(cell.y, 0.24, 0.90, pixel.y);
                winMask = gx * gy * hasWin;
                float mullion = 1.0 - gx;
                albedo = mix(albedo, frame, mullion / max(1.0 - winMask, 0.01));
                float pv = hash21(cid + vec2(seed * 71.0, seed * 29.0));
                glassTransmission = mix(vec3(1.0), tint / max(max(tint.r, tint.g), tint.b), 0.32)
                                  * mix(mix(0.96, 1.0, pv), 0.98, far);
                rough  = mix(rough, 0.06, winMask);
                metal  = mix(metal, 0.0, winMask);
                glassF0 = 0.08;
            } else {
                // ---------- punched / ribbon windows ----------
                vec4 wr;            // window rect in bay space: x0 x1 y0 y1
                vec3 wall;
                if (theme < 1.5) {
                    // brick with stone lintels + sills
                    wall = hs < 0.45 ? mix(vec3(0.38, 0.28, 0.23), vec3(0.53, 0.36, 0.28), hs / 0.45)
                         : mix(vec3(0.50, 0.43, 0.34), vec3(0.63, 0.55, 0.44), (hs - 0.45) / 0.55);
                    wr = vec4(0.24, 0.76, 0.16, 0.76);
                    vec2 bp = vec2(u / 0.215, localZ / 0.075);
                    bp.x += 0.5 * mod(floor(bp.y), 2.0);
                    vec2 bf = fract(bp);
                    float mort = max(1.0 - smoothstep(0.03, 0.09, bf.x), 1.0 - smoothstep(0.06, 0.16, bf.y));
                    float bv = hash21(floor(bp) + seed * 13.0);
                    vec3 bricks = wall * (0.82 + 0.36 * bv);
                    albedo = mix(wall, mix(bricks, vec3(0.60, 0.57, 0.52), mort * 0.85), detail);
                    rough = 0.9; metal = 0.0;
                } else if (theme < 2.5) {
                    // Cut-stone piers, coursing and taller recessed sash windows.
                    wall = stuccoPalette(hs);
                    wr = hs2 < 0.45 ? vec4(0.27, 0.73, 0.15, 0.86) : vec4(0.18, 0.82, 0.20, 0.84);
                    float course = 1.0 - band(localZ / 0.72, 0.008, 0.992, facadeFootprint.y / 0.72);
                    float pier = 1.0 - band(cell.x, 0.10, 0.90, pixel.x);
                    float belt = 1.0 - band(cell.y / 4.0, 0.022, 0.978, pixel.y / 4.0);
                    albedo = wall * (1.0 - 0.13 * course + 0.035 * pier - 0.08 * belt);
                    bounceAO *= 1.0 - 0.10 * band(cell.y, 0.90, 0.98, pixel.y) * hasWin;
                    rough = 0.82; metal = 0.0;
                } else if (theme < 3.5) {
                    // industrial cladding: corrugated ribs + clerestory strip
                    wall = (hs < 0.5) ? vec3(0.52, 0.53, 0.53)
                         : (hs < 0.8) ? vec3(0.43, 0.46, 0.47) : vec3(0.60, 0.58, 0.53);
                    wr = vec4(0.10, 0.90, 0.64, 0.82);
                    float rib = 0.5 + 0.5 * sin(u * 14.0);
                    albedo = wall * (0.86 + 0.20 * mix(0.5, rib, detail));
                    rough = 0.48; metal = 0.0;  // painted industrial cladding
                } else if (theme < 4.5) {
                    // Precast panels: narrow joints, coherent structural bays and slab bands.
                    wall = stuccoPalette(hs) * vec3(0.96, 0.97, 0.98);
                    wr = hs2 < 0.5 ? vec4(0.15, 0.85, 0.18, 0.86) : vec4(0.08, 0.92, 0.28, 0.80);
                    float joint = 1.0 - band(cell.x, 0.008, 0.992, pixel.x)
                                       * band(cell.y, 0.009, 0.991, pixel.y);
                    albedo = wall * (1.0 - 0.18 * joint);                    bounceAO *= 1.0 - 0.13 * band(cell.y, 0.90, 0.98, pixel.y) * hasWin;

                    rough = 0.88; metal = 0.0;
                } else if (theme < 5.5) {
                    // ---- pre-war brick walk-up: tall narrow openings, stone sills ----
                    wall = hs < .5 ? mix(vec3(.34, .22, .17), vec3(.47, .29, .21), hs / .5)
                         : mix(vec3(.40, .27, .21), vec3(.55, .36, .26), (hs - .5) / .5);
                    wr = vec4(0.28, 0.72, 0.10, 0.80);
                    vec2 bp = vec2(u / .20, localZ / .068);      // small weathered headers
                    bp.x += .5 * mod(floor(bp.y), 2.0);
                    vec2 bf = fract(bp);
                    float mort = max(1.0 - smoothstep(.03, .10, bf.x), 1.0 - smoothstep(.05, .15, bf.y));
                    float bv = hash21(floor(bp) + seed * 17.0);
                    albedo = mix(wall, mix(wall * (0.80 + 0.34 * bv), vec3(.56, .53, .47), mort * .8), detail);
                    // stone belt course every fourth floor, as pre-war walk-ups carry
                    float belt = 1.0 - band(cell.y / 4.0, 0.0, 0.045, pixel.y / 4.0);
                    albedo = mix(albedo, vec3(.60, .57, .50), belt * .85);
                    rough = .92; metal = 0.0;
                } else if (theme < 6.5) {
                    // ---- beaux-arts limestone: coursed stone, quoins, tall sashes ----
                    wall = stuccoPalette(hs) * vec3(1.02, .99, .93);
                    wr = hs2 < .4 ? vec4(0.30, 0.70, 0.12, 0.84) : vec4(0.22, 0.78, 0.16, 0.86);
                    float course = 1.0 - band(localZ / 0.66, 0.006, 0.994, facadeFootprint.y / 0.66);
                    float quoin = clamp(band(u, 0.0, 0.55, facadeFootprint.x)
                                      + band(u, L_wall - 0.55, L_wall, facadeFootprint.x), 0.0, 1.0);
                    float pier = 1.0 - band(cell.x, 0.06, 0.94, pixel.x);
                    albedo = wall * (1.0 - .15 * course + .05 * pier);
                    albedo = mix(albedo, vec3(.68, .65, .58), quoin * detail * .30);
                    bounceAO *= 1.0 - .11 * band(cell.y, .90, .98, pixel.y) * hasWin;
                    rough = .84; metal = 0.0;
                } else if (theme < 7.5) {
                    // ---- art-deco: recessed window ribbons between vertical piers ----
                    wall = hs < .5 ? vec3(.60, .58, .52) : vec3(.50, .48, .43);
                    wr = vec4(0.33, 0.67, 0.06, 0.94);
                    float pierW = band(cell.x, 0.0, 0.30, pixel.x);
                    float spandrel = band(cell.y, 0.0, 0.17, pixel.y);
                    float flute = 0.5 + 0.5 * sin(u * 26.0);
                    albedo = wall * (1.0 + .10 * pierW - .06 * spandrel)
                           * (1.0 + .05 * flute * detail * spandrel);
                    albedo = mix(albedo, vec3(.44, .42, .38), pierW * .35);
                    rough = .78; metal = 0.0;
                } else {
                    // ---- cast-iron / loft front: pilasters between tall arched bays ----
                    wall = hs < .45 ? vec3(.72, .71, .65)
                         : (hs < .8 ? vec3(.52, .54, .50) : vec3(.34, .35, .33));
                    wr = vec4(0.22, 0.78, 0.08, 0.88);
                    float pilaster = band(cell.x, 0.0, 0.16, pixel.x);
                    float capital = band(cell.y, 0.93, 0.99, pixel.y);
                    albedo = mix(wall * (1.0 - .10 * pilaster), wall * 1.10, capital);
                    rough = .55; metal = 0.0;   // painted cast iron
                }
                float wx = band(cell.x, wr.x, wr.y, pixel.x);
                // Limestone and cast-iron fronts carry round-arched heads.
                float arch = ((theme > 5.5 && theme < 6.5) || theme > 7.5)
                           ? .12 * max(0.0, 1.0 - pow(clamp((ffrac.x * 2.0 - 1.0) / .92, -1.0, 1.0), 2.0))
                           : 0.0;
                float wy = band(cell.y, wr.z, wr.w - arch, pixel.y);
                float outer = wx * wy * hasWin;
                float fw = 0.022;
                float ix = band(cell.x, wr.x + fw, wr.y - fw, pixel.x);
                float iy = band(cell.y, wr.z + fw, wr.w - fw, pixel.y);
                winMask = ix * iy * hasWin;

                // frame ring, sill and lintel
                vec3 frameCol = (hs2 < 0.35) ? vec3(0.52, 0.51, 0.48) : vec3(0.19, 0.20, 0.20);
                albedo = mix(albedo, frameCol, clamp((outer - winMask) / max(1.0 - winMask, 0.01), 0.0, 1.0));
                if (theme < 1.5 || (theme > 4.5 && theme < 5.5)) {
                    float sill = band(cell.y, wr.z - 0.035, wr.z, pixel.y) * wx * hasWin;
                    float lintel = band(cell.y, wr.w, wr.w + 0.05, pixel.y)
                                 * band(cell.x, wr.x - 0.03, wr.y + 0.03, pixel.x) * hasWin;
                    albedo = mix(albedo, vec3(0.62, 0.59, 0.53), max(sill, lintel));
                }
                // recess shadow along the top of every opening
                float recess = 0.55 + 0.45 * smoothstep(0.0, 0.16, (wr.w - ffrac.y) / (wr.w - wr.z));
                bounceAO *= mix(1.0, mix(recess, 0.96, far), winMask);
                rough = mix(rough, 0.05, winMask);
                glassRough = 0.05;
                metal = mix(metal, 0.0, winMask);
                glassF0 = 0.04;
                windowRect = wr;
            }
            glassMask = winMask;

            // roof-line trim: capped band + shadow line below it
            if (topD < 1.1) {
                float lip = step(topD, 0.5);
                bool prewar = theme > 4.5;
                vec3 trim = (theme < 0.5) ? vec3(0.22, 0.23, 0.23)
                          : (prewar ? vec3(0.66, 0.63, 0.56) : albedo * 0.96);
                albedo = mix(albedo, trim, step(topD, 1.1) * (1.0 - winMask));
                albedo *= 1.0 - 0.18 * smoothstep(0.5, 0.7, topD) * (1.0 - lip);
            }
            // weathering streaks and street-level grime
            if (dist < 350.0) {
                float streak = vnoise(vec2(u * 1.7 + seed * 9.0, fz * 0.06));
                albedo *= 1.0 - 0.10 * smoothstep(0.55, 0.92, streak) * (1.0 - smoothstep(180.0, 350.0, dist));
            }
            albedo *= mix(0.80, 1.0, smoothstep(2.0, 14.0, localZ));

            // --- OCCUPANCY AND ELECTRIC LIGHTING ---
            // Material theme is not usage: concrete office towers need occupied
            // daytime floors too, not just the all-glass architectural theme.
            bool  office = theme < 0.5 || (bH > 35.0 &&
                           ((theme > 3.5 && theme < 4.5) || (theme > 1.5 && theme < 2.5)));
            float unitCells = office && theme < .5
                            ? 2.0 + floor(hash21(vec2(cid.y + 71.0, seed * 39.0)) * 3.0) : 2.0;
            float unitId = floor(cid.x / unitCells);
            float floorId = cid.y;
            float roomRand  = hash21(vec2(unitId + seed * 131.0, floorId + seed * 29.0));
            float floorRand = hash21(vec2(floorId * 1.31 + 3.0, seed * 42.0 + 9.0));
            float winRand   = hash21(cid + vec2(seed * 137.0, seed * 91.0));

            // Stable per-suite daily schedules: seconds-long fades at independent
            // arrival/departure times, never a city-wide night-factor switch.
            float schedule = hash21(vec2(unitId + 7.0, floorId + seed * 5.0));
            float clock = mod(uHours + 24.0, 24.0);
            float arrival = office ? mix(7.0, 10.5, floorRand * .45 + schedule * .55)
                                   : mix(16.0, 20.5, schedule);
            float departure = office ? mix(17.0, 23.5, roomRand)
                                     : mix(21.0, 26.0, roomRand);
            float eveningClock = office ? clock : mod(clock + 8.0, 24.0) + 16.0;
            float eveningLit = smoothstep(arrival, arrival + .08, eveningClock)
                         * (1.0 - smoothstep(departure, departure + .10, eveningClock));
            // Short, staggered morning routines rather than every home remaining
            // lit from waking until 09:30. Only some households need electric light.
            float wake = mix(5.5, 8.5, schedule);
            float leave = wake + mix(.35, 1.1, roomRand);
            float morning = smoothstep(wake, wake + .08, clock)
                          * (1.0 - smoothstep(leave, leave + .10, clock))
                          * step(roomRand, .65);
            // Fraction of floors / suites that are ever occupied. A real tower at night shows
            // roughly a third to a half of its windows lit, not nearly all of them.
            float occupied = office ? step(floorRand, .84) * step(roomRand, .66)
                                    : step(floorRand, .70);
            // Workplaces commonly keep practical lights on during business hours;
            // use independent suite demand rather than a uniformly glowing floor.
            {
                // Daylight carries the room, so only a small share of suites leave lights on.
                float dayUse = mix(office ? .07 : .015, office ? .20 : .06,
                                   hash21(vec2(seed * 73.0 + 2.1, seed * 19.0 + 8.4)));
                float lampRand = hash21(vec2(unitId + seed * 83.0 + 5.2, floorId + seed * 47.0));
                float dayLit = step(lampRand, dayUse);
                float duskStart = mix(.15, .50, schedule);
                occupied *= mix(dayLit, 1.0, smoothstep(duskStart, duskStart + .20, uNightFactor));
            }
            float late = step(roomRand, (office ? .045 : .035) * mix(.22, 1.0, clamp(uNightFactor, 0.0, 1.0)));
            // A quiet, independently phased activity cycle also animates lights
            // when the solar clock is paused. Periods are several real minutes.
            float phase = fract(uTime / mix(240.0, 660.0, schedule) + roomRand);
            float away = smoothstep(.78, .79, phase) * (1.0 - smoothstep(.85, .86, phase));
            float atHome = step(schedule, .32) * smoothstep(9.0, 10.0, clock)
                         * (1.0 - smoothstep(16.0, 17.0, clock));
            float roomLit = max(late, occupied * max(eveningLit, office ? 0.0 : max(morning, atHome)) * (1.0 - away));
            float isOn = roomLit * (office ? 1.0 : step(winRand, .90));

            float bright = mix(0.35, 1.0, hash21(vec2(unitId * 3.1 + seed * 11.0, floorId * 1.7 + 2.0)));
            emisS = winMask * isOn * bright;

            float k = hash21(vec2(unitId * 5.3 + seed * 61.0 + 7.0, floorId * 2.9 + seed * 43.0));
            vec3 tungsten = vec3(1.00, 0.69, 0.42);
            vec3 warmLed  = vec3(1.00, 0.84, 0.69);
            vec3 coolLed  = vec3(0.78, 0.88, 1.00);
            vec3 col;
            if (office) col = mix(vec3(1.00, 0.96, 0.90), warmLed, step(0.86, k));
            else        col = k < 0.45 ? tungsten : (k < 0.80 ? warmLed : coolLed);
            emisC = col;

            // ---- finite 3D room behind the glass
            float idist = length(uCamPos - vWorld);
            float iw = winMask * hasWin;
            if (iw > 0.0 && idist < uFogFar) {
                float ux0 = floor(cid.x / unitCells) * unitCells;
                vec4 hh = roomHash(vec2(floor(cid.x / unitCells), cid.y), seed);
                vec2 ro = vec2((cid.x - ux0 + ffrac.x) * cw, ffrac.y * floorH - 0.25); // slab below, ceiling void above: clear height = floorH - 0.5
                vec3 rd = vec3(dot(Rv, Tu), Rv.z, dot(Rv, -N));
                float depthM = mix(3.0, 5.4, hh.w);
                float roomWidth = min(unitCells, nCols - ux0) * cw;
                vec3 rad = interiorRadiance(rd, ro, vec2(roomWidth, floorH - 0.5), depthM, hh,
                                            roomLit * bright, emisC, inAmb, office, facadeFootprint);
                rad = windowDressing(rad, rd, ffrac, vec2(cw, floorH), windowRect,
                                     hh, roomLit * bright, emisC, inAmb, office);
                interiorRad = rad;
                interiorW = iw;
            }
        }
    } else if (vPartKind < 1.5) {
        // ================= ROOF =================
        albedo = roofSurface(vWorld, vColor, N, seed);
        rough  = 0.88; metal = 0.0;
    } else if (vPartKind > 8.5 && vPartKind < 9.5) {
        // ================= ROOFTOP: parapet, plant rooms, AC units =================
        float n = vnoise(vWorld.xy * 1.3 + vWorld.z * 0.7);
        albedo = mix(vec3(0.52, 0.51, 0.48), vec3(dot(vColor, PHOTO_LUMA)), 0.20) * (0.96 + 0.08 * n);
        float louver = band(vWorld.z / 0.18, 0.12, 0.60, facadeFootprint.y / 0.18);
        albedo *= 1.0 - 0.16 * louver * (1.0 - abs(N.z));
        rough  = 0.84; metal = 0.0;
    } else if (vPartKind > 6.5 && vPartKind < 8.5) {
        // ================= STREET FURNITURE / PAINT =================
        albedo = vColor;
        rough  = 0.42; metal = 0.0;  // paint on street furniture / signs
        if (abs(theme-1.0)<.01) {
            // Woven, subtly striped awning fabric instead of saturated plastic sheets.
            float cloth=1.-smoothstep(.025,.13,max(fwidth(vUV.x),fwidth(vUV.y)));
            albedo *= 1.0-.10*cloth*(.5+.5*sin(vUV.x*20.0));
            rough=.94;
        } else if (abs(theme-2.0)<.01) {
            rough=.72;  // unlettered, powder-coated fascia
        } else if (theme > 2.5) {
            streetPropSurface(theme, seed, vUV, propFootprint, N, albedo, rough, metal);
        }
    } else if (vPartKind > 10.5) {
        // ================= ROAD PAINT =================
        albedo = vColor;
        rough  = 0.62; metal = 0.0;
    } else {
        // ================= MAST / BEACON =================
        albedo = vColor;
        rough  = 0.40; metal = 0.75;
        if (vEmissive > 0.5) {
            // aviation obstruction light: slow red blink, brighter at night
            float blink = 0.30 + 0.70 * step(0.5, fract(uTime * 0.6 + seed * 3.0));
            emisC = vec3(1.0, 0.05, 0.03);
            emisS = blink * (BEACON_NITS / WINDOW_NITS);
            albedo = vec3(0.6, 0.05, 0.04);
        }
    }

    // The architectural palettes are display RGB, not linear reflectance.
    albedo = linearColor(albedo);
    vec2 albRange = PAINT_REFLECTANCE;
    bool streetProp = vPartKind > 6.5 && vPartKind < 8.5 && theme > 2.5;
    bool roofFace = (!streetProp && N.z > 0.55) || (vPartKind > 0.5 && vPartKind < 1.5);
    if (roofFace) albRange = ROOF_REFLECTANCE;
    // Wall/frame coverage is already mixed: clamping it to the masonry minimum
    // would turn dark frames and spandrels into pale coloured borders.
    if (roofFace) {
        // One gain per roof, taken from its unpatterned colour.  Renormalising every
        // pixel to the same luminance (as reflectanceRange does) would erase the tile
        // courses, leaving a flat slab of the right hue.
        vec3 roofBase = linearColor(vColor);
        float y = dot(roofBase, PHOTO_LUMA);
        albedo *= clamp(y, albRange.x, albRange.y) / max(y, 1e-6);
    } else {
        albedo = mix(reflectanceRange(albedo, albRange), albedo, glassMask);
    }
    if (metal > 0.5) albedo = mix(albedo, vec3(0.56, 0.57, 0.58), metal);  // bare iron / steel

    // Wet facade + animated beads running down glazing.
    float wet = clamp(uWetness, 0.0, 1.0);
    float droplets = 0.0;
    if (glassMask > 0.01 && uRain > 0.01) {
        vec2 dp = vec2(u * 0.55, localZ * 1.35)
                + vec2(0.0, -uWeatherTime * 0.10);
        float bead = 0.5 + 0.5 * sin(dp.y * 9.0 + sin(dp.x * 5.0) * 2.0
                    + hash21(floor(dp * 3.0)) * 6.28);
        droplets = smoothstep(0.56, 0.92, bead) * glassMask * uRain;
    }
    albedo *= mix(1.0, 0.72, wet);
    rough *= (1.0 - 0.75 * wet);
    glassRough *= (1.0 - 0.75 * wet);
    albedo *= 1.0 - 0.18 * droplets;
    rough = mix(rough, 0.02, droplets * 0.55);
    glassRough = mix(glassRough, 0.02, droplets * 0.55);
    emisS *= 1.0 - 0.25 * droplets;

    vec3 F0 = mix(vec3(0.04), albedo, metal);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.001);
    // A tight GGX lobe peaks in the thousands on a mirror-aligned pane, which the bloom then
    // spreads into big glowing blobs.  Keep the floor wider and cap the peak.
    vec3 spec = min(spec_ggx(N, V, L, F0, max(rough, 0.20)), vec3(0.6)) * (1.0 - glassMask);
    vec3 kD   = (1.0 - F0) * (1.0 - metal) * (1.0 - glassMask);
    vec3 diff = kD * albedo / PI;
    vec3 direct = (diff + spec) * NdotL * uSunColor * sf;

    float streetAO = mix(0.45, 1.0, smoothstep(0.0, 25.0, vWorld.z));
    float ao = vAO * bounceAO * mix(streetAO, 1.0, clamp(N.z, 0.0, 1.0));

    // sky-dome fill from above, warm street bounce from below (low facades only)
    vec3  amb = uAmbColor * mix(0.55, 1.0, N.z * 0.5 + 0.5);
    float bounce = (1.0 - clamp(N.z, 0.0, 1.0)) * (1.0 - smoothstep(0.0, 45.0, vWorld.z));
    amb += vec3(0.085, 0.060, 0.038) * bounce * dayK;
    vec3 ambientLit = kD * albedo * amb * ao * 0.55;
    vec3 lit = direct + ambientLit;
    vec3 sunReflection = vec3(0.0);

    // Mirror reflection of the sky on glazing and polished metal.  Each pane is
    // nudged by a random tilt so a tower reads as many panels, not one flat mirror.
    // envSky() (noise-based clouds) is only needed where something reflects, so
    // it is skipped for plain walls and roofs -- the large majority of pixels.
    if (glassMask > 0.01 || metal > 0.5) {
        vec3 R = reflect(-V, N);
        if (glassMask > 0.01) {
            vec2 pc = floor(vec2(u / 1.5, localZ / 3.0));
            R = normalize(R + (vec3(hash21(pc + seed), hash21(pc + 17.0 + seed), hash21(pc + 41.0 + seed)) - 0.5)
                            * vec3(0.05, 0.05, 0.035) * (1.0 - smoothstep(90.0, 420.0, length(uCamPos - vWorld))));
        }
        vec3 env = envSky(R, L);
        float Fg = glassF0 + (1.0 - glassF0) * pow(1.0 - NdotV, 5.0);
        lit += env * Fg * glassMask * mix(0.55, 1.0, ao);
        // Reflect the actual solar radiance, attenuated by the glass Fresnel term.  The true
        // sun direction/intensity keeps this bright, compact glint exclusive to daylight.
        if (glassMask > 0.01 && uSkySunDir.z > 0.02 && uSunIntensity > 0.0) {
            float gdist = length(uCamPos - vWorld);
            float rl = max(dot(R, normalize(uSkySunDir)), 0.0);
            float angularBlur = 0.0015 + glassRough * glassRough * 0.25;
            float glint = sunRadiance(rl, angularBlur);
            float gfade = 1.0 - smoothstep(700.0, 2200.0, gdist);
            vec3 sunTint = uSunColor / max(uSunColor.r, 1e-4);
            float sunVisible = uSunIntensity * (1.0 - clamp(uCloudCover, 0.0, 1.0) * 0.95);
            // The mirrored solar disc needs an unobstructed sun. A partially shadowed
            // value (soft PCF penumbra, settling map, receiver bias) still renders a
            // white-hot glint after tone mapping, so the disc fades out well before the
            // point is fully lit. Diffuse light keeps the smooth `sf`.
            float sunUnblocked = smoothstep(0.55, 0.92, sf);
            sunReflection = sunTint * Fg * glint * glassMask * sunUnblocked * sunVisible * gfade * mix(0.6, 1.0, ao);
        }
        if (metal > 0.5) lit += env * F0 * (1.0 - rough) * 0.60 * mix(0.55, 1.0, ao);
    }

    float glassThrough = 1.0 - glassF0 - (1.0 - glassF0) * pow(1.0 - NdotV, 5.0);
    vec3 transmitted = interiorRad * glassTransmission * glassThrough;
    vec3 emissive = mix(emissionNits(emisC, WINDOW_NITS) * emisS, transmitted, interiorW);
    // Unresolved coverage uses the same subdued night fallback as before.  Both the
    // fallback and the resolved room fade with the window LOD above, so a distant
    // occupied suite stays a small lamp rather than a wall-sized white panel.
    if (vPartKind < .5) emissive = mix(emissionNits(emisC, WINDOW_NITS) * emisS
                                     * mix(1.0, .23, uNightFactor),
                                      transmitted, interiorW) * windowLod;

    // Street lamps at any distance: the interpolated per-vertex sum covers the
    // far field, the exact per-pixel loop takes over within a few hundred metres.
    vec3 sDiff = vec3(0.0), sSpec = vec3(0.0);
    if (uNightFactor > 0.02) {
        float nearW = spillNearWeight(length(uCamPos - vWorld));
        sDiff = vSpill;
        if (nearW > 0.002) {
            vec3 pd = vec3(0.0), ps = vec3(0.0);
            streetLights(vWorld, N, V, pd, ps);
            sDiff = mix(sDiff, pd, nearW);
            sSpec = ps * nearW;
        }
    }

    lit += (kD * albedo / PI * sDiff + sSpec * (0.02 + 0.35 * metal) * (1.0 - rough)) * ao * uNightFactor;
    // Ordinary signs keep their wash.  Billboards are emissive surfaces only; their
    // per-fragment area-light loop was removed because dense screen blocks lagged.
    lit += neonSpill(vWorld, N, V) * (0.20 + 0.80 * uNightFactor) * ao;
    lit += emissive;
    // Rough masonry has no clearcoat rim; glazing already includes Fresnel above.
    // Validate BEFORE min(): some GPUs turn min(NaN, 32) into a white-hot 32.
    if (any(isnan(lit)) || any(isinf(lit))) lit = albedo * uAmbColor * 0.05;
    if (any(isnan(sunReflection)) || any(isinf(sunReflection))) sunReflection = vec3(0.0);
    // Ordinary materials stay bounded; a direct solar reflection needs the sun's HDR headroom.
    lit = min(lit, vec3(32.0)) + sunReflection;
    lit = max(lit, albedo * uAmbColor * 0.04);

    vec3 mapped = sceneOut(lit);

    float f = cityFog(length(uCamPos - vWorld), vWorld.z, uFogNear, uFogFar);
    FragColor = vec4(mix(mapped, displayToScene(uFogColor), f), 1.0);
}
""")
