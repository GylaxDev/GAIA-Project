"""Screen space reflections.

Two pieces live here:

* ``mrt_fs``  - patches a scene fragment shader so it also writes a second
  colour output (``oMat``): world-space normal in RGB and a reflectivity
  factor in A.  Shaders that give no expression write reflectivity 0, which
   is what makes them *occluders* in the reflection pass (trees, poles, matte trim).
* ``SSR_FS``  - fullscreen pass.  For every reflective pixel it marches a ray
  through the depth buffer, and mixes the colour found at the hit point into
  the scene.  It outputs the complete image, so bloom runs on top of it.
"""
import re
from .ao import AO_RESOLVE
from .depth import SCENE_DEPTH

# Reflectivity expressions, evaluated at the very end of each shader's main().
# Everything they use (N, rough, metal, ...) is already in scope there.
BUILDING_MAT = ("N * 0.5 + 0.5, "
                "clamp((1.0 - rough) * max(metal, glassMask), 0.0, 1.0)")
TERRAIN_MAT = ("N * 0.5 + 0.5, "
               "paving * clamp(puddle * 0.28, 0.0, 0.28)")
WATER_MAT = "N * 0.5 + 0.5, .82 * (1.0 - foam) * (1.0 - f)"
VEHICLE_MAT = "N * 0.5 + 0.5, reflectivity * (1.0 - f)"


def mrt_fs(src, mat_expr=None, ambient_expr=None):
    """Add the ``oMat`` output to a fragment shader.

    ``oMat`` is initialised to "flat normal, not reflective" at the top of
    main(), so every code path (early returns included) writes something
    defined.  If ``mat_expr`` is given, ``oMat = vec4(<mat_expr>);`` is
    emitted just before main()'s closing brace.
    """
    decl = "out vec4 FragColor;"
    if decl not in src:
        raise ValueError("mrt_fs: shader has no 'out vec4 FragColor;'")
    src = src.replace(
        decl,
        "layout(location = 0) out vec4 FragColor;\n"
        "layout(location = 1) out vec4 oMat;", 1)
    src, n = re.subn(r"void\s+main\s*\(\s*\)\s*\{",
                     "void main() {\n    oMat = vec4(0.5, 0.5, 1.0, 0.0);",
                     src, count=1)
    if n != 1:
        raise ValueError("mrt_fs: could not find main()")
    end = src.rstrip().rfind("}")
    output = f"    oMat = vec4({mat_expr});\n" if mat_expr else ""
    # Scene alpha carries exposed ambient luminance, not opacity (opaque MRT passes).
    # Keeping an absolute amount lets additive lamps preserve it without getting occluded.
    output += f"    FragColor.a = {ambient_expr or '0.0'};\n"
    src = src[:end] + output + src[end:]
    return src


SSR_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uScene;   // exposed linear scene; alpha = ambient luminance
uniform sampler2D uDepth;   // scene depth
uniform sampler2D uMat;     // rgb = world normal * 0.5 + 0.5, a = reflectivity
uniform mat4 uView, uProj, uInvProj;
uniform vec2 uResolution;
uniform float uNear, uMaxDist, uStrength;
out vec4 FragColor;
""" + AO_RESOLVE + SCENE_DEPTH + """
const int STEPS  = 64;
const int REFINE = 6;
const float GROWTH = 1.07;

vec3 viewPos(vec2 uv, float d) {
    vec4 v = uInvProj * vec4(uv * 2.0 - 1.0, depthNdc(d), 1.0);
    return v.xyz / v.w;
}
// positive view-space depth from a [0,1] depth-buffer value
float linZ(float d) {
    return uProj[3][2] / (depthNdc(d) + uProj[2][2]);
}
vec3 decodeN(vec3 enc) {
    return normalize(mat3(uView) * (enc * 2.0 - 1.0));
}

void main() {
    vec4 source = texture(uScene, vUV);
    vec3 scene = source.rgb;
    FragColor = vec4(scene, 1.0);

    float d = texture(uDepth, vUV).r;
    if (depthSky(d)) return;
    vec3 P = viewPos(vUV, d);
    vec4 m = texture(uMat, vUV);
    vec3 Nv = decodeN(m.rgb);
    if (uAOOn > 0.5 && source.a > 1e-6 && -P.z < 650.0) {
        float occlusion = contactAO(P, Nv);
        float lum = dot(scene, vec3(0.2126, 0.7152, 0.0722));
        scene *= 1.0 - clamp(source.a * occlusion / max(lum, 1e-6), 0.0, 0.75);
        FragColor = vec4(scene, 1.0);
    }
    if (uStrength <= 0.0) return;
    float refl = m.a;
    if (refl < 0.02) return;                        // not a reflective surface

    vec3 Vv = normalize(P);                         // camera -> surface
    float NdotV = clamp(dot(Nv, -Vv), 0.0, 1.0);

    // Schlick fresnel; the base reflectance rises a little with reflectivity
    // (water ~ 0.07, glass/wet asphalt in between, matte surfaces ~ 0.03).
    float f0 = 0.03 + 0.04 * refl;
    float F  = (f0 + (1.0 - f0) * pow(1.0 - NdotV, 4.0)) * refl * uStrength;
    if (F < 0.004) return;

    vec3 R = reflect(Vv, Nv);
    if (R.z > 0.55) return;                         // heading back at the camera

    float zr0   = -P.z;
    vec3  pos   = P + Nv * (0.03 + 0.0015 * zr0);   // lift off the surface
    vec3  prev  = pos;
    // Deterministic intervals: per-pixel random step lengths produce hit/miss
    // speckles on glazing in a renderer without temporal accumulation.
    float stepL = max(0.20, 0.004 * zr0);
    float t     = 0.0;
    bool  hit   = false;
    vec2  huv   = vec2(0.0);

    for (int i = 0; i < STEPS; ++i) {
        prev   = pos;
        pos   += R * stepL;
        t     += stepL;
        float thick = 0.30 + 1.5 * stepL;
        stepL *= GROWTH;

        if (pos.z > -uNear || t > uMaxDist) break;
        vec4 c = uProj * vec4(pos, 1.0);
        vec2 uv = c.xy / c.w * 0.5 + 0.5;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;

        float sd = texture(uDepth, uv).r;
        if (depthSky(sd)) continue;
        float diff = (-pos.z) - linZ(sd);           // > 0: ray is behind the surface
        if (diff > 0.0 && diff < thick) {
            vec3 lo = prev, hi = pos;
            for (int j = 0; j < REFINE; ++j) {
                vec3 mid = 0.5 * (lo + hi);
                vec4 cm = uProj * vec4(mid, 1.0);
                vec2 um = cm.xy / cm.w * 0.5 + 0.5;
                if ((-mid.z) > linZ(texture(uDepth, um).r)) hi = mid; else lo = mid;
            }
            vec4 ch = uProj * vec4(hi, 1.0);
            huv = ch.xy / ch.w * 0.5 + 0.5;
            hit = true;
            break;
        }
    }
    if (!hit) return;

    // Reject hits on surfaces that face away from the ray (we found a back side).
    vec3 Nh = decodeN(texture(uMat, huv).rgb);
    float facing = dot(Nh, R);
    if (facing >= -0.02) return;
    // Refine against the hit texel's actual plane, then verify the projected
    // intersection still belongs to it. This rejects false silhouette hits.
    vec2 hitUV = (floor(huv * uResolution) + 0.5) / uResolution;
    vec3 surface = viewPos(hitUV, texture(uDepth, hitUV).r);
    vec3 origin = P + Nv * (0.03 + 0.0015 * zr0);
    float hitT = dot(Nh, surface - origin) / facing;
    if (hitT <= 0.0 || hitT > uMaxDist) return;
    vec3 hitP = origin + R * hitT;
    vec4 hitClip = uProj * vec4(hitP, 1.0);
    huv = hitClip.xy / hitClip.w * 0.5 + 0.5;
    if (any(lessThan(huv, vec2(0.0))) || any(greaterThan(huv, vec2(1.0)))) return;
    vec2 checkUV = (floor(huv * uResolution) + 0.5) / uResolution;
    float checkD = texture(uDepth, checkUV).r;
    if (depthSky(checkD)) return;
    vec3 checkP = viewPos(checkUV, checkD);
    float error = abs(dot(Nh, checkP - hitP));
    float valid = 1.0 - smoothstep(0.03, max(0.12, -hitP.z * 0.002), error);
    t = hitT;

    vec2  e    = min(huv, 1.0 - huv);
    float edge = smoothstep(0.0, 0.08, min(e.x, e.y));
    float dist = 1.0 - smoothstep(0.55, 1.0, t / uMaxDist);
    float back = 1.0 - smoothstep(0.15, 0.55, R.z);
    float conf = edge * dist * back * valid;

    // Soften the reflection: rougher surfaces and longer rays blur more.
    float rad = (0.8 + 5.0 * (1.0 - refl)) * (0.4 + 0.6 * smoothstep(0.0, 0.5, t / uMaxDist));
    vec2  px  = rad / uResolution;
    vec3 rc = texture(uScene, huv).rgb * 0.4
            + texture(uScene, huv + vec2( px.x,  px.y)).rgb * 0.15
            + texture(uScene, huv + vec2(-px.x,  px.y)).rgb * 0.15
            + texture(uScene, huv + vec2( px.x, -px.y)).rgb * 0.15
            + texture(uScene, huv + vec2(-px.x, -px.y)).rgb * 0.15;

    float a = clamp(F * conf, 0.0, 1.0);
    FragColor = vec4(mix(scene, min(rc, vec3(6.0)), a), 1.0);
}
"""
