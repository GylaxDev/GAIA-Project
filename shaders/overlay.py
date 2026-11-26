OVERLAY_VS = """
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
layout(location=2) in vec4 aColor;
uniform vec2 uScreen;
out vec2 vUV;
out vec2 vPos;
out vec4 vColor;
void main() {
    vUV = aUV;
    vPos = aPos;
    vColor = aColor;
    gl_Position = vec4(aPos.x / uScreen.x * 2.0 - 1.0, 1.0 - aPos.y / uScreen.y * 2.0, 0.0, 1.0);
}
"""

OVERLAY_BLUR_FS = """
#version 330 core
in vec2 vUV;
uniform sampler2D uTex;
uniform vec2 uDir;      // one half-res texel step (scaled) in texture units
uniform vec2 uScale;    // source uv extent (captured region may be smaller than the texture)
uniform float uFlip;    // 1 on the first pass: the framebuffer copy is stored bottom-up
out vec4 FragColor;

void main() {
    vec2 uv = vec2(vUV.x, uFlip > 0.5 ? 1.0 - vUV.y : vUV.y) * uScale;
    vec3 c = texture(uTex, uv).rgb * 0.2270270270;
    c += (texture(uTex, uv + uDir).rgb + texture(uTex, uv - uDir).rgb) * 0.3162162162;
    c += (texture(uTex, uv + uDir * 2.3846).rgb + texture(uTex, uv - uDir * 2.3846).rgb) * 0.0702702703;
    FragColor = vec4(c, 1.0);
}
"""

OVERLAY_FS = """
#version 330 core
in vec2 vUV;
in vec2 vPos;
in vec4 vColor;
uniform int uMode;          // 0 solid, 1 glyph, 2 map tile, 3 vertical gradient,
                            // 4 rounded box (+ border), 5 soft shadow, 6 anti-aliased line,
                            // 7 rounded textured image, 8 coloured triangles
uniform vec4 uColor;        // fill (top) colour / text colour / line colour
uniform vec4 uColor2;       // fill (bottom) colour for gradients and boxes
uniform vec4 uBorder;       // box border colour
uniform vec2 uHalf;         // box half extents (line: x = half width)
uniform float uRad;         // box corner radius
uniform float uBw;          // box border width
uniform float uSoft;        // shadow blur radius
uniform float uInvert;      // tiles: 0 = plain OSM, 1 = Apple-Maps dark style
uniform vec4 uClip;         // clip centre (xy) and size (zw) for coloured triangles
uniform float uClipRad;     // corner radius of the clip box
uniform vec4 uUV;           // sub-rectangle of the texture drawn by mode 9 (u0, v0, u1, v1)
uniform sampler2D uTex;
out vec4 FragColor;

// Re-colour a standard OSM raster tile the way Apple Maps' dark map looks:
// deep indigo water, muted teal land, darker green for parks and forest, and
// light linework (the source's black labels/borders become soft light ink).
vec3 appleTile(vec3 c) {
    float lum   = dot(c, vec3(0.299, 0.587, 0.114));
    float water = smoothstep(0.03, 0.09, c.b - c.r) * step(0.70, c.b);
    float green = smoothstep(0.03, 0.10, c.g - c.r) * smoothstep(0.02, 0.06, c.g - c.b) * (1.0 - water);
    vec3 land = vec3(0.235, 0.385, 0.345);
    vec3 grn  = vec3(0.180, 0.345, 0.250);
    vec3 wcol = vec3(0.125, 0.165, 0.440);
    vec3 col = mix(land, grn, green);
    col = mix(col, wcol, water);
    float ink = smoothstep(0.80, 0.30, lum) * (1.0 - 0.6 * water);
    col = mix(col, vec3(0.80, 0.88, 0.86), ink * 0.85);
    col += smoothstep(0.96, 1.0, lum) * (1.0 - water) * 0.07;
    return col;
}

// signed distance to a rounded box centred on the origin
float sdBox(vec2 p, vec2 h, float r) {
    vec2 q = abs(p) - h + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

void main() {
    if (uMode == 0) { FragColor = uColor; return; }
    if (uMode == 3) { FragColor = mix(uColor, uColor2, clamp(vUV.y, 0.0, 1.0)); return; }
    if (uMode == 4) {
        // Analytic anti-aliasing: one device pixel is fwidth(d), so edges stay
        // exactly one pixel soft at any UI scale (no polygon-fan stair-steps).
        float d   = sdBox(vUV, uHalf, uRad);
        float px  = max(fwidth(d), 1e-4);
        float cov = clamp(0.5 - d / px, 0.0, 1.0);
        float inn = uBw > 0.0 ? clamp(0.5 - (d + uBw) / px, 0.0, 1.0) : cov;
        float t   = clamp(vUV.y / (2.0 * uHalf.y) + 0.5, 0.0, 1.0);
        vec4 fill = mix(uColor, uColor2, t);
        float ring = max(cov - inn, 0.0);
        float a = fill.a * inn + uBorder.a * ring;
        if (a <= 0.0005) discard;
        vec3 rgb = (fill.rgb * fill.a * inn + uBorder.rgb * uBorder.a * ring) / a;
        FragColor = vec4(rgb, a);
        return;
    }
    if (uMode == 5) {
        float d = sdBox(vUV, uHalf, uRad);
        float a = 1.0 - smoothstep(-uSoft * 0.35, uSoft, d);
        FragColor = vec4(uColor.rgb, uColor.a * a * a);
        return;
    }
    if (uMode == 6) {
        float d  = abs(vUV.x) - uHalf.x;
        float px = max(fwidth(vUV.x), 1e-4);
        float cov = clamp(0.5 - d / px, 0.0, 1.0);
        if (cov <= 0.002) discard;
        FragColor = vec4(uColor.rgb, uColor.a * cov);
        return;
    }
    if (uMode == 7) {
        // Cover art with an anti-aliased rounded mask.  vUV holds quad-local
        // pixel offsets, so the SDF edge matches the box primitive exactly.
        float d = sdBox(vUV, uHalf, uRad);
        float px = max(fwidth(d), 1e-4);
        float cov = clamp(0.5 - d / px, 0.0, 1.0);
        if (cov <= 0.002) discard;
        vec2 uv = clamp(vUV / uHalf * 0.5 + 0.5, 0.0, 1.0);
        vec4 t = texture(uTex, uv);
        FragColor = vec4(t.rgb, t.a * uColor.a * cov);
        return;
    }
    if (uMode == 9) {
        // A rounded sub-rectangle of the frame's backdrop copy, blurred here while
        // the panel is drawn.  Blurring inline keeps the GPU from chaining the
        // framebuffer copy into render-target passes every frame.
        float d = sdBox(vUV, uHalf, uRad);
        float px = max(fwidth(d), 1e-4);
        float cov = clamp(0.5 - d / px, 0.0, 1.0);
        if (cov <= 0.002) discard;
        vec2 t01 = clamp(vUV / uHalf * 0.5 + 0.5, 0.0, 1.0);
        vec2 uv = mix(uUV.xy, uUV.zw, t01);
        // One gaussian step per device pixel, so the frost reads the same at any
        // UI scale; uSoft is the blur radius in device pixels.
        vec2 span = uUV.zw - uUV.xy;
        vec2 blur = span * fwidth(t01) * uSoft;    // one blur radius, in per-pixel uv steps
        vec2 dx = vec2(blur.x, 0.0);
        vec2 dy = vec2(0.0, blur.y);
        // Weights sum to 1: a 13-tap cross with corners, a two-axis gaussian.
        vec3 c = texture(uTex, uv).rgb * 0.20;
        c += (texture(uTex, uv + dx).rgb + texture(uTex, uv - dx).rgb
              + texture(uTex, uv + dy).rgb + texture(uTex, uv - dy).rgb) * 0.12;
        c += (texture(uTex, uv + dx * 2.0).rgb + texture(uTex, uv - dx * 2.0).rgb
              + texture(uTex, uv + dy * 2.0).rgb + texture(uTex, uv - dy * 2.0).rgb) * 0.05;
        c += (texture(uTex, uv + dx + dy).rgb + texture(uTex, uv + dx - dy).rgb
              + texture(uTex, uv - dx + dy).rgb + texture(uTex, uv - dx - dy).rgb) * 0.03;
        FragColor = vec4(c, uColor.a * cov);
        return;
    }
    if (uMode == 8) {
        // Batched per-vertex coloured triangles clipped to the map card.
        vec2 local = vPos - uClip.xy + uClip.zw * 0.5;
        float d = sdBox(local - uClip.zw * 0.5, uClip.zw * 0.5, uClipRad);
        float px = max(fwidth(d), 1e-4);
        float cov = clamp(0.5 - d / px, 0.0, 1.0);
        if (cov <= 0.002) discard;
        FragColor = vec4(vColor.rgb, vColor.a * cov);
        return;
    }
    vec4 t = texture(uTex, vUV);
    if (uMode == 1) {
        // light-on-dark text looks thin with plain coverage; a mild curve restores weight
        FragColor = vec4(uColor.rgb, uColor.a * pow(t.a, 0.80));
        return;
    }
    if (uInvert < 0.5) { FragColor = vec4(t.rgb, uColor.a); return; }
    FragColor = vec4(appleTile(t.rgb), uColor.a);
}
"""
