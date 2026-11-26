OVERLAY_VS = """
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
uniform vec2 uScreen;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos.x / uScreen.x * 2.0 - 1.0, 1.0 - aPos.y / uScreen.y * 2.0, 0.0, 1.0);
}
"""

OVERLAY_FS = """
#version 330 core
in vec2 vUV;
uniform int uMode;
uniform vec4 uColor;
uniform float uInvert;
uniform sampler2D uTex;
out vec4 FragColor;
void main() {
    if (uMode == 0) { FragColor = uColor; return; }
    vec4 t = texture(uTex, vUV);
    if (uMode == 1) { FragColor = vec4(uColor.rgb, uColor.a * t.a); return; }
    if (uInvert < 0.5) { FragColor = vec4(t.rgb, uColor.a); return; }
    float lum = dot(t.rgb, vec3(0.299, 0.587, 0.114));
    float v = 0.22 + 0.78 * smoothstep(0.0, 0.45, 1.0 - lum);
    v *= 1.0 - 0.70 * smoothstep(0.06, 0.18, t.b - t.r);
    vec3 col = mix(vec3(0.040, 0.046, 0.066), vec3(0.50, 0.62, 0.80), v * 0.85);
    FragColor = vec4(col, uColor.a);
}
"""