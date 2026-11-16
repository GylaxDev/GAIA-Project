"""Exactly shared Gerstner/Stokes surface evaluation for water vertices/normals."""
from ..world.water_waves import GRAVITY_WAVES, DETAIL_WAVES, WAVE_CHOP, WAVE_HARMONIC, WAVE_MESH_FLOOR


def _table(name, waves):
    rows = ',\n    '.join('vec4(' + ', '.join(f'{x:.9f}' for x in row) + ')' for row in waves)
    return f'const vec4 {name}[{len(waves)}] = vec4[{len(waves)}](\n    {rows});\n'


WATER_WAVES = (_table('gravityBands', GRAVITY_WAVES) + _table('detailBands', DETAIL_WAVES)
               + f'const int gravityCount = {len(GRAVITY_WAVES)}, detailCount = {len(DETAIL_WAVES)};\n'
               + f'const float waterChop = {WAVE_CHOP:.9f}, waterHarmonic = {WAVE_HARMONIC:.9f};\n'
               + f'const float waterMeshFloor = {WAVE_MESH_FLOOR:.9f};\n' + """
float waterSea() {
    return clamp(.68 + length(uWind) * .12 + uStorm * .20, .70, 1.6);
}
vec2 waterWindDirection() {
    float speed = length(uWind);
    return speed > .001 ? uWind / speed : vec2(.83646265, .54802394);
}
float waterAlignment(vec2 direction) {
    float preferred = dot(direction, vec2(.83646265, .54802394));
    float aligned = dot(direction, waterWindDirection());
    // Wind direction is undefined in a lull. Fade its energy weighting in rather
    // than letting a near-zero gust abruptly change an existing crest's height.
    return mix(preferred * preferred, aligned * aligned, smoothstep(.35, 2., length(uWind)));
}
void waterGravity(vec2 p, float time, float spacing, float footprint, float scale,
                  out vec3 displacement, out vec2 slope, out float crest, out float variance) {
    displacement = vec3(0.); slope = vec2(0.); variance = 0.;
    crest = 0.;
    if (uWaterWaves < .5 || scale < .001) return;
    vec2 dh = vec2(0.);
    mat2 J = mat2(1.);
    float energy = 0.;
    float filterSpacing = spacing > .001 ? max(spacing, waterMeshFloor) : 0.;
    for (int i = 0; i < gravityCount; ++i) {
        vec4 band = gravityBands[i];
        float wavelength = band.x, k = 6.283185307 / wavelength;
        float omega = sqrt(9.81 * k);
        vec2 direction = vec2(cos(band.z), sin(band.z));
        vec2 crossDir = vec2(-direction.y, direction.x);
        float meshFilter = 1. - smoothstep(wavelength * .20, wavelength * .45, filterSpacing);
        float resolved = 1. - smoothstep(wavelength * .12, wavelength * .45, footprint);
        float alignment = waterAlignment(direction);
        float base = band.y * waterSea() * scale * uWaterWaves
                   * (.45 + .55 * alignment) * meshFilter;
        // Analytic cross-wave phase modulation shortens the otherwise infinite
        // parallel crests. Its derivative is included in both the Jacobian and N.
        float crossPhase = k * .28 * dot(crossDir, p) - omega * .16 * time + band.w * 1.7;
        float phi = k * dot(direction, p) - omega * time + band.w + .85 * sin(crossPhase);
        vec2 phaseGrad = k * (direction + crossDir * .238 * cos(crossPhase));
        float s = sin(phi), c = cos(phi);
        float groupPhase = k * .11 * dot(crossDir, p) - omega * .07 * time + band.w * 2.1;
        float a = base * (.78 + .22 * sin(groupPhase));
        vec2 da = crossDir * (base * .22 * k * .11 * cos(groupPhase));
        variance += .5 * (a * k) * (a * k) * (1. + waterHarmonic * waterHarmonic) * (1. - resolved * resolved);
        a *= resolved;
        da *= resolved;
        displacement.xy += direction * (waterChop * a * c);
        // A zero-mean second harmonic narrows crests without sharp sawtooth waves.
        float height = s + waterHarmonic * (s * s - .5);
        displacement.z += a * height;
        dh += phaseGrad * (a * c * (1. + 2. * waterHarmonic * s)) + da * height;
        vec2 horizontal = waterChop * (da * c - phaseGrad * (a * s));
        J[0] += direction * horizontal.x;
        J[1] += direction * horizontal.y;
        energy += a;
    }
    float detJ = max(determinant(J), .25);
    slope = vec2(J[1].y * dh.x - J[0].y * dh.y,
                 J[0].x * dh.y - J[1].x * dh.x) / detJ;
    crest = smoothstep(.05, .70, displacement.z / max(energy, .001));
}
void waterDetail(vec2 p, float time, float footprint, float scale,
                 inout vec2 slope, inout float variance, out float whitecaps) {
    whitecaps = 0.;
    if (scale < .001) return;
    for (int i = 0; i < detailCount; ++i) {
        vec4 band = detailBands[i];
        float k = 6.283185307 / band.x;
        vec2 direction = vec2(cos(band.z), sin(band.z));
        vec2 crossDir = vec2(-direction.y, direction.x);
        float alignment = waterAlignment(direction);
        float a = band.y * waterSea() * scale * (.55 + .45 * alignment);
        float resolved = 1. - smoothstep(band.x * .12, band.x * .50, footprint);
        float omega = sqrt(9.81 * k + .000074 * k * k * k);
        float crossPhase = k * .34 * dot(crossDir, p) - omega * .21 * time + band.w;
        float phi = k * dot(direction, p) - omega * time + band.w + .72 * sin(crossPhase);
        float s = sin(phi), c = cos(phi);
        float harmonic = waterHarmonic * (1. - smoothstep(band.x * .06, band.x * .22, footprint));
        slope += (direction + crossDir * .2448 * cos(crossPhase)) * a * c * (1. + 2. * harmonic * s) * resolved;
        variance += .5 * a * a * (1. + waterHarmonic * waterHarmonic) * (1. - resolved * resolved);
        if (i < 3) {
            // Thin ribbons follow actual short-wave crests, not round noise islands.
            // Subpixel ribbons converge to coverage instead of flickering on/off.
            // Keep froth width in metres, not a fraction of the wavelength: even
            // a sharp sine-power mask made metre-wide white tape on long crests.
            float crestDistance = abs(c) / k;
            float ribbon = (1. - smoothstep(.035, .15, crestDistance)) * smoothstep(.85, .98, s);
            float ribbonFilter = 1. - smoothstep(.035, .18, footprint);
            ribbon = mix(.185 / band.x, ribbon, ribbonFilter);
            float packet = smoothstep(.20, .85, .5 + .5 * sin(crossPhase * .73 + band.w));
            ribbon *= smoothstep(.025, .10, a) * packet;
            whitecaps = max(whitecaps, ribbon * .8);
        }
    }
}
void waterDetail(vec2 p, float time, float footprint, float scale,
                 inout vec2 slope, inout float variance) {
    float whitecaps;
    waterDetail(p, time, footprint, scale, slope, variance, whitecaps);
}
""")
