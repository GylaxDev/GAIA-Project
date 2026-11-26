"""Equatorial Milky Way and phase-correct Moon shared by sky / reflections."""

CELESTIAL_SKY = """
#ifndef GAIA_CELESTIAL_SKY
#define GAIA_CELESTIAL_SKY
uniform mat3 uWorldToEquatorial;
uniform vec3 uSkyMoonDir;
uniform float uMoonIntensity, uMoonPhase, uMoonRadius, uMoonBrightness, uMilkyWayVisibility;

float celestialHash(vec2 p) {
    p = fract(p * vec2(127.13, 311.79));
    p += dot(p, p + 37.17);
    return fract(p.x * p.y);
}
float celestialNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p), u = f * f * (3. - 2. * f);
    return mix(mix(celestialHash(i), celestialHash(i + vec2(1., 0.)), u.x),
               mix(celestialHash(i + vec2(0., 1.)), celestialHash(i + vec2(1., 1.)), u.x), u.y);
}
vec3 galacticSky(vec3 dir) {
    if (uMilkyWayVisibility <= .001 || dir.z <= 0.) return vec3(0.);
    vec3 eq = normalize(uWorldToEquatorial * dir);
    // IAU J2000 equatorial -> Galactic rotation, with the actual Galactic pole.
    vec3 gal = vec3(dot(eq, vec3(-.0548755604, -.8734370902, -.4838350155)),
                    dot(eq, vec3(.4941094279, -.4448296300, .7469822445)),
                    dot(eq, vec3(-.8676661490, -.1980763734, .4559837762)));
    float longitude = atan(gal.y, gal.x), latitude = asin(clamp(gal.z, -1., 1.));
    float band = exp(-latitude * latitude * 110.);
    float bulge = exp(-longitude * longitude * 2.5 - latitude * latitude * 32.);
    vec2 p = vec2(gal.x, gal.y) * 17. + vec2(gal.z * 23.);
    float grain = .65 * celestialNoise(p) + .35 * celestialNoise(p * 2.7 + 8.3);
    float dust = exp(-pow((latitude + .018 * sin(longitude * 5.)) / .023, 2.));
    vec3 light = emissionNits(vec3(.70, .76, .91), .004 * band)
               + emissionNits(vec3(.91, .81, .64), .014 * bulge);
    return light * (.42 + .78 * grain) * (1. - dust * .62) * uMilkyWayVisibility;
}
// discGain 1 = the sky pass, 0 = a reflected skydome (lunarSkyGain).  Reflections are sampled per
// pixel through an orange-peel / flake normal, so a hard-edged disc there bursts into
// single-pixel sparkle wherever the mirrored direction crosses it; the smooth halo is
// all a reflected moon can contribute anyway.
vec3 lunarSkyGain(vec3 dir, vec3 sunDir, float discGain) {
    if (uMoonIntensity <= .001 || uMoonRadius <= 0. || uSkyMoonDir.z < -.025) return vec3(0.);
    vec3 moonDir = normalize(uSkyMoonDir);
    vec3 pole = transpose(uWorldToEquatorial) * vec3(0., 0., 1.);
    vec3 east = cross(pole, moonDir);
    if (length(east) < .01) east = cross(vec3(0., 1., 0.), moonDir);
    east = normalize(east);
    vec3 north = normalize(cross(moonDir, east));
    vec2 p = vec2(dot(dir, east), dot(dir, north)) / uMoonRadius;
    float radius2 = dot(p, p);
    float aa = max(fwidth(radius2), .005);
    float disc = (1. - smoothstep(1. - aa, 1. + aa, radius2))
               * step(0., dot(dir, moonDir)) * discGain;
    float mu = max(dot(dir, moonDir), 0.);
    float halo = pow(mu, 850.) * nits(.035) * uMoonBrightness;
    vec3 surface = east * p.x + north * p.y - moonDir * sqrt(max(1. - radius2, 0.));
    float lit = max(dot(surface, normalize(sunDir)), 0.);
    float phaseAngle = acos(clamp(2. * uMoonPhase - 1., -1., 1.));
    float lambert = (sin(phaseAngle) + (3.14159265 - phaseAngle) * cos(phaseAngle)) / 3.14159265;
    float phaseGain = clamp(uMoonBrightness / max(lambert, .0001), 0., 2.);
    // Procedural maria/crater relief, not a surveyed lunar texture.
    float maria = .67 * celestialNoise(p * 3.4 + 2.7) + .33 * celestialNoise(p * 8.2 + 11.3);
    vec2 craterCell = floor(p * 11.), cf = fract(p * 11.) - .5;
    float crater = exp(-pow((length(cf) - .22) / .075, 2.)) * celestialHash(craterCell);
    float albedo = .70 + .28 * smoothstep(.26, .73, maria) + crater * .06;
    float earthshine = .0005 * (1. - uMoonPhase);
    float altitude = max(degrees(asin(clamp(moonDir.z, -1., 1.))), 0.);
    float airMass = 1. / max(moonDir.z + .50572 * pow(altitude + 6.07995, -1.6364), .025);
    vec3 attenuation = exp(-vec3(.10, .15, .25) * max(airMass - 1., 0.) * max(uAerosolDensity, .4));
    return vec3(.94, .96, 1.) * attenuation * uMoonIntensity
         * (disc * nits(2500.) * albedo * (lit * phaseGain + earthshine) + halo);
}
vec3 lunarSky(vec3 dir, vec3 sunDir) {
    return lunarSkyGain(dir, sunDir, 1.0);
}
vec3 celestialSky(vec3 dir, vec3 sunDir) {
    return galacticSky(dir) + lunarSky(dir, sunDir);
}
// Same sky without resolvable point sources, for mirror-direction lookups.
vec3 celestialSkyReflected(vec3 dir, vec3 sunDir) {
    return galacticSky(dir) + lunarSkyGain(dir, sunDir, 0.0);
}
#endif
"""
