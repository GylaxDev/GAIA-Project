"""Continuous displaced water, absorbing refraction and sunlit subsurface crests."""
from .depth import SCENE_DEPTH
from .env import ENV_SKY
from .hdr import HDR_CONV, HDR_OUT
from .rain_surface import RAIN_EXPOSURE, RAIN_RIPPLES
from .shadow import SHADOW_PCF
from .water_ripples import WATER_RIPPLES
from .water_waves import WATER_WAVES

WATER_VS = ("""
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=4) in vec3 aStyle;
uniform mat4 uView, uProj;
uniform vec3 uOffset, uCamPos;
uniform vec2 uScale;
uniform vec2 uWind;
uniform float uTime, uStorm, uWaterWaves;
out vec3 vWorld, vShadowRelative;
out vec3 vRest;
out vec2 vWaveParams;
""" + WATER_WAVES + """
void main() {
    vRest = vec3(aPos.xy * uScale, aPos.z) + uOffset;
    vWaveParams = aStyle.xy;
    vec3 displacement; vec2 slope; float crest, variance;
    waterGravity(vRest.xy, uTime, aStyle.y, 0., aStyle.x, displacement, slope, crest, variance);
    vec3 p = vRest + displacement;
    vWorld = p;
    vShadowRelative = p - uCamPos;
    // Camera-relative projection avoids cancellation between tile positions and
    // the view translation. The wave phase itself always stays in world metres.
    gl_Position = uProj * vec4(mat3(uView) * vShadowRelative, 1.);
}
""")

WATER_FS = ("""
#version 330 core
in vec3 vWorld;
in vec3 vRest;
in vec2 vWaveParams;
uniform vec3 uCamPos, uSunDir, uFogColor, uSunColor, uAmbColor;
uniform float uFogNear, uFogFar, uTime, uNightFactor, uExposure, uRain, uWeatherTime;
uniform sampler2D uDepthMap, uRefractionMap;
uniform float uRefractionOn;
uniform float uWaterWaves, uWaterSSS;
uniform vec2 uResolution;
uniform mat4 uView, uProj, uInvViewProj;
out vec4 FragColor;
""" + ENV_SKY + HDR_CONV + HDR_OUT + SCENE_DEPTH + SHADOW_PCF + RAIN_EXPOSURE + RAIN_RIPPLES + WATER_WAVES + WATER_RIPPLES + """

vec3 waterWorld(vec2 uv, float depth) {
    vec4 p = uInvViewProj * vec4(uv * 2. - 1., depthNdc(depth), 1.);
    return p.xyz / p.w;
}
vec3 waterScene(vec2 uv) {
    vec3 c = max(texture(uRefractionMap, uv).rgb, vec3(0.));
    return uHdrOut > .5 ? c / max(uExposure, 1e-4)
                       : acesInv(pow(c, vec3(2.2))) / max(uExposure, 1e-4);
}
vec3 waterSpecular(vec3 N, vec3 V, vec3 L, float rough) {
    vec3 hv = L + V;
    vec3 H = hv / max(length(hv), 1e-5);
    float NoL = max(dot(N, L), 0.);
    float NoV = max(dot(N, V), .001);
    float NoH = max(dot(N, H), 0.);
    float VoH = max(dot(V, H), 0.);
    float a = rough * rough, a2 = a * a;
    float d = NoH * NoH * (a2 - 1.) + 1.;
    float D = a2 / max(3.14159265 * d * d, 1e-8);
    float k = (rough + 1.) * (rough + 1.) * .125;
    float G = (NoV / (NoV * (1. - k) + k)) * (NoL / max(NoL * (1. - k) + k, .001));
    float F = .0204 + .9796 * pow(1. - VoH, 5.);
    return vec3(min(D * G * F / max(4. * NoV * NoL, .001), 8.)) * NoL;
}

vec3 waterEnvironment(vec3 R, vec3 L, vec3 neighbouringWater) {
    // A downward ray sees neighbouring water, not a grey city/ground hemisphere.
    // This bounded multiple-reflection fallback supplies no sky below the horizon.
    return mix(neighbouringWater, envSky(R, L), smoothstep(-.045, .025, R.z));
}

vec3 waterCrestScatter(vec3 N, vec3 V, vec3 L, float crest, float shadow) {
    if (uWaterSSS < .5) return vec3(0.);
    // Forward single-scattering approximation through a thinner illuminated
    // crest. It is incident-light dependent, shadowed, and never emissive/cyan
    // albedo painted over the whole surface. Fresnel still owns the interface.
    vec3 bentLight = normalize(L + N * .22);
    float forward = pow(max(dot(-V, bentLight), 0.), 4.);
    float sunUp = smoothstep(0., .18, L.z) * (1. - uNightFactor);
    float thickness = mix(3., .45, crest);
    vec3 transmitted = exp(-vec3(.46, .31, .27) * thickness);
    return uSunColor * vec3(.05, .32, .24) * transmitted
         * (uWaterSSS * .24 * crest * forward * shadow * sunUp);
}

void main() {
    vec2 uv = gl_FragCoord.xy / uResolution;
    vec3 toEye = uCamPos - vWorld;
    float dist = max(length(toEye), .001);
    vec3 V = toEye / dist, L = normalize(uSunDir);
    float footprint = max(length(dFdx(vShadowRelative.xy)), length(dFdy(vShadowRelative.xy)));
    float windSpeed = length(uWind);
    vec3 displacement; float variance, crest; vec2 slope;
    waterGravity(vRest.xy, uTime, vWaveParams.y, footprint, vWaveParams.x,
                 displacement, slope, crest, variance);
    float whitecaps;
    waterDetail(vRest.xy, uTime, footprint, vWaveParams.x, slope, variance, whitecaps);
    float rippleGrain;
    waterRipples(vRest.xy, uTime, dFdx(vShadowRelative.xy), dFdy(vShadowRelative.xy),
                 vWaveParams.x, slope, variance, rippleGrain);
    // The cached rain roof stores the mean water datum. Testing a trough against
    // that plane would mistakenly treat the water itself as a roof.
    float exposed = rainExposure(vRest);
    if (uRain > .01)
        slope += rainRipples(vWorld.xy, uWeatherTime, footprint) * uRain * exposed;
    vec3 N = normalize(vec3(-slope, 1.));
    float NoV = clamp(dot(N, V), 0., 1.);
    float fres = .0204 + .9796 * pow(1. - NoV, 5.);  // IOR 1.333, no turquoise paint
    float sf = shadowPCF(vWorld, N, L);

    float sceneDepth = texture(uDepthMap, uv).r;
    bool hasScene = !depthSky(sceneDepth);
    float opticalDepth = 30., waterDepth = 25.;
    vec3 bed = vec3(.075, .073, .064) * (uAmbColor * .65 + uSunColor * .318 * max(L.z, 0.) * sf);
    if (hasScene) {
        vec3 bottom = waterWorld(uv, sceneDepth);
        opticalDepth = clamp(dot(bottom - vWorld, -V), 0., 80.);
        waterDepth = clamp(vWorld.z - bottom.z, 0., 80.);
        if (uRefractionOn > .5) {
            vec2 bend = (uView * vec4(N.xy, 0., 0.)).xy * min(waterDepth, 2.) * 8. / uResolution;
            vec2 refractUV = clamp(uv + bend, vec2(.5) / uResolution, 1. - vec2(.5) / uResolution);
            float refractDepth = texture(uDepthMap, refractUV).r;
            // Never bend a dry bank/building into the water as a fake seabed.
            if (!depthSky(refractDepth) && waterWorld(refractUV, refractDepth).z <= vWorld.z + .02)
                bed = waterScene(refractUV);
            else bed = waterScene(uv);
        }
    }

    // Representative slightly turbid urban/coastal water. Beer-Lambert attenuation
    // follows the actual ray length; shallows transmit the real bed, not cyan/sand paint.
    vec3 transmission = exp(-vec3(.46, .31, .27) * opticalDepth);
    vec3 scatter = vec3(.012, .032, .046)
                 * (uAmbColor * .75 + uSunColor * .318 * max(L.z, 0.) * sf);
    vec3 body = bed * transmission + scatter * (1. - transmission);
    vec3 neighbouringWater = body * .75;
    body += waterCrestScatter(N, V, L, crest, sf);
    // Resolved ripples break the solar path into glints; only UNRESOLVED energy
    // broadens the highlight. A permanently broad lobe made every crest look soft.
    float rough = clamp(sqrt(.0036 + variance * .55) + uRain * .025, .06, .26);
    vec3 reflected = reflect(-V, N);
    vec3 reflection = waterEnvironment(reflected, L, neighbouringWater);
    if (abs(reflected.z) < rough * 2.) {
        // Filter the environment across the unresolved microfacet cone near the
        // horizon instead of turning each tiny trough into a hard black stripe.
        vec3 upper = waterEnvironment(normalize(reflected + vec3(0., 0., rough)), L, neighbouringWater);
        vec3 lower = waterEnvironment(normalize(reflected - vec3(0., 0., rough)), L, neighbouringWater);
        reflection = reflection * .4 + (upper + lower) * .3;
    }
    vec3 col = mix(body, reflection, fres);
    col += waterSpecular(N, V, L, rough) * uSunColor * sf;

    // Broken whitecap lace sits on narrow wind-wave ridges. Pools/canals remain
    // quiet, and shoreline froth still requires a genuinely shallow seabed.
    float foam = 0.;
    float breaking = smoothstep(3.2, 7.5, windSpeed) * smoothstep(.35, .85, vWaveParams.x);
    float shallow = hasScene ? 1. - smoothstep(.08, .45, waterDepth) : 0.;
    if (breaking > .001 || shallow > .001) {
        vec2 foamPos = vRest.xy - uWindOffset * .13;
        float coarse = cloudNoise(foamPos * .8);
        float fineFilter = 1. - smoothstep(.10, .65, footprint * 3.1);
        float fine = mix(.5, cloudNoise(foamPos * 3.1), fineFilter);
        float grain = coarse * .25 + fine * .40 + rippleGrain * .35;
        float caps = whitecaps * smoothstep(.45, .58, grain) * breaking;
        float shore = shallow * smoothstep(.52, .82, grain) * (.06 + .16 * breaking);
        foam = clamp(caps * 1.6 + shore, 0., .90);
    }
    vec3 foamLight = vec3(.62, .64, .65) * (uAmbColor * .70 + uSunColor * .318 * max(L.z, 0.) * sf);
    col = mix(col, foamLight, foam);
    float f = cityFog(dist, vWorld.z, uFogNear, uFogFar);
    FragColor = vec4(mix(sceneOut(max(col, vec3(0.))), displayToScene(uFogColor), f), 1.);
}
""")
