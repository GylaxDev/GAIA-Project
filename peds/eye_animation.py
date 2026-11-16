"""Deterministic facial motion, shared with the GPU and offline pose previews."""
import math


def _hash(x):
    # Quantized integer avalanche matches GLSL exactly. The former sin hash
    # amplified float32 trig error into different CPU/GPU blink schedules.
    h = math.floor(x * 64. + .5) & 0xffffffff
    h = ((h ^ (h >> 16)) * 0x7feb352d) & 0xffffffff
    h = ((h ^ (h >> 15)) * 0x846ca68b) & 0xffffffff
    h ^= h >> 16
    return (h >> 8) / 16777216.


def _smooth(a, b, x):
    t = max(0., min(1., (x - a) / (b - a)))
    return t * t * (3. - 2. * t)


def blink_closure(t, seed):
    period = 3.2 + 2.8 * _hash(seed * 19.7 + 11.)
    clock = t + seed * period * 13.
    cycle = math.floor(clock / period)
    local = clock % period
    pulse = lambda x: _smooth(0., .075, x) * (1. - _smooth(.11, .26, x))
    double = _hash(cycle + seed * 71.) < .16
    return max(pulse(local), pulse(local - .34) if double else 0.)


def gaze_angles(t, seed, head_yaw=0., phone=False):
    interval = 1.3 + .8 * _hash(seed * 23. + 7.)
    clock = (t + seed * 17.) / interval
    cycle = math.floor(clock)
    blend = _smooth(0., .11, (clock - cycle) * interval)
    target = lambda c: ((_hash(c + seed * 31.) - .5) * .22,
                        (_hash(c * 1.73 + seed * 47.) - .5) * .12)
    a, b = target(cycle - 1.), target(cycle)
    yaw, pitch = [a[k] + (b[k] - a[k]) * blend for k in range(2)]
    return max(-.28, min(.28, head_yaw * .22 + yaw)), pitch - (.12 if phone else 0.)


EYE_ANIM_GLSL = """
float facialHash(float x) {
    uint h = uint(int(floor(x * 64.0 + 0.5)));
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h ^= h >> 16u;
    return float(h >> 8u) / 16777216.0;
}
float blinkPulse(float x) { return sstep(0.0, 0.075, x) * (1.0 - sstep(0.11, 0.26, x)); }
float blinkClosure(float t, float seed) {
    float period = 3.2 + 2.8 * facialHash(seed * 19.7 + 11.0);
    float clock = t + seed * period * 13.0;
    float cycle = floor(clock / period), local = mod(clock, period);
    float second = facialHash(cycle + seed * 71.0) < 0.16 ? blinkPulse(local - 0.34) : 0.0;
    return max(blinkPulse(local), second);
}
vec2 gazeTarget(float cycle, float seed) {
    return vec2((facialHash(cycle + seed * 31.0) - 0.5) * 0.22,
                (facialHash(cycle * 1.73 + seed * 47.0) - 0.5) * 0.12);
}
vec2 gazeAngles(float t, float seed, float headYaw, bool phone) {
    float interval = 1.3 + 0.8 * facialHash(seed * 23.0 + 7.0);
    float clock = (t + seed * 17.0) / interval, cycle = floor(clock);
    float blend = sstep(0.0, 0.11, fract(clock) * interval);
    vec2 gaze = mix(gazeTarget(cycle - 1.0, seed), gazeTarget(cycle, seed), blend);
    gaze.x = clamp(gaze.x + headYaw * 0.22, -0.28, 0.28);
    gaze.y -= phone ? 0.12 : 0.0;
    return gaze;
}
"""
