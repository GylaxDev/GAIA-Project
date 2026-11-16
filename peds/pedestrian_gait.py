"""Gait maths shared by the CPU (phase rate, stride) and mirrored in shaders/pedestrian.py.

Legs are solved with analytic two-bone IK so planted feet stay planted: each foot follows a
stance line (constant ground speed relative to the pelvis), a heel-strike -> flat -> toe-off
roll, then a lifted swing. The pelvis height comes from what the stance leg can reach, giving
the natural bob; sway, tilt and rotation follow the stance leg.  Model units (1.78 m figure).
"""
from __future__ import annotations

import math

import numpy as np

from .pedestrian_models import HEIGHT, HIP_Y, HIP_Z, KNEE_Z, ANKLE_Z

L1 = HIP_Z - KNEE_Z            # thigh
L2 = KNEE_Z - ANKLE_Z          # shin
LMAX = .995 * (L1 + L2)
S_CAP = .50 * (L1 + L2)        # half stance excursion the legs can sweep comfortably


def smoothstep(a, b, x):
    t = min(max((x - a) / (b - a), 0.), 1.)
    return t * t * (3. - 2. * t)


def speed_factor(speed):
    return min(max((speed - 1.2) / 4.3, 0.), 1.)


def duty(sf):
    """Fraction of the cycle a foot is on the ground: .62 walking .. .25 sprinting."""
    return .62 - .37 * sf ** .8


def cadence(speed, height):
    base = 1.0 + .55 * speed if speed <= 2.6 else 2.43 + .18 * (speed - 2.6)
    return base * math.sqrt(HEIGHT / height)


def gait(speed, height, width=1.0):
    """-> (S_model, speed_factor, phase_rate) for a walking speed in m/s.

    S_model is the half stance excursion in model units (the shader scales it by the
    body); phase_rate is rad/s for the full two-step cycle.  The rate keeps the stance
    foot's ground speed equal to the body speed (no foot sliding).
    """
    if speed < .12:
        return 0., 0., 0.
    sf = speed_factor(speed)
    k = height / HEIGHT * width                     # model -> world scale along the walking direction
    d = duty(sf)
    s_world = min(d * speed / cadence(speed, height), S_CAP * k)
    return s_world / k, sf, math.pi * d * speed / s_world


def foot_target(u, side, S, sf, idle_shift=0.):
    """Ankle target (x forward, y lateral, z height) and foot pitch psi for cycle position u.

    idle_shift in [-1, 1] is the standing weight shift (positive = weight on the left leg);
    the unloaded foot rests a little forward and turned out.
    """
    on = smoothstep(.05, .25, S)
    d = duty(sf)
    front, rear = .85 * S, 1.15 * S
    lift = (.055 + .16 * sf) * on
    if u < d:
        t = u / d
        x = front + (-rear - front) * t
        z = 0.
        psi = (.28 * (1. - smoothstep(0., .22, t)) - .75 * smoothstep(.55, 1., t)) * on
    else:
        t = (u - d) / (1. - d)
        x = -rear + (front + rear) * smoothstep(0., 1., t)
        z = lift * math.sin(math.pi * t ** .85)
        psi = (-.75 + 1.05 * smoothstep(0., .85, t)) * on
    unload = min(max(-side * idle_shift, 0.), 1.) * (1. - on)
    x += .055 * unload
    roll = ANKLE_Z * math.cos(psi) + (.07 * math.sin(psi) if psi > 0 else .17 * math.sin(-psi))
    return x, side * (HIP_Y + .012 + .012 * unload), roll + z, psi


def pelvis(phase, S, sf, idle_sway=0.):
    """-> (sway, yaw, roll, height) of the pelvis for a phase. idle_sway is metres of standing sway."""
    on = smoothstep(.05, .25, S)
    k = min(S / S_CAP, 1.)
    u0 = (phase / (2 * math.pi)) % 1.
    shift = math.cos(2 * math.pi * (u0 - .31))
    sway = .028 * k * shift + idle_sway * (1. - on)
    yaw = -.075 * k * math.cos(2 * math.pi * u0)
    roll = .030 * k * shift
    reach = []
    for side, u in ((1., u0), (-1., (u0 + .5) % 1.)):
        x, y, z, _ = foot_target(u, side, S, sf, idle_sway / .03)
        dyl = y - (side * HIP_Y + sway)
        reach.append(z + math.sqrt(max(LMAX ** 2 - x * x - dyl * dyl, .01)))
    h = min(HIP_Z - .008, min(reach))
    return sway, yaw, roll, max(h, HIP_Z - .16)


def leg_solve(hip, target):
    """Two-bone IK. hip/target are (x, y, z) -> (thigh pitch, knee flexion, roll, reach_ratio)."""
    ax, ay, az = target[0] - hip[0], target[1] - hip[1], hip[2] - target[2]
    roll = math.atan2(ay, az)
    dy = math.hypot(az, ay)
    d = math.hypot(ax, dy)
    dc = min(max(d, .12), LMAX)
    cg = min(max((L1 * L1 + dc * dc - L2 * L2) / (2 * L1 * dc), -1.), 1.)
    ck = min(max((L1 * L1 + L2 * L2 - dc * dc) / (2 * L1 * L2), -1.), 1.)
    gamma, kappa = math.acos(cg), math.acos(ck)
    alpha = math.atan2(ax, dy)
    return alpha + gamma, math.pi - kappa, roll, d / LMAX
