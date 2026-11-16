"""Canonical, bounded wave spectrum and CPU reference for displacement contracts.

The shaders generate their constants from this table. Directions/phases do not
rotate with gusts: wind smoothly changes energy, never the world-space phase.
"""
from __future__ import annotations

import math
import numpy as np

# wavelength (m), amplitude (m), direction (radians), phase (radians).
# Most height energy belongs to wind waves, not a few enormous rounded swells.
# The existing lattice filters bands it cannot displace; it need not be enlarged.
GRAVITY_WAVES = ((62., .12, .58, .7), (38., .17, .24, 2.3),
                 (25., .18, .87, 4.9), (17., .18, -.06, 1.4),
                 (12., .145, .53, 3.8), (8.5, .09, 1.19, 5.6),
                 (6., .055, .22, 2.9), (4.3, .03, -.43, .3))
# wavelength (m), slope, direction, phase. Directional, non-harmonic wavelengths
# give the surface long broken ridges. The spectral ripple maps supply capillary
# wrinkles; evaluating those again analytically loses float32 phase precision at
# city coordinates and duplicates their slope energy.
DETAIL_WAVES = ((15.7, .090, .58, 2.1), (10.4, .085, .28, 5.7),
                (7.1, .080, .82, 3.3), (5.1, .075, -.19, .9),
                (3.65, .085, .51, 4.4), (2.7, .070, 1.14, 1.8),
                (1.95, .055, .16, 5.2), (1.41, .055, .74, 3.9),
                (1., .045, -.38, 2.7), (.71, .040, .42, 5.9))
WAVE_CHOP = .9
WAVE_HARMONIC = .28
# The z14/512-cell lattice is <= 4.78 m at the equator. A shared conservative
# cutoff prevents minute north/south tile-spacing differences changing wave energy.
WAVE_MESH_FLOOR = 5.


def _smoothstep(lo, hi, value):
    x = np.clip((np.asarray(value) - lo) / (hi - lo), 0., 1.)
    return x * x * (3. - 2. * x)


def sample_waves(points, time, wind=(1.8, .8), storm=0., scale=1., spacing=0., footprint=0.):
    """Return displacement, world normal, crest mask, variance (float64 reference)."""
    p = np.asarray(points, np.float64)
    wind = np.asarray(wind, np.float64)
    speed = float(np.linalg.norm(wind))
    wind_dir = wind / speed if speed > .001 else np.array([math.cos(.58), math.sin(.58)])
    preferred_wind = np.array([math.cos(.58), math.sin(.58)])
    directional = float(_smoothstep(.35, 2., speed))
    sea = float(np.clip(.68 + speed * .12 + storm * .20, .70, 1.6))
    shape = p.shape[:-1]
    filter_spacing = np.where(np.asarray(spacing) > .001, np.maximum(spacing, WAVE_MESH_FLOOR), 0.)
    disp, dh = np.zeros((*shape, 3)), np.zeros((*shape, 2))
    j00, j01, j10, j11 = np.ones(shape), np.zeros(shape), np.zeros(shape), np.ones(shape)
    energy, variance = np.zeros(shape), np.zeros(shape)
    for wavelength, amplitude, angle, phase in GRAVITY_WAVES:
        direction = np.array([math.cos(angle), math.sin(angle)])
        cross = np.array([-direction[1], direction[0]])
        k, omega = 2. * math.pi / wavelength, math.sqrt(9.81 * 2. * math.pi / wavelength)
        mesh_filter = 1. - _smoothstep(wavelength * .20, wavelength * .45, filter_spacing)
        pixel_filter = 1. - _smoothstep(wavelength * .12, wavelength * .45, footprint)
        alignment = ((1. - directional) * float(direction @ preferred_wind) ** 2
                     + directional * float(direction @ wind_dir) ** 2)
        base = amplitude * sea * np.asarray(scale) * (.45 + .55 * alignment) * mesh_filter
        q = k * .28 * (p @ cross) - omega * .16 * time + phase * 1.7
        phi = k * (p @ direction) - omega * time + phase + .85 * np.sin(q)
        grad = k * (direction + cross * (.238 * np.cos(q))[..., None])
        s, c = np.sin(phi), np.cos(phi)
        group_phase = k * .11 * (p @ cross) - omega * .07 * time + phase * 2.1
        group = .78 + .22 * np.sin(group_phase)
        a = base * group
        da = cross * (base * .22 * k * .11 * np.cos(group_phase))[..., None]
        variance += .5 * (a * k) ** 2 * (1. + WAVE_HARMONIC ** 2) * (1. - pixel_filter * pixel_filter)
        a, da = a * pixel_filter, da * np.asarray(pixel_filter)[..., None]
        # Positive cosine moves the shoulders TOWARD a crest. The previous sign
        # stretched crests and compressed troughs, making the sea look inflated.
        disp[..., :2] += direction * (WAVE_CHOP * a * c)[..., None]
        height = s + WAVE_HARMONIC * (s * s - .5)
        disp[..., 2] += a * height
        dh += grad * (a * c * (1. + 2. * WAVE_HARMONIC * s))[..., None] + da * height[..., None]
        horizontal = WAVE_CHOP * (da * c[..., None] - grad * (a * s)[..., None])
        j00 += direction[0] * horizontal[..., 0]
        j01 += direction[0] * horizontal[..., 1]
        j10 += direction[1] * horizontal[..., 0]
        j11 += direction[1] * horizontal[..., 1]
        energy += a
    determinant = np.maximum(j00 * j11 - j01 * j10, .25)
    slope = np.stack((j11 * dh[..., 0] - j10 * dh[..., 1],
                      j00 * dh[..., 1] - j01 * dh[..., 0]), axis=-1) / determinant[..., None]
    normal = np.concatenate((-slope, np.ones((*shape, 1))), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    crest = _smoothstep(.05, .70, disp[..., 2] / np.maximum(energy, .001))
    return disp, normal, crest, variance


def sample_detail(points, time, wind=(1.8, .8), storm=0., scale=1., footprint=0.):
    """Return the filtered detail slope/variance used on top of gravity normals."""
    p = np.asarray(points, np.float64)
    wind = np.asarray(wind, np.float64)
    speed = float(np.linalg.norm(wind))
    preferred = np.array([math.cos(.58), math.sin(.58)])
    wind_dir = wind / speed if speed > .001 else preferred
    directional = float(_smoothstep(.35, 2., speed))
    sea = float(np.clip(.68 + speed * .12 + storm * .20, .70, 1.6))
    slope, variance = np.zeros_like(p), np.zeros(p.shape[:-1])
    for wavelength, amplitude, angle, phase in DETAIL_WAVES:
        direction = np.array([math.cos(angle), math.sin(angle)])
        cross = np.array([-direction[1], direction[0]])
        k = 2. * math.pi / wavelength
        alignment = ((1. - directional) * float(direction @ preferred) ** 2
                     + directional * float(direction @ wind_dir) ** 2)
        a = amplitude * sea * np.asarray(scale) * (.55 + .45 * alignment)
        resolved = 1. - _smoothstep(wavelength * .12, wavelength * .50, footprint)
        omega = math.sqrt(9.81 * k + .000074 * k * k * k)
        q = k * .34 * (p @ cross) - omega * .21 * time + phase
        phi = k * (p @ direction) - omega * time + phase + .72 * np.sin(q)
        s, c = np.sin(phi), np.cos(phi)
        harmonic = WAVE_HARMONIC * (1. - _smoothstep(wavelength * .06, wavelength * .22, footprint))
        slope += (direction + cross * (.2448 * np.cos(q))[..., None]) * (a * c * (1. + 2. * harmonic * s) * resolved)[..., None]
        variance += .5 * a * a * (1. + WAVE_HARMONIC ** 2) * (1. - resolved * resolved)
    return slope, variance
