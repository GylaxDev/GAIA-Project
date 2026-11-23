"""CPU companion to the shared atmospheric GLSL, in scene-linear radiance.

Spherical Rayleigh/Mie/ozone single scattering, with a bounded empirical
multiple-scattering twilight term. Fog uses the azimuthal horizon average, not
an arbitrary east-facing sample. This is not a full atmospheric transport solver.
"""
from __future__ import annotations

import math
import numpy as np

from .photometry import NITS_PER_SCENE_UNIT, SUN_IRRADIANCE

RE, RA, HR, HM = 6371e3, 6471e3, 8000., 1200.
BETA_R = np.array([5.802e-6, 13.558e-6, 33.1e-6])
BETA_M = np.array([3.996e-6] * 3)
BETA_O = np.array([.650e-6, 1.881e-6, .085e-6])
LUMA = np.array([.2126, .7152, .0722])
N_VIEW, N_SUN = 20, 6


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x) - a) / (b - a), 0., 1.)
    return t * t * (3. - 2. * t)


def _tint(color):
    color = np.asarray(color, np.float64)
    return color / max(1e-8, float(color @ LUMA))


def twilight_radiance(directions, sun_dir):
    directions = np.asarray(directions, np.float64)
    sun = np.asarray(sun_dir, np.float64)
    degrees = math.degrees(math.asin(np.clip(sun[2], -1., 1.)))
    elevation = np.maximum(directions[..., 2], 0.)
    ray_xy = directions[..., :2] / np.maximum(np.linalg.norm(directions[..., :2], axis=-1, keepdims=True), 1e-6)
    sun_xy = sun[:2] / max(float(np.linalg.norm(sun[:2])), 1e-6)
    toward = np.clip(ray_xy @ sun_xy, -1., 1.)
    blue = _tint([.24, .46, 1.])
    purple = _tint([.47, .32, .82])
    warm = _tint([1., .32, .09])
    violet = np.exp(-((elevation - .16) / .17) ** 2) * np.maximum(toward, 0.) ** 2
    color = blue + (purple - blue) * violet[..., None] * .32
    rim = (np.exp(-elevation * 14.) * np.maximum(toward, 0.) ** 4
           * smoothstep(-10., -1., degrees) * (1. - smoothstep(0., 6., degrees)))
    color = color + (warm - color) * rim[..., None] * .68
    # Antisolar pink band above Earth's rising blue shadow (Belt of Venus).
    belt_height = math.radians(4. + max(-degrees, 0.) * .65)
    belt = (np.exp(-((elevation - belt_height) / .065) ** 2) * np.maximum(-toward, 0.) ** 3
            * smoothstep(-8., -2., degrees) * (1. - smoothstep(0., 4., degrees)))
    color += _tint([.85, .38, .48]) * belt[..., None] * .22
    shadow = (np.exp(-elevation * 28.) * np.maximum(-toward, 0.) ** 2
              * smoothstep(-.5, -6., degrees))
    color *= 1. - shadow[..., None] * .48
    gain = (2. ** (min(degrees, 0.) * .86) * (1. - smoothstep(0., 7., degrees))
            * smoothstep(-20., -17., degrees))
    return color * (450. / NITS_PER_SCENE_UNIT) * gain


# Night-city sky glow: smog and low cloud lit from below.  Blue overhead, warm smog glow at
# the horizon (the original, natural tints).  Shared with the GLSL (shaders/atmosphere.py) so the fog colour, the sky and its
# reflections always agree.  The gain lifts the glow from a barely-visible 0.1 cd/m2 to a
# haze-lit ~1 cd/m2, which is what lets the skyline read as silhouettes against a glowing sky.
URBAN_ZENITH = (.30, .42, .65)
URBAN_HORIZON = (.75, .64, .48)
URBAN_GLOW_GAIN = 6.0


def urban_radiance(directions, night, glow):
    directions = np.asarray(directions, np.float64)
    h = np.exp(-np.maximum(directions[..., 2], 0.) * 4.)
    zenith, horizon = np.array(URBAN_ZENITH), np.array(URBAN_HORIZON)
    tint = zenith + (horizon - zenith) * h[..., None]
    tint /= np.maximum(tint @ LUMA, 1e-5)[..., None]
    return (tint * ((.11 + .22 * h) * URBAN_GLOW_GAIN * float(glow) / .3)[..., None]
            + _tint([.36, .52, .50]) * .00025) * float(night) / NITS_PER_SCENE_UNIT


def atmosphere_radiance(directions, sun_dir, intensity=1., *, height=100., aerosol=1.):
    source = np.asarray(directions, np.float64)
    shape = source.shape
    rays = source.reshape(-1, 3)
    if float(intensity) == 0.:
        return np.zeros(shape, np.float64)  # dark-night fog needs only airglow, not scattering integration
    rays = rays / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-9)
    sun = np.asarray(sun_dir, np.float64)
    sun = sun / max(float(np.linalg.norm(sun)), 1e-9)
    height, aerosol = np.clip(float(height), 2., 80000.), np.clip(float(aerosol), .1, 8.)
    origin = np.array([0., 0., RE + height])
    b = rays @ origin
    tmax = -b + np.sqrt(np.maximum(0., b * b + RA * RA - origin @ origin))
    earth_disc = b * b + RE * RE - origin @ origin
    hit = (b < 0.) & (earth_disc >= 0.)
    tmax = np.where(hit, np.minimum(tmax, -b - np.sqrt(np.maximum(earth_disc, 0.))), tmax)
    tmax = np.clip(tmax, 0., 350000.)
    edges = (np.arange(N_VIEW + 1) / N_VIEW) ** 2
    ds = tmax[:, None] * np.diff(edges)
    mid = tmax[:, None] * ((edges[:-1] + edges[1:]) * .5)
    points = origin + rays[:, None, :] * mid[..., None]
    h = np.maximum(np.linalg.norm(points, axis=2) - RE, 0.)
    rho = np.stack((np.exp(-h / HR), np.exp(-h / HM), np.maximum(0., 1. - np.abs(h - 25000.) / 15000.)), axis=-1)
    optical = rho * ds[..., None]
    view_optical = np.cumsum(optical, axis=1) - .5 * optical
    pp = np.einsum('vij,vij->vi', points, points)
    sb = points @ sun
    lit = ~((sb < 0.) & (sb * sb - (pp - RE * RE) > 0.))
    sun_max = -sb + np.sqrt(np.maximum(0., sb * sb - (pp - RA * RA)))
    se = (np.arange(N_SUN + 1) / N_SUN) ** 2
    sds = sun_max[..., None] * np.diff(se)
    smid = sun_max[..., None] * ((se[:-1] + se[1:]) * .5)
    sp = points[:, :, None, :] + sun * smid[..., None]
    sh = np.maximum(np.linalg.norm(sp, axis=3) - RE, 0.)
    srho = np.stack((np.exp(-sh / HR), np.exp(-sh / HM), np.maximum(0., 1. - np.abs(sh - 25000.) / 15000.)), axis=-1)
    sun_optical = (srho * sds[..., None]).sum(axis=2)
    totals = view_optical + sun_optical
    tau = totals[..., 0, None] * BETA_R + totals[..., 1, None] * BETA_M * aerosol * 1.1 + totals[..., 2, None] * BETA_O
    attenuation = np.exp(-tau) * lit[..., None]
    scatter_r = (optical[..., 0, None] * attenuation).sum(axis=1) * BETA_R
    scatter_m = (optical[..., 1, None] * attenuation).sum(axis=1) * BETA_M * aerosol
    mu = np.clip(rays @ sun, -1., 1.)
    phase_r = 3. / (16. * math.pi) * (1. + mu * mu)
    g = .76
    phase_m = (1. - g * g) / (4. * math.pi * np.maximum(1. + g * g - 2. * g * mu, 1e-4) ** 1.5)
    radiance = (scatter_r * phase_r[:, None] + scatter_m * phase_m[:, None]) * SUN_IRRADIANCE
    radiance += twilight_radiance(rays, sun)
    return (np.maximum(radiance, 0.) * float(intensity)).reshape(shape)


def horizon_display(sun_dir, intensity, exposure, *, height=100., aerosol=1., night=0., glow=0.):
    azimuth = np.arange(8) * (math.pi / 4.)
    rays = np.column_stack((np.cos(azimuth), np.sin(azimuth), np.full(8, .012)))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    color = atmosphere_radiance(rays, sun_dir, intensity, height=height, aerosol=aerosol)
    color += urban_radiance(rays, night, glow)
    x = np.maximum(color.mean(axis=0) * float(exposure), 0.)
    return np.clip(x * (2.51 * x + .03) / (x * (2.43 * x + .59) + .14), 0., 1.) ** (1. / 2.2)
