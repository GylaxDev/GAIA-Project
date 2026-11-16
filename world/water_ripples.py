"""Small, periodic spectral ripple fields baked once, not simulated per frame.

Two independent fields at each of two scales supply irregular directional
wrinkles between the analytic wind waves. Signed slopes and their second moment
are mipmapped together, so unresolved detail becomes roughness, not sparkle noise.
These are shading textures, not another displaced mesh or a dynamic FFT ocean.
"""
from __future__ import annotations

import math

import numpy as np

WATER_RIPPLE_TILE = 256
WATER_RIPPLE_SPANS = (64., 8.)
WATER_RIPPLE_SLOPES = (.180, .100)
WATER_RIPPLE_LAYERS = 4


def build_water_ripples(size=WATER_RIPPLE_TILE, seed=631):
    """Return RGBA32F: signed XY slope, crest grain, squared slope magnitude."""
    if size < 32 or size & (size - 1):
        raise ValueError('water ripple tile must be a power of two >= 32')
    result = []
    for layer in range(WATER_RIPPLE_LAYERS):
        cascade = layer // 2
        span = WATER_RIPPLE_SPANS[cascade]
        rng = np.random.default_rng(seed + layer * 101)
        freq = np.fft.fftfreq(size, d=span / size) * math.tau
        kx, ky = np.meshgrid(freq, freq)
        k = np.maximum(np.hypot(kx, ky), 1e-6)
        alignment = (kx * math.cos(.58) + ky * math.sin(.58)) / k
        peak = math.tau / (1.8 if cascade == 0 else .30)
        band = np.exp(-.5 * (np.log(k / peak) / .62) ** 2)
        direction = .10 + .90 * alignment ** 6
        spectrum = np.fft.fft2(rng.standard_normal((size, size))) * np.sqrt(band * direction) / k
        spectrum[0, 0] = 0.
        # Odd spectral derivatives must be zero on the self-conjugate Nyquist
        # axes to preserve real-valued, genuinely periodic gradients.
        dx, dy = kx.copy(), ky.copy()
        dx[:, size // 2] = 0.
        dy[size // 2, :] = 0.
        height = np.fft.ifft2(spectrum).real
        sx = np.fft.ifft2(spectrum * (1j * dx)).real
        sy = np.fft.ifft2(spectrum * (1j * dy)).real
        hnorm = height / max(float(height.std()), 1e-8)
        # A bounded Stokes-like sharpening puts extra wrinkles on positive crests
        # without turning the underlying Gaussian field into rounded noise blobs.
        t = np.tanh(hnorm)
        sharpening = 1. + .56 * t * (1. - t * t)
        slopes = np.stack((sx, sy), axis=-1) * sharpening[..., None]
        slopes -= slopes.mean(axis=(0, 1))
        rms = math.sqrt(float(np.mean(np.sum(slopes * slopes, axis=-1))))
        slopes *= WATER_RIPPLE_SLOPES[cascade] / max(rms, 1e-8)
        curvature = np.fft.ifft2(spectrum * k * k).real
        crest = .5 + .5 * np.tanh(curvature / max(float(curvature.std()), 1e-8) * .55)
        tile = np.empty((size, size, 4), np.float32)
        tile[..., :2] = slopes
        tile[..., 2] = crest
        tile[..., 3] = np.sum(slopes * slopes, axis=-1)
        result.append(tile)
    return np.ascontiguousarray(result, np.float32)


def build_water_ripple_mipmaps(tiles):
    """Area-average slopes AND energy; half-float payload including the 1x1 mip."""
    data = np.asarray(tiles, np.float32)
    if (data.ndim != 4 or data.shape[0] != WATER_RIPPLE_LAYERS or data.shape[-1] != 4
            or data.shape[1] != data.shape[2] or data.shape[1] < 1
            or data.shape[1] & (data.shape[1] - 1) or not np.isfinite(data).all()):
        raise ValueError('water mips require four finite square power-of-two RGBA fields')
    levels = [np.ascontiguousarray(data, np.float16)]
    side = data.shape[1]
    while side > 1:
        data = data.reshape(WATER_RIPPLE_LAYERS, side // 2, 2, side // 2, 2, 4).mean(axis=(2, 4))
        side //= 2
        levels.append(np.ascontiguousarray(data, np.float16))
    return levels
