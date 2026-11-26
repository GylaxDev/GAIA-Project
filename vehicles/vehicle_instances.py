"""Shared, packed vehicle stream format (float32, 96 bytes per instance).

The first 20 values retain the parked/cache format: column-major model matrix,
paint RGB and integer light flags. The tail is wheel phase, steering (radians),
speed (m/s) and visibility. Model entries 3/7/11, unused by affine xyz transforms,
carry yaw rate, pitch rate and wheel radius for short render-time prediction.
Player rows may set ``SUSPENSION``: the matrix is then the *sprung body* pose and
model entries 3/7/11/15 carry the vertical offset (metres, body frame) of the
front-left, front-right, rear-left and rear-right wheel assemblies; the flag keeps
those channels separate from the AI prediction rates. Old 20-float records are accepted at the renderer
boundary and remain stationary.
"""
from __future__ import annotations

import numpy as np

LEGACY_VEHICLE_STRIDE = 20
VEHICLE_STRIDE = 28
VEHICLE_BYTES = VEHICLE_STRIDE * 4
INSTANCE_ATTRIBUTES = ((4, 4, 0), (5, 4, 16), (6, 4, 32), (7, 4, 48),
                       (8, 3, 64), (9, 1, 76), (11, 4, 80), (12, 4, 96))
LAMPS = 1
BRAKE = 2
LEFT = 4
RIGHT = 8
# Bits 4..11 hold a stable, per-driver indicator phase (1/256 of a flash cycle).
PHASE_SHIFT = 4
FLASH_PERIOD = .8
# Player-only wheel travel. AI rows leave this clear, so 3/7/11 keep their
# prediction-rate meaning and 15 keeps its homogeneous matrix value.
SUSPENSION = 0x1000

# Shared by the traffic physics and GPU animation, not inferred from body boxes.
LENGTHS = np.array([4.885, 5.050, 4.460, 11., 7.5, 4.885, 5.691, 5.890, 4.465])
WHEELBASES = np.array([2.825, 3.025, 2.700, 5.65, 4.5, 2.825, 3.429, 3.683, 2.670])
WHEEL_RADII = np.array([.33415, .39425, .317, .50, .40, .33415, .38695, .43756, .34126])
HEAVY = np.array([False, False, False, True, True, False, False, False, False])

# Body envelopes for player picking/collision (mirrors are not solid obstacles).
WIDTHS = np.array([1.840, 2.004, 1.745, 2.550, 2.200, 1.840, 2.012, 2.100, 1.833])
HEIGHTS = np.array([1.445, 1.783, 1.490, 3.445, 3.200, 1.570, 2.159, 1.912, 1.610])
# Approximate kerb masses for player suspension. Heavy kinds are never drivable.
MASSES = np.array([1500., 2050., 1250., 12000., 7500., 1500., 2300., 2300., 1500.])
VEHICLE_NAMES = ('Sedan', 'SUV', 'Prius', 'Bus', 'Truck', 'Taxi', 'Van', 'Pickup', 'BMW X1')


def normalize_instances(rows):
    """Upgrade old parked/inspection rows without changing their transforms."""
    a = np.asarray(rows, np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2 or a.shape[1] not in (LEGACY_VEHICLE_STRIDE, 24, VEHICLE_STRIDE):
        raise ValueError('vehicle instances must have 20, 24, or 28 float32 columns')
    if a.shape[1] == VEHICLE_STRIDE:
        return np.ascontiguousarray(a)
    out = np.zeros((len(a), VEHICLE_STRIDE), np.float32)
    if a.shape[1] == 24:
        out[:, :24] = a
    else:
        out[:, :LEGACY_VEHICLE_STRIDE] = a
        out[:, 23] = 1.
    out[:, 24:28] = 0.
    return out


def indicator_brightness(clock, flags):
    """A 75 flashes/minute lamp, with short rise/fall ramps (also used by GLSL)."""
    phase = (float(clock) / FLASH_PERIOD + np.floor(np.asarray(flags) / 16.) / 256.) % 1.
    return np.minimum(np.clip(phase / .06, 0., 1.), np.clip((.5 - phase) / .06, 0., 1.))


def predict_instances(rows, seconds):
    """Bounded vector prediction; no pathfinding, sorting or per-car Python work.

    The worker limits speed at stop lines/obstacles before publishing. Prediction
    is capped at three ticks so an overloaded worker cannot drive cars through a
    junction. Parked records have zero speed and motion rates.
    """
    a = normalize_instances(rows)
    dt = float(np.clip(seconds, 0., .1))
    if not dt or not len(a):
        return a
    out = a.copy()
    yaw = a[:, 3] * dt
    c, s = np.cos(yaw), np.sin(yaw)
    for col in (0, 4, 8):
        out[:, col] = c * a[:, col] - s * a[:, col + 1]
        out[:, col + 1] = s * a[:, col] + c * a[:, col + 1]
    # Midpoint integration follows the arc rather than a turn's initial tangent.
    out[:, 12:15] += (a[:, :3] + out[:, :3]) * (.5 * a[:, 22] * dt)[:, None]
    out[:, 20] = np.mod(a[:, 20] + a[:, 22] * dt / np.maximum(a[:, 11], .05), 2 * np.pi)
    return out


class VehicleFrame(list):
    """Packed streams plus timestamp and optional identity/odometer side tables.

    IDs encode driver life and slot, not row order. The small side tables are not
    GPU attributes and keep 24-float and legacy 20-float instance formats intact.
    """
    def __init__(self, rows=(), *, clock=0., generation=0, sequence=0, counts=None, signals=None,
                 ids=None, distances=None):
        super().__init__(rows)
        self.clock = float(clock)
        self.generation = int(generation)
        self.sequence = int(sequence)
        self.counts = counts or {}
        self.signals = signals
        self.ids = ids
        self.distances = distances
