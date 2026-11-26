"""One carriageway layout for imported widths, driving lanes and road paint.

Zero lane counts mean unspecified, not a tagged single lane. Explicit OSM counts
and widths win; only missing counts are inferred from the usable road width.
Offsets are right-positive and ordered left-to-right in the travel direction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..core.config import ROAD_WIDTHS

MIN_LANE_WIDTH = 2.8
DEFAULT_LANE_WIDTH = 3.2
MAX_LANE_WIDTH = 3.6


def _count(value):
    try:
        return max(0, min(8, int(value)))
    except (TypeError, ValueError):
        return 0


def _width_hint(road):
    try:
        width = float(getattr(road, 'width', 0.))
    except (TypeError, ValueError):
        width = 0.
    if math.isfinite(width) and width > 0.:
        return width
    # Named promenades are mapped as ordinary pedestrian ways without a width.
    # The Mid Beach Boardwalk is a ~6 m promenade, not a 2.5 m footpath.
    name = str(getattr(road, 'name', '') or '').casefold()
    if 'boardwalk' in name and road.highway in ('pedestrian', 'footway', 'path'):
        return 5.5
    return float(ROAD_WIDTHS.get(road.highway, 6.))


def lane_counts(road):
    total = _count(getattr(road, 'lanes', 0))
    forward = _count(getattr(road, 'lanes_forward', 0))
    backward = _count(getattr(road, 'lanes_backward', 0))
    if road.oneway:
        return forward or total or max(1, min(8, int(_width_hint(road) / DEFAULT_LANE_WIDTH))), 0
    if forward and backward:
        return forward, backward
    if not total:
        # Unknown bidirectional roads get balanced pairs, not an invented
        # three-lane road with an arbitrary extra direction.
        total = max(2, min(8, 2 * int(_width_hint(road) / (2. * DEFAULT_LANE_WIDTH))))
    if forward:
        return forward, max(0 if _count(getattr(road, 'lanes', 0)) > 1 else 1, total - forward)
    if backward:
        return max(0 if _count(getattr(road, 'lanes', 0)) > 1 else 1, total - backward), backward
    return max(1, (total + 1) // 2), max(1, total // 2)


def shared_lane(road):
    # Two directed routing records, but only ONE physical lane on the ground.
    return (not road.oneway and _count(getattr(road, 'lanes', 0)) == 1
            and max(_count(getattr(road, 'lanes_forward', 0)), _count(getattr(road, 'lanes_backward', 0))) <= 1
            and not (_count(getattr(road, 'lanes_forward', 0))
                     and _count(getattr(road, 'lanes_backward', 0))))


def road_width(road):
    width = _width_hint(road)
    tagged = getattr(road, 'width', 0.)
    try:
        if math.isfinite(float(tagged)) and float(tagged) > 0.:
            return width
    except (TypeError, ValueError):
        pass
    if road.highway not in ('footway', 'path', 'pedestrian', 'steps', 'track', 'cycleway', 'bridleway', 'railway'):
        if any(_count(getattr(road, name, 0)) for name in ('lanes', 'lanes_forward', 'lanes_backward')):
            total = 1 if shared_lane(road) else sum(lane_counts(road))
            # A class-wide 14 m default must not turn a tagged single lane into
            # an apparent four-lane avenue. Conversely, tagged parallel lanes
            # need enough room for ordinary vehicle bodies.
            width = max(total * MIN_LANE_WIDTH, min(width, total * MAX_LANE_WIDTH))
    return width


def road_surface_lift(highway, index=0, *, footpath=False):
    """Shared ribbon/decor datum, including the millimetre overlap tie-break.

    Footways are cut out of every carriageway before they are meshed, so they
    only need to clear the terrain and the exact land-cover overlay.  They sit
    on the shared road datum; the old 0.10 m lift left promenades such as the
    Mid Beach Boardwalk floating on a tall black kerb above the sand.
    """
    if footpath: return 0.
    rank = {'motorway': 5, 'trunk': 5, 'primary': 4, 'secondary': 3, 'tertiary': 2}.get(highway, 1)
    return .006*rank + .002*((index*2654435761) % 97)/97.


def lane_offsets(width, own, opposite=0):
    if own <= 0: return ()
    spacing = float(width) / (own + opposite)
    return tuple(-float(width) * .5 + (opposite + i + .5) * spacing for i in range(own))


@dataclass(frozen=True, slots=True)
class RoadLayout:
    width: float
    forward: int
    backward: int
    shared: bool = False

    @classmethod
    def from_road(cls, road):
        return cls(road_width(road), *lane_counts(road), shared=shared_lane(road))

    def offsets(self, forward=True):
        if self.shared: return (0.,)
        own, opposite = ((self.forward, self.backward) if forward else (self.backward, self.forward))
        return lane_offsets(self.width, own, opposite) if own else ()

    @property
    def divider_uv(self):
        # Ribbon U increases to the SOURCE's left, the opposite of lane offsets.
        return self.forward / (self.forward + self.backward) if self.forward and self.backward and not self.shared else None

    @property
    def separators_uv(self):
        if self.shared: return ()
        total = self.forward + self.backward
        return tuple(i / total for i in range(1, total) if self.divider_uv is None or i != self.forward)

    @property
    def paint_counts(self):
        # Shared single lanes have no exclusive directional allocations to paint.
        return (0, 0) if self.shared else (self.forward, self.backward)
