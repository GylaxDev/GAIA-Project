"""Traffic simulation, routing and signals.

``TrafficCore``, ``TrafficSimulator`` and the other public names remain
available from ``osm3d.traffic`` for compatibility with existing callers and
historical fixtures; the implementation lives in :mod:`osm3d.traffic.core`.
"""
from .core import *  # noqa: F401,F403
