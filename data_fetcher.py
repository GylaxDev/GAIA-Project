"""Compatibility shim for historical caches.

The implementation moved to :mod:`osm3d.world.data_fetcher`; old mesh and
traffic fixtures still reference ``osm3d.data_fetcher`` classes.
"""
from .world.data_fetcher import *  # noqa: F401,F403
