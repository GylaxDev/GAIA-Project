"""Entry point: CLI parsing and application bootstrap."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

from .app import App
from ..core.astronomy import parse_date
from ..core.config import DEFAULT_TILE_URL, StreamConfig, parse_clock
from ..world.data_fetcher import OSMDataFetcher, Road, demo_buildings, road_paths_from_roads
from ..world.mesh_extruder import MeshExtruder
from ..core.projection import LocalProjection
from ..world.places import build_demo_places


def build_demo_scene(with_places=False):
    """Offline blocks with streets between them, usable by the driving controls."""
    from shapely.geometry import LineString
    roads = []
    for i in range(-6, 6):
        axis = i * 55. + 27.5
        ordinal = ("1st", "2nd", "3rd")[i + 6] if i < -3 else f"{i + 7}th"
        roads.extend((Road('residential', LineString([(-390., axis), (390., axis)]), width=10.5, lanes=2,
                           name=f"{ordinal} Street"),
                      Road('residential', LineString([(axis, -390.), (axis, 390.)]), width=10.5, lanes=2,
                           name=f"{ordinal} Avenue")))
    paths = road_paths_from_roads(roads)
    buildings = demo_buildings()
    mesh = MeshExtruder().build(buildings, roads, [], [], [], paths=paths,
                               bbox=(-400., -400., 400., 400.))
    return (mesh, paths, build_demo_places(buildings)) if with_places else (mesh, paths)


def main():
    ap = argparse.ArgumentParser(
        description="Interactive 3D OSM buildings with a world map and "
                    "real-time chunk streaming (PyOpenGL + Pygame)")
    ap.add_argument("--lat", type=float, default=40.7580)
    ap.add_argument("--lon", type=float, default=-73.9855)
    ap.add_argument("--radius", type=float, default=350,
                    help="initial viewing radius around the start point (m)")
    ap.add_argument("--place", type=str, help="geocode a place name instead of --lat/--lon")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("S", "W", "N", "E"),
                    help="start at the centre of this box")
    ap.add_argument("--world", action="store_true", help="start on the world map")
    ap.add_argument("--tile-url", default=DEFAULT_TILE_URL)
    ap.add_argument("--no-tile-invert", action="store_true")
    ap.add_argument("--default-height", type=float, default=9.0)
    ap.add_argument("--level-height", type=float, default=3.2)
    ap.add_argument("--cache-dir", type=Path, default=Path(".osm_cache"))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--demo", action="store_true",
                    help="offline synthetic city, no network")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 800), metavar=("W", "H"))
    ap.add_argument("--fullscreen", action="store_true",
                    help="start in fullscreen (F11 toggles it)")
    ap.add_argument("--skip-menu", action="store_true",
                    help="go straight into the city instead of opening the main menu")
    g = ap.add_argument_group("chunk streaming")
    g.add_argument("--chunk-zoom", type=int, default=14, choices=range(14, 19), metavar="Z")
    g.add_argument("--load-radius", type=int, default=1)
    g.add_argument("--retain-radius", type=int, default=None)
    g.add_argument("--workers", type=int, default=3)
    g.add_argument("--vram-mb", type=float, default=512.0)
    g.add_argument("--max-fps", type=int, default=60, help="frame-rate/power cap (0: uncapped)")
    g.add_argument("--prefetch-tiles", type=float, default=1.5, metavar="N")
    g.add_argument("--no-prefetch", action="store_true")
    g.add_argument("--no-mesh-cache", action="store_true")
    g.add_argument("--no-process-pool", action="store_true")
    g.add_argument("--no-terrain", action="store_true",
                   help="flatten the ground instead of sampling AWS DEM elevation")
    n = ap.add_argument_group("network")
    n.add_argument("--max-requests", type=int, default=3)
    n.add_argument("--min-interval", type=float, default=0.5)
    n.add_argument("--connect-timeout", type=float, default=6.0, metavar="S")
    n.add_argument("--http-timeout", type=float, default=45.0, metavar="S")
    n.add_argument("--hedge-delay", type=float, default=8.0, metavar="S")
    n.add_argument("--no-race", action="store_true")
    li = ap.add_argument_group("atmosphere, astronomy & lighting")
    li.add_argument("--time", type=parse_clock, default="13:00", metavar="HH:MM",
                    help="local mean solar time, or civil time when --utc-offset is supplied")
    li.add_argument("--date", type=parse_date, default=None, metavar="YYYY-MM-DD",
                    help="sky calendar date (default: today; advances across midnight)")
    li.add_argument("--utc-offset", type=float, default=None, metavar="H",
                    help="civil time minus UTC, e.g. -5 for New York standard time; no automatic DST")
    li.add_argument("--light-pollution", type=float, default=.35, metavar="0..1",
                    help="night sky glow / faint-star suppression (0: dark sky, 1: bright city)")
    li.add_argument("--constellations", action="store_true",
                    help="show optional catalog constellation guides (I toggles)")
    li.add_argument("--day-speed", type=float, default=0.05, metavar="H/S")
    li.add_argument("--shadow-size", type=int, default=4096, choices=(1024, 2048, 4096, 8192),
                    help="cached city shadow resolution (default: 4096; GPU-limited fallback)")
    li.add_argument("--near-shadow-size", type=int, default=2048, choices=(0, 512, 1024, 2048, 4096),
                    help="close rooftop/contact shadow resolution (0 disables the close map)")
    li.add_argument("--vehicle-shadow-size", type=int, default=4096, choices=(0, 512, 1024, 2048, 4096, 8192),
                    help="separate car shadow resolution (0 disables car casting)")
    li.add_argument("--no-shadows", action="store_true")
    # Accept old launch shortcuts, but they cannot re-enable the removed blade path.
    li.add_argument("--grass-shadow-size", type=int, default=None, help=argparse.SUPPRESS)
    li.add_argument("--no-grass", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not all(math.isfinite(x) for x in (args.time, args.day_speed, args.light_pollution, args.lat, args.lon)):
        ap.error('time, day speed, light pollution and coordinates must be finite')
    if not 0. <= args.light_pollution <= 1.:
        ap.error('--light-pollution must be between 0 and 1')
    if args.utc_offset is not None and (not math.isfinite(args.utc_offset) or not -14. <= args.utc_offset <= 14.):
        ap.error('--utc-offset must be finite and between -14 and 14 hours')
    if not -90. <= args.lat <= 90. or not -180. <= args.lon <= 180.:
        ap.error('--lat/--lon must be within [-90, 90] / [-180, 180]')
    if args.no_grass or args.grass_shadow_size is not None:
        print("[materials] 3D grass and blade shadows have been removed; legacy grass flags "
              "have no effect. Flat ground textures are always used.", file=sys.stderr)

    fetcher = OSMDataFetcher(
        args.cache_dir, args.default_height, args.level_height,
        args.max_requests, args.min_interval,
        http_timeout=args.http_timeout,
        connect_timeout=args.connect_timeout,
        hedge_delay=args.hedge_delay,
        race=1 if args.no_race else None,
        terrain_enabled=not args.no_terrain)  # real ground relief from AWS DEM tiles
    fetcher.refresh = args.refresh
    cfg = StreamConfig(
        args.chunk_zoom, args.load_radius, args.retain_radius, args.workers,
        args.vram_mb,
        mesh_cache=not args.no_mesh_cache and not args.refresh,
        process_pool=not args.no_process_pool,
        prefetch=not args.no_prefetch,
        prefetch_tiles=args.prefetch_tiles,
        max_fps=max(0, args.max_fps))
    demo_mesh = demo_paths = demo_places = start = None
    if args.demo:
        demo_mesh, demo_paths, demo_places = build_demo_scene(with_places=True)
    elif not args.world:
        if args.bbox:
            s, w, n, e = args.bbox
            lat0, lon0 = (s + n) / 2, (w + e) / 2
            label = f"{lat0:.4f}, {lon0:.4f}"
        elif args.place:
            lat0, lon0, label = fetcher.geocode_full(args.place)
        else:
            lat0, lon0, label = args.lat, args.lon, f"{args.lat:.4f}, {args.lon:.4f}"
        start = (lat0, lon0, label)

    App(fetcher, args.radius, cfg, demo_mesh, start, tuple(args.size),
        args.tile_url, not args.no_tile_invert, args.cache_dir,
        start_map=args.world,
        start_hours=args.time, day_speed=args.day_speed,
        sky_date=args.date, utc_offset=args.utc_offset, light_pollution=args.light_pollution,
        constellations=args.constellations, sky_location=(args.lat, args.lon),
        shadow_size=args.shadow_size, shadows=not args.no_shadows,
         vehicle_shadow_size=args.vehicle_shadow_size,
         near_shadow_size=args.near_shadow_size,
        fullscreen=args.fullscreen, demo_paths=demo_paths, demo_places=demo_places,
        skip_menu=args.skip_menu).run()


if __name__ == "__main__":
    main()
