"""Entry point: CLI parsing and application bootstrap."""
from __future__ import annotations

import argparse
from pathlib import Path

from .app import App
from .config import DEFAULT_TILE_URL, StreamConfig, parse_clock
from .data_fetcher import OSMDataFetcher, demo_buildings
from .mesh_extruder import MeshExtruder
from .projection import LocalProjection


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
    g = ap.add_argument_group("chunk streaming")
    g.add_argument("--chunk-zoom", type=int, default=14, choices=range(14, 19), metavar="Z")
    g.add_argument("--load-radius", type=int, default=1)
    g.add_argument("--retain-radius", type=int, default=None)
    g.add_argument("--workers", type=int, default=3)
    g.add_argument("--vram-mb", type=float, default=512.0)
    g.add_argument("--prefetch-tiles", type=float, default=1.5, metavar="N")
    g.add_argument("--no-prefetch", action="store_true")
    g.add_argument("--no-mesh-cache", action="store_true")
    g.add_argument("--no-process-pool", action="store_true")
    n = ap.add_argument_group("network")
    n.add_argument("--max-requests", type=int, default=3)
    n.add_argument("--min-interval", type=float, default=0.5)
    n.add_argument("--connect-timeout", type=float, default=6.0, metavar="S")
    n.add_argument("--http-timeout", type=float, default=45.0, metavar="S")
    n.add_argument("--hedge-delay", type=float, default=8.0, metavar="S")
    n.add_argument("--no-race", action="store_true")
    li = ap.add_argument_group("lighting & shadows")
    li.add_argument("--time", default="13:00", metavar="HH:MM")
    li.add_argument("--day-speed", type=float, default=0.05, metavar="H/S")
    li.add_argument("--shadow-size", type=int, default=2048, choices=(1024, 2048, 4096))
    li.add_argument("--no-shadows", action="store_true")
    args = ap.parse_args()

    fetcher = OSMDataFetcher(
        args.cache_dir, args.default_height, args.level_height,
        args.max_requests, args.min_interval,
        http_timeout=args.http_timeout,
        connect_timeout=args.connect_timeout,
        hedge_delay=args.hedge_delay,
        race=1 if args.no_race else None)
    fetcher.refresh = args.refresh
    cfg = StreamConfig(
        args.chunk_zoom, args.load_radius, args.retain_radius, args.workers,
        args.vram_mb,
        mesh_cache=not args.no_mesh_cache and not args.refresh,
        process_pool=not args.no_process_pool,
        prefetch=not args.no_prefetch,
        prefetch_tiles=args.prefetch_tiles)
    demo_mesh = start = None
    if args.demo:
        demo_mesh = MeshExtruder().build(demo_buildings(), [], [], [], [])
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
        start_hours=parse_clock(args.time), day_speed=args.day_speed,
        shadow_size=args.shadow_size, shadows=not args.no_shadows).run()


if __name__ == "__main__":
    main()