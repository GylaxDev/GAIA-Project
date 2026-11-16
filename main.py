"""Entry point shim: ``python -m osm3d.main`` delegates to the application package."""
from .application.main import build_demo_scene, main

__all__ = ["build_demo_scene", "main"]


if __name__ == "__main__":
    main()
