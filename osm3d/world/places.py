"""Small, named OSM places shared by destination search and both map views.

Places contain only strings and coordinates: resident tiles never retain raw
OSM objects, building meshes or staged upload buffers just to show a map pin.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from ..core.projection import lonlat_to_tile


CATEGORY_LABELS = {
    "coffee": "Coffee", "food": "Food & drink", "shopping": "Shopping",
    "fuel": "Fuel & charging", "parking": "Parking", "health": "Health",
    "transit": "Transit", "park": "Parks & outdoors", "hotel": "Hotel",
    "landmark": "Landmark", "civic": "Community",
}
CATEGORY_COLORS = {
    "coffee": (0.76, 0.49, 0.30, 1.), "food": (1., 0.58, 0.20, 1.),
    "shopping": (0.95, 0.54, 0.74, 1.), "fuel": (0.35, 0.67, 0.95, 1.),
    "parking": (0.35, 0.61, 0.96, 1.), "health": (1., 0.40, 0.45, 1.),
    "transit": (0.40, 0.65, 1., 1.), "park": (0.33, 0.76, 0.48, 1.),
    "hotel": (0.77, 0.58, 0.95, 1.), "landmark": (0.81, 0.65, 0.97, 1.),
    "civic": (0.62, 0.73, 0.82, 1.),
}


@dataclass(frozen=True)
class Place:
    key: str
    name: str
    category: str
    x: float
    y: float

    @property
    def position(self):
        return (self.x, self.y)


def _category(tags):
    amenity = tags.get("amenity", "")
    if amenity == "cafe": return "coffee", "Cafe"
    if amenity in ("restaurant", "fast_food", "food_court", "bar", "pub", "biergarten"):
        return "food", amenity.replace("_", " ").title()
    if amenity in ("fuel", "charging_station"): return "fuel", "Fuel" if amenity == "fuel" else "Charging station"
    if amenity in ("parking", "parking_entrance"): return "parking", "Parking"
    if amenity in ("hospital", "clinic", "doctors", "dentist", "pharmacy"):
        return "health", amenity.title()
    if tags.get("shop") not in (None, "", "no", "vacant"):
        return "shopping", str(tags["shop"]).replace("_", " ").title()
    if (tags.get("railway") in ("station", "halt", "subway_entrance")
            or tags.get("public_transport") == "station"):
        return "transit", "Station"
    if tags.get("leisure") in ("park", "garden", "playground", "stadium", "sports_centre"):
        return "park", str(tags["leisure"]).replace("_", " ").title()
    if tags.get("natural") == "beach": return "park", "Beach"
    if tags.get("tourism") in ("hotel", "motel", "hostel", "guest_house"):
        return "hotel", "Hotel"
    if (tags.get("tourism") in ("museum", "attraction", "viewpoint", "gallery")
            or tags.get("historic") in ("monument", "memorial", "castle", "ruins")):
        return "landmark", str(tags.get("tourism") or tags["historic"]).replace("_", " ").title()
    if amenity in ("school", "university", "college", "library", "bank", "cinema", "theatre",
                   "police", "fire_station", "place_of_worship", "community_centre", "townhall"):
        return "civic", amenity.replace("_", " ").title()
    if tags.get("building") not in (None, "", "no") and tags.get("name"):
        return "landmark", "Building"
    return None


def _position(element, proj):
    if "lat" in element and "lon" in element:
        return proj.to_local(float(element["lat"]), float(element["lon"]))
    centre = element.get("center") or {}
    if "lat" in centre and "lon" in centre:
        return proj.to_local(float(centre["lat"]), float(centre["lon"]))
    # An interior point, not a polygon's average (which can lie in a courtyard).
    from .data_fetcher import item_polygons
    polygons = item_polygons(element, proj)
    if polygons:
        point = max(polygons, key=lambda p: p.area).representative_point()
        return np.array((point.x, point.y))
    geometry = element.get("geometry")
    if isinstance(geometry, dict) and geometry.get("type") == "Point":
        lon, lat = geometry["coordinates"][:2]
        return proj.to_local(float(lat), float(lon))
    if isinstance(geometry, list) and geometry:
        points = proj.to_local([float(p["lat"]) for p in geometry],
                               [float(p["lon"]) for p in geometry])
        return points.mean(axis=0)
    return None


def _deduplicate(places):
    seen, buckets, out = set(), {}, []
    for place in places:
        if place.key in seen:
            continue
        seen.add(place.key)
        name = " ".join(place.name.casefold().split())
        cell = (math.floor(place.x / 12.), math.floor(place.y / 12.))
        duplicate = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in buckets.get((cell[0] + dx, cell[1] + dy, name), ()):
                    compatible = place.category == other.category or "landmark" in (place.category, other.category)
                    if compatible and math.hypot(place.x - other.x, place.y - other.y) < 12.:
                        duplicate = True
        if not duplicate:
            buckets.setdefault((*cell, name), []).append(place)
            out.append(place)
    return tuple(out)


def parse_places(raw, proj, owner=None):
    """Extract useful POIs without changing the geometry importer's return shape."""
    out = []
    elements = raw.get("elements", ()) if isinstance(raw, dict) else ()
    elements = [e for e in elements if isinstance(e, dict) and isinstance(e.get("tags") or {}, dict)]
    # Prefer a shop's explicit node over a similarly named surrounding building.
    for element in sorted(elements, key=lambda e: e.get("type") != "node"):
        tags = element.get("tags") or {}
        category = _category(tags)
        if category is None:
            continue
        name = str(tags.get("name") or tags.get("brand") or category[1])
        name = " ".join("".join(c for c in name if c.isprintable()).split())[:100]
        if not name:
            continue
        try:
            xy = _position(element, proj)
            if xy is None or not np.isfinite(xy).all() or np.linalg.norm(xy) > 5000.:
                continue
            if owner is not None and lonlat_to_tile(*proj.to_latlon(*xy), owner[0]) != tuple(owner[1:]):
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        key = f"{element.get('type', 'place')}:{element.get('id', len(out))}"
        out.append(Place(key, name, category[0], float(xy[0]), float(xy[1])))
    return _deduplicate(out)


def scene_places(scene):
    """Cached, deduplicated world coordinates; rebuilt only on tile residency changes."""
    if scene is None:
        return ()
    generation = getattr(scene, "generation", None)
    cached = getattr(scene, "_map_places", None)
    if generation is not None and cached is not None and cached[0] == generation:
        return cached[1]
    places = []
    for tile in getattr(scene, "tiles", {}).values():
        for place in getattr(tile, "places", ()):
            places.append(replace(place, x=place.x * tile.scale[0] + tile.origin[0],
                                  y=place.y * tile.scale[1] + tile.origin[1]))
    result = _deduplicate(places)
    if generation is not None:
        scene._map_places = (generation, result)
    return result


def build_demo_places(buildings):
    """Named offline destinations anchored to real demo footprints, not invented roads."""
    from shapely.geometry import Point
    choices = (("Corner Coffee", "coffee", (-55., 0.)),
               ("City Market", "shopping", (55., 55.)),
               ("Garden Kitchen", "food", (0., 110.)),
               ("Central Library", "civic", (-110., 110.)),
               ("Riverside Hotel", "hotel", (165., -55.)),
               ("Community Clinic", "health", (-165., -55.)),
               ("North Gallery", "landmark", (55., 220.)),
               ("South Parking", "parking", (55., -165.)))
    available, out = list(buildings), []
    for name, category, target in choices:
        if not available: break
        building = min(available, key=lambda b: b.footprint.centroid.distance(Point(target)))
        available = [b for b in available if b is not building]
        point = building.footprint.representative_point()
        out.append(Place(f"demo:{building.id}", name, category, point.x, point.y))
    return tuple(out)
