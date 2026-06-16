"""OpenStreetMap fusion: replace guessed roads/buildings/water with authoritative
vector geometry where a map already exists.

The whole point of this module is the PM insight: don't detect what you can look
up. For a geo-referenced image we query the OSM Overpass API for roads, building
footprints, and water in the image's bounding box, rasterize that geometry onto
the image grid, and overlay it on top of the color-based detection. Detection
still covers everything OSM doesn't know about (bare ground, off-map tracks,
vegetation), so we get the best of both: authoritative accuracy where the map
exists, imagery-based traversability where it doesn't.

Only the Python standard library is used for the request (urllib + json), so no
heavy geo stack (osmnx/geopandas) is required.
"""

import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request

import cv2
import numpy as np

from .terrain import ROAD, BUILDING, WATER

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Approx draw widths (in "base pixels", scaled by image resolution) per road class.
ROAD_WIDTHS = {
    "motorway": 6, "trunk": 6, "primary": 5, "secondary": 4, "tertiary": 3,
    "residential": 2, "unclassified": 2, "service": 2, "track": 2, "path": 1,
    "_default": 2,
}


def _cache_path(bounds) -> str:
    key = hashlib.md5((",".join(f"{x:.5f}" for x in bounds)).encode()).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"bog_osm_{key}.json")


def fetch_osm(bounds, timeout: int = 30, use_cache: bool = True):
    """Fetch OSM ways (roads/buildings/water) inside bounds=(west,south,east,north).

    Returns a list of element dicts (each with 'tags' and inline 'geometry'),
    or None on any failure (no internet, API down, parse error)."""
    west, south, east, north = bounds
    cache = _cache_path(bounds)
    if use_cache and os.path.exists(cache):
        try:
            with open(cache, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass

    query = (
        "[out:json][timeout:25];("
        f'way["highway"]({south},{west},{north},{east});'
        f'way["building"]({south},{west},{north},{east});'
        f'way["natural"="water"]({south},{west},{north},{east});'
        f'way["waterway"]({south},{west},{north},{east});'
        ");out geom;"
    )
    try:
        body = urllib.parse.urlencode({"data": query}).encode("utf-8")
        req = urllib.request.Request(OVERPASS_URL, data=body, headers={"User-Agent": "BOG/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        elements = payload.get("elements", [])
        if use_cache:
            try:
                with open(cache, "w", encoding="utf-8") as fh:
                    json.dump(elements, fh)
            except Exception:
                pass
        return elements
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[osm] Overpass fetch failed: {exc}")
        return None


def _geo_to_px(lat, lon, shape, bounds):
    h, w = shape
    west, south, east, north = bounds
    col = (lon - west) / (east - west) * (w - 1) if east != west else 0
    row = (north - lat) / (north - south) * (h - 1) if north != south else 0
    return int(round(row)), int(round(col))


def rasterize(elements, shape, bounds) -> dict:
    """Burn OSM ways onto boolean masks at the image's resolution."""
    h, w = shape
    road = np.zeros((h, w), np.uint8)
    building = np.zeros((h, w), np.uint8)
    water = np.zeros((h, w), np.uint8)
    counts = {"road": 0, "building": 0, "water": 0}
    scale = max(1, int(round(min(h, w) / 512)))

    for el in elements or []:
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        tags = el.get("tags", {})
        pts = np.array(
            [[c, r] for (r, c) in (_geo_to_px(p["lat"], p["lon"], shape, bounds) for p in geom)],
            dtype=np.int32,
        )
        if "building" in tags:
            cv2.fillPoly(building, [pts], 255)
            counts["building"] += 1
        elif "highway" in tags:
            width = ROAD_WIDTHS.get(tags.get("highway"), ROAD_WIDTHS["_default"])
            cv2.polylines(road, [pts], False, 255, thickness=max(1, width * scale))
            counts["road"] += 1
        elif tags.get("natural") == "water":
            cv2.fillPoly(water, [pts], 255)
            counts["water"] += 1
        elif "waterway" in tags:
            cv2.polylines(water, [pts], False, 255, thickness=max(1, 3 * scale))
            counts["water"] += 1

    return {"road": road > 0, "building": building > 0, "water": water > 0, "counts": counts}


def fuse_osm(mask, data) -> tuple:
    """Overlay authoritative OSM geometry on the detection mask.

    Returns (new_mask, info). info['available'] is False (with a 'reason') when
    fusion can't run, so the UI can explain why and fall back to detection.
    """
    bounds = data.get("bounds_wgs84")
    if not bounds:
        return mask, {"available": False, "reason": "image is not geo-referenced"}

    elements = fetch_osm(bounds)
    if elements is None:
        return mask, {"available": False, "reason": "could not reach OpenStreetMap (offline?)"}

    layers = rasterize(elements, mask.shape, bounds)
    out = mask.copy()
    out[layers["water"]] = WATER
    out[layers["building"]] = BUILDING
    out[layers["road"]] = ROAD  # roads drawn last so they win at crossings
    info = {"available": True, "counts": layers["counts"], "total": len(elements)}
    return out, info
