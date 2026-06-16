"""NASA SRTM elevation + slope.

Only useful for geo-referenced GeoTIFFs (we need real lat/lon to look up
elevation). Uses the pure-Python `srtm` package, which downloads and caches
SRTM tiles on demand. Everything is wrapped so that being offline or passing a
plain photo simply returns None and the app continues without elevation.
"""

import os
import tempfile

import numpy as np

GRID = 80  # how many samples per side when building the small DEM


def fetch_elevation(bounds_wgs84, grid: int = GRID):
    """Return a (grid x grid) float32 elevation array for the bbox, or None.

    bounds_wgs84 = (west, south, east, north) in degrees. Row 0 is the north edge.
    """
    if bounds_wgs84 is None:
        return None
    try:
        west, south, east, north = (float(v) for v in bounds_wgs84)
    except Exception:
        return None

    if not np.all(np.isfinite([west, south, east, north])):
        return None
    if west >= east or south >= north:
        return None
    if not (-180 <= west <= 180 and -180 <= east <= 180 and -90 <= south <= 90 and -90 <= north <= 90):
        return None

    try:
        import srtm

        cache_dir = os.path.join(tempfile.gettempdir(), "bog_elevation_cache")
        os.makedirs(cache_dir, exist_ok=True)
        data = srtm.get_data(local_cache_dir=cache_dir)

        lats = np.linspace(north, south, grid)
        lons = np.linspace(west, east, grid)
        elev = np.full((grid, grid), np.nan, dtype="float32")
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                value = data.get_elevation(lat, lon)
                if value is not None:
                    elev[i, j] = value

        if np.isnan(elev).all():
            return None
        fill = float(np.nanmean(elev))
        elev = np.where(np.isnan(elev), fill, elev)
        elev = np.where((elev < -9000) | (elev > 9000), fill, elev)
        return elev.astype("float32")
    except Exception as exc:  # offline, missing tiles, etc.
        print(f"[elevation] SRTM fetch failed ({exc}); continuing without elevation")
        return None


def compute_slope(elevation_array, pixel_size_meters: float = 30.0):
    """Slope in degrees from an elevation grid (steepness of the surface)."""
    if elevation_array is None:
        return None
    dy, dx = np.gradient(elevation_array.astype("float32"), pixel_size_meters, pixel_size_meters)
    slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
    return np.clip(slope, 0, 90).astype("float32")


def profile(elevation_array, path, image_shape):
    """Elevation value at each step along a path (path is (row,col) pixels)."""
    if elevation_array is None or not path:
        return None
    import cv2

    h, w = image_shape
    elev = cv2.resize(elevation_array, (w, h), interpolation=cv2.INTER_LINEAR)
    values = []
    for r, c in path:
        r = min(max(r, 0), h - 1)
        c = min(max(c, 0), w - 1)
        values.append(float(elev[r, c]))
    return np.array(values, dtype="float32")


def gain(profile_array):
    """Min/max elevation plus cumulative gain/loss/net along the profile."""
    if profile_array is None or len(profile_array) == 0:
        return None
    diffs = np.diff(profile_array)
    return {
        "min_elevation": float(np.min(profile_array)),
        "max_elevation": float(np.max(profile_array)),
        "total_gain": float(np.sum(diffs[diffs > 0])),
        "total_loss": float(-np.sum(diffs[diffs < 0])),
        "net_change": float(profile_array[-1] - profile_array[0]),
    }
