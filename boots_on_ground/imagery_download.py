"""Download Sentinel-2 and OpenAerialMap tiles into data/satellite/."""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds
from rasterio.warp import transform_bounds

from config.satellite_scenes import REAL_DIR, REAL_SCENES, scene_path

MAX_SIDE = 1024
OAM_SEARCH = "https://api.openaerialmap.org/meta"
STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"


def _http_json(url, data=None, timeout=60):
    if data is not None:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "BOG/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _cap_window(win: Window, max_side: int = MAX_SIDE) -> Window:
    col, row, w, h = float(win.col_off), float(win.row_off), float(win.width), float(win.height)
    if w <= max_side and h <= max_side:
        return Window(int(col), int(row), int(w), int(h))
    scale = min(max_side / w, max_side / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    ncol = int(col + (w - nw) / 2)
    nrow = int(row + (h - nh) / 2)
    return Window(ncol, nrow, nw, nh)


def _stretch_band(band: np.ndarray) -> np.ndarray:
    band = band.astype("float32")
    lo, hi = np.percentile(band, (2, 98))
    if hi <= lo:
        lo, hi = float(band.min()), float(band.max())
    if hi <= lo:
        return np.zeros(band.shape, dtype=np.uint8)
    return (np.clip((band - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _write_geotiff(path, arrays, crs, transform):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": arrays[0].shape[0],
        "width": arrays[0].shape[1],
        "count": len(arrays),
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        for i, band in enumerate(arrays, start=1):
            dst.write(band, i)


def download_oam(label: str, meta: dict, out_path: str) -> None:
    west, south, east, north = meta["bbox"]
    qs = urllib.parse.urlencode({"bbox": f"{west},{south},{east},{north}", "limit": "8"})
    results = _http_json(f"{OAM_SEARCH}?{qs}", timeout=30).get("results", [])
    if not results:
        raise RuntimeError(f"No OpenAerialMap imagery for {label}")

    last_err = None
    for cand in results:
        remote_url = cand["uuid"]
        try:
            with rasterio.open(remote_url) as src:
                wins = []
                try:
                    native = transform_bounds("EPSG:4326", src.crs, west, south, east, north)
                    win = window_from_bounds(*native, transform=src.transform)
                    win = win.intersection(Window(0, 0, src.width, src.height))
                    if win.width >= 64 and win.height >= 64:
                        wins.append(_cap_window(win))
                except Exception:
                    pass
                side = min(MAX_SIDE, src.width, src.height)
                wins.append(Window((src.width - side) // 2, (src.height - side) // 2, side, side))

                for win in wins:
                    bands = src.read([1, 2, 3], window=win)
                    if bands.dtype != np.uint8:
                        bands = np.stack([_stretch_band(bands[i]) for i in range(3)])
                    if float(bands.mean()) < 5.0:
                        continue
                    transform = src.window_transform(win)
                    _write_geotiff(out_path, [bands[i] for i in range(3)], src.crs, transform)
                    print(f"  OAM source: {cand.get('title', remote_url[:60])}")
                    return
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"All OAM candidates were empty for {label}: {last_err}")


def _stac_sentinel(bbox):
    body = json.dumps({
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": "2023-01-01T00:00:00Z/2024-12-31T00:00:00Z",
        "query": {"eo:cloud_cover": {"lt": 25}},
        "limit": 3,
    }).encode()
    fc = _http_json(STAC_SEARCH, data=body, timeout=60)
    feats = fc.get("features", [])
    if not feats:
        raise RuntimeError("No Sentinel-2 scene found (try another bbox or date range)")
    return feats[0]["assets"]


def download_sentinel(label: str, meta: dict, out_path: str) -> None:
    bbox = meta["bbox"]
    assets = _stac_sentinel(bbox)
    band_keys = ["red", "green", "blue", "nir"]
    urls = [assets[k]["href"] for k in band_keys]
    print(f"  Sentinel-2 scene: {assets['red']['href'].split('/')[-2]}")

    with rasterio.open(urls[0]) as ref:
        native = transform_bounds("EPSG:4326", ref.crs, *bbox)
        win = window_from_bounds(*native, transform=ref.transform)
        win = win.intersection(Window(0, 0, ref.width, ref.height))
        win = _cap_window(win)
        transform = ref.window_transform(win)
        crs = ref.crs

    stacked = []
    for url in urls:
        with rasterio.open(url) as src:
            raw = src.read(1, window=win)
            stacked.append(_stretch_band(raw))

    _write_geotiff(out_path, stacked, crs, transform)


def download_scene(label: str, force=False) -> str:
    if label not in REAL_SCENES:
        raise KeyError(f"Unknown scene: {label}")
    meta = REAL_SCENES[label]
    out = scene_path(label)
    if os.path.isfile(out) and not force:
        print(f"[skip] {label} -> {out}")
        return out

    print(f"[download] {label} ({meta['source']}) ...")
    if meta["source"] == "oam":
        download_oam(label, meta, out)
    elif meta["source"] == "sentinel":
        download_sentinel(label, meta, out)
    else:
        raise ValueError(meta["source"])

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"  saved {out} ({size_mb:.1f} MB)")
    return out


def download_all(force=False):
    os.makedirs(REAL_DIR, exist_ok=True)
    ok, fail = [], []
    for label in REAL_SCENES:
        try:
            download_scene(label, force=force)
            ok.append(label)
        except Exception as exc:
            print(f"  FAILED {label}: {exc}", file=sys.stderr)
            fail.append(label)
    print(f"\nDone: {len(ok)} ok, {len(fail)} failed.")
    if fail:
        print("Failed:", ", ".join(fail))
    return ok, fail
