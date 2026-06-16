"""Image loading for BOG.

Accepts a file path or a Streamlit uploaded-file object and returns a small,
uniform dict describing the image. GeoTIFFs are read with rasterio (so we can
recover geographic bounds and an optional near-infrared band); ordinary PNG/JPG
photos are read with Pillow.
"""

import io
import os

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds
    HAS_RASTERIO = True
except Exception:  # pragma: no cover - rasterio optional at import time
    HAS_RASTERIO = False

MAX_SIDE = 1500


def _percentile_stretch(band: np.ndarray) -> np.ndarray:
    """Stretch a float band to uint8 [0, 255] using the 2-98 percentile range."""
    band = band.astype("float32")
    lo, hi = np.percentile(band, (2, 98))
    if hi <= lo:
        lo, hi = float(band.min()), float(band.max())
    if hi <= lo:
        return np.zeros(band.shape, dtype=np.uint8)
    out = np.clip((band - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def _maybe_resize(rgb: np.ndarray, nir=None, max_side: int = MAX_SIDE):
    """Shrink very large images so downstream CV/pathfinding stays fast."""
    import cv2

    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return rgb, nir
    scale = max_side / longest
    new_size = (int(w * scale), int(h * scale))
    rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
    if nir is not None:
        nir = cv2.resize(nir, new_size, interpolation=cv2.INTER_AREA)
    return rgb, nir


def _read_input(file_or_path):
    """Return (filename, raw_bytes) for either a path string or an uploaded file."""
    if isinstance(file_or_path, (str, os.PathLike)):
        path = str(file_or_path)
        with open(path, "rb") as fh:
            return os.path.basename(path), fh.read()
    name = getattr(file_or_path, "name", "uploaded_image")
    if hasattr(file_or_path, "seek"):
        try:
            file_or_path.seek(0)
        except Exception:
            pass
    return name, file_or_path.read()


def _load_geotiff(data: bytes, filename: str) -> dict:
    with MemoryFile(data) as mem:
        with mem.open() as src:
            count = src.count
            bands = src.read()  # shape (count, H, W)

            if count >= 3:
                rgb = np.dstack([
                    _percentile_stretch(bands[0]),
                    _percentile_stretch(bands[1]),
                    _percentile_stretch(bands[2]),
                ])
            else:
                gray = _percentile_stretch(bands[0])
                rgb = np.dstack([gray, gray, gray])

            nir = bands[3].astype("float32") if count >= 4 else None
            has_nir = nir is not None

            bounds_wgs84 = None
            try:
                if src.crs is not None:
                    b = src.bounds
                    bounds_wgs84 = tuple(
                        transform_bounds(src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
                    )
            except Exception:
                bounds_wgs84 = None

    rgb, nir = _maybe_resize(rgb, nir)
    return {
        "rgb": rgb.astype(np.uint8),
        "is_geotiff": True,
        "bounds_wgs84": bounds_wgs84,
        "has_nir": has_nir,
        "nir": nir,
        "filename": filename,
    }


def load_image(file_or_path) -> dict:
    """Load an image into a uniform dict.

    Keys: rgb (H,W,3 uint8), is_geotiff, bounds_wgs84 (or None), has_nir,
    nir (HxW float32 or None), filename.
    """
    filename, data = _read_input(file_or_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".tif", ".tiff") and HAS_RASTERIO:
        try:
            return _load_geotiff(data, filename)
        except Exception as exc:  # fall back to Pillow on any rasterio error
            print(f"[loader] rasterio could not read {filename} ({exc}); using Pillow")

    img = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    rgb, _ = _maybe_resize(rgb)
    return {
        "rgb": rgb,
        "is_geotiff": False,
        "bounds_wgs84": None,
        "has_nir": False,
        "nir": None,
        "filename": filename,
    }


def image_info(data: dict) -> str:
    """Short human-readable description for the UI."""
    h, w = data["rgb"].shape[:2]
    kind = "GeoTIFF" if data["is_geotiff"] else "Image"
    geo = "geo-referenced" if data["bounds_wgs84"] else "no geo-reference"
    nir = "with NIR band" if data["has_nir"] else "RGB only"
    return f"{data['filename']} | {kind} {w}x{h}px | {geo} | {nir}"
