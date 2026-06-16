"""Catalog of real satellite and orthophoto scenes."""

from boots_on_ground.paths import SATELLITE

REAL_DIR = str(SATELLITE)

# west, south, east, north (WGS84)
REAL_SCENES = {
    "San Francisco (drone ortho)": {
        "source": "oam",
        "bbox": (-122.45, 37.75, -122.40, 37.80),
        "file": "oam_san_francisco.tif",
        "note": "High-res community orthophoto (~2 cm). RGB, geo-referenced.",
    },
    "Haiti (disaster ortho)": {
        "source": "oam",
        "bbox": (-72.35, 18.50, -72.30, 18.55),
        "file": "oam_haiti.tif",
        "note": "Post-disaster HOT orthophoto. Good for damage / change demos.",
    },
    "Kenya (Nairobi area)": {
        "source": "oam",
        "bbox": (36.80, -1.30, 36.85, -1.25),
        "file": "oam_kenya.tif",
        "note": "Urban / semi-urban orthophoto (~30 cm).",
    },
    "California coast (Sentinel-2)": {
        "source": "sentinel",
        "bbox": (-122.45, 37.75, -122.40, 37.80),
        "file": "s2_california.tif",
        "note": "Sentinel-2 with NIR band. Enables NDVI/NDWI + SRTM + OSM.",
    },
    "Netherlands delta (Sentinel-2)": {
        "source": "sentinel",
        "bbox": (4.85, 52.35, 4.95, 52.42),
        "file": "s2_netherlands.tif",
        "note": "Canals, water, and urban mix near Amsterdam.",
    },
    "Amazon river (Sentinel-2)": {
        "source": "sentinel",
        "bbox": (-60.05, -3.15, -59.95, -3.05),
        "file": "s2_amazon.tif",
        "note": "Dense forest and river — strong vegetation/water signal.",
    },
    "Dubai coast (Sentinel-2)": {
        "source": "sentinel",
        "bbox": (55.10, 25.05, 55.20, 25.15),
        "file": "s2_dubai.tif",
        "note": "Desert, coast, and urban — varied terrain classes.",
    },
}


def scene_path(label: str) -> str:
    return str(SATELLITE / REAL_SCENES[label]["file"])


def is_downloaded(label: str) -> bool:
    return SATELLITE.joinpath(REAL_SCENES[label]["file"]).is_file()


def downloaded_labels() -> list:
    return [lbl for lbl in REAL_SCENES if is_downloaded(lbl)]
