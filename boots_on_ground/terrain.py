"""Terrain classification and the traversability cost map.

We turn an image into an integer mask where every pixel is one of five classes,
then turn that mask into a float "cost map" telling the pathfinder how expensive
it is to step on each pixel.

Class codes:
    0 = Bare land, 1 = Road, 2 = Building, 3 = Vegetation, 4 = Water
"""

import cv2
import numpy as np

BARE, ROAD, BUILDING, VEG, WATER = 0, 1, 2, 3, 4

NAMES = {0: "Bare Land", 1: "Road", 2: "Building", 3: "Vegetation", 4: "Water"}

# Display colors (RGB)
COLORS = {
    0: (210, 180, 140),
    1: (128, 128, 128),
    2: (220, 50, 50),
    3: (34, 139, 34),
    4: (30, 144, 255),
}

# How hard each class is to cross. Road is cheapest; water is effectively a wall.
DEFAULT_COSTS = {0: 3.0, 1: 1.0, 2: 8.0, 3: 5.0, 4: 999.0}


def _detect_roads(rgb: np.ndarray) -> np.ndarray:
    """Roads are long, straight, gray features: find edges, fit line segments,
    and keep only the gray-colored pixels under those lines."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10
    )
    line_img = np.zeros_like(gray)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_img, (x1, y1), (x2, y2), 255, thickness=4)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    gray_pixels = ((s < 40) & (v > 50) & (v < 220)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    line_img = cv2.dilate(line_img, kernel, iterations=1)

    road = cv2.bitwise_and(line_img, gray_pixels)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return road > 0


def _detect_buildings(rgb: np.ndarray) -> np.ndarray:
    """Buildings are compact, roughly rectangular blobs: threshold for local
    contrast, then keep connected components in a sensible size/shape range."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    max_area = int(rgb.shape[0] * rgb.shape[1] * 0.28)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100 or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if 0.4 <= aspect <= 2.5:
            cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return mask > 0


def _veg_water_rgb(rgb: np.ndarray):
    """Color-rule vegetation/water for ordinary photos using the HSV color space
    (hue separates color from brightness, which makes thresholds robust)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    veg = (h >= 35) & (h <= 85) & (s > 40) & (v > 40)
    water = ((h >= 95) & (h <= 135) & (s > 30) & (v > 20)) | ((h >= 100) & (h <= 140) & (s > 50))
    return veg, water


def _refine_water(water: np.ndarray, rgb: np.ndarray,
                  large_area_frac: float = 0.01, min_area: int = 80,
                  var_thresh: float = 170.0, irregular_extent: float = 0.72) -> np.ndarray:
    """Color alone says 'blue', not 'water'. A blue-painted rooftop is blue too.

    So we require a candidate to also look like water *structurally*:
      - smooth: water has low local texture/variance (roofs have edges/detail);
      - large OR irregular: real water bodies are either sizeable, or have ragged
        natural outlines. Rooftops are small and rectangular (they fill their
        bounding box), so a small, boxy blue blob is rejected as a likely roof.

    This kills the classic false positive (small/ textured/ rectangular blue roof)
    while keeping lakes, rivers, and irregular ponds. It still cannot resolve a
    *large, smooth, blue* rooftop from RGB alone - that genuinely needs height
    (nDSM/LiDAR), a vector source (OSM), or a trained segmentation model.
    """
    if not water.any():
        return water
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype("float32")
    mean = cv2.blur(gray, (7, 7))
    var = cv2.blur(gray * gray, (7, 7)) - mean * mean
    smooth = var < var_thresh

    cand = (water & smooth).astype("uint8")
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    large_area = max(min_area, int(large_area_frac * water.shape[0] * water.shape[1]))
    keep = np.zeros(water.shape, dtype=bool)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        extent = area / max(w * h, 1)   # ~1.0 for a filled rectangle (roof-like)
        if area >= large_area or extent < irregular_extent:
            keep[labels == i] = True
    return keep


def _veg_water_nir(rgb: np.ndarray, nir: np.ndarray):
    """The proper remote-sensing way when a near-infrared band is available.

    NDVI is high for healthy plants (they reflect lots of NIR); NDWI is high for
    open water (which absorbs NIR)."""
    nir = nir.astype("float32")
    red = rgb[:, :, 0].astype("float32")
    green = rgb[:, :, 1].astype("float32")
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndwi = (green - nir) / (green + nir + 1e-8)
    return ndvi > 0.3, ndwi > 0.2


def _separate_road_building(road: np.ndarray, building: np.ndarray):
    """Roads and buildings are both gray man-made structures, so detectors
    overlap. Resolve by *shape*: a connected blob that is long and thin (high
    elongation) is road-like; a compact blob is building-like. This reassigns
    misfiled pixels instead of trusting color/edges alone.
    """
    combined = (road | building).astype("uint8")
    if not combined.any():
        return road, building
    num, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    out_road = np.zeros(road.shape, dtype=bool)
    out_building = np.zeros(building.shape, dtype=bool)
    for i in range(1, num):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        comp = labels == i
        elongation = max(w, h) / max(min(w, h), 1)
        extent = area / max(w * h, 1)        # how fully it fills its bounding box
        # Long/thin, or sparse-in-its-box => road; compact & filled => building.
        if elongation >= 3.0 or extent < 0.35:
            out_road[comp] = True
        else:
            out_building[comp] = True
    return out_road, out_building


def water_mask(data: dict) -> np.ndarray:
    """Validated open-water mask for one image (color + smoothness rules)."""
    rgb = data["rgb"]
    if data.get("has_nir") and data.get("nir") is not None:
        _, water = _veg_water_nir(rgb, data["nir"])
    else:
        _, water = _veg_water_rgb(rgb)
        water = _refine_water(water, rgb)
    return water


def extract_terrain(data: dict) -> np.ndarray:
    """Build the integer terrain mask from a loader dict."""
    rgb = data["rgb"]
    if data.get("has_nir") and data.get("nir") is not None:
        veg, water = _veg_water_nir(rgb, data["nir"])
    else:
        veg, water = _veg_water_rgb(rgb)
        water = _refine_water(water, rgb)

    road = _detect_roads(rgb)
    building = _detect_buildings(rgb)
    road, building = _separate_road_building(road, building)

    # Compose in increasing priority so the last assignment wins:
    # water > road > building > vegetation > bare.
    mask = np.zeros(rgb.shape[:2], dtype=np.int32)
    mask[veg] = VEG
    mask[building] = BUILDING
    mask[road] = ROAD
    mask[water] = WATER
    return mask


def colorize(mask: np.ndarray) -> np.ndarray:
    """Integer mask -> RGB image for display."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for code, color in COLORS.items():
        out[mask == code] = color
    return out


def stats(mask: np.ndarray) -> dict:
    """Per-class pixel counts and percentages."""
    total = int(mask.size)
    breakdown = {}
    for code, name in NAMES.items():
        count = int(np.count_nonzero(mask == code))
        breakdown[name] = {"count": count, "percentage": 100.0 * count / total}
    return {"total_pixels": total, "breakdown": breakdown}


def build_cost_map(mask, slope=None, costs=None, slope_factor=0.5) -> np.ndarray:
    """Convert the class mask into a per-pixel movement cost (float32)."""
    costs = costs or DEFAULT_COSTS
    cost = np.zeros(mask.shape, dtype="float32")
    for code, value in costs.items():
        cost[mask == code] = float(value)

    if slope is not None:
        if slope.shape != mask.shape:
            slope = cv2.resize(
                slope.astype("float32"), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        penalty = np.clip(slope, 0, 45) * slope_factor
        passable = cost < 900
        cost[passable] = cost[passable] + penalty[passable]
    return cost


def colorize_cost(cost_map: np.ndarray) -> np.ndarray:
    """Green (cheap) to red (expensive); water dark blue, buildings dark red."""
    try:
        from matplotlib import colormaps
        cmap = colormaps["RdYlGn_r"]
    except Exception:  # pragma: no cover - older matplotlib
        import matplotlib.cm as cm
        cmap = cm.get_cmap("RdYlGn_r")

    norm = np.clip((cost_map - 1.0) / (10.0 - 1.0), 0.0, 1.0)
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)

    water = cost_map >= 900
    building = (cost_map >= 7) & (~water)
    rgb[building] = (139, 0, 0)
    rgb[water] = (0, 0, 139)
    return rgb
