"""Before/after change detection for satellite and aerial imagery.

Combines pixel-level image differencing with before-anchored land-cover rules.
Water, vegetation, structure, and surface changes are validated on both dates
so classifier noise (e.g. shadows read as water) does not pollute results.
Works for floods, structural damage, fires, construction, and other events.
"""

import cv2
import numpy as np

from . import terrain as terrain_mod
from .terrain import BARE, ROAD, BUILDING, VEG, WATER, NAMES

# Change categories
NOCHANGE = 0
NEW_BUILDING = 1
VEG_LOSS = 2
VEG_GAIN = 3
NEW_WATER = 4
WATER_LOSS = 5
NEW_ROAD = 6
SURFACE_CHANGE = 7
STRUCTURE_DAMAGE = 8

# Severity levels (how strong the pixel-level change is)
SEVERITY_NONE = 0
SEVERITY_MINOR = 1
SEVERITY_MODERATE = 2
SEVERITY_MAJOR = 3

SEVERITY_NAMES = {
    SEVERITY_MINOR: "Minor",
    SEVERITY_MODERATE: "Moderate",
    SEVERITY_MAJOR: "Major",
}

# Pink → black → red as change intensity increases
SEVERITY_COLORS = {
    SEVERITY_MINOR: (255, 192, 203),
    SEVERITY_MODERATE: (25, 25, 25),
    SEVERITY_MAJOR: (220, 20, 20),
}

SEVERITY_ALPHA = {
    SEVERITY_MINOR: 0.36,
    SEVERITY_MODERATE: 0.50,
    SEVERITY_MAJOR: 0.62,
}

CHANGE_NAMES = {
    NEW_BUILDING: "New construction",
    VEG_LOSS: "Vegetation loss",
    VEG_GAIN: "Vegetation gain",
    NEW_WATER: "Inundation / new water",
    WATER_LOSS: "Water recession",
    NEW_ROAD: "New road / cleared route",
    SURFACE_CHANGE: "Surface change",
    STRUCTURE_DAMAGE: "Structure damage",
}

CHANGE_COLORS = {
    NEW_BUILDING: (220, 50, 50),
    VEG_LOSS: (255, 140, 0),
    VEG_GAIN: (0, 200, 0),
    NEW_WATER: (30, 144, 255),
    WATER_LOSS: (160, 100, 40),
    NEW_ROAD: (210, 210, 210),
    SURFACE_CHANGE: (255, 220, 50),
    STRUCTURE_DAMAGE: (255, 0, 255),
}

# Categories shown on the primary satellite overlay (excludes noisy minor classes).
HIGHLIGHT_CATEGORIES = (
    STRUCTURE_DAMAGE,
    NEW_WATER,
    WATER_LOSS,
    VEG_LOSS,
    VEG_GAIN,
    NEW_BUILDING,
    NEW_ROAD,
    SURFACE_CHANGE,
)


def align_after_rgb(before_rgb: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
    """Resize the after image to match the before image (same pixel grid)."""
    h, w = before_rgb.shape[:2]
    if after_rgb.shape[:2] == (h, w):
        return after_rgb
    return cv2.resize(after_rgb, (w, h), interpolation=cv2.INTER_AREA)


def _change_score(before_rgb: np.ndarray, after_rgb: np.ndarray) -> np.ndarray:
    """Pixel change strength from intensity and texture differences."""
    after_rgb = align_after_rgb(before_rgb, after_rgb)
    g1 = cv2.GaussianBlur(cv2.cvtColor(before_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(after_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    intensity = cv2.absdiff(g1, g2).astype(np.float32)
    lap1 = np.abs(cv2.Laplacian(g1, cv2.CV_32F))
    lap2 = np.abs(cv2.Laplacian(g2, cv2.CV_32F))
    return intensity + 0.6 * np.abs(lap1 - lap2)


def detect_roof_footprints(before_rgb: np.ndarray) -> np.ndarray:
    """Label large roof footprints on the before image (0 = none, 1..N = roof id)."""
    h, w = before_rgb.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(before_rgb, cv2.COLOR_RGB2GRAY), (7, 7), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = int(h * w * 0.025)
    max_area = int(h * w * 0.22)
    labels = np.zeros((h, w), dtype=np.int32)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > 0.92 * w and bh > 0.92 * h:
            continue
        if (bw * bh) > 0.28 * h * w:
            continue
        aspect = bw / max(bh, 1)
        if not (0.35 <= aspect <= 3.0):
            continue
        candidates.append((area, c))
    candidates.sort(key=lambda item: item[0], reverse=True)

    roof_id = 1
    for _, contour in candidates[:6]:
        roof_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(roof_mask, [contour], -1, 255, cv2.FILLED)
        if np.count_nonzero(labels[roof_mask > 0]) > 0.3 * np.count_nonzero(roof_mask):
            continue
        labels[roof_mask > 0] = roof_id
        roof_id += 1
    return labels


def _structure_coherence(gray: np.ndarray, win: int = 9) -> np.ndarray:
    """Ridge/edge coherence in [0, 1] — high on regular corrugated roofs."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    j11 = cv2.blur(gx * gx, (win, win))
    j22 = cv2.blur(gy * gy, (win, win))
    j12 = cv2.blur(gx * gy, (win, win))
    tr = j11 + j22
    det = j11 * j22 - j12 * j12
    tmp = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    l1 = tr / 2.0 + tmp
    l2 = tr / 2.0 - tmp
    return (l1 - l2) / (l1 + l2 + 1e-6)


def _roof_damage_index(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    roof_mask: np.ndarray,
    change_score: np.ndarray,
) -> float:
    """Combined roof damage score in [0, 1] — structure loss, collapse, and change."""
    after_rgb = align_after_rgb(before_rgb, after_rgb)
    g1 = cv2.GaussianBlur(cv2.cvtColor(before_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)
    g2 = cv2.GaussianBlur(cv2.cvtColor(after_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)
    vals1 = g1[roof_mask]
    vals2 = g2[roof_mask]
    if vals1.size < 50:
        return 0.0

    coherence_loss = _structure_coherence(g1) - _structure_coherence(g2)
    struct_loss = float(np.clip(coherence_loss[roof_mask].mean() * 2.5, 0.0, 1.0))
    struct_frac = float(np.mean(coherence_loss[roof_mask] > 0.02))

    darkening = vals1 - vals2
    collapse_frac = float(np.mean(darkening > 12.0))

    local = change_score[roof_mask]
    hot_frac = float(np.mean(local >= np.percentile(local, 58)))

    return float(np.clip(
        0.45 * struct_loss + 0.30 * struct_frac + 0.15 * collapse_frac + 0.10 * hot_frac,
        0.0, 1.0,
    ))


def _roof_correlation_damage(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    roof_mask: np.ndarray,
) -> float:
    """Backward-compatible alias."""
    return _roof_damage_index(before_rgb, after_rgb, roof_mask, _change_score(before_rgb, after_rgb))


def assess_roof_footprint_damage(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    footprint_labels: np.ndarray,
    change_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect damaged roofs via correlation; return (damage_mask, severity_map)."""
    h, w = before_rgb.shape[:2]
    damage_mask = np.zeros((h, w), dtype=bool)
    severity = np.zeros((h, w), dtype=np.uint8)

    roof_ids = [int(i) for i in np.unique(footprint_labels) if i > 0]
    if not roof_ids:
        return damage_mask, severity

    roof_damages = {}
    struct_loss_fracs = {}
    g1 = cv2.GaussianBlur(cv2.cvtColor(before_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)
    g2 = cv2.GaussianBlur(cv2.cvtColor(align_after_rgb(before_rgb, after_rgb), cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)
    dark_map = g1 - g2
    coherence_loss = _structure_coherence(g1) - _structure_coherence(g2)

    for rid in roof_ids:
        roof_mask = footprint_labels == rid
        roof_damages[rid] = _roof_damage_index(before_rgb, after_rgb, roof_mask, change_score)
        struct_loss_fracs[rid] = float(np.mean(coherence_loss[roof_mask] > 0.02))

    if not roof_damages:
        return damage_mask, severity

    ranked = sorted(roof_damages.items(), key=lambda item: item[1], reverse=True)
    best_rid, best_damage = ranked[0]
    second_damage = ranked[1][1] if len(ranked) > 1 else 0.0

    for rid, damage in roof_damages.items():
        struct_frac = struct_loss_fracs[rid]
        is_primary = rid == best_rid and (best_damage >= 0.18 or best_damage - second_damage >= 0.05)
        if not is_primary:
            continue

        roof_mask = footprint_labels == rid
        interior = roof_mask.copy()
        interior_u8 = interior.astype(np.uint8) * 255
        interior_u8 = cv2.erode(
            interior_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1,
        )
        interior = interior_u8 > 0
        if not np.any(interior):
            interior = roof_mask

        local_scores = change_score[interior]
        struct_hit = interior & (coherence_loss > 0.015)
        dark_hit = interior & (dark_map > 5)
        score_hit = interior & (change_score >= float(np.percentile(local_scores, 25)))
        hit = struct_hit | (dark_hit & score_hit)

        if damage >= 0.28 or struct_frac >= 0.55:
            hit = struct_hit | dark_hit | score_hit
            level = SEVERITY_MAJOR
        elif damage >= 0.18:
            level = SEVERITY_MODERATE
        else:
            level = SEVERITY_MINOR

        if not np.any(hit):
            hit = interior & (coherence_loss > 0.01)

        damage_mask |= hit
        severity[hit] = np.maximum(severity[hit], level)

    return damage_mask, severity


def _water_score(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel water-likeness in [0, 1] from hue, saturation, and brightness."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue_ok = np.exp(-((h - 115.0) ** 2) / (2.0 * 25.0 ** 2))
    return hue_ok * (s / 255.0) * np.clip((v - 35.0) / 100.0, 0.0, 1.0)


def _validated_water_change(before_data: dict, after_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Water appearing or receding only when both dates support the transition.

    Rejects shadow and classifier noise: a pixel must gain (or lose) water-like
    color on the after image, not merely be misclassified on one date alone.
    """
    before_rgb = before_data["rgb"]
    after_rgb = align_after_rgb(before_rgb, after_data["rgb"])
    after_payload = {**after_data, "rgb": after_rgb}

    before_water = terrain_mod.water_mask(before_data)
    after_water = terrain_mod.water_mask(after_payload)
    before_score = _water_score(before_rgb)
    after_score = _water_score(after_rgb)
    after_v = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2HSV)[:, :, 2]

    score_rise = (after_score - before_score) > 0.12
    clearly_inundated = (after_score > 0.25) & (before_score < 0.12)
    bright_enough = after_v > 42

    new_water = (
        after_water
        & ~before_water
        & (score_rise | clearly_inundated)
        & bright_enough
    )

    score_drop = (before_score - after_score) > 0.12
    clearly_receded = (before_score > 0.25) & (after_score < 0.12)
    water_loss = before_water & ~after_water & (score_drop | clearly_receded)
    return new_water, water_loss


def structural_change_mask(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    before_mask: np.ndarray,
    *,
    sensitivity: float = 0.92,
) -> np.ndarray:
    """Significant pixel-level change, gated by the before-scene geography."""
    after_rgb = align_after_rgb(before_rgb, after_rgb)
    score = _change_score(before_rgb, after_rgb)

    # Anchor analysis to what was present before (avoids after-only classifier noise).
    site = (
        (before_mask == BUILDING)
        | (before_mask == BARE)
        | (before_mask == ROAD)
        | (before_mask == VEG)
        | (before_mask == WATER)
    )
    if not np.any(site):
        return np.zeros(before_rgb.shape[:2], dtype=bool)

    building = before_mask == BUILDING
    score = np.where(building, score * 1.3, score)
    score = np.where(before_mask == VEG, score * 0.9, score)

    pct = float(np.clip(sensitivity, 0.5, 0.99)) * 100.0
    thresh = float(np.percentile(score[site], pct))
    changed = (score > thresh) & site

    u8 = changed.astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, k3)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k7)

    min_area = max(80, int(before_rgb.shape[0] * before_rgb.shape[1] * 0.00025))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    out = np.zeros_like(u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out > 0


def _categorize_changes(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    structural: np.ndarray,
    new_water: np.ndarray,
    water_loss: np.ndarray,
) -> np.ndarray:
    """Assign each changed pixel a category using before-anchored, validated rules."""
    cat = np.full(before_mask.shape, NOCHANGE, dtype=np.uint8)
    assigned = np.zeros(before_mask.shape, dtype=bool)

    def _apply(code, mask):
        nonlocal assigned
        take = mask & ~assigned
        cat[take] = code
        assigned |= take

    _apply(NEW_WATER, new_water)
    _apply(WATER_LOSS, water_loss)

    _apply(
        VEG_LOSS,
        (before_mask == VEG) & (after_mask != VEG) & (structural | new_water),
    )
    _apply(
        VEG_GAIN,
        (before_mask != VEG) & (after_mask == VEG) & structural,
    )
    _apply(
        NEW_BUILDING,
        (before_mask != BUILDING) & (after_mask == BUILDING) & structural,
    )
    _apply(
        NEW_ROAD,
        (before_mask != ROAD) & (after_mask == ROAD) & structural,
    )
    _apply(
        STRUCTURE_DAMAGE,
        structural
        & (before_mask == BUILDING)
        & ~assigned,
    )
    _apply(
        SURFACE_CHANGE,
        structural
        & ((before_mask == BARE) | (before_mask == ROAD))
        & ~assigned,
    )
    _apply(
        SURFACE_CHANGE,
        structural & (before_mask == VEG) & ~assigned,
    )
    return cat


def detect_changes(before_mask: np.ndarray, after_mask: np.ndarray) -> dict:
    """Low-level mask comparison (used internally; prefer detect_changes_from_data)."""
    if before_mask.shape != after_mask.shape:
        after_mask = cv2.resize(
            after_mask.astype("int32"),
            (before_mask.shape[1], before_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(before_mask.dtype)
    category = np.full(before_mask.shape, NOCHANGE, dtype=np.uint8)
    category[before_mask != after_mask] = SURFACE_CHANGE
    return {
        "category": category,
        "changed": category != NOCHANGE,
        "after_mask": after_mask,
    }


def detect_changes_from_data(
    before_data: dict,
    after_data: dict,
    *,
    include_structural: bool = True,
    structural_sensitivity: float = 0.92,
) -> dict:
    """Full pipeline: align, validate water, score image change, categorize."""
    before_rgb = before_data["rgb"]
    after_rgb = align_after_rgb(before_rgb, after_data["rgb"])
    after_aligned = {**after_data, "rgb": after_rgb}

    before_mask = before_data.get("mask")
    if before_mask is None:
        before_mask = terrain_mod.extract_terrain(before_data)
    after_mask = terrain_mod.extract_terrain(after_aligned)

    new_water, water_loss = _validated_water_change(before_data, after_aligned)
    change_score = _change_score(before_rgb, after_rgb)
    footprint_labels = detect_roof_footprints(before_rgb)
    roof_damage, roof_severity = assess_roof_footprint_damage(
        before_rgb, after_rgb, footprint_labels, change_score,
    )

    structural = (
        structural_change_mask(
            before_rgb, after_rgb, before_mask,
            sensitivity=structural_sensitivity,
        )
        if include_structural
        else np.zeros(before_rgb.shape[:2], dtype=bool)
    )
    structural |= roof_damage

    category = _categorize_changes(
        before_mask, after_mask, structural, new_water, water_loss,
    )
    if np.any(roof_damage):
        category[roof_damage] = STRUCTURE_DAMAGE

    changed = category != NOCHANGE
    specific = specific_change_mask(category, change_score, always_include=roof_damage)
    severity = compute_severity_map(change_score, specific, severity_hint=roof_severity)

    if np.any(footprint_labels > 0) and np.any(roof_damage):
        damaged_ids = set(int(i) for i in np.unique(footprint_labels[roof_damage]) if i > 0)
        if damaged_ids:
            intact_roofs = footprint_labels > 0
            for rid in damaged_ids:
                intact_roofs &= footprint_labels != rid
            if np.any(intact_roofs):
                keep = (change_score >= np.percentile(change_score[intact_roofs], 92)) if np.any(intact_roofs) else np.zeros_like(specific)
                suppress = intact_roofs & ~keep
                specific[suppress] = False
                severity[suppress] = SEVERITY_NONE

    return {
        "category": category,
        "changed": changed,
        "specific": specific,
        "severity": severity,
        "change_score": change_score,
        "before_mask": before_mask,
        "after_mask": after_mask,
        "before_rgb": before_rgb,
        "after_rgb": after_rgb,
        "structural": structural,
        "new_water": new_water,
        "water_loss": water_loss,
        "roof_footprints": footprint_labels,
        "roof_damage": roof_damage,
    }


def specific_change_mask(
    category: np.ndarray,
    score: np.ndarray,
    *,
    keep_percentile: float = 42.0,
    always_include: np.ndarray = None,
) -> np.ndarray:
    """Keep the strongest changed pixels inside each region (not whole zones)."""
    refined = np.zeros(category.shape, dtype=bool)
    if always_include is not None:
        refined |= always_include

    per_code_percentile = {
        STRUCTURE_DAMAGE: 18.0,
        NEW_WATER: 30.0,
        WATER_LOSS: 30.0,
    }

    for code in HIGHLIGHT_CATEGORIES:
        comp_u8 = (category == code).astype(np.uint8) * 255
        if not np.any(comp_u8):
            continue
        pct = per_code_percentile.get(code, keep_percentile)
        num, labels, _, _ = cv2.connectedComponentsWithStats(comp_u8, connectivity=8)
        for i in range(1, num):
            region = labels == i
            region_scores = score[region]
            if region_scores.size < 12:
                refined |= region
                continue
            cutoff = float(np.percentile(region_scores, pct))
            core = region & (score >= cutoff)
            if code == STRUCTURE_DAMAGE and np.count_nonzero(core) < 0.15 * region_scores.size:
                core = region & (score >= float(np.percentile(region_scores, 10)))
            core_u8 = core.astype(np.uint8) * 255
            if code != STRUCTURE_DAMAGE:
                core_u8 = cv2.morphologyEx(
                    core_u8, cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                )
            refined |= core_u8 > 0
    return refined


def compute_severity_map(
    score: np.ndarray,
    specific_mask: np.ndarray,
    severity_hint: np.ndarray = None,
) -> np.ndarray:
    """Classify each specific changed pixel as minor, moderate, or major."""
    severity = np.zeros(score.shape, dtype=np.uint8)
    if severity_hint is not None:
        severity = np.maximum(severity, severity_hint)

    if not np.any(specific_mask):
        return severity

    vals = score[specific_mask]
    p33, p66 = np.percentile(vals, (33, 66))
    if p66 <= p33:
        p66 = p33 + 1e-6

    computed = np.zeros(score.shape, dtype=np.uint8)
    computed[specific_mask & (score <= p33)] = SEVERITY_MINOR
    computed[specific_mask & (score > p33) & (score <= p66)] = SEVERITY_MODERATE
    computed[specific_mask & (score > p66)] = SEVERITY_MAJOR
    return np.maximum(severity, computed)


def change_zones(
    category: np.ndarray,
    specific_mask: np.ndarray,
    severity: np.ndarray,
    *,
    min_area: int = 40,
) -> list[dict]:
    """Tight change clusters on specific pixels, with dominant type and severity."""
    zones = []
    if not np.any(specific_mask):
        return zones

    combined_u8 = specific_mask.astype(np.uint8) * 255
    combined_u8 = cv2.morphologyEx(
        combined_u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_u8, connectivity=8)

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cluster = labels == i
        cluster &= specific_mask
        if not np.any(cluster):
            continue

        type_codes, type_counts = np.unique(category[cluster], return_counts=True)
        dominant_code = int(type_codes[int(np.argmax(type_counts))])
        sev_in_cluster = severity[cluster]
        sev_nonzero = sev_in_cluster[sev_in_cluster > 0]
        if sev_nonzero.size:
            sev_vals, sev_counts = np.unique(sev_nonzero, return_counts=True)
            dominant_sev = int(sev_vals[int(np.argmax(sev_counts))])
        else:
            dominant_sev = SEVERITY_MINOR

        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        zones.append({
            "type": CHANGE_NAMES.get(dominant_code, "Change"),
            "code": dominant_code,
            "severity": SEVERITY_NAMES.get(dominant_sev, "Minor"),
            "severity_code": dominant_sev,
            "area_px": int(np.count_nonzero(cluster)),
            "area_pct": 100.0 * int(np.count_nonzero(cluster)) / category.size,
            "center": (cx, cy),
        })

    zones.sort(key=lambda z: (z["severity_code"], z["area_pct"]), reverse=True)
    for idx, z in enumerate(zones, start=1):
        z["id"] = idx
    return zones


def colorize_severity(severity: np.ndarray, base_rgb: np.ndarray) -> np.ndarray:
    """Severity-only view on a dimmed backdrop."""
    h, w = severity.shape
    if base_rgb.shape[:2] != (h, w):
        base_rgb = cv2.resize(base_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
    out = cv2.cvtColor((gray * 0.55).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for level, color in SEVERITY_COLORS.items():
        out[severity == level] = color
    return out


def colorize_change(category: np.ndarray, base_rgb: np.ndarray = None) -> np.ndarray:
    """Render the change map on a dimmed satellite backdrop."""
    h, w = category.shape
    if base_rgb is not None:
        if base_rgb.shape[:2] != (h, w):
            base_rgb = cv2.resize(base_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)
        out = cv2.cvtColor((gray * 0.45).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        out = np.full((h, w, 3), 40, dtype=np.uint8)
    for code, color in CHANGE_COLORS.items():
        out[category == code] = color
    return out


def highlight_changes_on_image(
    base_rgb: np.ndarray,
    severity: np.ndarray,
    specific_mask: np.ndarray,
    zones: list[dict] = None,
) -> np.ndarray:
    """Mark only specific changed pixels by severity — pink / black / red.

    Unchanged areas stay fully visible. Tight contours outline each cluster;
  no large bounding boxes over whole zones.
    """
    h, w = severity.shape
    if base_rgb.shape[:2] != (h, w):
        base_rgb = cv2.resize(base_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    out = base_rgb.astype(np.float32)
    for level in (SEVERITY_MINOR, SEVERITY_MODERATE, SEVERITY_MAJOR):
        mask = severity == level
        if not np.any(mask):
            continue
        color = np.array(SEVERITY_COLORS[level], dtype=np.float32)
        alpha = SEVERITY_ALPHA[level]
        out[mask] = out[mask] * (1.0 - alpha) + color * alpha

    out = np.clip(out, 0, 255).astype(np.uint8)
    line_w = max(1, min(h, w) // 220)

    for level in (SEVERITY_MINOR, SEVERITY_MODERATE, SEVERITY_MAJOR):
        level_u8 = (severity == level).astype(np.uint8) * 255
        if not np.any(level_u8):
            continue
        color = SEVERITY_COLORS[level]
        contours, _ = cv2.findContours(level_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 20:
                continue
            if level == SEVERITY_MODERATE:
                cv2.drawContours(out, [cnt], -1, (255, 255, 255), line_w + 1, cv2.LINE_AA)
            cv2.drawContours(out, [cnt], -1, color, line_w, cv2.LINE_AA)

    font_scale = max(0.4, min(h, w) / 1000.0)
    zones = zones or []
    for zone in zones[:10]:
        if zone.get("severity_code", 0) < SEVERITY_MODERATE:
            continue
        cx, cy = zone["center"]
        label = f"#{zone['id']}"
        cv2.putText(
            out, label, (cx + 6, cy),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            out, label, (cx + 6, cy),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            SEVERITY_COLORS.get(zone["severity_code"], (220, 20, 20)), 1, cv2.LINE_AA,
        )
    return out


def change_stats(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    category: np.ndarray,
    *,
    new_water: np.ndarray = None,
    water_loss: np.ndarray = None,
    specific_mask: np.ndarray = None,
    severity: np.ndarray = None,
) -> dict:
    """Validated change summary — not raw per-date classifier totals."""
    total = int(before_mask.size)
    specific_mask = specific_mask if specific_mask is not None else (category != NOCHANGE)
    severity = severity if severity is not None else np.zeros(category.shape, dtype=np.uint8)

    categories = []
    for code, name in CHANGE_NAMES.items():
        count = int(np.count_nonzero(category == code))
        if count:
            categories.append({
                "name": name,
                "code": code,
                "count": count,
                "percentage": 100.0 * count / total,
            })
    categories.sort(key=lambda c: c["percentage"], reverse=True)

    specific_pct = 100.0 * int(np.count_nonzero(specific_mask)) / total
    changed_pct = specific_pct

    summary = [f"{c['name']}: {c['percentage']:.1f}% of scene" for c in categories]
    if not summary:
        summary = ["No significant change detected between the two dates."]

    severity_breakdown = []
    for level, name in SEVERITY_NAMES.items():
        pct = 100.0 * int(np.count_nonzero(severity == level)) / total
        if pct > 0:
            severity_breakdown.append({"name": name, "percentage": pct})

    validated_water_pct = 0.0
    if new_water is not None:
        validated_water_pct = 100.0 * int(np.count_nonzero(new_water)) / total

    zones = change_zones(category, specific_mask, severity)

    return {
        "total_changed_pct": changed_pct,
        "specific_changed_pct": specific_pct,
        "categories": categories,
        "summary": summary,
        "severity_breakdown": severity_breakdown,
        "zones": zones,
        "validated_water_pct": validated_water_pct,
        "structure_damage_pct": next(
            (c["percentage"] for c in categories if c["code"] == STRUCTURE_DAMAGE), 0.0,
        ),
        "veg_loss_pct": next(
            (c["percentage"] for c in categories if c["code"] == VEG_LOSS), 0.0,
        ),
        "surface_change_pct": next(
            (c["percentage"] for c in categories if c["code"] == SURFACE_CHANGE), 0.0,
        ),
        "major_pct": next((s["percentage"] for s in severity_breakdown if s["name"] == "Major"), 0.0),
        "moderate_pct": next((s["percentage"] for s in severity_breakdown if s["name"] == "Moderate"), 0.0),
        "minor_pct": next((s["percentage"] for s in severity_breakdown if s["name"] == "Minor"), 0.0),
    }


def summarize_detection(result: dict) -> dict:
    """Build the stats dict from a ``detect_changes_from_data`` result."""
    return change_stats(
        result["before_mask"],
        result["after_mask"],
        result["category"],
        new_water=result.get("new_water"),
        water_loss=result.get("water_loss"),
        specific_mask=result.get("specific"),
        severity=result.get("severity"),
    )
