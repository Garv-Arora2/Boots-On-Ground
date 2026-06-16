"""Mission presets, area stats, and route summary metrics for the UI."""

import math

import numpy as np

from . import planning, routing, terrain
from .planning import circle_mask, risk_map
from .terrain import ROAD, VEG, WATER, BUILDING, BARE

OBJECTIVES = [
    "Reach Destination",
    "Search & Rescue",
    "Deliver Supplies",
    "Recon Mission",
    "Evacuation",
]

# Display metadata for agent cards (sidebar + comparison table).
AGENT_DISPLAY = {
    "Foot soldier": {"speed": "Medium", "terrain": "Excellent", "label": "Soldier"},
    "Light vehicle (jeep)": {"speed": "High", "terrain": "Poor", "label": "Vehicle"},
    "Truck / convoy": {"speed": "Medium", "terrain": "Poor", "label": "Convoy"},
    "Tank": {"speed": "Low", "terrain": "Good", "label": "Tank"},
}

# One-click mission presets (synthetic scenes — always available offline).
MISSION_TEMPLATES = {
    "Search & Rescue": {
        "scene": "Forest & lake",
        "area": "Sector Alpha",
        "objective": "Search & Rescue",
        "agent": "Foot soldier",
        "risk": 0.65,
        "start": (70, 70),
        "end": (440, 440),
        "threats": [(280, 260, 48)],
        "avoids": [],
    },
    "Flood Response": {
        "scene": "River crossing",
        "area": "Delta Sector",
        "objective": "Evacuation",
        "agent": "Foot soldier",
        "risk": 0.5,
        "start": (60, 220),
        "end": (450, 220),
        "threats": [(210, 256, 52)],
        "avoids": [(200, 320, 38)],
    },
    "Military Recon": {
        "scene": "Mixed terrain",
        "area": "Grid 7-Bravo",
        "objective": "Recon Mission",
        "agent": "Foot soldier",
        "risk": 0.85,
        "start": (50, 50),
        "end": (460, 460),
        "threats": [(256, 256, 58), (160, 340, 42)],
        "avoids": [],
    },
    "Disaster Relief": {
        "scene": "Urban grid",
        "area": "City Block C",
        "objective": "Deliver Supplies",
        "agent": "Truck / convoy",
        "risk": 0.35,
        "start": (80, 80),
        "end": (430, 430),
        "threats": [(300, 300, 45)],
        "avoids": [(200, 200, 55)],
    },
    "Supply Delivery": {
        "scene": "Desert outpost",
        "area": "Outpost Nine",
        "objective": "Deliver Supplies",
        "agent": "Light vehicle (jeep)",
        "risk": 0.25,
        "start": (60, 400),
        "end": (400, 120),
        "threats": [(200, 250, 40)],
        "avoids": [],
    },
    "General transit": {
        "scene": "Mixed terrain",
        "area": "Open sector",
        "objective": "Reach Destination",
        "agent": "Foot soldier",
        "risk": 0.0,
        "start": (60, 60),
        "end": (450, 450),
        "threats": [],
        "avoids": [],
    },
}


def meters_per_pixel(bounds_wgs84, shape):
    """Approximate ground resolution; 1.0 px = 1 m when not geo-referenced."""
    if not bounds_wgs84:
        return 1.0
    west, south, east, north = bounds_wgs84
    h, w = shape
    lat = (south + north) / 2.0
    m_lon = 111_320 * math.cos(math.radians(lat))
    return ((east - west) * m_lon / max(w, 1) + (north - south) * 111_320 / max(h, 1)) / 2.0


def path_length_m(path, mpp):
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        total += math.hypot(dr, dc) * mpp
    return total


# Terrain-class speed (m/s) for travel-time estimates.
_SPEED = {ROAD: 1.4, BARE: 1.0, VEG: 0.7, BUILDING: 0.5, WATER: 0.35}


def estimate_travel_time(path, mask, agent_name, mpp):
    if not path:
        return 0.0
    agent = planning.AGENTS[agent_name]
    mult = 1.0 if "vehicle" in agent_name.lower() or "truck" in agent_name.lower() else 1.0
    if "Tank" in agent_name:
        mult = 0.85
    elif "vehicle" in agent_name.lower():
        mult = 2.5
    t = 0.0
    for i in range(1, len(path)):
        r, c = path[i]
        code = int(mask[r, c]) if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] else BARE
        spd = max(_SPEED.get(code, 1.0) * mult, 0.2)
        step_m = math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) * mpp
        t += step_m / spd
    return t


def format_duration(seconds):
    if seconds <= 0:
        return "-"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _level(score, low, high):
    if score < low:
        return "Low"
    if score < high:
        return "Medium"
    return "High"


def recommend_agent(stats):
    """Heuristic agent pick from area composition."""
    b = stats["breakdown"]
    water = b.get("Water", {}).get("percentage", 0)
    road = b.get("Road", {}).get("percentage", 0)
    veg = b.get("Vegetation", {}).get("percentage", 0)
    if water > 12:
        return "Foot soldier"
    if road > 8 and veg < 25:
        return "Light vehicle (jeep)"
    if veg > 35:
        return "Foot soldier"
    return "Foot soldier"


def area_intelligence(mask, stats):
    """Hero panel: traversability and hazard overview."""
    b = stats["breakdown"]
    water_pct = b.get("Water", {}).get("percentage", 0)
    road_pct = b.get("Road", {}).get("percentage", 0)
    veg_pct = b.get("Vegetation", {}).get("percentage", 0)
    impassable = water_pct + b.get("Building", {}).get("percentage", 0) * 0.5
    traversable = max(0.0, 100.0 - impassable)
    return {
        "traversable_pct": traversable,
        "terrain_risk": _level(veg_pct * 0.4 + water_pct * 0.8, 8, 20),
        "road_access": _level(road_pct, 3, 10),
        "water_hazard": _level(water_pct, 4, 12),
        "recommended_agent": recommend_agent(stats),
    }


def terrain_difficulty(stats):
    b = stats["breakdown"]
    weights = {"Road": 1, "Bare Land": 3, "Vegetation": 5, "Building": 8, "Water": 10}
    avg_cost_proxy = sum(
        b.get(name, {}).get("percentage", 0) * w / 100.0 for name, w in weights.items()
    )
    return _level(avg_cost_proxy, 2.5, 4.5)


def threat_exposure(path, threat_zones, shape):
    if not path or not threat_zones:
        return 0.0, "Low"
    risk = risk_map(shape, threat_zones)
    vals = [float(risk[r, c]) for r, c in path if 0 <= r < shape[0] and 0 <= c < shape[1]]
    if not vals:
        return 0.0, "Low"
    peak = 10.0
    avg = sum(vals) / len(vals)
    pct = 100.0 * avg / peak
    return pct, _level(pct, 15, 40)


def route_terrain_shares(path, mask):
    if not path:
        return {}
    counts = {}
    for r, c in path:
        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
            code = int(mask[r, c])
            counts[code] = counts.get(code, 0) + 1
    total = max(sum(counts.values()), 1)
    return {terrain.NAMES.get(k, str(k)): 100.0 * v / total for k, v in counts.items()}


def route_quality(path, mask, threat_zones, found):
    if not found or not path:
        return "Low"
    _, exp_level = threat_exposure(path, threat_zones, mask.shape)
    shares = route_terrain_shares(path, mask)
    road = shares.get("Road", 0)
    water = shares.get("Water", 0)
    score = 50 + road * 0.4 - water * 2
    if exp_level == "Low":
        score += 15
    elif exp_level == "High":
        score -= 20
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def explain_route_why(path, mask, threat_zones, avoid_zones, risk_weight, shares):
    """Short bullet points explaining why the route looks the way it does."""
    bullets = []
    if not path:
        return ["No route generated."]

    shape = mask.shape
    risk_layer = risk_map(shape, threat_zones) if threat_zones else None
    entered_threat = False
    if risk_layer is not None:
        for r, c in path:
            if risk_layer[r, c] > 1.0:
                entered_threat = True
                break
    if threat_zones and not entered_threat:
        bullets.append(f"Avoided {len(threat_zones)} threat zone(s)")
    elif threat_zones and risk_weight > 0:
        bullets.append(f"Minimized exposure across {len(threat_zones)} threat zone(s)")

    if shares.get("Water", 0) < 1:
        bullets.append("Avoided water bodies")
    elif shares.get("Water", 0) > 0:
        bullets.append(f"Crossed water for {shares['Water']:.0f}% of the route (fording)")

    if avoid_zones:
        bullets.append(f"Routed around {len(avoid_zones)} no-go zone(s)")

    road = shares.get("Road", 0)
    if road >= 5:
        bullets.append(f"Used roads for {road:.0f}% of travel")
    veg = shares.get("Vegetation", 0)
    if veg >= 5:
        bullets.append(f"Crossed vegetation for {veg:.0f}% of travel")

    if risk_weight >= 0.5 and threat_zones:
        bullets.append("Prioritized safety over shortest distance")

    if not bullets:
        bullets.append("Balanced terrain cost for the shortest passable path")
    return bullets


def mission_analysis(path, mask, stats, agent_name, threat_zones, risk_weight, mpp, found):
    shares = route_terrain_shares(path, mask)
    exp_pct, exp_level = threat_exposure(path, threat_zones, mask.shape)
    return {
        "terrain_difficulty": terrain_difficulty(stats),
        "threat_exposure": exp_level,
        "threat_exposure_pct": exp_pct,
        "travel_time": format_duration(estimate_travel_time(path, mask, agent_name, mpp)),
        "travel_seconds": estimate_travel_time(path, mask, agent_name, mpp),
        "water_crossings": 1 if shares.get("Water", 0) > 0 else 0,
        "road_pct": shares.get("Road", 0),
        "veg_pct": shares.get("Vegetation", 0),
        "route_quality": route_quality(path, mask, threat_zones, found),
        "recommended_algo": "A*",
        "distance_m": path_length_m(path, mpp),
        "shares": shares,
    }


def compare_agents(mask, start, end, slope, avoid_zones, threat_zones, risk_weight, mpp):
    """Run A* for every agent profile; return rows for the comparison table."""
    rows = []
    for name in planning.AGENT_NAMES:
        agent = planning.AGENTS[name]
        cost_map = planning.compose_cost(
            mask, agent, slope=slope,
            avoid_zones=avoid_zones, threat_zones=threat_zones,
            risk_weight=risk_weight,
        )
        result = routing.astar(cost_map, start, end)
        if not result["found"]:
            rows.append({
                "Agent": AGENT_DISPLAY[name]["label"],
                "Cost": "Blocked",
                "Time": "-",
                "Status": "No path",
            })
        else:
            t = estimate_travel_time(result["path"], mask, name, mpp)
            rows.append({
                "Agent": AGENT_DISPLAY[name]["label"],
                "Cost": f"{result['total_cost']:.0f}",
                "Time": format_duration(t),
                "Status": "OK",
            })
    return rows


def change_impact(cstats):
    """Headline metrics for the change-detection tab (validated categories only)."""
    cats = {c["name"]: c["percentage"] for c in cstats.get("categories", [])}
    return {
        "area_changed_pct": cstats.get("specific_changed_pct", cstats.get("total_changed_pct", 0)),
        "structure_damage_pct": cstats.get("structure_damage_pct", 0),
        "water_change_pct": cstats.get("validated_water_pct", 0),
        "vegetation_lost_pct": cstats.get("veg_loss_pct", 0),
        "surface_change_pct": cstats.get("surface_change_pct", 0),
        "new_construction_pct": cats.get("New construction", 0),
        "zone_count": len(cstats.get("zones", [])),
        "major_pct": cstats.get("major_pct", 0),
        "moderate_pct": cstats.get("moderate_pct", 0),
        "minor_pct": cstats.get("minor_pct", 0),
    }


def disaster_impact(cstats):
    """Backward-compatible alias for change_impact."""
    impact = change_impact(cstats)
    return {
        **impact,
        "road_access_label": "-",
        "buildings_damaged_pct": impact["structure_damage_pct"] + impact["new_construction_pct"],
    }
