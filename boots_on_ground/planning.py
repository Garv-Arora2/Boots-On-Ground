"""Mission planning layer: agent profiles, operator zones, and the layered
cost model that turns terrain + constraints into a single cost map.

This is where the "everything is a cost layer" idea lives. The terrain mask
(from terrain.py) and the search (routing.py) stay simple; this module composes:

    base terrain cost (per agent)
    + slope penalty (and slope-limit -> impassable per agent)
    + risk from threat zones (weighted by a fast<->safe slider)
    + operator avoid zones -> impassable

Different agents (foot/vehicle/tank) have different cost tables and limits, so a
foot soldier can wade shallow water while a truck cannot - the same map, a
different profile.
"""

from dataclasses import dataclass

import cv2
import numpy as np

IMPASSABLE = 999.0

# Terrain codes (mirror terrain.py): 0 bare, 1 road, 2 building, 3 veg, 4 water


@dataclass(frozen=True)
class Agent:
    name: str
    costs: dict          # terrain code -> base movement cost
    max_slope_deg: float  # steeper than this is impassable for this agent
    note: str = ""


# Note how water cost encodes "fording ability": the foot soldier can wade
# (high but finite), vehicles cannot (IMPASSABLE). Tanks crush veg/buildings.
AGENTS = {
    "Foot soldier": Agent(
        "Foot soldier",
        {0: 2.0, 1: 1.0, 2: 8.0, 3: 4.0, 4: 12.0},
        max_slope_deg=35.0,
        note="On foot: can wade shallow water (slowly) and handle steep ground.",
    ),
    "Light vehicle (jeep)": Agent(
        "Light vehicle (jeep)",
        {0: 3.0, 1: 1.0, 2: IMPASSABLE, 3: 8.0, 4: IMPASSABLE},
        max_slope_deg=25.0,
        note="Road-biased; cannot cross water or buildings; moderate slope limit.",
    ),
    "Truck / convoy": Agent(
        "Truck / convoy",
        {0: 5.0, 1: 1.0, 2: IMPASSABLE, 3: 12.0, 4: IMPASSABLE},
        max_slope_deg=15.0,
        note="Heavily road-dependent; low slope limit; no off-road water.",
    ),
    "Tank": Agent(
        "Tank",
        {0: 2.0, 1: 1.0, 2: 6.0, 3: 3.0, 4: IMPASSABLE},
        max_slope_deg=30.0,
        note="Can crush vegetation and buildings; still stopped by deep water.",
    ),
}

AGENT_NAMES = list(AGENTS.keys())


def circle_mask(shape, zones):
    """Boolean mask = union of circular zones. Each zone is (row, col, radius)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if not zones:
        return mask
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    for r, c, rad in zones:
        rad = max(int(rad), 1)
        mask |= ((yy - r) ** 2 + (xx - c) ** 2) <= rad ** 2
    return mask


def risk_map(shape, threat_zones, peak=10.0):
    """Continuous danger score: highest at each threat center, fading to 0 at its
    radius. Overlapping zones take the max (kept bounded and easy to reason about).
    """
    h, w = shape
    risk = np.zeros((h, w), dtype="float32")
    if not threat_zones:
        return risk
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    for r, c, rad in threat_zones:
        rad = max(int(rad), 1)
        dist = np.sqrt((yy - r) ** 2 + (xx - c) ** 2)
        contrib = np.clip(1.0 - dist / rad, 0.0, 1.0) * peak
        risk = np.maximum(risk, contrib)
    return risk


def compose_cost(mask, agent, slope=None, avoid_zones=None, threat_zones=None,
                 risk_weight=0.0, slope_factor=0.5):
    """Build the final cost map from terrain + agent + constraints.

    risk_weight: 0 = ignore threats (fastest route), higher = avoid danger (safest).
    """
    h, w = mask.shape
    cost = np.zeros((h, w), dtype="float32")
    for code, value in agent.costs.items():
        cost[mask == code] = float(value)

    passable = cost < 900

    if slope is not None:
        if slope.shape != mask.shape:
            slope = cv2.resize(slope.astype("float32"), (w, h), interpolation=cv2.INTER_LINEAR)
        penalty = np.clip(slope, 0, 45) * slope_factor
        cost[passable] = cost[passable] + penalty[passable]
        too_steep = passable & (slope > agent.max_slope_deg)
        cost[too_steep] = IMPASSABLE
        passable = cost < 900

    if threat_zones and risk_weight > 0:
        risk = risk_map((h, w), threat_zones)
        cost[passable] = cost[passable] + risk_weight * risk[passable]

    if avoid_zones:
        cost[circle_mask((h, w), avoid_zones)] = IMPASSABLE

    return cost
