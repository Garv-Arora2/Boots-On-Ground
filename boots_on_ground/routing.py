"""A* and Dijkstra on a 2D cost grid (8-connected, diagonal steps cost sqrt(2))."""

import heapq
import math
import time

import cv2
import numpy as np

MAX_GRID = 384          # cap the working grid so pure-Python search stays fast
IMPASSABLE = 900.0      # cells at/above this cost are walls (e.g. water = 999)
SQRT2 = math.sqrt(2.0)

# (row offset, col offset, step distance) for the 8 movement directions.
NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
]


def _downsample(cost_map: np.ndarray):
    """Shrink the cost map to at most MAX_GRID on its long side.

    We use nearest-neighbor so impassable water cells (999) keep their value and
    never get averaged into something walkable.
    """
    h, w = cost_map.shape
    longest = max(h, w)
    if longest <= MAX_GRID:
        return cost_map.astype("float32"), (h, w)
    scale = MAX_GRID / longest
    gh, gw = max(1, int(h * scale)), max(1, int(w * scale))
    grid = cv2.resize(cost_map.astype("float32"), (gw, gh), interpolation=cv2.INTER_NEAREST)
    return grid, (h, w)


def _to_grid(pt, full_shape, grid_shape):
    """Map a full-image (row, col) point onto the downsampled grid."""
    fr, fc = pt
    fh, fw = full_shape
    gh, gw = grid_shape
    gr = int(round(fr * (gh - 1) / max(fh - 1, 1)))
    gc = int(round(fc * (gw - 1) / max(fw - 1, 1)))
    return (min(max(gr, 0), gh - 1), min(max(gc, 0), gw - 1))


def _upscale_path(path, full_shape, grid_shape):
    """Map a grid path back to full-image pixel coordinates for drawing."""
    fh, fw = full_shape
    gh, gw = grid_shape
    out = []
    for gr, gc in path:
        r = int(round(gr * (fh - 1) / max(gh - 1, 1)))
        c = int(round(gc * (fw - 1) / max(gw - 1, 1)))
        out.append((r, c))
    return out


def _reconstruct(came_from, end):
    """Walk the came_from links backward from the goal to rebuild the path."""
    path = [end]
    while end in came_from:
        end = came_from[end]
        path.append(end)
    path.reverse()
    return path


def _search(cost_map, start, end, use_heuristic):
    t0 = time.perf_counter()
    grid, full_shape = _downsample(cost_map)
    gh, gw = grid.shape
    s = _to_grid(start, full_shape, (gh, gw))
    e = _to_grid(end, full_shape, (gh, gw))

    # The heuristic must never overestimate the true remaining cost (this is the
    # "admissibility" condition that keeps A* optimal). The cheapest any single
    # step can cost is min_cost, so straight-line distance * min_cost is a safe,
    # always-optimistic guess.
    passable_vals = grid[grid < IMPASSABLE]
    min_cost = float(passable_vals.min()) if passable_vals.size else 1.0

    def heuristic(node):
        if not use_heuristic:
            return 0.0  # h = 0 turns A* into Dijkstra
        return math.hypot(node[0] - e[0], node[1] - e[1]) * min_cost

    open_heap = []
    counter = 0  # tie-breaker so the heap never has to compare (row, col) tuples
    g_score = {s: 0.0}
    came_from = {}
    visited = set()
    nodes_visited = 0
    heapq.heappush(open_heap, (heuristic(s), counter, s))

    found = False
    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        nodes_visited += 1

        if current == e:
            found = True
            break

        cr, cc = current
        for dr, dc, step in NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nc < 0 or nr >= gh or nc >= gw:
                continue
            cell = grid[nr, nc]
            if cell >= IMPASSABLE:
                continue  # can't walk through water/walls
            tentative_g = g_score[current] + cell * step
            if tentative_g < g_score.get((nr, nc), math.inf):
                g_score[(nr, nc)] = tentative_g
                came_from[(nr, nc)] = current
                counter += 1
                f = tentative_g + heuristic((nr, nc))
                heapq.heappush(open_heap, (f, counter, (nr, nc)))

    runtime_ms = (time.perf_counter() - t0) * 1000.0

    if found:
        grid_path = _reconstruct(came_from, e)
        path = _upscale_path(grid_path, full_shape, (gh, gw))
        total_cost = g_score[e]
    else:
        path = []
        total_cost = math.inf

    return {
        "path": path,
        "total_cost": total_cost,
        "path_length_px": len(path),
        "nodes_visited": nodes_visited,
        "runtime_ms": runtime_ms,
        "algorithm": "A*" if use_heuristic else "Dijkstra",
        "found": found,
    }


def astar(cost_map, start, end):
    """A* search. start/end are (row, col) in full-image pixels."""
    return _search(cost_map, start, end, use_heuristic=True)


def dijkstra(cost_map, start, end):
    """Dijkstra search (A* with a zero heuristic)."""
    return _search(cost_map, start, end, use_heuristic=False)


def compare(a: dict, d: dict) -> dict:
    """Summarize A* vs Dijkstra for the report/stats."""
    both_found = a["found"] and d["found"]
    if both_found:
        cost_diff = abs(a["total_cost"] - d["total_cost"])
        denom = max(a["total_cost"], 1.0)
        same_cost = cost_diff < 0.001 * denom
    else:
        cost_diff = math.inf
        same_cost = False
    eff = (d["nodes_visited"] / a["nodes_visited"]) if a["nodes_visited"] else 1.0
    return {
        "astar_cost": a["total_cost"],
        "dijkstra_cost": d["total_cost"],
        "cost_difference": cost_diff,
        "astar_nodes": a["nodes_visited"],
        "dijkstra_nodes": d["nodes_visited"],
        "efficiency_ratio": eff,
        "astar_time_ms": a["runtime_ms"],
        "dijkstra_time_ms": d["runtime_ms"],
        "same_cost": same_cost,
    }


def draw_path(image, path, color_rgb, thickness=2):
    """Draw a route on a copy of an RGB image; green start dot, red end dot."""
    out = image.copy()
    if not path:
        return out
    pts = np.array([[c, r] for r, c in path], dtype=np.int32)  # cv2 wants (x, y)
    cv2.polylines(out, [pts], isClosed=False, color=color_rgb, thickness=thickness)
    (r0, c0), (r1, c1) = path[0], path[-1]
    cv2.circle(out, (c0, r0), 6, (0, 255, 0), -1)
    cv2.circle(out, (c1, r1), 6, (255, 0, 0), -1)
    return out


def explain_path(path, mask, elev_stats=None) -> dict:
    """Plain-language account of what a route actually crosses, for the report
    and UI. Counts the terrain class under each step and flags notable features.
    """
    from .terrain import NAMES, WATER, BUILDING  # local import avoids a cycle

    if not path:
        return {"breakdown": [], "dominant": None, "fords": 0, "buildings": 0, "text": "No route to explain."}

    counts = {}
    for r, c in path:
        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
            code = int(mask[r, c])
            counts[code] = counts.get(code, 0) + 1

    total = max(sum(counts.values()), 1)
    breakdown = sorted(
        ((NAMES.get(code, str(code)), 100.0 * n / total) for code, n in counts.items()),
        key=lambda kv: kv[1], reverse=True,
    )
    dominant = breakdown[0][0] if breakdown else None
    fords = counts.get(WATER, 0)
    buildings = counts.get(BUILDING, 0)

    parts = [f"The route runs mostly over {dominant.lower()} ({breakdown[0][1]:.0f}% of its length)."]
    if len(breakdown) > 1:
        rest = ", ".join(f"{name.lower()} {pct:.0f}%" for name, pct in breakdown[1:])
        parts.append(f"It also crosses {rest}.")
    if fords:
        parts.append(f"It fords water in {fords} cells (only possible for agents that can wade).")
    if buildings:
        parts.append(f"It passes through {buildings} built-up cells.")
    if elev_stats:
        parts.append(
            f"Elevation ranges {elev_stats['min_elevation']:.0f}-{elev_stats['max_elevation']:.0f} m "
            f"with {elev_stats['total_gain']:.0f} m of total climb."
        )
    return {
        "breakdown": breakdown,
        "dominant": dominant,
        "fords": fords,
        "buildings": buildings,
        "text": " ".join(parts),
    }


def random_passable_points(mask, water_code=4, building_code=2, seed=None):
    """Pick two random points that are not on water or buildings (for the UI)."""
    rng = np.random.default_rng(seed)
    candidates = np.argwhere((mask != water_code) & (mask != building_code))
    if len(candidates) < 2:
        h, w = mask.shape
        return (0, 0), (h - 1, w - 1)
    i, j = rng.choice(len(candidates), size=2, replace=False)
    start = tuple(int(x) for x in candidates[i])
    end = tuple(int(x) for x in candidates[j])
    return start, end
