"""HTML route report export."""

import base64
import io
import math
from datetime import datetime

import numpy as np
from jinja2 import Template
from PIL import Image

_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>BOG Route Intelligence Report</title>
<style>
  :root{--bg:#0f1117;--surface:#1a1d27;--border:#2d3147;--accent:#00d4aa;--accent2:#ff8800;--text:#e8eaf0;--muted:#8892b0}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:2rem}
  .header{text-align:center;padding:1.5rem 0;border-bottom:1px solid var(--border);margin-bottom:1.5rem}
  .title{font-size:1.8rem;font-weight:700;color:var(--accent);letter-spacing:2px}
  .subtitle{color:var(--muted);margin-top:.4rem}
  .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.5rem}
  .grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.5rem;margin-bottom:1.5rem}
  .card h2{font-size:.75rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:1rem}
  .stat-value{font-size:1.8rem;font-weight:700;color:var(--accent)}
  .stat-label{font-size:.8rem;color:var(--muted);margin-top:.25rem}
  .metric-row{display:flex;justify-content:space-between;padding:.5rem 0;border-bottom:1px solid var(--border)}
  .metric-row:last-child{border-bottom:none}
  .metric-label{color:var(--muted);font-size:.9rem}
  .astar{color:#00ff88}.dijkstra{color:var(--accent2)}
  .bar-row{margin-bottom:.7rem}
  .bar-label{display:flex;justify-content:space-between;margin-bottom:.25rem;font-size:.85rem}
  .bar-bg{background:var(--border);border-radius:4px;height:8px;overflow:hidden}
  .bar-fill{height:100%;border-radius:4px}
  .img-container img{max-width:100%;border-radius:6px;border:1px solid var(--border)}
  .img-label{font-size:.8rem;color:var(--muted);margin-top:.5rem;text-align:center}
  .highlight{background:rgba(0,212,170,.1);border:1px solid var(--accent);border-radius:6px;padding:1rem;margin-top:1rem;color:var(--accent);font-size:.9rem}
  .footer{text-align:center;padding:1.5rem 0;color:var(--muted);font-size:.8rem;border-top:1px solid var(--border);margin-top:1rem}
</style></head><body>
<div class="header">
  <div class="title">BOOTS ON GROUND</div>
  <div class="subtitle">Terrain Intelligence &amp; Route Planning Report</div>
  <div class="subtitle" style="font-size:.8rem">Generated {{ timestamp }} | Source: {{ filename }}</div>
  {% if mission %}<div class="subtitle" style="font-size:.8rem">Agent: {{ mission.agent }} | Avoid zones: {{ mission.avoid }} | Threat zones: {{ mission.threat }}</div>{% endif %}
</div>

<div class="grid-2">
  <div class="card"><h2>Satellite Image</h2>
    <div class="img-container"><img src="data:image/png;base64,{{ image_b64 }}">
    <div class="img-label">Original input</div></div></div>
  <div class="card"><h2>Terrain Classification</h2>
    <div class="img-container"><img src="data:image/png;base64,{{ terrain_b64 }}">
    <div class="img-label">Classified terrain layers</div></div></div>
</div>

<div class="card"><h2>Terrain Composition</h2>
  {% for name, pct, color in terrain_bars %}
  <div class="bar-row"><div class="bar-label"><span>{{ name }}</span><span>{{ "%.1f"|format(pct) }}%</span></div>
  <div class="bar-bg"><div class="bar-fill" style="width:{{ pct }}%;background:{{ color }}"></div></div></div>
  {% endfor %}
</div>

<div class="grid-2">
  <div class="card"><h2>A* Route</h2>
    <div class="metric-row"><span class="metric-label">Found</span><span class="astar">{{ astar.found_str }}</span></div>
    <div class="metric-row"><span class="metric-label">Total Cost</span><span class="astar">{{ astar.cost }}</span></div>
    <div class="metric-row"><span class="metric-label">Path Length</span><span class="astar">{{ astar.length }} px</span></div>
    <div class="metric-row"><span class="metric-label">Nodes Visited</span><span class="astar">{{ astar.nodes }}</span></div>
    <div class="metric-row"><span class="metric-label">Runtime</span><span class="astar">{{ astar.runtime }} ms</span></div>
  </div>
  <div class="card"><h2>Dijkstra Route</h2>
    <div class="metric-row"><span class="metric-label">Found</span><span class="dijkstra">{{ dijkstra.found_str }}</span></div>
    <div class="metric-row"><span class="metric-label">Total Cost</span><span class="dijkstra">{{ dijkstra.cost }}</span></div>
    <div class="metric-row"><span class="metric-label">Path Length</span><span class="dijkstra">{{ dijkstra.length }} px</span></div>
    <div class="metric-row"><span class="metric-label">Nodes Visited</span><span class="dijkstra">{{ dijkstra.nodes }}</span></div>
    <div class="metric-row"><span class="metric-label">Runtime</span><span class="dijkstra">{{ dijkstra.runtime }} ms</span></div>
  </div>
</div>

<div class="card"><h2>Algorithm Comparison</h2>
  <div class="grid-3">
    <div><div class="stat-value">{{ comparison.efficiency }}x</div><div class="stat-label">A* efficiency (fewer nodes)</div></div>
    <div><div class="stat-value">{{ comparison.cost_diff }}</div><div class="stat-label">Cost difference</div></div>
    <div><div class="stat-value">{{ comparison.optimality }}</div><div class="stat-label">Route optimality</div></div>
  </div>
  <div class="highlight">{{ comparison.takeaway }}</div>
</div>

{% if elevation %}
<div class="card"><h2>Elevation Analysis (NASA SRTM)</h2>
  <div class="grid-3">
    <div><div class="stat-value">{{ elevation.min }} m</div><div class="stat-label">Min elevation</div></div>
    <div><div class="stat-value">{{ elevation.max }} m</div><div class="stat-label">Max elevation</div></div>
    <div><div class="stat-value">{{ elevation.gain }} m</div><div class="stat-label">Elevation gain</div></div>
  </div>
</div>
{% endif %}

{% if route_explain %}
<div class="card"><h2>Route Explanation</h2>
  <p style="line-height:1.6">{{ route_explain.text }}</p>
  <div style="margin-top:1rem">
  {% for name, pct in route_explain.breakdown %}
  <div class="bar-row"><div class="bar-label"><span>{{ name }}</span><span>{{ "%.0f"|format(pct) }}%</span></div>
  <div class="bar-bg"><div class="bar-fill" style="width:{{ pct }}%;background:var(--accent)"></div></div></div>
  {% endfor %}
  </div>
</div>
{% endif %}

<div class="footer">
  <p>Boots on Ground (BOG) — © 2026 Garv Arora · <a href="https://github.com/Garv-Arora2">github.com/Garv-Arora2</a></p>
  <p>A* + Dijkstra pathfinding | OpenCV terrain classification | NASA SRTM elevation</p>
</div>
</body></html>"""
)

_BAR_COLORS = {
    "Road": "#808080", "Vegetation": "#228B22", "Water": "#1E90FF",
    "Building": "#FF3232", "Bare Land": "#D2B48C",
}


def _img_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fmt_cost(value) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.1f}"


def _result_ctx(result: dict) -> dict:
    return {
        "found_str": "Yes" if result["found"] else "No",
        "cost": _fmt_cost(result["total_cost"]),
        "length": result["path_length_px"],
        "nodes": result["nodes_visited"],
        "runtime": f"{result['runtime_ms']:.1f}",
    }


def generate_report(rgb_image, terrain_colored, terrain_stats, astar_result,
                    dijkstra_result, comparison, elevation_stats, image_filename,
                    agent_name=None, avoid_count=0, threat_count=0,
                    route_explain=None) -> str:
    bars = []
    for name, info in terrain_stats["breakdown"].items():
        bars.append((name, info["percentage"], _BAR_COLORS.get(name, "#888")))

    if comparison["same_cost"]:
        optimality = "Equivalent"
    elif comparison["astar_cost"] == comparison["dijkstra_cost"]:
        optimality = "Equivalent"
    else:
        optimality = "Different"

    eff = comparison["efficiency_ratio"]
    if comparison["astar_nodes"] and eff > 1.01:
        saved = comparison["dijkstra_nodes"] - comparison["astar_nodes"]
        reduction = (1 - 1 / eff) * 100 if eff else 0
        takeaway = (
            f"A* visited {saved} fewer nodes than Dijkstra - a {reduction:.0f}% smaller search "
            "space - while finding an equivalent-cost route, thanks to its goal-directed heuristic."
        )
    else:
        takeaway = "Both algorithms explored a similar search space on this terrain configuration."

    elevation_ctx = None
    if elevation_stats:
        elevation_ctx = {
            "min": f"{elevation_stats['min_elevation']:.0f}",
            "max": f"{elevation_stats['max_elevation']:.0f}",
            "gain": f"{elevation_stats['total_gain']:.0f}",
        }

    return _TEMPLATE.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        filename=image_filename,
        mission=({"agent": agent_name, "avoid": avoid_count, "threat": threat_count}
                 if agent_name else None),
        image_b64=_img_b64(rgb_image),
        terrain_b64=_img_b64(terrain_colored),
        terrain_bars=bars,
        astar=_result_ctx(astar_result),
        dijkstra=_result_ctx(dijkstra_result),
        comparison={
            "efficiency": f"{eff:.1f}",
            "cost_diff": _fmt_cost(comparison["cost_difference"]),
            "optimality": optimality,
            "takeaway": takeaway,
        },
        elevation=elevation_ctx,
        route_explain=route_explain,
    )
