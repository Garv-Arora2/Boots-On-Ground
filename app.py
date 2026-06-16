"""Streamlit UI for BOG."""

import os
import importlib

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_image_coordinates import streamlit_image_coordinates

from boots_on_ground import demo_assets, elevation as elev, imagery_download
from boots_on_ground import change, loader, missions, osm, planning, report, routing, synthetic_scenes, terrain, visualize
import boots_on_ground.change as _change_module
import boots_on_ground.missions as _missions_module
importlib.reload(_change_module)
importlib.reload(_missions_module)
change = _change_module
missions = _missions_module
from config import satellite_scenes


def draw_scene(rgb, start, end, avoid_zones=None, threat_zones=None):
    """Return a copy of the image with avoid zones (orange), threat zones (red,
    translucent), and the start (green) / end (red) markers drawn on top."""
    out = rgb.copy()
    h, w = out.shape[:2]

    # Threat zones: translucent red fill.
    if threat_zones:
        overlay = out.copy()
        for r, c, rad in threat_zones:
            cv2.circle(overlay, (int(c), int(r)), int(rad), (255, 40, 40), -1)
        out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        for r, c, rad in threat_zones:
            cv2.circle(out, (int(c), int(r)), int(rad), (255, 0, 0), 2)

    # Avoid zones: orange outline (hard no-go).
    if avoid_zones:
        for r, c, rad in avoid_zones:
            cv2.circle(out, (int(c), int(r)), int(rad), (255, 165, 0), 3)

    radius = max(6, min(h, w) // 60)
    for (r, c), color, label in ((start, (0, 255, 0), "S"), (end, (255, 0, 0), "E")):
        r = int(min(max(r, 0), h - 1))
        c = int(min(max(c, 0), w - 1))
        cv2.circle(out, (c, r), radius + 3, (255, 255, 255), 2)
        cv2.circle(out, (c, r), radius, color, -1)
        cv2.putText(out, label, (c + radius + 2, r + radius // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, radius / 12.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def build_route_view(rgb, start, end, path, avoid_zones=None, threat_zones=None):
    """Scene + a thick, high-contrast route line (the main thing users look for)."""
    base = draw_scene(rgb, start, end, avoid_zones, threat_zones)
    if not path:
        return base
    out = base.copy()
    h, w = out.shape[:2]
    thick = max(4, min(h, w) // 80)
    pts = np.array([[c, r] for r, c in path], dtype=np.int32)
    cv2.polylines(out, [pts], False, (255, 255, 255), thick + 3, cv2.LINE_AA)
    cv2.polylines(out, [pts], False, (0, 255, 100), thick, cv2.LINE_AA)
    return out


def get_viewport(H, W, zoom, center_r, center_c):
    """Crop window for zoom/pan on the click-map. zoom=1 shows full image."""
    zoom = max(1.0, min(8.0, float(zoom)))
    if zoom <= 1.01:
        return 0, 0, H, W
    span = max(80, int(min(H, W) / zoom))
    vh = min(span, H)
    vw = min(span, W)
    r0 = int(np.clip(center_r - vh // 2, 0, H - vh))
    c0 = int(np.clip(center_c - vw // 2, 0, W - vw))
    return r0, c0, vh, vw


def build_map_crop(rgb, r0, c0, vh, vw, start, end, path=None, avoid_zones=None, threat_zones=None):
    """Scene + route on a zoomed crop (same map the user clicks on)."""
    crop = rgb[r0:r0 + vh, c0:c0 + vw]
    sv = (start[0] - r0, start[1] - c0)
    ev = (end[0] - r0, end[1] - c0)
    path_local = [(r - r0, c - c0) for r, c in (path or [])]
    az = [(r - r0, c - c0, rad) for r, c, rad in (avoid_zones or [])]
    tz = [(r - r0, c - c0, rad) for r, c, rad in (threat_zones or [])]
    return build_route_view(crop, sv, ev, path_local, az, tz)


def click_to_full_px(click, disp_w, disp_h, r0, c0, vh, vw):
    col = c0 + int(min(max(click[0] / disp_w * vw, 0), vw - 1))
    row = r0 + int(min(max(click[1] / disp_h * vh, 0), vh - 1))
    return row, col



st.set_page_config(
    page_title="BOG - Terrain Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background-color: #0f1117; }
      h1, h2, h3 { color: #e8eaf0 !important; }
      .stTabs [data-baseweb="tab"] { color: #8892b0; }
      .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #00d4aa; }
      .cmd-header {
        border: 1px solid #2d3147; border-radius: 8px; padding: 1rem 1.25rem;
        margin-bottom: 1rem; background: linear-gradient(90deg, #1a1d27 0%, #12151f 100%);
      }
      .cmd-title { color: #00d4aa; font-size: 1.35rem; font-weight: 700; letter-spacing: 2px; }
      .cmd-sub { color: #8892b0; font-size: 0.85rem; margin-top: 0.25rem; }
      .intel-card {
        background: #1a1d27; border: 1px solid #2d3147; border-radius: 8px;
        padding: 0.75rem 1rem; text-align: center;
      }
      .intel-val { color: #00d4aa; font-size: 1.5rem; font-weight: 700; }
      .intel-lbl { color: #8892b0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULTS = {
    "image_data": None,
    "terrain_mask": None,
    "terrain_stats": None,
    "cost_map": None,
    "astar_result": None,
    "dijkstra_result": None,
    "comparison": None,
    "elevation_stats": None,
    "folium_html": None,
    "processing_done": False,
    "pt_start": (50, 50),
    "pt_end": (200, 200),
    "click_target": "Start",
    "last_click": None,
    "avoid_zones": [],
    "threat_zones": [],
    "after_data": None,
    "after_mask": None,
    "route_explain": None,
    "base_mask": None,
    "use_osm": False,
    "osm_info": None,
    "mission_name": "Custom",
    "area_name": "Unassigned",
    "mission_objective": "Reach Destination",
    "agent_name_select": "Foot soldier",
    "risk_slider_value": 0.0,
    "mission_analysis": None,
    "route_bullets": None,
    "agent_comparison": None,
    "map_zoom": 1.0,
    "map_center_r": 256,
    "map_center_c": 256,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def load_into_state(file_or_path):
    data = loader.load_image(file_or_path)
    st.session_state.image_data = data
    st.session_state.base_mask = terrain.extract_terrain(data)
    st.session_state.osm_info = None
    rebuild_terrain()
    # Reset any previous routing results.
    for k in ("cost_map", "astar_result", "dijkstra_result", "comparison",
              "elevation_stats", "folium_html"):
        st.session_state[k] = None
    st.session_state.processing_done = False
    h, w = data["rgb"].shape[:2]
    st.session_state.map_zoom = 1.0
    st.session_state.map_center_r = h // 2
    st.session_state.map_center_c = w // 2


def rebuild_terrain():
    """Derive the working terrain mask from the detection base, optionally fused
    with authoritative OpenStreetMap geometry."""
    base = st.session_state.base_mask
    data = st.session_state.image_data
    if st.session_state.use_osm and data is not None and data.get("bounds_wgs84"):
        fused, info = osm.fuse_osm(base, data)
        st.session_state.terrain_mask = fused
        st.session_state.osm_info = info
    else:
        st.session_state.terrain_mask = base
        st.session_state.osm_info = None
    st.session_state.terrain_stats = terrain.stats(st.session_state.terrain_mask)


def apply_mission_template(name: str):
    tpl = missions.MISSION_TEMPLATES[name]
    load_into_state(synthetic_scenes.ensure_scene(tpl["scene"]))
    st.session_state.mission_name = name
    st.session_state.area_name = tpl["area"]
    st.session_state.mission_objective = tpl["objective"]
    st.session_state.agent_name_select = tpl["agent"]
    st.session_state.pt_start = tpl["start"]
    st.session_state.pt_end = tpl["end"]
    st.session_state.threat_zones = [tuple(z) for z in tpl["threats"]]
    st.session_state.avoid_zones = [tuple(z) for z in tpl["avoids"]]
    st.session_state.risk_slider_value = tpl["risk"]
    st.session_state.map_center_r = (tpl["start"][0] + tpl["end"][0]) // 2
    st.session_state.map_center_c = (tpl["start"][1] + tpl["end"][1]) // 2
    st.session_state.map_zoom = 1.5


def run_pathfinding(agent_name: str, risk_weight: float) -> bool:
    data = st.session_state.image_data
    start_px = st.session_state.pt_start
    end_px = st.session_state.pt_end
    if start_px == end_px:
        return False

    mask = st.session_state.terrain_mask
    agent = planning.AGENTS[agent_name]
    avoid_zones = st.session_state.avoid_zones
    threat_zones = st.session_state.threat_zones
    dem = slope = elev_stats = None

    if data["is_geotiff"] and data["bounds_wgs84"]:
        dem = elev.fetch_elevation(data["bounds_wgs84"])
        if dem is not None:
            slope = elev.compute_slope(dem)

    cost_map = planning.compose_cost(
        mask, agent, slope=slope,
        avoid_zones=avoid_zones, threat_zones=threat_zones,
        risk_weight=risk_weight,
    )
    a = routing.astar(cost_map, start_px, end_px)
    d = routing.dijkstra(cost_map, start_px, end_px)
    cmp = routing.compare(a, d)

    if dem is not None and a["found"]:
        prof = elev.profile(dem, a["path"], data["rgb"].shape[:2])
        elev_stats = elev.gain(prof)

    mpp = missions.meters_per_pixel(data.get("bounds_wgs84"), mask.shape)
    analysis = missions.mission_analysis(
        a["path"] if a["found"] else [], mask, st.session_state.terrain_stats,
        agent_name, threat_zones, risk_weight, mpp, a["found"],
    )
    shares = analysis.get("shares", {})
    bullets = missions.explain_route_why(
        a["path"] if a["found"] else [], mask, threat_zones, avoid_zones, risk_weight, shares,
    )
    route_explain = routing.explain_path(a["path"], mask, elev_stats) if a["found"] else None

    scene = draw_scene(data["rgb"], start_px, end_px, avoid_zones, threat_zones)
    fmap = visualize.build_map(
        scene, terrain.colorize(mask), terrain.colorize_cost(cost_map),
        a["path"] if a["found"] else None, d["path"] if d["found"] else None,
        data["bounds_wgs84"], start_px, end_px,
    )

    st.session_state.agent_name = agent_name
    st.session_state.route_explain = route_explain
    st.session_state.mission_analysis = analysis
    st.session_state.route_bullets = bullets
    st.session_state.agent_comparison = missions.compare_agents(
        mask, start_px, end_px, slope, avoid_zones, threat_zones, risk_weight, mpp,
    )
    st.session_state.cost_map = cost_map
    st.session_state.astar_result = a
    st.session_state.dijkstra_result = d
    st.session_state.comparison = cmp
    st.session_state.elevation_stats = elev_stats
    st.session_state.folium_html = visualize.map_to_html(fmap)
    st.session_state.start_px = start_px
    st.session_state.end_px = end_px
    st.session_state.processing_done = True
    return a["found"]


def render_command_header():
    status = "Complete" if st.session_state.processing_done else "Ready"
    intel = missions.area_intelligence(
        st.session_state.terrain_mask, st.session_state.terrain_stats,
    ) if st.session_state.terrain_mask is not None else None
    tr = intel["terrain_risk"] if intel else "-"
    st.markdown(
        f"""<div class="cmd-header">
          <div class="cmd-title">BOOTS ON GROUND</div>
          <div class="cmd-sub">Area: <b>{st.session_state.area_name}</b> &nbsp;|&nbsp;
          Mission: <b>{st.session_state.mission_name}</b> &nbsp;|&nbsp;
          Objective: <b>{st.session_state.mission_objective}</b></div>
          <div class="cmd-sub">Terrain Risk: <b>{tr}</b> &nbsp;|&nbsp;
          Threat Level: <b>{'Active' if st.session_state.threat_zones else 'Clear'}</b> &nbsp;|&nbsp;
          Status: <b>{status}</b></div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_terrain_cards(stats):
    colors = {"Road": "#808080", "Vegetation": "#228B22", "Water": "#1E90FF",
              "Building": "#FF3232", "Bare Land": "#D2B48C"}
    cols = st.columns(5)
    for col, (name, info) in zip(cols, stats["breakdown"].items()):
        pct = info["percentage"]
        col.markdown(
            f"""<div class="intel-card" style="border-top: 3px solid {colors.get(name, '#888')}">
              <div class="intel-val">{pct:.0f}%</div>
              <div class="intel-lbl">{name}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ----------------------------- Sidebar -----------------------------
st.sidebar.title("BOOTS ON GROUND")
st.sidebar.caption("Terrain Intelligence & Route Planning")

st.sidebar.subheader("Quick setups")
st.sidebar.caption("Loads scene, points & agent only — click **Run pathfinder** after.")
tpl_cols = st.sidebar.columns(2)
for i, name in enumerate(missions.MISSION_TEMPLATES):
    with tpl_cols[i % 2]:
        if st.button(name, key=f"tpl_{name}", use_container_width=True):
            apply_mission_template(name)
            st.rerun()

st.sidebar.divider()
st.sidebar.subheader("1. Image")
img_source = st.sidebar.radio("Source", ["Real satellite", "Synthetic demo"], horizontal=True)

if img_source == "Real satellite":
    real_scene = st.sidebar.selectbox("Real satellite scene", list(satellite_scenes.REAL_SCENES.keys()))
    meta = satellite_scenes.REAL_SCENES[real_scene]
    st.sidebar.caption(meta["note"])
    have = satellite_scenes.is_downloaded(real_scene)
    if not have:
        st.sidebar.warning("Not on disk yet (~1–3 MB, needs internet).")
    rc1, rc2 = st.sidebar.columns(2)
    if rc1.button("Download scene", disabled=have):
        with st.spinner("Downloading from Sentinel-2 / OpenAerialMap..."):
            try:
                imagery_download.download_scene(real_scene)
                st.rerun()
            except Exception as exc:
                st.sidebar.error(str(exc))
    if rc2.button("Download all", help="Fetch all 7 real scenes (~10 MB total)"):
        with st.spinner("Downloading all real scenes (may take a few minutes)..."):
            imagery_download.download_all()
            st.rerun()
    if st.sidebar.button("Load real scene", type="primary", disabled=not have):
        load_into_state(satellite_scenes.scene_path(real_scene))
else:
    scene = st.sidebar.selectbox("Demo scene", list(synthetic_scenes.SCENES.keys()))
    if st.sidebar.button("Load demo scene", type="primary"):
        load_into_state(synthetic_scenes.ensure_scene(scene))

uploaded = st.sidebar.file_uploader(
    "Or upload your own image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded is not None:
    current = st.session_state.image_data
    if current is None or current["filename"] != uploaded.name:
        try:
            load_into_state(uploaded)
        except Exception as exc:
            st.sidebar.error(f"Could not read that image: {exc}")

data = st.session_state.image_data

if data is not None:
    st.sidebar.success(loader.image_info(data))
    H, W = data["rgb"].shape[:2]

    osm_supported = bool(data["bounds_wgs84"])
    use_osm = st.sidebar.checkbox(
        "Fuse OpenStreetMap roads/buildings", value=st.session_state.use_osm,
        disabled=not osm_supported,
        help="Overlay authoritative OSM geometry instead of guessing from pixels. "
             "Requires a geo-referenced GeoTIFF and internet.")
    if not osm_supported:
        st.sidebar.caption("OSM fusion needs a geo-referenced GeoTIFF.")
    if use_osm != st.session_state.use_osm:
        st.session_state.use_osm = use_osm
        with st.spinner("Fetching OpenStreetMap data..."):
            rebuild_terrain()
        st.session_state.processing_done = False
        st.rerun()
    info = st.session_state.osm_info
    if info is not None:
        if info.get("available"):
            c = info["counts"]
            st.sidebar.caption(f"OSM: {c['road']} roads, {c['building']} buildings, "
                               f"{c['water']} water features fused.")
        else:
            st.sidebar.warning(f"OSM unavailable: {info.get('reason', 'unknown')}")

    st.sidebar.subheader("2. Mission & agent")
    objectives = missions.OBJECTIVES
    obj_idx = objectives.index(st.session_state.mission_objective) if st.session_state.mission_objective in objectives else 0
    st.session_state.mission_objective = st.sidebar.radio(
        "Mission objective", objectives, index=obj_idx,
    )

    st.sidebar.caption("Select agent")
    ac1, ac2 = st.sidebar.columns(2)
    for idx, name in enumerate(planning.AGENT_NAMES):
        meta = missions.AGENT_DISPLAY[name]
        col = ac1 if idx % 2 == 0 else ac2
        selected = name == st.session_state.agent_name_select
        if col.button(
            f"{meta['label']} — {meta['speed']} speed",
            key=f"agent_{name}",
            type="primary" if selected else "secondary",
            use_container_width=True,
        ):
            st.session_state.agent_name_select = name
            st.rerun()
    agent_name = st.session_state.agent_name_select
    agent = planning.AGENTS[agent_name]
    st.sidebar.caption(agent.note)

    risk_slider = st.sidebar.slider(
        "Threat avoidance (fast 0 - safe 1)", 0.0, 1.0,
        float(st.session_state.risk_slider_value), 0.1,
        help="0 ignores threat zones (fastest). Higher routes around danger (safest).",
    )
    st.session_state.risk_slider_value = risk_slider
    risk_weight = risk_slider * 12.0

    st.sidebar.subheader("3. Start, end & zones")
    st.sidebar.caption("Pick points on **Area Analysis** (zoom in for accuracy).")
    zone_radius = st.sidebar.slider("Zone radius (pixels)", 10, max(20, min(H, W) // 2), 40, 5)
    zc1, zc2 = st.sidebar.columns(2)
    if zc1.button("Clear avoid"):
        st.session_state.avoid_zones = []
        st.rerun()
    if zc2.button("Clear threats"):
        st.session_state.threat_zones = []
        st.rerun()
    st.sidebar.caption(
        f"Avoid zones: {len(st.session_state.avoid_zones)} (no-go)  |  "
        f"Threat zones: {len(st.session_state.threat_zones)} (risk)")
    if st.sidebar.button("Random points"):
        s, e = routing.random_passable_points(st.session_state.terrain_mask)
        st.session_state.pt_start = s
        st.session_state.pt_end = e
        st.rerun()

    # Clamp current points to the loaded image size.
    cur_s = (min(st.session_state.pt_start[0], H - 1), min(st.session_state.pt_start[1], W - 1))
    cur_e = (min(st.session_state.pt_end[0], H - 1), min(st.session_state.pt_end[1], W - 1))

    col_a, col_b = st.sidebar.columns(2)
    sr = col_a.number_input("Start row", 0, H - 1, value=int(cur_s[0]))
    sc = col_b.number_input("Start col", 0, W - 1, value=int(cur_s[1]))
    er = col_a.number_input("End row", 0, H - 1, value=int(cur_e[0]))
    ec = col_b.number_input("End col", 0, W - 1, value=int(cur_e[1]))
    st.session_state.pt_start = (int(sr), int(sc))
    st.session_state.pt_end = (int(er), int(ec))
    if st.session_state.processing_done:
        if st.session_state.pt_start != st.session_state.get("start_px") or \
                st.session_state.pt_end != st.session_state.get("end_px"):
            st.session_state.processing_done = False

    st.sidebar.subheader("4. Pathfinder")
    run = st.sidebar.button("Run pathfinder", type="primary")

    if run:
        with st.spinner("Computing route..."):
            ok = run_pathfinding(agent_name, risk_weight)
        if not ok:
            st.sidebar.warning("No valid route. Move start/end off water or buildings.")
        else:
            st.sidebar.success("Route computed — see green line on the map.")

    if st.session_state.processing_done:
        st.sidebar.subheader("4. Export")
        html = report.generate_report(
            data["rgb"], terrain.colorize(st.session_state.terrain_mask),
            st.session_state.terrain_stats, st.session_state.astar_result,
            st.session_state.dijkstra_result, st.session_state.comparison,
            st.session_state.elevation_stats, data["filename"],
            agent_name=st.session_state.get("agent_name"),
            avoid_count=len(st.session_state.avoid_zones),
            threat_count=len(st.session_state.threat_zones),
            route_explain=st.session_state.route_explain,
        )
        st.sidebar.download_button(
            "Download HTML report", data=html.encode("utf-8"),
            file_name="bog_route_report.html", mime="text/html",
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "© 2026 [Garv Arora](https://github.com/Garv-Arora2) · BOG"
    )


# ----------------------------- Main panel -----------------------------
if data is None:
    st.markdown("### Quick setup — pick a scenario")
    st.caption("Loads the area and default points. Then open **Area Analysis**, place points, "
               "zoom in, and click **Run pathfinder**.")
    mc1, mc2, mc3 = st.columns(3)
    cols = [mc1, mc2, mc3, mc1, mc2, mc3]
    for i, name in enumerate(missions.MISSION_TEMPLATES):
        with cols[i % len(cols)]:
            if st.button(name, key=f"main_tpl_{name}", use_container_width=True):
                apply_mission_template(name)
                st.rerun()
    st.info("Or use the sidebar: **Real satellite** / **Synthetic demo**, then **Run pathfinder**.")
    st.caption("© 2026 Garv Arora · [github.com/Garv-Arora2](https://github.com/Garv-Arora2)")
    st.stop()

render_command_header()

intel = missions.area_intelligence(st.session_state.terrain_mask, st.session_state.terrain_stats)
st.markdown("#### Area overview")
h1, h2, h3, h4, h5 = st.columns(5)
h1.metric("Traversable area", f"{intel['traversable_pct']:.0f}%")
h2.metric("Terrain risk", intel["terrain_risk"])
h3.metric("Road access", intel["road_access"])
h4.metric("Water hazard", intel["water_hazard"])
h5.metric("Recommended agent", missions.AGENT_DISPLAY[intel["recommended_agent"]]["label"])

if st.session_state.processing_done and st.session_state.mission_analysis:
    ma = st.session_state.mission_analysis
    st.markdown("#### Route summary")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Route distance", f"{ma['distance_m']:.0f} m")
    d2.metric("Travel time", ma["travel_time"])
    d3.metric("Terrain difficulty", ma["terrain_difficulty"])
    d4.metric("Threat exposure", ma["threat_exposure"])
    d5.metric("Route quality", ma["route_quality"])
    if not (st.session_state.astar_result and st.session_state.astar_result.get("found")):
        st.warning("No valid route — adjust points on **Area Analysis** and click **Run pathfinder**.")

tab_area, tab_threat, tab_change, tab_plan, tab_report = st.tabs([
    "1. Area Analysis",
    "2. Threat Assessment",
    "3. Change Detection",
    "4. Mission Planning",
    "5. Mission Report",
])

with tab_area:
    rgb = data["rgb"]
    H, W = rgb.shape[:2]

    path_full = None
    route_ok = (
        st.session_state.processing_done
        and st.session_state.astar_result
        and st.session_state.astar_result.get("found")
        and st.session_state.pt_start == st.session_state.get("start_px")
        and st.session_state.pt_end == st.session_state.get("end_px")
    )
    if route_ok:
        path_full = st.session_state.astar_result["path"]

    st.subheader("Interactive map — place points & view route")
    MODES = ["Start", "End", "Avoid zone", "Threat zone"]
    st.session_state.click_target = st.radio(
        "Next click places:", MODES, horizontal=True,
        index=MODES.index(st.session_state.click_target)
        if st.session_state.click_target in MODES else 0,
    )

    z1, z2, z3, z4 = st.columns([2, 1, 1, 1])
    st.session_state.map_zoom = z1.slider(
        "Zoom in", 1.0, 8.0, float(st.session_state.map_zoom), 0.5,
        help="Zoom in to place points accurately on small or detailed imagery.",
    )
    if z2.button("Center on S/E", help="Center view between start and end"):
        st.session_state.map_center_r = (st.session_state.pt_start[0] + st.session_state.pt_end[0]) // 2
        st.session_state.map_center_c = (st.session_state.pt_start[1] + st.session_state.pt_end[1]) // 2
        st.rerun()
    if z3.button("Center on start"):
        st.session_state.map_center_r = st.session_state.pt_start[0]
        st.session_state.map_center_c = st.session_state.pt_start[1]
        st.rerun()
    if z4.button("Fit full image"):
        st.session_state.map_zoom = 1.0
        st.session_state.map_center_r = H // 2
        st.session_state.map_center_c = W // 2
        st.rerun()

    r0, c0, vh, vw = get_viewport(
        H, W, st.session_state.map_zoom,
        st.session_state.map_center_r, st.session_state.map_center_c,
    )
    if st.session_state.map_zoom > 1.05:
        p1, p2 = st.columns(2)
        st.session_state.map_center_r = p1.slider(
            "Pan vertically (row)", 0, H - 1, int(st.session_state.map_center_r), key="pan_r",
        )
        st.session_state.map_center_c = p2.slider(
            "Pan horizontally (col)", 0, W - 1, int(st.session_state.map_center_c), key="pan_c",
        )
        r0, c0, vh, vw = get_viewport(
            H, W, st.session_state.map_zoom,
            st.session_state.map_center_r, st.session_state.map_center_c,
        )

    preview = build_map_crop(
        rgb, r0, c0, vh, vw,
        st.session_state.pt_start, st.session_state.pt_end,
        path=path_full if route_ok else None,
        avoid_zones=st.session_state.avoid_zones,
        threat_zones=st.session_state.threat_zones,
    )
    DISPLAY_W = 950
    if route_ok:
        st.caption("**Green line** = computed route from S to E. Zoom/pan, place points, then **Run pathfinder** to refresh.")
    else:
        st.caption("Place **Start** and **End**, zoom in for accuracy, then click **Run pathfinder** in the sidebar.")

    coords = streamlit_image_coordinates(preview, width=DISPLAY_W, key="point_picker")
    if coords is not None:
        click = (coords["x"], coords["y"])
        if click != st.session_state.last_click:
            st.session_state.last_click = click
            disp_w = coords.get("width") or DISPLAY_W
            disp_h = coords.get("height") or (DISPLAY_W * vh / max(vw, 1))
            row, col = click_to_full_px(click, disp_w, disp_h, r0, c0, vh, vw)
            target = st.session_state.click_target
            if target == "Start":
                st.session_state.pt_start = (row, col)
            elif target == "End":
                st.session_state.pt_end = (row, col)
            elif target == "Avoid zone":
                st.session_state.avoid_zones.append((row, col, int(zone_radius)))
            elif target == "Threat zone":
                st.session_state.threat_zones.append((row, col, int(zone_radius)))
            st.session_state.processing_done = False
            st.rerun()

    st.write(
        f"Start row {st.session_state.pt_start[0]}, col {st.session_state.pt_start[1]}  |  "
        f"End row {st.session_state.pt_end[0]}, col {st.session_state.pt_end[1]}  |  "
        f"View {vh}×{vw} px (zoom {st.session_state.map_zoom:.1f}×)  |  "
        f"Route: {'shown' if route_ok else 'not computed'}"
    )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Terrain (current view)")
        crop_mask = st.session_state.terrain_mask[r0:r0 + vh, c0:c0 + vw]
        st.image(terrain.colorize(crop_mask), use_container_width=True)
        st.caption("Gray = road, red = building, green = veg, blue = water, tan = bare.")
    with c2:
        st.subheader("Land cover (full area)")
        render_terrain_cards(st.session_state.terrain_stats)

with tab_threat:
    st.subheader("Threat assessment")
    st.caption("Threat zones (red) add risk cost; avoid zones (orange) are hard no-go.")
    path_threat = None
    if st.session_state.processing_done and st.session_state.astar_result:
        ar = st.session_state.astar_result
        if ar.get("found"):
            path_threat = ar["path"]
    if path_threat:
        preview = build_route_view(
            data["rgb"], st.session_state.pt_start, st.session_state.pt_end,
            path_threat, st.session_state.avoid_zones, st.session_state.threat_zones,
        )
    else:
        preview = draw_scene(
            data["rgb"], st.session_state.pt_start, st.session_state.pt_end,
            st.session_state.avoid_zones, st.session_state.threat_zones,
        )
    st.image(preview, use_container_width=True)
    t1, t2, t3 = st.columns(3)
    t1.metric("Threat zones", len(st.session_state.threat_zones))
    t2.metric("No-go zones", len(st.session_state.avoid_zones))
    t3.metric("Avoidance level", f"{st.session_state.risk_slider_value:.0%}")
    if st.session_state.threat_zones:
        st.markdown("**Active threats:**")
        for i, (r, c, rad) in enumerate(st.session_state.threat_zones, 1):
            st.markdown(f"- Threat {i}: center row {r}, col {c}, radius {rad} px")
    else:
        st.info("No threat zones placed. Use a quick setup or draw on Area Analysis.")

with tab_plan:
    if not st.session_state.processing_done:
        st.info("Click **Run pathfinder** in the sidebar or launch a **Quick setup** first.")
    else:
        ma = st.session_state.mission_analysis
        st.subheader("Mission analysis")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Terrain difficulty", ma["terrain_difficulty"])
        p2.metric("Threat exposure", ma["threat_exposure"])
        p3.metric("Travel time", ma["travel_time"])
        p4.metric("Route quality", ma["route_quality"])
        p5, p6, p7, p8 = st.columns(4)
        p5.metric("Water crossings", ma["water_crossings"])
        p6.metric("Road usage", f"{ma['road_pct']:.0f}%")
        p7.metric("Vegetation", f"{ma['veg_pct']:.0f}%")
        p8.metric("Algorithm", ma["recommended_algo"])

        if st.session_state.route_bullets:
            st.subheader("Why this route?")
            for line in st.session_state.route_bullets:
                st.markdown(f"- ✓ {line}")

        st.subheader("Compare agents")
        if st.session_state.agent_comparison:
            st.dataframe(st.session_state.agent_comparison, use_container_width=True, hide_index=True)

        a = st.session_state.astar_result
        d = st.session_state.dijkstra_result
        base = draw_scene(data["rgb"], st.session_state.start_px, st.session_state.end_px,
                          st.session_state.avoid_zones, st.session_state.threat_zones)
        thick = max(4, min(data["rgb"].shape[0], data["rgb"].shape[1]) // 80)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("A* route (recommended)")
            st.image(routing.draw_path(base, a["path"], (0, 255, 100), thickness=thick),
                     use_container_width=True)
        with c2:
            st.caption("Dijkstra route (baseline)")
            st.image(routing.draw_path(base, d["path"], (255, 136, 0), thickness=thick),
                     use_container_width=True)

        expl = st.session_state.route_explain
        if expl:
            st.caption(expl["text"])

with tab_change:
    st.subheader("Change detection (before / after)")
    st.caption(
        "Compare two images of the **same area** from different dates. BOG aligns the pair, "
        "validates changes on both images, and marks what actually changed — floods, structural "
        "damage, vegetation loss, new construction, and surface disruption."
    )

    def load_after(file_or_path):
        adata = loader.load_image(file_or_path)
        st.session_state.after_data = adata
        st.session_state.after_mask = None

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        if st.button("Load satellite demo pair", use_container_width=True):
            try:
                bpath, apath = demo_assets.ensure_satellite_pair()
                load_into_state(bpath)
                load_after(apath)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load satellite demo: {exc}")
    with dc2:
        if st.button("Load synthetic demo", use_container_width=True):
            demo_assets.ensure_before()
            load_into_state(str(demo_assets.SAMPLE_BEFORE))
            demo_assets.ensure_after()
            load_after(str(demo_assets.SAMPLE_AFTER))
            st.rerun()
    with dc3:
        after_file = st.file_uploader(
            "Upload after image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="after_uploader",
            label_visibility="collapsed",
        )

    if after_file is not None:
        cur = st.session_state.after_data
        if cur is None or cur["filename"] != after_file.name:
            try:
                load_after(after_file)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read that image: {exc}")

    before_data = st.session_state.image_data
    after_data = st.session_state.after_data
    if before_data is None:
        st.info("Load a **before** image from the sidebar, or click **Load satellite demo pair**.")
    elif after_data is None:
        st.info("Upload an **after** image or load a demo pair to run change detection.")
    else:
        sensitivity = st.slider(
            "Change sensitivity",
            min_value=0.75,
            max_value=0.98,
            value=0.94,
            step=0.01,
            help="Higher = only the strongest changes. Lower = more sensitive "
                 "(may include minor lighting or season differences).",
        )
        try:
            before_payload = {**before_data, "mask": st.session_state.terrain_mask}
            result = change.detect_changes_from_data(
                before_payload,
                after_data,
                include_structural=True,
                structural_sensitivity=sensitivity,
            )
        except Exception as exc:
            st.error(f"Change detection failed: {exc}")
            st.stop()

        cat = result["category"]
        cstats = change.summarize_detection(result)
        zones = cstats["zones"]
        highlighted = change.highlight_changes_on_image(
            result["after_rgb"],
            result["severity"],
            result["specific"],
            zones=zones,
        )
        severity_view = change.colorize_severity(result["severity"], result["after_rgb"])

        st.markdown("#### Imagery — severity-marked changes")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.caption("Before")
            st.image(result["before_rgb"], use_container_width=True)
        with sc2:
            st.caption("After")
            st.image(result["after_rgb"], use_container_width=True)
        with sc3:
            st.caption("After — specific changes by severity")
            st.image(highlighted, use_container_width=True)
        with sc4:
            st.caption("Severity map")
            st.image(severity_view, use_container_width=True)

        impact = missions.change_impact(cstats)
        st.markdown("#### Change summary")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Specific change", f"{impact.get('area_changed_pct', 0):.1f}%")
        m2.metric("Major (red)", f"{cstats.get('major_pct', 0):.1f}%")
        m3.metric("Moderate (black)", f"{cstats.get('moderate_pct', 0):.1f}%")
        m4.metric("Minor (pink)", f"{cstats.get('minor_pct', 0):.1f}%")
        m5.metric("Structure damage", f"{impact.get('structure_damage_pct', 0):.1f}%")
        m6.metric("Change zones", impact.get("zone_count", len(zones)))

        st.markdown("**Detected changes:**")
        for line in cstats["summary"]:
            st.markdown(f"- {line}")

        if zones:
            st.markdown("#### Change zones (specific pixels only)")
            st.dataframe(
                [{
                    "Zone": z["id"],
                    "Type": z["type"],
                    "Severity": z["severity"],
                    "Area": f"{z['area_pct']:.2f}%",
                    "Location": f"({z['center'][0]}, {z['center'][1]})",
                } for z in zones],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Technical — terrain classification (reference only)"):
            st.caption(
                "Automatic land-cover maps for each date. Change results above use "
                "validated rules on both images — raw class totals can differ due to "
                "lighting and are not used for water or damage reporting."
            )
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.caption("Before — terrain")
                st.image(terrain.colorize(result["before_mask"]), use_container_width=True)
            with tc2:
                st.caption("After — terrain")
                st.image(terrain.colorize(result["after_mask"]), use_container_width=True)
            with tc3:
                st.caption("Change map")
                st.image(change.colorize_change(cat, result["after_rgb"]), use_container_width=True)

        st.caption(
            "Severity on after image: **pink** = minor change, **black** = moderate change, "
            "**red** = major change. Only specific changed pixels are marked — unchanged "
            "areas stay clear. Labels (#) appear on moderate and major zones."
        )

with tab_report:
    if not st.session_state.processing_done or not st.session_state.folium_html:
        st.info("Click **Run pathfinder** in the sidebar to generate the report.")
    else:
        cmp = st.session_state.comparison
        a = st.session_state.astar_result
        st.markdown("#### Algorithm validation")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("A* cost", f"{a['total_cost']:.1f}" if a["found"] else "-")
        r2.metric("A* nodes", a["nodes_visited"])
        r3.metric("A* efficiency", f"{cmp['efficiency_ratio']:.1f}x")
        r4.metric("Optimal match", "Yes" if cmp["same_cost"] else "No")
        if not data["bounds_wgs84"]:
            st.caption("No geo-reference in this image - map coordinates are relative.")
        components.html(st.session_state.folium_html, height=550, scrolling=False)
        st.caption("Download the full HTML report from the sidebar (**Export**).")
