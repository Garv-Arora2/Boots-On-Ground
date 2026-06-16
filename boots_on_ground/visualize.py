"""Folium map construction (kept tiny and separate from the Streamlit UI)."""

import folium


def _px_to_geo(path, shape, bounds):
    h, w = shape
    west, south, east, north = bounds
    return [[north - (r / h) * (north - south), west + (c / w) * (east - west)] for r, c in path]


def build_map(rgb, terrain_colored, cost_colored, astar_path, dijkstra_path,
              bounds_wgs84, start_px, end_px):
    """Return a Folium map with image/terrain/cost overlays and both routes.

    If the image has no geo-reference we lay it over a unit square so everything
    still renders (coordinates are then relative, not real-world).
    """
    h, w = rgb.shape[:2]
    bounds = bounds_wgs84 if bounds_wgs84 else (0.0, 0.0, 1.0, 1.0)
    west, south, east, north = bounds
    img_bounds = [[south, west], [north, east]]
    center = [(south + north) / 2.0, (west + east) / 2.0]

    m = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron")

    folium.raster_layers.ImageOverlay(
        rgb, bounds=img_bounds, opacity=0.9, name="Satellite Image"
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        terrain_colored, bounds=img_bounds, opacity=0.6, name="Terrain Map", show=False
    ).add_to(m)
    folium.raster_layers.ImageOverlay(
        cost_colored, bounds=img_bounds, opacity=0.6, name="Traversability Map", show=False
    ).add_to(m)

    if astar_path:
        folium.PolyLine(
            _px_to_geo(astar_path, (h, w), bounds), color="#00FF00", weight=3, opacity=0.9,
            tooltip="A* Route",
        ).add_to(m)
    if dijkstra_path:
        folium.PolyLine(
            _px_to_geo(dijkstra_path, (h, w), bounds), color="#FF8800", weight=3, opacity=0.9,
            dash_array="10", tooltip="Dijkstra Route",
        ).add_to(m)

    if start_px is not None:
        folium.CircleMarker(
            _px_to_geo([start_px], (h, w), bounds)[0], radius=8, color="#00FF00",
            fill=True, fill_opacity=1.0, tooltip="Start",
        ).add_to(m)
    if end_px is not None:
        folium.CircleMarker(
            _px_to_geo([end_px], (h, w), bounds)[0], radius=8, color="#FF4444",
            fill=True, fill_opacity=1.0, tooltip="End",
        ).add_to(m)

    folium.LayerControl().add_to(m)
    try:
        m.fit_bounds(img_bounds)
    except Exception:
        pass
    return m


def map_to_html(m) -> str:
    return m.get_root().render()
