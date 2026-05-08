"""
Stage 4: Interactive Route Map Generator (Fixed Layout)
========================================================
Multi-Factor Optimal Routing using Temporal Fusion Transformer
NIT Karnataka, Surathkal

Changes from previous version:
  - All routes shown simultaneously on map
  - Click route line or legend item to highlight (others dim)
  - Click again to reset all routes
  - Route comparison table moved to bottom-right (no layer control overlap)
  - Layer control stays top-right unobstructed
  - Colors match route_radar.png exactly

Usage:
    python stage4_map.py
    python stage4_map.py --origin "12.9716,77.5946" --dest "12.9352,77.6245"
"""

import argparse
import pickle
import warnings
import json
import pandas as pd
import networkx as nx
import folium
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PROCESSED_DIR = Path(r"C:\Dhawal\Datasets\processed")
PLOTS_DIR     = Path(r"C:\Dhawal\plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Colors matching route_radar.png EXACTLY
# ROUTE_COLORS = {
#     "balanced"     : "#2C3E50",
#     "recommended"  : "#2C3E50",
#     "fastest"      : "#E74C3C",
#     "most certain" : "#3498DB",
#     "safest"       : "#2ECC71",
#     "cleanest"     : "#9B59B6",
#     "speed+safety" : "#E67E22",
#     "balanced-v2"  : "#1ABC9C",
#     "default"      : "#95A5A6",
# }
ROUTE_COLORS = {
    "balanced"     : "#E74C3C",   # red
    "recommended"  : "#E74C3C",   # red
    "fastest"      : "#3498DB",   # blue
    "most certain" : "#3498DB",   # blue
    "safest"       : "#2ECC71",   # green
    "cleanest"     : "#9B59B6",   # purple
    "speed+safety" : "#E67E22",   # orange
    "balanced-v2"  : "#1ABC9C",   # teal
    "default"      : "#95A5A6",   # gray
}

ROUTE_BASE_WEIGHTS = {
    "balanced"    : 6,
    "recommended" : 6,
    "fastest"     : 5,
    "safest"      : 5,
    "default"     : 4,
}


def get_route_color(label: str) -> str:
    for key, color in ROUTE_COLORS.items():
        if key in label.lower():
            return color
    return ROUTE_COLORS["default"]


def get_route_weight(label: str) -> int:
    for key, w in ROUTE_BASE_WEIGHTS.items():
        if key in label.lower():
            return w
    return 4


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_graph() -> nx.MultiDiGraph:
    graph_path = PROCESSED_DIR / "osm_graph.pkl"
    if not graph_path.exists():
        raise FileNotFoundError("OSM graph not found. Run: python stage4_routing.py first")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    print(f"  [OK] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def load_routes() -> pd.DataFrame:
    routes_path = PROCESSED_DIR / "pareto_routes.parquet"
    if not routes_path.exists():
        raise FileNotFoundError("Routes not found. Run: python stage4_routing.py first")
    return pd.read_parquet(routes_path)


# ─────────────────────────────────────────────
# RECOMPUTE FULL PATHS
# ─────────────────────────────────────────────

def recompute_paths(
    G          : nx.MultiDiGraph,
    routes_df  : pd.DataFrame,
    origin_str : str,
    dest_str   : str,
) -> list:
    from stage4_routing import parse_location, get_nearest_node

    origin_lat, origin_lon = parse_location(origin_str)
    dest_lat,   dest_lon   = parse_location(dest_str)
    origin_node = get_nearest_node(G, origin_lat, origin_lon)
    dest_node   = get_nearest_node(G, dest_lat,   dest_lon)

    weight_sets = [
        (0.40, 0.20, 0.25, 0.15, "balanced"),
        (0.90, 0.05, 0.03, 0.02, "fastest"),
        (0.05, 0.90, 0.03, 0.02, "most certain"),
        (0.05, 0.05, 0.88, 0.02, "safest"),
        (0.05, 0.05, 0.05, 0.85, "cleanest"),
        (0.50, 0.20, 0.20, 0.10, "speed+safety"),
    ]

    # valid_labels = set(routes_df["label"].str.lower().tolist())
    # results      = []
    # seen_paths   = set()

    # for a, b, g, d, label in weight_sets:
    #     if not any(label.split()[0] in vl or vl in label for vl in valid_labels):
    #         continue
    valid_labels = set(routes_df["label"].str.lower().tolist())
    results      = []
    seen_paths   = set()

    # print for debugging
    print(f"  [DEBUG] Valid labels in parquet: {valid_labels}")

    for a, b, g, d, label in weight_sets:
        # match if ANY word in the weight_set label appears in ANY valid label
        label_words = label.lower().split()
        matched = any(
            any(word in vl for word in label_words)
            for vl in valid_labels
        )
        if not matched:
            print(f"  [SKIP] No match for weight_set label: '{label}'")
            continue
        print(f"  [MATCH] '{label}' matched in valid labels")



        for u, v, key in G.edges(keys=True):
            data = G[u][v][key]
            te   = data.get("weight_te", 0)
            ue   = data.get("weight_ue", 0)
            re   = data.get("weight_re", 0)
            pe   = data.get("weight_pe", 0)
            tt   = data.get("travel_time_s", 1)
            G[u][v][key]["weight_temp"] = (a*te + b*ue + g*re + d*pe) * tt

        try:
            path = nx.shortest_path(G, origin_node, dest_node, weight="weight_temp")
        except nx.NetworkXNoPath:
            continue

        sig = tuple(path[::max(1, len(path)//8)])
        if sig in seen_paths:
            continue
        seen_paths.add(sig)

        # try each word in the label until a match is found
        match = pd.DataFrame()
        for word in label.lower().split():
            match = routes_df[routes_df["label"].str.lower().str.contains(word, na=False)]
            if len(match) > 0:
                break
        if len(match) == 0:
            # fallback: use the route with closest composite score
            match = routes_df.iloc[[i % len(routes_df)]]
        row = match.iloc[0]


        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]

        results.append({
            "label"           : row["label"],
            "coords"          : coords,
            "color"           : get_route_color(row["label"]),
            "weight"          : get_route_weight(row["label"]),
            "travel_time_min" : float(row["travel_time_min"]),
            "distance_km"     : float(row["distance_km"]),
            "Te"              : float(row["Te_mean"]),
            "Ue"              : float(row["Ue_mean"]),
            "Re"              : float(row["Re_mean"]),
            "Pe"              : float(row["Pe_mean"]),
            "composite"       : float(row["composite"]),
        })

    results.sort(key=lambda x: x["composite"])
    return results


# ─────────────────────────────────────────────
# BUILD MAP
# ─────────────────────────────────────────────

def build_map(
    G          : nx.MultiDiGraph,
    routes     : list,
    origin_str : str,
    dest_str   : str,
) -> folium.Map:
    from stage4_routing import parse_location

    origin_lat, origin_lon = parse_location(origin_str)
    dest_lat,   dest_lon   = parse_location(dest_str)
    centre_lat = (origin_lat + dest_lat) / 2
    centre_lon = (origin_lon + dest_lon) / 2

    m = folium.Map(
        location   = [centre_lat, centre_lon],
        zoom_start = 14,
        tiles      = "CartoDB positron",
    )

    # ── draw all routes as polylines ──
    for i, route in enumerate(routes):
        coords = route["coords"]
        if len(coords) < 2:
            continue

        popup_html = f"""
        <div style="font-family:Arial;font-size:13px;min-width:210px;">
          <b style="color:{route['color']};font-size:15px;">{route['label']}</b><br>
          <hr style="margin:5px 0;border:0;border-top:1px solid #eee;">
          <b>ETA:</b> {route['travel_time_min']:.1f} min
          &nbsp;&nbsp;
          <b>Distance:</b> {route['distance_km']:.2f} km<br>
          <hr style="margin:5px 0;border:0;border-top:1px solid #eee;">
          <table style="width:100%;font-size:12px;">
            <tr><td style="color:#888;">Te (travel time)</td>
                <td style="text-align:right;">{route['Te']:.3f}</td></tr>
            <tr><td style="color:#888;">Ue (uncertainty)</td>
                <td style="text-align:right;">{route['Ue']:.3f}</td></tr>
            <tr><td style="color:#888;">Re (incident risk)</td>
                <td style="text-align:right;">{route['Re']:.3f}</td></tr>
            <tr><td style="color:#888;">Pe (pollution)</td>
                <td style="text-align:right;">{route['Pe']:.3f}</td></tr>
          </table>
          <hr style="margin:5px 0;border:0;border-top:1px solid #eee;">
          <b>Composite score:</b> {route['composite']:.4f}
        </div>
        """

        folium.PolyLine(
            locations = coords,
            color     = route["color"],
            weight    = route["weight"],
            opacity   = 0.80,
            tooltip   = f"{route['label']}: {route['travel_time_min']:.1f} min",
            popup     = folium.Popup(popup_html, max_width=280),
        ).add_to(m)

    # ── origin marker ──
    folium.Marker(
        location = [origin_lat, origin_lon],
        popup    = folium.Popup(f"<b>Origin</b><br>{origin_str}", max_width=220),
        icon     = folium.Icon(color="red", icon="plus-sign", prefix="glyphicon"),
        tooltip  = f"Origin: {origin_str}",
    ).add_to(m)

    # ── destination marker ──
    folium.Marker(
        location = [dest_lat, dest_lon],
        popup    = folium.Popup(f"<b>Destination</b><br>{dest_str}", max_width=220),
        icon     = folium.Icon(color="green", icon="flag", prefix="glyphicon"),
        tooltip  = f"Destination: {dest_str}",
    ).add_to(m)

    n_routes   = len(routes)
    colors_js  = json.dumps([r["color"]  for r in routes])
    weights_js = json.dumps([r["weight"] for r in routes])

    # ── legend — bottom LEFT, clear of layer control ──
    legend_items = "".join([
        f"""
        <div id="leg_{i}"
             onclick="highlightRoute({i})"
             style="display:flex;align-items:center;gap:10px;
                    margin:4px 0;cursor:pointer;padding:5px 8px;
                    border-radius:6px;transition:background 0.15s;"
             onmouseover="this.style.background='#f5f5f5'"
             onmouseout="if({i}!==activeRoute)this.style.background='transparent'">
          <div style="flex-shrink:0;width:34px;height:5px;
                      background:{r['color']};border-radius:3px;"></div>
          <div>
            <span style="font-weight:600;font-size:12px;
                         color:#333;">{r['label']}</span><br>
            <span style="font-size:11px;color:#888;">
              {r['travel_time_min']:.1f} min &nbsp;·&nbsp; {r['distance_km']:.2f} km
            </span>
          </div>
        </div>"""
        for i, r in enumerate(routes)
    ])

    legend_html = f"""
    <div style="
        position:fixed; bottom:28px; left:15px;
        z-index:9997;
        background:white; padding:14px 16px;
        border-radius:10px; border:1px solid #ddd;
        font-family:Arial,sans-serif;
        box-shadow:0 2px 10px rgba(0,0,0,0.12);
        min-width:250px; max-width:300px;">
      <div style="font-weight:700;font-size:13px;margin-bottom:8px;
                  display:flex;justify-content:space-between;align-items:center;">
        <span>Pareto-Optimal Routes</span>
        <span style="font-size:10px;color:#bbb;font-weight:400;">click to highlight</span>
      </div>
      {legend_items}
      <div style="margin-top:10px;padding-top:8px;
                  border-top:1px solid #eee;
                  display:flex;gap:14px;font-size:11px;color:#666;">
        <span>
          <span style="display:inline-block;width:9px;height:9px;
                background:#E74C3C;border-radius:50%;
                vertical-align:middle;margin-right:3px;"></span>Origin
        </span>
        <span>
          <span style="display:inline-block;width:9px;height:9px;
                background:#2ECC71;border-radius:50%;
                vertical-align:middle;margin-right:3px;"></span>Destination
        </span>
      </div>
      <div style="margin-top:7px;font-size:10px;color:#ccc;">
        Click route line or item to highlight · click again to reset
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── route comparison table — bottom RIGHT ──
    # kept at bottom-right so it never overlaps the layer control (top-right)
    table_rows = "".join([
        f"""
        <tr id="trow_{i}"
            onclick="highlightRoute({i})"
            style="cursor:pointer;transition:background 0.15s;"
            onmouseover="this.style.background='#f9f9f9'"
            onmouseout="if({i}!==activeRoute)this.style.background='transparent'">
          <td style="padding:4px 8px;">
            <span style="display:inline-block;width:12px;height:12px;
                  background:{r['color']};border-radius:3px;
                  vertical-align:middle;margin-right:5px;"></span>
            <span style="font-size:12px;">{r['label']}</span>
          </td>
          <td style="padding:4px 8px;text-align:right;font-size:12px;">
            {r['travel_time_min']:.1f}
          </td>
          <td style="padding:4px 8px;text-align:right;font-size:12px;">
            {r['distance_km']:.2f}
          </td>
          <td style="padding:4px 8px;text-align:right;font-size:12px;">
            {r['Re']:.3f}
          </td>
          <td style="padding:4px 8px;text-align:right;font-size:12px;color:#666;">
            {r['composite']:.4f}
          </td>
        </tr>"""
        for i, r in enumerate(routes)
    ])

    summary_html = f"""
    <div style="
        position:fixed; bottom:28px; right:15px;
        z-index:9997;
        background:white; padding:12px 14px;
        border-radius:10px; border:1px solid #ddd;
        font-family:Arial,sans-serif;
        box-shadow:0 2px 10px rgba(0,0,0,0.12);
        max-width:420px;">
      <div style="font-weight:700;font-size:13px;margin-bottom:8px;">
        Route Comparison
      </div>
      <table style="border-collapse:collapse;width:100%;">
        <tr style="color:#aaa;font-size:11px;border-bottom:1px solid #eee;">
          <th style="padding:3px 8px;text-align:left;font-weight:500;">Route</th>
          <th style="padding:3px 8px;text-align:right;font-weight:500;">ETA</th>
          <th style="padding:3px 8px;text-align:right;font-weight:500;">Dist</th>
          <th style="padding:3px 8px;text-align:right;font-weight:500;">Risk</th>
          <th style="padding:3px 8px;text-align:right;font-weight:500;">Score</th>
        </tr>
        {table_rows}
      </table>
      <div style="margin-top:7px;font-size:10px;color:#bbb;
                  border-top:1px solid #eee;padding-top:6px;">
        {origin_str[:38]} &rarr; {dest_str[:38]}
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(summary_html))

    # ── JavaScript: highlight on click, dim others, reset on second click ──
    js = f"""
    <script>
    var activeRoute = -1;
    var allPolylines = [];
    var routeColors  = {colors_js};
    var routeWeights = {weights_js};

    document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(collectPolylines, 900);
    }});

    function collectPolylines() {{
        var paths = document.querySelectorAll('path.leaflet-interactive');
        var idx   = 0;
        paths.forEach(function(el) {{
            var sw = el.getAttribute('stroke-width');
            if (sw && parseFloat(sw) >= 4 && idx < {n_routes}) {{
                allPolylines.push(el);
                (function(i) {{
                    el.addEventListener('click', function(e) {{
                        highlightRoute(i);
                        e.stopPropagation();
                    }});
                }})(idx);
                idx++;
            }}
        }});
    }}

    function highlightRoute(idx) {{
        if (activeRoute === idx) {{
            // reset everything
            activeRoute = -1;
            allPolylines.forEach(function(el, i) {{
                el.style.opacity      = '0.80';
                el.style.strokeWidth  = routeWeights[i] + 'px';
                el.setAttribute('stroke-width', routeWeights[i]);
            }});
            for (var i = 0; i < {n_routes}; i++) {{
                var leg = document.getElementById('leg_'  + i);
                var row = document.getElementById('trow_' + i);
                if (leg) leg.style.background = 'transparent';
                if (row) row.style.background = 'transparent';
            }}
        }} else {{
            activeRoute = idx;
            allPolylines.forEach(function(el, i) {{
                if (i === idx) {{
                    el.style.opacity = '1.0';
                    el.setAttribute('stroke-width', routeWeights[i] + 4);
                }} else {{
                    el.style.opacity = '0.15';
                    el.setAttribute('stroke-width', routeWeights[i]);
                }}
            }});
            for (var i = 0; i < {n_routes}; i++) {{
                var leg = document.getElementById('leg_'  + i);
                var row = document.getElementById('trow_' + i);
                if (leg) leg.style.background = (i === idx) ? '#eef6ff' : 'transparent';
                if (row) row.style.background = (i === idx) ? '#eef6ff' : 'transparent';
            }}
        }}
    }}
    </script>
    """
    m.get_root().html.add_child(folium.Element(js))

    return m


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def generate_map(
    origin_str : str = "Manipal Hospital, Bangalore",
    dest_str   : str = "Victoria Hospital, Bangalore",
) -> str:
    print("\n" + "="*60)
    print("  STAGE 4 — Interactive Route Map (Fixed Layout)")
    print("="*60)
    print(f"\n  Origin      : {origin_str}")
    print(f"  Destination : {dest_str}")

    print("\n>>> Loading data...")
    G         = load_graph()
    routes_df = load_routes()

    print("\n>>> Recomputing full route paths...")
    routes = recompute_paths(G, routes_df, origin_str, dest_str)
    print(f"  [OK] {len(routes)} routes with full geometry")
    for r in routes:
        print(f"  {r['label']:20s}: {len(r['coords'])} nodes  "
              f"ETA={r['travel_time_min']:.1f}min  color={r['color']}")

    print("\n>>> Building interactive map...")
    m   = build_map(G, routes, origin_str, dest_str)
    out = PLOTS_DIR / "route_map.html"
    m.save(str(out))

    print(f"\n  [SAVED] {out}")
    print("  Open in Chrome or Firefox")
    print("  All routes visible — click any line or legend item to highlight")
    print("  Click again to reset all routes to full opacity")
    print("="*60)
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", type=str,
                        default="Manipal Hospital, Bangalore")
    parser.add_argument("--dest",   type=str,
                        default="Victoria Hospital, Bangalore")
    args = parser.parse_args()
    generate_map(args.origin, args.dest)
