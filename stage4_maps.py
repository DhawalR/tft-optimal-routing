"""
Stage 4: Interactive Route Map — All Routes Visible
====================================================
Multi-Factor Optimal Routing using Temporal Fusion Transformer
NIT Karnataka, Surathkal

Draws all Pareto-optimal routes simultaneously on one map.
Each route has a distinct color. Click any route to highlight it.

Color scheme (fixed, matches route_radar.png):
    balanced  → RED    #E74C3C
    fastest   → BLUE   #3498DB
    safest    → GREEN  #2ECC71
    cleanest  → PURPLE #9B59B6

Usage:
    python stage4_map.py
    python stage4_map.py --origin "Manipal Hospital, Bangalore" --dest "Victoria Hospital, Bangalore"
"""

import argparse
import pickle
import warnings
import json
import networkx as nx
import pandas as pd
import folium
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PROCESSED_DIR = Path(r"C:\Dhawal\Datasets\processed")
PLOTS_DIR     = Path(r"C:\Dhawal\plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ── FIXED color scheme — label keyword → color ──
# Order matters: checked top to bottom, first match wins
LABEL_COLORS = [
    ("safest",       "#2ECC71", 6),   # green,  weight 6
    ("safe",         "#2ECC71", 6),
    ("fastest",      "#3498DB", 5),   # blue,   weight 5
    ("fast",         "#3498DB", 5),
    ("low-te",       "#3498DB", 5),
    ("certain",      "#9B59B6", 4),   # purple, weight 4
    ("clean",        "#9B59B6", 4),
    ("speed+safety", "#E67E22", 5),   # orange, weight 5
    ("balanced",     "#E74C3C", 7),   # red,    weight 7  ← recommended
    ("recommended",  "#E74C3C", 7),
]

DEFAULT_COLOR  = "#95A5A6"
DEFAULT_WEIGHT = 4


def get_color_weight(label: str):
    ll = label.lower()
    for keyword, color, weight in LABEL_COLORS:
        if keyword in ll:
            return color, weight
    return DEFAULT_COLOR, DEFAULT_WEIGHT


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_graph() -> nx.MultiDiGraph:
    path = PROCESSED_DIR / "osm_graph.pkl"
    if not path.exists():
        raise FileNotFoundError(
            "osm_graph.pkl not found.\n"
            "Run: python stage4_routing.py"
        )
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"  [OK] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def load_routes() -> pd.DataFrame:
    path = PROCESSED_DIR / "pareto_routes.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "pareto_routes.parquet not found.\n"
            "Run: python stage4_routing.py"
        )
    df = pd.read_parquet(path)
    print(f"  [OK] {len(df)} Pareto routes loaded")
    print(f"  [OK] Labels: {df['label'].tolist()}")
    return df


# ─────────────────────────────────────────────
# EXTRACT COORDINATES FROM STORED PATHS
# ─────────────────────────────────────────────

def extract_route_coords(
    routes_df : pd.DataFrame,
    G         : nx.MultiDiGraph,
    origin_str: str,
    dest_str  : str,
) -> list:
    """
    For each route in routes_df, recompute the full node path and
    extract lat/lon coordinates.

    Strategy:
      1. Try to parse path_nodes column (JSON list of node IDs)
      2. Fall back to recomputing via shortest_path with stored weights
    """
    from stage4_routing import parse_location, get_nearest_node

    origin_lat, origin_lon = parse_location(origin_str)
    dest_lat,   dest_lon   = parse_location(dest_str)
    origin_node = get_nearest_node(G, origin_lat, origin_lon)
    dest_node   = get_nearest_node(G, dest_lat,   dest_lon)

    # weight sets — one per route label, in same order as stage4_routing.py
    weight_sets = {
        "balanced"     : (0.40, 0.20, 0.25, 0.15),
        "recommended"  : (0.40, 0.20, 0.25, 0.15),
        "fastest"      : (0.90, 0.05, 0.03, 0.02),
        "low-te"       : (0.90, 0.05, 0.03, 0.02),
        "most certain" : (0.05, 0.90, 0.03, 0.02),
        "low-ue"       : (0.05, 0.90, 0.03, 0.02),
        "safest"       : (0.05, 0.05, 0.88, 0.02),
        "low-re"       : (0.05, 0.05, 0.88, 0.02),
        "cleanest"     : (0.05, 0.05, 0.05, 0.85),
        "speed+safety" : (0.50, 0.20, 0.20, 0.10),
    }

    results    = []
    seen_sigs  = set()

    for _, row in routes_df.iterrows():
        label  = row["label"]
        color, lw = get_color_weight(label)

        # ── Try method 1: parse stored path_nodes ──
        coords = []
        path   = []
        try:
            raw = row.get("path_nodes", "")
            if raw and raw != "nan" and str(raw).startswith("["):
                node_ids = json.loads(str(raw))
                # validate nodes exist in graph
                valid = [n for n in node_ids if G.has_node(n)]
                if len(valid) > 1:
                    path   = valid
                    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
                    print(f"  [PATH] '{label}': {len(coords)} nodes from stored path")
        except Exception as e:
            print(f"  [WARN] Could not parse stored path for '{label}': {e}")

        # ── Fall back to method 2: recompute shortest path ──
        if len(coords) < 2:
            print(f"  [RECOMPUTE] '{label}': running shortest_path...")

            # find matching weight set
            ws = None
            ll = label.lower()
            for key, vals in weight_sets.items():
                if key in ll or ll in key:
                    ws = vals
                    break
            if ws is None:
                ws = (0.40, 0.20, 0.25, 0.15)   # default to balanced
                print(f"  [WARN] No weight set for '{label}', using balanced weights")

            a, b, g, d = ws

            # set temp weights
            for u, v, key in G.edges(keys=True):
                data = G[u][v][key]
                te   = data.get("weight_te", 0)
                ue   = data.get("weight_ue", 0)
                re   = data.get("weight_re", 0)
                pe   = data.get("weight_pe", 0)
                tt   = data.get("travel_time_s", 1)
                G[u][v][key]["_w"] = (a*te + b*ue + g*re + d*pe) * tt

            try:
                path   = nx.shortest_path(G, origin_node, dest_node, weight="_w")
                coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
                print(f"  [PATH] '{label}': {len(coords)} nodes via shortest_path")
            except nx.NetworkXNoPath:
                print(f"  [SKIP] '{label}': no path found")
                continue

        if len(coords) < 2:
            print(f"  [SKIP] '{label}': too few coordinates")
            continue

        # deduplicate by path signature
        sig = tuple(path[::max(1, len(path)//10)])
        if sig in seen_sigs:
            print(f"  [DEDUP] '{label}' is duplicate of earlier route — keeping both with label")
            # don't skip — different labels even if same path
        seen_sigs.add(sig)

        results.append({
            "label"           : label,
            "coords"          : coords,
            "color"           : color,
            "weight"          : lw,
            "travel_time_min" : float(row["travel_time_min"]),
            "distance_km"     : float(row["distance_km"]),
            "Te"              : float(row["Te_mean"]),
            "Ue"              : float(row["Ue_mean"]),
            "Re"              : float(row["Re_mean"]),
            "Pe"              : float(row["Pe_mean"]),
            "composite"       : float(row["composite"]),
        })

    print(f"\n  [OK] {len(results)} routes with full geometry")
    return results


# ─────────────────────────────────────────────
# BUILD MAP
# ─────────────────────────────────────────────

def build_map(
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

    n_routes   = len(routes)
    colors_js  = json.dumps([r["color"]  for r in routes])
    weights_js = json.dumps([r["weight"] for r in routes])

    # ── draw routes — thinner routes first, thicker on top ──
    for i, route in enumerate(sorted(routes, key=lambda r: r["weight"])):
        coords = route["coords"]
        color  = route["color"]
        lw     = route["weight"]
        label  = route["label"]
        eta    = route["travel_time_min"]
        dist   = route["distance_km"]

        popup_html = f"""
        <div style="font-family:Arial;font-size:13px;min-width:220px;">
          <b style="color:{color};font-size:15px;">{label}</b><br>
          <hr style="margin:5px 0;border:0;border-top:1px solid #eee;">
          <b>ETA:</b> {eta:.1f} min &nbsp;&nbsp;
          <b>Distance:</b> {dist:.2f} km<br>
          <hr style="margin:5px 0;border:0;border-top:1px solid #eee;">
          <table style="width:100%;font-size:12px;border-collapse:collapse;">
            <tr>
              <td style="color:#888;padding:2px 4px;">Te (travel time)</td>
              <td style="text-align:right;padding:2px 4px;">{route['Te']:.3f}</td>
            </tr>
            <tr>
              <td style="color:#888;padding:2px 4px;">Ue (uncertainty)</td>
              <td style="text-align:right;padding:2px 4px;">{route['Ue']:.3f}</td>
            </tr>
            <tr>
              <td style="color:#888;padding:2px 4px;">Re (incident risk)</td>
              <td style="text-align:right;padding:2px 4px;">{route['Re']:.3f}</td>
            </tr>
            <tr>
              <td style="color:#888;padding:2px 4px;">Pe (pollution)</td>
              <td style="text-align:right;padding:2px 4px;">{route['Pe']:.3f}</td>
            </tr>
          </table>
          <hr style="margin:5px 0;border:0;border-top:1px solid #eee;">
          <b>Composite score:</b> {route['composite']:.4f}
        </div>
        """

        folium.PolyLine(
            locations = coords,
            color     = color,
            weight    = lw,
            opacity   = 0.85,
            tooltip   = f"{label}: {eta:.1f} min · {dist:.2f} km",
            popup     = folium.Popup(popup_html, max_width=280),
        ).add_to(m)

    # ── origin marker (red plus) ──
    folium.Marker(
        location = [origin_lat, origin_lon],
        popup    = folium.Popup(
            f"<b>Origin</b><br>{origin_str}", max_width=220
        ),
        icon     = folium.Icon(color="red", icon="plus-sign", prefix="glyphicon"),
        tooltip  = f"Origin: {origin_str}",
    ).add_to(m)

    # ── destination marker (green flag) ──
    folium.Marker(
        location = [dest_lat, dest_lon],
        popup    = folium.Popup(
            f"<b>Destination</b><br>{dest_str}", max_width=220
        ),
        icon     = folium.Icon(color="green", icon="flag", prefix="glyphicon"),
        tooltip  = f"Destination: {dest_str}",
    ).add_to(m)

    # ── legend — bottom left ──
    legend_items = "".join([
        f"""
        <div id="leg_{i}"
             onclick="highlightRoute({i})"
             style="display:flex;align-items:center;gap:10px;
                    margin:5px 0;cursor:pointer;padding:6px 8px;
                    border-radius:6px;transition:background 0.15s;"
             onmouseover="this.style.background='#f5f5f5'"
             onmouseout="this.style.background='transparent'">
          <div style="flex-shrink:0;width:36px;height:6px;
                      background:{r['color']};border-radius:3px;"></div>
          <div>
            <span style="font-weight:700;font-size:13px;color:#1B2A4A;">
              {r['label']}
            </span><br>
            <span style="font-size:11px;color:#8C9BAB;">
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
        background:white; padding:14px 18px;
        border-radius:12px; border:1px solid #ddd;
        font-family:Arial,sans-serif;
        box-shadow:0 4px 12px rgba(0,0,0,0.12);
        min-width:240px;">
      <div style="font-weight:700;font-size:14px;color:#1B2A4A;
                  margin-bottom:10px;
                  display:flex;justify-content:space-between;align-items:center;">
        <span>Pareto-Optimal Routes</span>
        <span style="font-size:10px;color:#bbb;font-weight:400;">
          click to highlight
        </span>
      </div>
      {legend_items}
      <div style="margin-top:10px;padding-top:8px;
                  border-top:1px solid #eee;
                  display:flex;gap:16px;font-size:11px;color:#666;">
        <span>
          <span style="display:inline-block;width:10px;height:10px;
                background:#E74C3C;border-radius:50%;
                vertical-align:middle;margin-right:3px;"></span>
          Origin
        </span>
        <span>
          <span style="display:inline-block;width:10px;height:10px;
                background:#2ECC71;border-radius:50%;
                vertical-align:middle;margin-right:3px;"></span>
          Destination
        </span>
      </div>
      <div style="margin-top:6px;font-size:10px;color:#ccc;">
        Click route line or legend item to highlight.<br>
        Click again to reset all routes.
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── comparison table — bottom right ──
    table_rows = "".join([
        f"""
        <tr id="trow_{i}"
            onclick="highlightRoute({i})"
            style="cursor:pointer;transition:background 0.15s;"
            onmouseover="this.style.background='#f9f9f9'"
            onmouseout="if({i}!==activeRoute)this.style.background='transparent'">
          <td style="padding:5px 8px;">
            <span style="display:inline-block;width:14px;height:14px;
                  background:{r['color']};border-radius:3px;
                  vertical-align:middle;margin-right:6px;"></span>
            <span style="font-size:12px;font-weight:600;">{r['label']}</span>
          </td>
          <td style="padding:5px 8px;text-align:right;font-size:12px;">
            {r['travel_time_min']:.1f}
          </td>
          <td style="padding:5px 8px;text-align:right;font-size:12px;">
            {r['distance_km']:.2f}
          </td>
          <td style="padding:5px 8px;text-align:right;font-size:12px;">
            {r['Re']:.3f}
          </td>
          <td style="padding:5px 8px;text-align:right;font-size:12px;color:#888;">
            {r['composite']:.4f}
          </td>
        </tr>"""
        for i, r in enumerate(routes)
    ])

    summary_html = f"""
    <div style="
        position:fixed; bottom:28px; right:15px;
        z-index:9997;
        background:white; padding:12px 16px;
        border-radius:12px; border:1px solid #ddd;
        font-family:Arial,sans-serif;
        box-shadow:0 4px 12px rgba(0,0,0,0.12);
        max-width:440px;">
      <div style="font-weight:700;font-size:14px;color:#1B2A4A;margin-bottom:8px;">
        Route Comparison
      </div>
      <table style="border-collapse:collapse;width:100%;">
        <tr style="color:#aaa;font-size:11px;border-bottom:1px solid #eee;">
          <th style="padding:3px 8px;text-align:left;font-weight:500;">Route</th>
          <th style="padding:3px 8px;text-align:right;font-weight:500;">ETA (min)</th>
          <th style="padding:3px 8px;text-align:right;font-weight:500;">Dist (km)</th>
          <th style="padding:3px 8px;text-align:right;font-weight:500;">Risk (Re)</th>
          <th style="padding:3px 8px;text-align:right;font-weight:500;">Score</th>
        </tr>
        {table_rows}
      </table>
      <div style="margin-top:8px;font-size:10px;color:#bbb;
                  border-top:1px solid #eee;padding-top:6px;">
        {origin_str[:40]} &rarr; {dest_str[:40]}
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(summary_html))

    # ── JavaScript highlight-on-click ──
    js = f"""
    <script>
    var activeRoute = -1;
    var allPolylines = [];
    var routeColors  = {colors_js};
    var routeWeights = {weights_js};

    document.addEventListener('DOMContentLoaded', function() {{
        setTimeout(collectPolylines, 1000);
    }});

    function collectPolylines() {{
        var paths = document.querySelectorAll('path.leaflet-interactive');
        var idx = 0;
        paths.forEach(function(el) {{
            var sw = parseFloat(el.getAttribute('stroke-width') || '0');
            if (sw >= 4 && idx < {n_routes}) {{
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
        console.log('Route map: collected', allPolylines.length, 'polylines');
    }}

    function highlightRoute(idx) {{
        if (activeRoute === idx) {{
            activeRoute = -1;
            allPolylines.forEach(function(el, i) {{
                el.style.opacity     = '0.85';
                el.setAttribute('stroke-width', routeWeights[i]);
            }});
            for (var i = 0; i < {n_routes}; i++) {{
                var leg  = document.getElementById('leg_'  + i);
                var row  = document.getElementById('trow_' + i);
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
                    el.style.opacity = '0.12';
                    el.setAttribute('stroke-width', routeWeights[i]);
                }}
            }});
            for (var i = 0; i < {n_routes}; i++) {{
                var leg  = document.getElementById('leg_'  + i);
                var row  = document.getElementById('trow_' + i);
                if (leg) leg.style.background = (i===idx) ? '#eef6ff' : 'transparent';
                if (row) row.style.background = (i===idx) ? '#eef6ff' : 'transparent';
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
    print("  STAGE 4 — Route Map Generator")
    print("="*60)
    print(f"\n  Origin      : {origin_str}")
    print(f"  Destination : {dest_str}")

    print("\n>>> Loading OSM graph...")
    G = load_graph()

    print("\n>>> Loading Pareto routes...")
    routes_df = load_routes()

    print("\n>>> Extracting route coordinates...")
    routes = extract_route_coords(G, routes_df, origin_str, dest_str)

    if len(routes) == 0:
        print("\n  [ERROR] No routes could be drawn. Check that stage4_routing.py ran successfully.")
        return ""

    print(f"\n>>> Routes to draw:")
    for r in routes:
        print(f"  {r['label']:22s}  color={r['color']}  nodes={len(r['coords'])}  ETA={r['travel_time_min']:.1f}min")

    print("\n>>> Building map...")
    m   = build_map(routes, origin_str, dest_str)
    out = PLOTS_DIR / "route_map.html"
    m.save(str(out))

    print(f"\n  [SAVED] {out}")
    print("\n  Instructions:")
    print("  - Open route_map.html in Chrome or Firefox")
    print("  - All routes are visible simultaneously")
    print("  - Click any route LINE or LEGEND ITEM to highlight it")
    print("  - Click again to reset all routes")
    print(f"\n  Colors:")
    for r in routes:
        print(f"  {r['color']}  {r['label']}")
    print("="*60)
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multi-route map for emergency routing"
    )
    parser.add_argument(
        "--origin", type=str,
        default="Manipal Hospital, Bangalore"
    )
    parser.add_argument(
        "--dest", type=str,
        default="Victoria Hospital, Bangalore"
    )
    args = parser.parse_args()
    generate_map(args.origin, args.dest)
