"""
Stage 5: Emergency Routing Dashboard (Streamlit)
=================================================
Multi-Factor Optimal Routing using Temporal Fusion Transformer
NIT Karnataka, Surathkal

Route-focused dashboard that reads from Stage 4 parquet outputs.
Shows Pareto-optimal routes with weight sliders, origin/dest picker,
embedded Folium map, and metric cards.

Usage:
    streamlit run stage5_dashboard.py

Requirements:
    pip install streamlit plotly folium streamlit-folium
"""

import warnings
import pickle
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PROCESSED_DIR = Path(r"C:\Dhawal\Datasets\processed")
CKPT_DIR      = Path(r"C:\Dhawal\checkpoints")
PLOTS_DIR     = Path(r"C:\Dhawal\plots")

# Route colors — identical to route_radar.png and route_map.html
ROUTE_COLORS = {
    "balanced"     : "#2C3E50",
    "recommended"  : "#2C3E50",
    "fastest"      : "#E74C3C",
    "most certain" : "#3498DB",
    "safest"       : "#2ECC71",
    "cleanest"     : "#9B59B6",
    "speed+safety" : "#E67E22",
    "balanced-v2"  : "#1ABC9C",
    "default"      : "#95A5A6",
}

WEIGHT_DEFAULTS = dict(alpha=0.40, beta=0.20, gamma=0.25, delta=0.15)

KNOWN_LOCATIONS = {
    "Manipal Hospital, Bangalore"     : (12.9352, 77.6245),
    "Victoria Hospital, Bangalore"    : (12.9659, 77.5694),
    "St. John's Hospital, Bangalore"  : (12.9250, 77.6227),
    "NIMHANS, Bangalore"              : (12.9396, 77.5953),
    "Fortis Hospital, Bannerghatta"   : (12.8931, 77.5969),
    "Narayana Health City, Bommasandra": (12.8340, 77.6540),
    "Custom (enter coordinates below)": None,
}


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title = "Emergency Routing Dashboard — NIT Karnataka",
    page_icon  = "🚑",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1B2A4A 0%, #2E4170 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        border-left: 6px solid #D4722A;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .main-header p {
        color: #C8D4E3;
        margin: 0.2rem 0 0;
        font-size: 0.9rem;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        border: 1px solid #DADEEA;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #8C9BAB;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1B2A4A;
        line-height: 1.1;
    }
    .metric-card .note {
        font-size: 0.75rem;
        color: #D4722A;
        margin-top: 0.2rem;
    }

    /* Route card */
    .route-card {
        background: white;
        border: 1px solid #DADEEA;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border-left: 5px solid #95A5A6;
    }
    .route-card.recommended {
        border-left: 5px solid #2C3E50;
        background: #F0F4F8;
    }
    .route-card h4 {
        margin: 0 0 0.4rem;
        font-size: 1.0rem;
        color: #1B2A4A;
    }
    .route-card .route-meta {
        font-size: 0.82rem;
        color: #8C9BAB;
    }
    .route-card .route-scores {
        display: flex;
        gap: 0.8rem;
        margin-top: 0.5rem;
        flex-wrap: wrap;
    }
    .score-pill {
        background: #F4F5F7;
        border-radius: 20px;
        padding: 0.15rem 0.6rem;
        font-size: 0.78rem;
        color: #1B2A4A;
    }

    /* Section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1B2A4A;
        border-bottom: 2px solid #D4722A;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }

    /* Sidebar */
    .sidebar-section {
        background: #F4F5F7;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }

    /* Warning / info banners */
    .info-banner {
        background: #EBF0F8;
        border-left: 4px solid #1B2A4A;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 0.8rem;
        font-size: 0.85rem;
        color: #1B2A4A;
        margin-bottom: 0.8rem;
    }
    .warning-banner {
        background: #FFF3E8;
        border-left: 4px solid #D4722A;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 0.8rem;
        font-size: 0.85rem;
        color: #7A3D12;
        margin-bottom: 0.8rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_routes():
    path = PROCESSED_DIR / "pareto_routes.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)

@st.cache_data(ttl=300)
def load_edges():
    path = PROCESSED_DIR / "weighted_edges.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)

@st.cache_data(ttl=300)
def load_ate():
    path = PROCESSED_DIR / "routing_ate_lookup.parquet"
    if path.exists():
        return pd.read_parquet(path)
    path2 = PROCESSED_DIR / "ate_results.parquet"
    if path2.exists():
        return pd.read_parquet(path2)
    return None

@st.cache_data(ttl=300)
def load_traffic():
    path = PROCESSED_DIR / "stage1_merged.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # only load a summary — full parquet is 1M rows
    summary = (
        df.groupby(["Area Name", "Road/Intersection Name"])
          .agg(
              lat          = ("Latitude",        "mean"),
              lon          = ("Longitude",       "mean"),
              avg_speed    = ("Average Speed",   "mean"),
              congestion   = ("Congestion Level","mean"),
              incidents    = ("Incident Reports","mean"),
              pm25         = ("pm2_5_mean",      "mean"),
              flood_risk   = ("flood_risk_score","mean"),
          )
          .reset_index()
    )
    summary["unique_id"] = summary["Area Name"].str.strip() + "__" + summary["Road/Intersection Name"].str.strip()
    return summary

@st.cache_resource
def load_osm_graph():
    path = PROCESSED_DIR / "osm_graph.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_route_color(label: str) -> str:
    ll = label.lower()
    for k, c in ROUTE_COLORS.items():
        if k in ll:
            return c
    return ROUTE_COLORS["default"]


# ─────────────────────────────────────────────
# ROUTING (calls stage4_routing.py)
# ─────────────────────────────────────────────

def run_routing(origin: str, dest: str, alpha: float, beta: float,
                gamma: float, delta: float) -> bool:
    """Call stage4_routing.py and stage4_map.py as subprocesses."""
    try:
        routing_script = Path(r"C:\Dhawal\stage4_routing.py")
        map_script     = Path(r"C:\Dhawal\stage4_map.py")

        with st.spinner("Computing Pareto-optimal routes..."):
            result = subprocess.run(
                [sys.executable, str(routing_script),
                 "--origin", origin, "--dest", dest,
                 "--alpha", str(alpha), "--beta", str(beta),
                 "--gamma", str(gamma), "--delta", str(delta)],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                st.error(f"Routing failed:\n{result.stderr[-500:]}")
                return False

        with st.spinner("Generating route map..."):
            subprocess.run(
                [sys.executable, str(map_script),
                 "--origin", origin, "--dest", dest],
                capture_output=True, text=True, timeout=120
            )

        st.cache_data.clear()
        return True

    except subprocess.TimeoutExpired:
        st.error("Routing timed out (>5 min). Try coordinates instead of place name.")
        return False
    except Exception as e:
        st.error(f"Error: {e}")
        return False


# ─────────────────────────────────────────────
# BUILD FOLIUM MAP
# ─────────────────────────────────────────────

def build_folium_map(routes_df: pd.DataFrame, G, traffic_df: pd.DataFrame) -> folium.Map:
    """Build the route map directly inside the dashboard."""
    import networkx as nx

    # centre on Bangalore
    centre = [12.9716, 77.5946]
    m = folium.Map(location=centre, zoom_start=13, tiles="CartoDB positron")

    # ── congestion heatmap circles ──
    if traffic_df is not None:
        for _, row in traffic_df.iterrows():
            cong   = float(row.get("congestion", 0.5))
            r      = int(min(255, cong * 510))
            g_val  = int(min(255, (1 - cong) * 510))
            color  = f"#{r:02x}{g_val:02x}00"
            folium.CircleMarker(
                location    = [row["lat"], row["lon"]],
                radius      = 8 + cong * 10,
                color       = color,
                fill        = True,
                fill_color  = color,
                fill_opacity= 0.55,
                tooltip     = (
                    f"{row['Road/Intersection Name']}<br>"
                    f"Congestion: {cong:.2f}<br>"
                    f"Avg Speed: {row['avg_speed']:.1f} km/h<br>"
                    f"PM2.5: {row['pm25']:.1f}"
                ),
            ).add_to(m)

    if routes_df is None or len(routes_df) == 0 or G is None:
        return m

    # ── draw route polylines ──
    try:
        import sys
        sys.path.insert(0, str(Path(r"C:\Dhawal")))
        from stage4_routing import parse_location, get_nearest_node
        import networkx as nx_lib

        origin_str = str(routes_df["origin"].iloc[0])
        dest_str   = str(routes_df["destination"].iloc[0])

        olat, olon = parse_location(origin_str)
        dlat, dlon = parse_location(dest_str)
        o_node = get_nearest_node(G, olat, olon)
        d_node = get_nearest_node(G, dlat, dlon)

        weight_sets = [
            (0.40, 0.20, 0.25, 0.15, "balanced"),
            (0.90, 0.05, 0.03, 0.02, "fastest"),
            (0.05, 0.05, 0.88, 0.02, "safest"),
            (0.05, 0.90, 0.03, 0.02, "most certain"),
            (0.05, 0.05, 0.05, 0.85, "cleanest"),
        ]

        valid_labels = set(routes_df["label"].str.lower())
        seen = set()
        route_idx = 0

        for a, b, g, d, label in weight_sets:
            if not any(label.split()[0] in vl or vl in label for vl in valid_labels):
                continue

            for u, v, key in G.edges(keys=True):
                data = G[u][v][key]
                te   = data.get("weight_te", 0)
                ue   = data.get("weight_ue", 0)
                re   = data.get("weight_re", 0)
                pe   = data.get("weight_pe", 0)
                tt   = data.get("travel_time_s", 1)
                G[u][v][key]["weight_temp"] = (a*te + b*ue + g*re + d*pe) * tt

            try:
                path = nx_lib.shortest_path(G, o_node, d_node, weight="weight_temp")
            except Exception:
                continue

            sig = tuple(path[::max(1, len(path)//8)])
            if sig in seen:
                continue
            seen.add(sig)

            match = routes_df[routes_df["label"].str.lower().str.contains(
                label.split()[0], na=False
            )]
            if len(match) == 0:
                continue
            row = match.iloc[0]

            coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
            color  = get_route_color(row["label"])
            lw     = 7 if "balanced" in row["label"].lower() or "recommended" in row["label"].lower() else 5

            folium.PolyLine(
                locations = coords,
                color     = color,
                weight    = lw,
                opacity   = 0.85,
                tooltip   = f"{row['label']}: {row['travel_time_min']:.1f} min · {row['distance_km']:.2f} km",
                popup     = folium.Popup(
                    f"<b style='color:{color}'>{row['label']}</b><br>"
                    f"ETA: {row['travel_time_min']:.1f} min<br>"
                    f"Distance: {row['distance_km']:.2f} km<br>"
                    f"Re (risk): {row['Re_mean']:.3f}<br>"
                    f"Pe (PM2.5): {row['Pe_mean']:.3f}<br>"
                    f"Composite: {row['composite']:.4f}",
                    max_width=240
                ),
            ).add_to(m)
            route_idx += 1

        # markers
        folium.Marker(
            location = [olat, olon],
            icon     = folium.Icon(color="red", icon="plus-sign", prefix="glyphicon"),
            tooltip  = f"Origin: {origin_str}",
        ).add_to(m)
        folium.Marker(
            location = [dlat, dlon],
            icon     = folium.Icon(color="green", icon="flag", prefix="glyphicon"),
            tooltip  = f"Destination: {dest_str}",
        ).add_to(m)

        # re-centre map between origin and destination
        m.location = [(olat + dlat) / 2, (olon + dlon) / 2]

    except Exception as e:
        st.warning(f"Could not draw route lines: {e}")

    return m


# ─────────────────────────────────────────────
# RADAR CHART
# ─────────────────────────────────────────────

def make_radar(routes_df: pd.DataFrame) -> go.Figure:
    categories = ["Te (travel time)", "Ue (uncertainty)",
                  "Re (incident risk)", "Pe (pollution)"]
    cols       = ["Te_mean", "Ue_mean", "Re_mean", "Pe_mean"]

    fig = go.Figure()
    for _, row in routes_df.iterrows():
        color = get_route_color(row["label"])
        vals  = [row[c] for c in cols]
        vals += vals[:1]
        cats  = categories + categories[:1]
        fig.add_trace(go.Scatterpolar(
            r     = vals,
            theta = cats,
            fill  = "toself",
            name  = row["label"],
            line  = dict(color=color, width=2),
            fillcolor = color,
            opacity   = 0.18,
        ))

    fig.update_layout(
        polar  = dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))),
        showlegend    = True,
        legend        = dict(orientation="h", y=-0.18),
        margin        = dict(t=20, b=60, l=30, r=30),
        height        = 340,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────
# BAR CHART — OBJECTIVE BREAKDOWN
# ─────────────────────────────────────────────

def make_bar_chart(routes_df: pd.DataFrame) -> go.Figure:
    cols   = ["Te_mean", "Ue_mean", "Re_mean", "Pe_mean"]
    labels = ["Te", "Ue", "Re", "Pe"]
    colors = ["#1B2A4A", "#2E4170", "#D4722A", "#9B59B6"]

    fig = go.Figure()
    for col, lbl, color in zip(cols, labels, colors):
        fig.add_trace(go.Bar(
            name      = lbl,
            x         = routes_df["label"],
            y         = routes_df[col],
            marker_color = color,
            opacity   = 0.85,
        ))

    fig.update_layout(
        barmode       = "stack",
        xaxis_title   = "",
        yaxis_title   = "Composite score contribution",
        legend        = dict(orientation="h", y=1.12),
        margin        = dict(t=40, b=40, l=40, r=20),
        height        = 280,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
    )
    fig.update_xaxes(tickangle=-20)
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    # ── header ──
    st.markdown("""
    <div class="main-header">
        <h1>🚑  Emergency Routing Dashboard</h1>
        <p>Multi-Factor Optimal Routing using Temporal Fusion Transformer  ·  NIT Karnataka, Surathkal  ·  AINA 2026</p>
    </div>
    """, unsafe_allow_html=True)

    # ── load data ──
    routes_df  = load_routes()
    ate_df     = load_ate()
    traffic_df = load_traffic()
    G          = load_osm_graph()

    # ── sidebar ──
    with st.sidebar:
        st.markdown("### 🗺️ Route Configuration")

        # origin
        st.markdown("**Origin**")
        origin_choice = st.selectbox("Select origin", list(KNOWN_LOCATIONS.keys()), index=0, label_visibility="collapsed")
        if KNOWN_LOCATIONS[origin_choice] is None:
            origin_input = st.text_input("Origin (place name or lat,lon)", "12.9716,77.5946")
        else:
            origin_input = origin_choice

        # destination
        st.markdown("**Destination**")
        dest_choice = st.selectbox("Select destination", list(KNOWN_LOCATIONS.keys()), index=1, label_visibility="collapsed")
        if KNOWN_LOCATIONS[dest_choice] is None:
            dest_input = st.text_input("Destination (place name or lat,lon)", "12.9659,77.5694")
        else:
            dest_input = dest_choice

        st.divider()

        # weight sliders
        st.markdown("### ⚖️ Objective Weights")
        st.caption("Must sum to 1.0. Adjust to change routing priority.")

        alpha = st.slider("α — Te (travel time)",   0.0, 1.0, WEIGHT_DEFAULTS["alpha"], 0.05)
        beta  = st.slider("β — Ue (uncertainty)",   0.0, 1.0, WEIGHT_DEFAULTS["beta"],  0.05)
        gamma = st.slider("γ — Re (incident risk)", 0.0, 1.0, WEIGHT_DEFAULTS["gamma"], 0.05)
        delta = st.slider("δ — Pe (pollution)",     0.0, 1.0, WEIGHT_DEFAULTS["delta"], 0.05)

        total = round(alpha + beta + gamma + delta, 3)
        if abs(total - 1.0) > 0.01:
            st.markdown(f'<div class="warning-banner">⚠️ Weights sum to <b>{total}</b> — must equal 1.0</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-banner">✓ Weights sum to {total}</div>', unsafe_allow_html=True)

        st.divider()

        compute_btn = st.button(
            "🔄  Compute Routes",
            disabled=(abs(total - 1.0) > 0.01),
            use_container_width=True,
            type="primary",
        )

        if compute_btn:
            success = run_routing(origin_input, dest_input, alpha, beta, gamma, delta)
            if success:
                routes_df = load_routes()
                st.success("Routes updated!")
            st.rerun()

        st.divider()

        # ATE sidebar info
        st.markdown("### 📊 Causal ATE Summary")
        if ate_df is not None and "routing_ate" in ate_df.columns:
            valid_ate = ate_df.dropna(subset=["routing_ate"])
            st.metric("Global ATE", "+0.0389 TTI", help="Average causal delay from incidents across all Bangalore roads")
            st.metric("Segments estimated", len(valid_ate))
            st.metric("Refutation passed", f"{(valid_ate.get('refute_pass', pd.Series()) == True).sum()} / {len(valid_ate)}")
        else:
            st.caption("Run stage3_causal.py to see ATE results")

    # ── check if routes available ──
    if routes_df is None or len(routes_df) == 0:
        st.markdown("""
        <div class="warning-banner">
        ⚠️ No route results found. Click <b>Compute Routes</b> in the sidebar, or run:<br>
        <code>python stage4_routing.py</code>
        </div>
        """, unsafe_allow_html=True)

        # show blank map centred on Bangalore
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB positron")
        if traffic_df is not None:
            for _, row in traffic_df.iterrows():
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]], radius=10,
                    color="#1B2A4A", fill=True, fill_opacity=0.5,
                    tooltip=row["Road/Intersection Name"]
                ).add_to(m)
        st_folium(m, height=500, use_container_width=True)
        return

    # ── metric cards ──
    recommended = routes_df.loc[routes_df["composite"].idxmin()]
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Recommended ETA</div>
            <div class="value">{recommended['travel_time_min']:.1f} min</div>
            <div class="note">{recommended['label']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Distance</div>
            <div class="value">{recommended['distance_km']:.2f} km</div>
            <div class="note">recommended route</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Incident Risk (Re)</div>
            <div class="value">{recommended['Re_mean']:.3f}</div>
            <div class="note">lower = safer</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        pm25_val = traffic_df["pm25"].mean() if traffic_df is not None else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">City PM2.5</div>
            <div class="value">{pm25_val:.1f}</div>
            <div class="note">µg/m³ mean</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Pareto Routes</div>
            <div class="value">{len(routes_df)}</div>
            <div class="note">non-dominated</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── main layout: map left, details right ──
    map_col, detail_col = st.columns([3, 2], gap="medium")

    with map_col:
        st.markdown('<div class="section-header">🗺️ Route Map</div>', unsafe_allow_html=True)
        st.caption("All Pareto-optimal routes shown. Circle size = congestion level. Click any route line for details.")

        with st.spinner("Rendering map..."):
            m = build_folium_map(routes_df, G, traffic_df)
            st_folium(m, height=480, use_container_width=True)

        # legend
        legend_items = ""
        for _, row in routes_df.iterrows():
            color = get_route_color(row["label"])
            legend_items += (
                f'<span style="display:inline-flex;align-items:center;gap:6px;'
                f'margin-right:14px;font-size:0.82rem;">'
                f'<span style="display:inline-block;width:28px;height:4px;'
                f'background:{color};border-radius:2px;"></span>'
                f'{row["label"]} ({row["travel_time_min"]:.1f} min)</span>'
            )
        st.markdown(f'<div style="margin-top:0.4rem;">{legend_items}</div>', unsafe_allow_html=True)

    with detail_col:
        # ── route cards ──
        st.markdown('<div class="section-header">🏆 Pareto-Optimal Routes</div>', unsafe_allow_html=True)

        origin_str = str(routes_df["origin"].iloc[0])
        dest_str   = str(routes_df["destination"].iloc[0])
        st.caption(f"{origin_str} → {dest_str}")

        for _, row in routes_df.sort_values("composite").iterrows():
            color   = get_route_color(row["label"])
            is_rec  = row["label"] == recommended["label"]
            rec_tag = " ⭐ Recommended" if is_rec else ""

            st.markdown(f"""
            <div class="route-card {'recommended' if is_rec else ''}"
                 style="border-left-color:{color};">
                <h4 style="color:{color};">{row['label']}{rec_tag}</h4>
                <div class="route-meta">
                    {row['travel_time_min']:.1f} min &nbsp;·&nbsp; {row['distance_km']:.2f} km
                </div>
                <div class="route-scores">
                    <span class="score-pill">Te {row['Te_mean']:.3f}</span>
                    <span class="score-pill">Ue {row['Ue_mean']:.3f}</span>
                    <span class="score-pill" style="background:#FFF3E8;color:#7A3D12;">Re {row['Re_mean']:.3f}</span>
                    <span class="score-pill">Pe {row['Pe_mean']:.3f}</span>
                </div>
                <div style="margin-top:0.4rem;font-size:0.78rem;color:#8C9BAB;">
                    Composite score: {row['composite']:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # weight display
        st.markdown(f"""
        <div class="info-banner">
            Current weights: α={alpha:.2f} (Te) · β={beta:.2f} (Ue) · γ={gamma:.2f} (Re) · δ={delta:.2f} (Pe)
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── charts row ──
    chart_col1, chart_col2 = st.columns(2, gap="medium")

    with chart_col1:
        st.markdown('<div class="section-header">📡 Radar — Objective Comparison</div>', unsafe_allow_html=True)
        st.caption("Smaller area = better overall. Each axis is a routing objective (lower = better).")
        st.plotly_chart(make_radar(routes_df), use_container_width=True, config={"displayModeBar": False})

    with chart_col2:
        st.markdown('<div class="section-header">📊 Objective Breakdown per Route</div>', unsafe_allow_html=True)
        st.caption("Stacked bars show how much each objective contributes to the composite score.")
        st.plotly_chart(make_bar_chart(routes_df), use_container_width=True, config={"displayModeBar": False})

    st.divider()

    # ── ATE heatmap ──
    if ate_df is not None and traffic_df is not None:
        st.markdown('<div class="section-header">🔬 Causal Incident Delay — Per Road Segment</div>', unsafe_allow_html=True)
        st.caption("ATE = Average Treatment Effect. How many TTI units does a high-incident condition CAUSE on each road? (Stage 3 output)")

        ate_col1, ate_col2 = st.columns([2, 1], gap="medium")

        with ate_col1:
            # merge ATE with traffic for lat/lon
            seg_col = "segment" if "segment" in ate_df.columns else ate_df.columns[0]
            ate_plot = ate_df.dropna(subset=["routing_ate" if "routing_ate" in ate_df.columns else "ate"]).copy()
            ate_val_col = "routing_ate" if "routing_ate" in ate_df.columns else "ate"

            # try to get lat/lon
            if "Latitude" in ate_plot.columns and "Longitude" in ate_plot.columns:
                lat_col, lon_col = "Latitude", "Longitude"
            else:
                # merge with traffic summary
                ate_plot = ate_plot.rename(columns={seg_col: "unique_id"})
                if "unique_id" in traffic_df.columns:
                    ate_plot = ate_plot.merge(
                        traffic_df[["unique_id", "lat", "lon"]], on="unique_id", how="left"
                    )
                    lat_col, lon_col = "lat", "lon"
                else:
                    lat_col = lon_col = None

            if lat_col and lat_col in ate_plot.columns:
                fig_map = px.scatter_mapbox(
                    ate_plot,
                    lat    = lat_col,
                    lon    = lon_col,
                    color  = ate_val_col,
                    size   = ate_plot[ate_val_col].abs(),
                    hover_name  = seg_col if seg_col in ate_plot.columns else None,
                    color_continuous_scale = "RdYlGn_r",
                    color_continuous_midpoint = 0,
                    zoom   = 11,
                    height = 320,
                    mapbox_style = "carto-positron",
                    title  = "ATE per road segment (red = high incident delay)",
                )
                fig_map.update_layout(margin=dict(t=30, b=0, l=0, r=0),
                                       paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Lat/lon not available in ATE results — run stage3_evaluate.py")

        with ate_col2:
            # ATE table
            display_cols = [seg_col, ate_val_col]
            if "refute_pass" in ate_df.columns:
                display_cols.append("refute_pass")
            if "ci_lower" in ate_df.columns:
                display_cols += ["ci_lower", "ci_upper"]

            show_df = ate_df[display_cols].dropna(subset=[ate_val_col]).sort_values(ate_val_col, ascending=False)
            show_df = show_df.head(10)
            show_df.columns = [c.replace("routing_ate", "ATE").replace("refute_pass", "Robust?") for c in show_df.columns]
            st.dataframe(show_df, use_container_width=True, height=320, hide_index=True)

    st.divider()

    # ── footer ──
    st.markdown("""
    <div style="text-align:center;color:#8C9BAB;font-size:0.8rem;padding:0.5rem 0;">
        Multi-Factor Optimal Routing using TFT  ·  NIT Karnataka, Surathkal  ·  AINA 2026  ·
        Madhusudhan R & Dhawal Ramdham
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
