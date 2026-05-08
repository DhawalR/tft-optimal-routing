"""
Stage 4: Routing Evaluation & Visualisation
============================================
Multi-Factor Optimal Routing using Temporal Fusion Transformer
NIT Karnataka, Surathkal

Loads Pareto-optimal routes from stage4_routing.py and produces
publication-ready plots and an interactive HTML map.

Plots produced (saved to C:\\UserName\\plots\\):
  1. pareto_front.png         2D Pareto front (ETA vs risk)
  2. route_radar.png          radar chart comparing all routes
  3. objective_breakdown.png  stacked bar of We components per route
  4. route_map.html           interactive Folium map with all routes

Usage:
    python stage4_evaluate.py
    (Run after stage4_routing.py has completed)
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

PROCESSED_DIR = Path(r"C:\Dhawal\Datasets\processed")
PLOTS_DIR     = Path(r"C:\Dhawal\plots")
PLOTS_DIR.mkdir(exist_ok=True)

COLORS = {
    "balanced"     : "#2C3E50",
    "fastest"      : "#E74C3C",
    "most certain" : "#3498DB",
    "safest"       : "#2ECC71",
    "cleanest"     : "#9B59B6",
    "speed+safety" : "#E67E22",
    "balanced-v2"  : "#1ABC9C",
    "default"      : "#95A5A6",
    "grid"         : "#ECF0F1",
}

ROUTE_COLORS_MAP = [
    "#E74C3C", "#3498DB", "#2ECC71",
    "#9B59B6", "#E67E22", "#1ABC9C", "#2C3E50"
]

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.color"       : COLORS["grid"],
    "grid.linewidth"   : 0.8,
    "figure.dpi"       : 150,
})


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_data() -> tuple:
    routes_path = PROCESSED_DIR / "pareto_routes.parquet"
    edges_path  = PROCESSED_DIR / "weighted_edges.parquet"
    graph_path  = PROCESSED_DIR / "osm_graph.pkl"

    if not routes_path.exists():
        raise FileNotFoundError(
            f"Routes not found at {routes_path}\n"
            f"Run: python stage4_routing.py first"
        )

    routes_df = pd.read_parquet(routes_path)
    edges_df  = pd.read_parquet(edges_path) if edges_path.exists() else None

    G = None
    if graph_path.exists():
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

    print(f"  [OK] Loaded {len(routes_df)} Pareto-optimal routes")
    return routes_df, edges_df, G


# ─────────────────────────────────────────────
# PLOT 1: PARETO FRONT
# ─────────────────────────────────────────────

def plot_pareto_front(routes_df: pd.DataFrame):
    """
    2D Pareto front plot: ETA vs Incident Risk.
    Each point is a route — points on the Pareto front are
    non-dominated (no other route is better on both axes).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Pareto-Optimal Route Fronts", fontsize=13, fontweight="bold")

    pairs = [
        ("travel_time_min", "Re_mean",  "ETA (minutes)", "Incident Risk (Re)"),
        ("travel_time_min", "Pe_mean",  "ETA (minutes)", "Pollution Exposure (Pe)"),
    ]

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
        for i, row in routes_df.iterrows():
            color = ROUTE_COLORS_MAP[i % len(ROUTE_COLORS_MAP)]
            ax.scatter(row[xcol], row[ycol], s=180,
                       color=color, zorder=5, edgecolors="white", lw=1)
            ax.annotate(
                row["label"],
                (row[xcol], row[ycol]),
                fontsize=9, alpha=0.85,
                xytext=(6, 6), textcoords="offset points"
            )

        # draw Pareto front line
        sorted_df = routes_df.sort_values(xcol)
        pareto_x, pareto_y = [], []
        best_y = float("inf")
        for _, row in sorted_df.iterrows():
            if row[ycol] < best_y:
                pareto_x.append(row[xcol])
                pareto_y.append(row[ycol])
                best_y = row[ycol]

        if len(pareto_x) > 1:
            ax.plot(pareto_x, pareto_y, color="#2C3E50",
                    lw=1.5, linestyle="--", alpha=0.6, label="Pareto front")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs {xlabel}")
        ax.legend(fontsize=9)

    out = PLOTS_DIR / "pareto_front.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 2: RADAR CHART
# ─────────────────────────────────────────────

def plot_radar(routes_df: pd.DataFrame):
    """
    Radar chart comparing all routes across four objectives.
    Best route = smallest area (all objectives minimised).
    """
    objectives = ["Te_mean", "Ue_mean", "Re_mean", "Pe_mean"]
    labels     = ["Travel Time\n(Te)", "Uncertainty\n(Ue)",
                  "Incident Risk\n(Re)", "Pollution\n(Pe)"]
    n_obj      = len(objectives)

    angles = np.linspace(0, 2 * np.pi, n_obj, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (_, row) in enumerate(routes_df.iterrows()):
        values = [row[o] for o in objectives]
        values += values[:1]
        color  = ROUTE_COLORS_MAP[i % len(ROUTE_COLORS_MAP)]
        ax.plot(angles, values, color=color, lw=2, label=row["label"])
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Route Comparison — All Objectives\n(smaller area = better overall)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    out = PLOTS_DIR / "route_radar.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 3: OBJECTIVE BREAKDOWN
# ─────────────────────────────────────────────

def plot_objective_breakdown(routes_df: pd.DataFrame):
    """
    Stacked bar chart showing how each We component contributes
    to the composite score of each route.
    """
    objectives = ["Te_mean", "Ue_mean", "Re_mean", "Pe_mean"]
    obj_labels = ["Te (travel time)", "Ue (uncertainty)",
                  "Re (incident risk)", "Pe (pollution)"]
    obj_colors = ["#3498DB", "#E67E22", "#E74C3C", "#9B59B6"]

    route_labels = routes_df["label"].tolist()
    x = np.arange(len(route_labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Route Objective Breakdown", fontsize=13, fontweight="bold")

    # stacked bar — absolute values
    bottom = np.zeros(len(routes_df))
    for obj, label, color in zip(objectives, obj_labels, obj_colors):
        vals = routes_df[obj].values
        axes[0].bar(x, vals, bottom=bottom, label=label,
                    color=color, alpha=0.85)
        bottom += vals

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(route_labels, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Composite score components (mean per edge)")
    axes[0].set_title("Absolute objective values per route")
    axes[0].legend(fontsize=9)

    # ETA comparison bar
    colors = [ROUTE_COLORS_MAP[i % len(ROUTE_COLORS_MAP)]
              for i in range(len(routes_df))]
    axes[1].bar(x, routes_df["travel_time_min"], color=colors, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(route_labels, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Estimated travel time (minutes)")
    axes[1].set_title("ETA comparison across Pareto routes")

    # annotate best route
    best_idx = routes_df["composite"].idxmin()
    axes[1].bar(best_idx, routes_df.loc[best_idx, "travel_time_min"],
                color=ROUTE_COLORS_MAP[0], alpha=1.0,
                edgecolor="black", lw=2, label="Recommended route")
    axes[1].legend(fontsize=9)

    out = PLOTS_DIR / "objective_breakdown.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# # ─────────────────────────────────────────────
# # PLOT 4: INTERACTIVE MAP
# # ─────────────────────────────────────────────

# def plot_route_map(routes_df: pd.DataFrame, G):
#     """
#     Interactive Folium map showing all Pareto-optimal routes
#     overlaid on the Bangalore street map.
#     """
#     try:
#         import folium
#         import osmnx as ox
#     except ImportError:
#         print("  [SKIP] folium/osmnx not available for map plot")
#         return

#     if G is None:
#         print("  [SKIP] OSM graph not available for map plot")
#         return

#     # centre map on Bangalore
#     m = folium.Map(
#         location=[12.9716, 77.5946],
#         zoom_start=13,
#         tiles="CartoDB positron"
#     )

#     # plot weighted edges as heatmap background
#     edges_path = PROCESSED_DIR / "weighted_edges.parquet"
#     if edges_path.exists():
#         edges_df = pd.read_parquet(edges_path)
#         # sample edges for performance
#         sample = edges_df.sample(min(2000, len(edges_df)), random_state=42)
#         for _, row in sample.iterrows():
#             # colour by composite weight
#             we = float(row.get("we", 0.5))
#             r  = int(min(255, we * 510))
#             g  = int(min(255, (1 - we) * 510))
#             color = f"#{r:02x}{g:02x}00"
#             folium.CircleMarker(
#                 location=[row["lat"], row["lon"]],
#                 radius=2,
#                 color=color,
#                 fill=True,
#                 fill_opacity=0.4,
#                 popup=f"We={we:.3f} Te={row['te']:.3f}",
#             ).add_to(m)

#     # plot routes
#     route_colors_folium = [
#         "red", "blue", "green", "purple",
#         "orange", "darkred", "cadetblue"
#     ]

#     for i, (_, row) in enumerate(routes_df.iterrows()):
#         color = route_colors_folium[i % len(route_colors_folium)]

#         # extract path node coordinates
#         try:
#             path_str = str(row.get("path_nodes", ""))
#             # We stored truncated path — use node positions from graph
#             # Draw route as a line between origin and destination
#             origin    = row.get("origin", "")
#             dest      = row.get("destination", "")
#         except:
#             continue

#         label     = row["label"]
#         eta       = row["travel_time_min"]
#         dist      = row["distance_km"]
#         composite = row["composite"]

#         # add route marker
#         folium.Marker(
#             location=[12.9716 + i * 0.002, 77.5946 + i * 0.002],
#             popup=folium.Popup(
#                 f"<b>{label}</b><br>"
#                 f"ETA: {eta:.1f} min<br>"
#                 f"Distance: {dist:.2f} km<br>"
#                 f"Composite: {composite:.4f}<br>"
#                 f"Te={row['Te_mean']:.3f} Ue={row['Ue_mean']:.3f}<br>"
#                 f"Re={row['Re_mean']:.3f} Pe={row['Pe_mean']:.3f}",
#                 max_width=250
#             ),
#             icon=folium.Icon(color=color, icon="ambulance", prefix="fa"),
#         ).add_to(m)

#     # legend
#     legend_html = """
#     <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
#                 background: white; padding: 12px; border-radius: 8px;
#                 border: 1px solid #ccc; font-size: 12px;">
#         <b>Pareto-Optimal Routes</b><br>
#         <span style="color:red">&#9632;</span> Balanced (recommended)<br>
#         <span style="color:blue">&#9632;</span> Fastest<br>
#         <span style="color:green">&#9632;</span> Safest<br>
#         <span style="color:purple">&#9632;</span> Cleanest<br>
#         <span style="color:orange">&#9632;</span> Most Certain<br>
#         <br>
#         <b>Edge colour:</b> green=low risk, red=high risk
#     </div>
#     """
#     m.get_root().html.add_child(folium.Element(legend_html))

#     out = PLOTS_DIR / "route_map.html"
#     m.save(str(out))
#     print(f"  [MAP]  {out}  (open in browser)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def evaluate():
    print("\n" + "="*60)
    print("  STAGE 4 — Routing Evaluation & Visualisation")
    print("="*60)

    print("\n>>> Loading routing results...")
    routes_df, edges_df, G = load_data()

    print("\n>>> Generating plots...")
    plot_pareto_front(routes_df)
    plot_radar(routes_df)
    plot_objective_breakdown(routes_df)
    plot_route_map(routes_df, G)

    # print final summary
    best = routes_df.iloc[0]
    print("\n" + "="*60)
    print("  STAGE 4 EVALUATION COMPLETE")
    print("="*60)
    print(f"  Pareto routes    : {len(routes_df)}")
    print(f"  Recommended      : {best['label']}")
    print(f"  ETA              : {best['travel_time_min']:.1f} minutes")
    print(f"  Distance         : {best['distance_km']:.2f} km")
    print(f"  Composite score  : {best['composite']:.4f}")
    print(f"  All plots saved  : {PLOTS_DIR}")
    print("="*60)
    return routes_df


if __name__ == "__main__":
    routes_df = evaluate()
