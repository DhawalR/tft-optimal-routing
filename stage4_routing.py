"""
Stage 4: Multi-Objective Emergency Routing Engine
==================================================
Multi-Factor Optimal Routing using Temporal Fusion Transformer
NIT Karnataka, Surathkal

Builds a weighted directed graph from the Bangalore OSM road network,
assigns composite edge weights from Stage 2 and Stage 3 outputs,
and computes Pareto-optimal emergency routes.

Edge weight equation (from paper):
    We = alpha*Te + beta*Ue + gamma*Re + delta*Pe

Where:
    Te : Expected travel time       (TFT median forecast)
    Ue : Uncertainty                (TFT quantile band width)
    Re : Incident risk              (DoWhy ATE per segment)
    Pe : PM2.5 pollution exposure   (air quality lookup)

    alpha + beta + gamma + delta = 1.0

Outputs (written to processed/):
    - osm_graph.pkl          serialised OSMnx graph
    - weighted_edges.parquet edge weights for all road segments
    - pareto_routes.parquet  all Pareto-optimal routes
    - best_route.parquet     single recommended route

Usage:
    # default weights
    python stage4_routing.py --origin "Manipal Hospital, Bangalore" --dest "Victoria Hospital, Bangalore"

    # custom weights (must sum to 1.0)
    python stage4_routing.py --origin "12.9716,77.5946" --dest "12.9352,77.6245" --alpha 0.5 --beta 0.2 --gamma 0.2 --delta 0.1

    # exclude pollution (delta=0, redistribute to alpha)
    python stage4_routing.py --origin "12.9716,77.5946" --dest "12.9352,77.6245" --alpha 0.5 --beta 0.25 --gamma 0.25 --delta 0.0
"""

import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from pathlib import Path
from scipy.spatial import cKDTree
import json

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATA_DIR      = Path(r"C:\Dhawal\Datasets")
PROCESSED_DIR = DATA_DIR / "processed"
CKPT_DIR      = Path(r"C:\Dhawal\checkpoints")
PLOTS_DIR     = Path(r"C:\Dhawal\plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Bangalore bounding box for OSM download
BANGALORE_PLACE = "Bangalore, Karnataka, India"

# Default route weights (alpha + beta + gamma + delta = 1.0)
DEFAULT_ALPHA = 0.40   # Te: travel time (highest priority for emergency)
DEFAULT_BETA  = 0.20   # Ue: uncertainty
DEFAULT_GAMMA = 0.25   # Re: incident risk
DEFAULT_DELTA = 0.15   # Pe: pollution exposure

# Speed fallback when TFT forecast unavailable (km/h)
FALLBACK_SPEED_KMH = 30.0

# Global ATE fallback for segments without reliable causal estimate
GLOBAL_ATE_FALLBACK = 0.0389   # from Stage 3 output

# PM2.5 normalisation ceiling (ug/m3) — above this = max exposure
PM25_MAX = 200.0


# ─────────────────────────────────────────────
# STEP 1: DOWNLOAD / LOAD OSM GRAPH
# ─────────────────────────────────────────────

def get_osm_graph(force_download: bool = False) -> nx.MultiDiGraph:
    """
    Download Bangalore road network from OpenStreetMap via OSMnx.
    Caches to disk so subsequent runs are instant.

    Returns a directed multigraph where:
      - nodes = intersections (with lat/lon)
      - edges = road segments (with length, speed, geometry)
    """
    cache_path = PROCESSED_DIR / "osm_graph.pkl"

    if cache_path.exists() and not force_download:
        print(f"  [CACHE] Loading OSM graph from {cache_path}")
        with open(cache_path, "rb") as f:
            G = pickle.load(f)
        print(f"  [OK]    Nodes: {G.number_of_nodes():,}  Edges: {G.number_of_edges():,}")
        return G

    print(f"  [OSM]   Downloading Bangalore road network...")
    print(f"          This takes 2-5 minutes on first run only.")

    # drive network — roads usable by emergency vehicles
    G = ox.graph_from_place(
        BANGALORE_PLACE,
        network_type = "drive",
        simplify     = True,
    )

    # add travel time based on speed limits
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    print(f"  [OK]    Downloaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    with open(cache_path, "wb") as f:
        pickle.dump(G, f)
    print(f"  [SAVED] {cache_path}")
    return G


# ─────────────────────────────────────────────
# STEP 2: LOAD STAGE 2/3 OUTPUTS
# ─────────────────────────────────────────────

def load_tft_forecasts() -> pd.DataFrame:
    """
    Load TFT predictions from Stage 2.
    Uses stage1_merged as a proxy for current conditions
    since we need per-segment speed and uncertainty estimates.

    In a production system this would call the live TFT model.
    Here we use the validation set predictions saved during Stage 2.
    """
    df = pd.read_parquet(PROCESSED_DIR / "stage1_merged.parquet")

    # build unique_id
    df["unique_id"] = (
        df["Area Name"].str.strip() + "__" +
        df["Road/Intersection Name"].str.strip()
    )

    # aggregate to segment level: use recent observations as forecast proxy
    # In production: run TFT.predict() on current conditions
    segment_stats = (
        df.groupby("unique_id")
          .agg(
              lat          = ("Latitude",              "mean"),
              lon          = ("Longitude",             "mean"),
              speed_median = ("Average Speed",         "median"),   # Te proxy
              speed_q10    = ("Average Speed",         lambda x: x.quantile(0.10)),
              speed_q90    = ("Average Speed",         lambda x: x.quantile(0.90)),
              congestion   = ("Congestion Level",      "mean"),
              pm25         = ("pm2_5_mean",            "mean"),
              flood_risk   = ("flood_risk_score",      "mean"),
          )
          .reset_index()
    )

    # Ue = coefficient of variation (std/mean) — how variable is speed?
    # This measures true uncertainty, not just band width
    # Higher CoV = less predictable road = higher uncertainty penalty
    df_std = df.groupby("unique_id")["Average Speed"].std().reset_index()
    df_std.columns = ["unique_id", "speed_std"]
    segment_stats = segment_stats.merge(df_std, on="unique_id", how="left")
    segment_stats["speed_std"] = segment_stats["speed_std"].fillna(0)

    cov = segment_stats["speed_std"] / (segment_stats["speed_median"] + 1e-9)
    cov_max = cov.max() + 1e-9
    segment_stats["uncertainty"] = (cov / cov_max).clip(0, 1)

    # Te = normalised inverse speed
    # Use min-max normalisation so range is spread across [0, 1]
    spd_min = segment_stats["speed_median"].min()
    spd_max = segment_stats["speed_median"].max()
    spd_range = spd_max - spd_min + 1e-9
    segment_stats["te_score"] = (
        1.0 - (segment_stats["speed_median"] - spd_min) / spd_range
    ).clip(0, 1)

    # Pe = PM2.5 normalised exposure
    segment_stats["pe_score"] = (
        segment_stats["pm25"] / PM25_MAX
    ).clip(0, 1)

    print(f"  [TFT]   Loaded forecasts for {len(segment_stats)} road segments")
    return segment_stats


def load_ate_lookup() -> pd.DataFrame:
    """Load routing-ready ATE lookup from Stage 3."""
    ate_path = PROCESSED_DIR / "routing_ate_lookup.parquet"

    if not ate_path.exists():
        # fallback: use ate_results.parquet directly
        ate_path = PROCESSED_DIR / "ate_results.parquet"

    if not ate_path.exists():
        print("  [WARN]  No ATE lookup found — using global fallback for all segments")
        return pd.DataFrame(columns=["segment", "routing_ate"])

    df = pd.read_parquet(ate_path)

    # ensure routing_ate column exists
    if "routing_ate" not in df.columns:
        df["routing_ate"] = df["ate"].fillna(GLOBAL_ATE_FALLBACK)

    # normalise ATE to [0,1] for use as edge weight
    ate_min = df["routing_ate"].min()
    ate_max = df["routing_ate"].max()
    df["re_score"] = (
        (df["routing_ate"] - ate_min) / (ate_max - ate_min + 1e-9)
    ).clip(0, 1)

    print(f"  [ATE]   Loaded {len(df)} segment ATE values")
    return df


# ─────────────────────────────────────────────
# STEP 3: ASSIGN EDGE WEIGHTS
# ─────────────────────────────────────────────

def assign_edge_weights(
    G           : nx.MultiDiGraph,
    forecasts   : pd.DataFrame,
    ate_lookup  : pd.DataFrame,
    alpha       : float,
    beta        : float,
    gamma       : float,
    delta       : float,
) -> nx.MultiDiGraph:
    """
    Assign composite weight We to every edge in the OSM graph.

    Strategy:
      1. Build a KD-tree of our known segment lat/lons
      2. For each OSM edge midpoint, find nearest known segment
      3. Look up Te, Ue, Re, Pe scores for that segment
      4. Compute We = alpha*Te + beta*Ue + gamma*Re + delta*Pe
      5. Scale by edge length so longer roads cost proportionally more
    """
    print(f"\n>>> Assigning edge weights...")
    print(f"    Weights: alpha={alpha} beta={beta} gamma={gamma} delta={delta}")

    # build KD-tree from known segment locations
    seg_coords = forecasts[["lat", "lon"]].values
    tree       = cKDTree(seg_coords)

    # build ATE lookup dict for fast access
    ate_dict = {}
    if "segment" in ate_lookup.columns:
        ate_dict = dict(zip(
            ate_lookup["segment"],
            ate_lookup["re_score"]
        ))

    n_edges    = 0
    edge_records = []

    for u, v, key, data in G.edges(keys=True, data=True):
        # get edge midpoint lat/lon
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        mid_lat = (u_data["y"] + v_data["y"]) / 2
        mid_lon = (u_data["x"] + v_data["x"]) / 2

        # find nearest known road segment
        dist, idx = tree.query([mid_lat, mid_lon], k=1)
        seg_row   = forecasts.iloc[idx]

        # get component scores
        te = float(seg_row["te_score"])
        ue = float(seg_row["uncertainty"])
        pe = float(seg_row["pe_score"])

        # Re: use ATE lookup if available, else segment forecast
        seg_name = str(seg_row["unique_id"])
        re = ate_dict.get(seg_name, float(seg_row.get("flood_risk", 0.0)))
        re = float(re)

        # composite weight
        we = alpha * te + beta * ue + gamma * re + delta * pe

        # scale by travel time (seconds) so longer roads cost more
        travel_time = data.get("travel_time", data.get("length", 100) / (FALLBACK_SPEED_KMH / 3.6))
        we_scaled   = we * travel_time

        # store all components for analysis
        G[u][v][key]["weight_te"]  = te
        G[u][v][key]["weight_ue"]  = ue
        G[u][v][key]["weight_re"]  = re
        G[u][v][key]["weight_pe"]  = pe
        G[u][v][key]["weight_we"]  = we
        G[u][v][key]["weight"]     = we_scaled   # used by shortest path
        G[u][v][key]["travel_time_s"] = travel_time

        edge_records.append({
            "u": u, "v": v,
            "lat": mid_lat, "lon": mid_lon,
            "te": te, "ue": ue, "re": re, "pe": pe,
            "we": we, "travel_time_s": travel_time,
            "length_m": data.get("length", 0),
            "segment": seg_name,
        })
        n_edges += 1

    edges_df = pd.DataFrame(edge_records)
    edges_df.to_parquet(PROCESSED_DIR / "weighted_edges.parquet", index=False)
    print(f"  [OK]    Weighted {n_edges:,} edges")
    print(f"  [SAVED] weighted_edges.parquet")
    return G, edges_df


# ─────────────────────────────────────────────
# STEP 4: GEOCODE LOCATIONS
# ─────────────────────────────────────────────

def parse_location(location_str: str) -> tuple:
    """
    Parse a location string into (lat, lon).
    Accepts either:
      - "lat,lon"  e.g. "12.9716,77.5946"
      - Place name e.g. "Manipal Hospital, Bangalore"
    """
    location_str = location_str.strip()

    # try lat,lon format first
    parts = location_str.split(",")
    if len(parts) == 2:
        try:
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass

    # geocode place name
    print(f"  [GEO]   Geocoding: {location_str}")
    try:
        location = ox.geocode(location_str)
        lat, lon = location
        print(f"  [GEO]   Found: ({lat:.4f}, {lon:.4f})")
        return lat, lon
    except Exception as e:
        raise ValueError(
            f"Could not geocode '{location_str}': {e}\n"
            f"Try using lat,lon format instead: '12.9716,77.5946'"
        )


def get_nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Find the nearest OSM node to a given lat/lon."""
    node = ox.nearest_nodes(G, X=lon, Y=lat)
    node_data = G.nodes[node]
    print(f"  [NODE]  Nearest node: {node} at ({node_data['y']:.4f}, {node_data['x']:.4f})")
    return node


# ─────────────────────────────────────────────
# STEP 5: PARETO-OPTIMAL ROUTING
# ─────────────────────────────────────────────

def compute_route_objectives(
    G    : nx.MultiDiGraph,
    path : list,
) -> dict:
    """
    Compute the four objective values for a given route (list of nodes).
    Returns total Te, Ue, Re, Pe and travel time for the full path.
    """
    total_te = total_ue = total_re = total_pe = 0.0
    total_tt = total_dist = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        if not G.has_edge(u, v):
            continue
        # use minimum weight edge if multiple edges exist
        edges = G[u][v]
        best  = min(edges.values(), key=lambda d: d.get("weight", 999))
        total_te   += best.get("weight_te", 0)
        total_ue   += best.get("weight_ue", 0)
        total_re   += best.get("weight_re", 0)
        total_pe   += best.get("weight_pe", 0)
        total_tt   += best.get("travel_time_s", 0)
        total_dist += best.get("length", 0)

    n = max(len(path) - 1, 1)
    return {
        "n_nodes"      : len(path),
        "Te_total"     : total_te,
        "Ue_total"     : total_ue,
        "Re_total"     : total_re,
        "Pe_total"     : total_pe,
        "Te_mean"      : total_te / n,
        "Ue_mean"      : total_ue / n,
        "Re_mean"      : total_re / n,
        "Pe_mean"      : total_pe / n,
        "travel_time_s": total_tt,
        "travel_time_min": total_tt / 60,
        "distance_m"   : total_dist,
        "distance_km"  : total_dist / 1000,
    }


def find_pareto_routes(
    G          : nx.MultiDiGraph,
    origin_node: int,
    dest_node  : int,
    alpha      : float,
    beta       : float,
    gamma      : float,
    delta      : float,
    n_routes   : int = 5,
) -> list:
    """
    Compute Pareto-optimal routes by generating k candidate routes
    with different objective emphasis, then filtering for Pareto dominance.

    Strategy:
      1. Generate k routes, each optimising a different single objective
         (fastest, safest, cleanest, most certain) plus the composite
      2. Evaluate all four objectives for each route
      3. Filter to the Pareto-optimal subset (no route dominates another)

    Returns list of (path, objectives_dict) tuples, sorted by composite score.
    """
    print(f"\n>>> Computing Pareto-optimal routes...")

    candidate_routes = []

    # weight sets to generate diverse candidate routes
    weight_sets = [
        # (alpha, beta, gamma, delta, label)
        (alpha,  beta,   gamma,  delta,  "balanced"),
        (0.90,   0.05,   0.03,   0.02,   "fastest"),
        (0.05,   0.90,   0.03,   0.02,   "most certain"),
        (0.05,   0.05,   0.88,   0.02,   "safest"),
        (0.05,   0.05,   0.05,   0.85,   "cleanest"),
        (0.50,   0.20,   0.20,   0.10,   "speed+safety"),
        (0.30,   0.30,   0.30,   0.10,   "balanced-v2"),
    ]

    for a, b, g, d, label in weight_sets:
        # temporarily set edge weights for this objective emphasis
        for u, v, key in G.edges(keys=True):
            data = G[u][v][key]
            te   = data.get("weight_te", 0)
            ue   = data.get("weight_ue", 0)
            re   = data.get("weight_re", 0)
            pe   = data.get("weight_pe", 0)
            tt   = data.get("travel_time_s", 1)
            G[u][v][key]["weight_temp"] = (a*te + b*ue + g*re + d*pe) * tt

        try:
            path = nx.shortest_path(
                G, origin_node, dest_node,
                weight="weight_temp"
            )
            if len(path) > 1:
                objs = compute_route_objectives(G, path)
                objs["label"]       = label
                objs["alpha_used"]  = a
                objs["beta_used"]   = b
                objs["gamma_used"]  = g
                objs["delta_used"]  = d
                # composite score with original weights
                objs["composite"] = (
                    alpha * objs["Te_mean"] +
                    beta  * objs["Ue_mean"] +
                    gamma * objs["Re_mean"] +
                    delta * objs["Pe_mean"]
                )
                candidate_routes.append((path, objs))
                print(f"  [ROUTE] {label:15s}: {objs['travel_time_min']:.1f} min  "
                      f"Te={objs['Te_mean']:.3f}  Ue={objs['Ue_mean']:.3f}  "
                      f"Re={objs['Re_mean']:.3f}  Pe={objs['Pe_mean']:.3f}")
        except nx.NetworkXNoPath:
            print(f"  [SKIP]  {label}: no path found")
        except Exception as e:
            print(f"  [SKIP]  {label}: {e}")

    if not candidate_routes:
        raise RuntimeError("No valid routes found between origin and destination")

    # deduplicate routes by path signature
    seen_paths  = set()
    unique_routes = []
    for path, objs in candidate_routes:
        sig = tuple(path[::max(1, len(path)//10)])   # sampled signature
        if sig not in seen_paths:
            seen_paths.add(sig)
            unique_routes.append((path, objs))

    # Pareto filter: route A dominates B if A is better on ALL objectives
    def dominates(a_objs, b_objs):
        return (
            a_objs["Te_mean"] <= b_objs["Te_mean"] and
            a_objs["Ue_mean"] <= b_objs["Ue_mean"] and
            a_objs["Re_mean"] <= b_objs["Re_mean"] and
            a_objs["Pe_mean"] <= b_objs["Pe_mean"] and
            (
                a_objs["Te_mean"] < b_objs["Te_mean"] or
                a_objs["Ue_mean"] < b_objs["Ue_mean"] or
                a_objs["Re_mean"] < b_objs["Re_mean"] or
                a_objs["Pe_mean"] < b_objs["Pe_mean"]
            )
        )

    pareto_routes = []
    for i, (path_i, objs_i) in enumerate(unique_routes):
        dominated = False
        for j, (path_j, objs_j) in enumerate(unique_routes):
            if i != j and dominates(objs_j, objs_i):
                dominated = True
                break
        if not dominated:
            pareto_routes.append((path_i, objs_i))

    # sort by composite score
    pareto_routes.sort(key=lambda x: x[1]["composite"])

    print(f"\n  [PARETO] {len(unique_routes)} unique candidates → "
          f"{len(pareto_routes)} Pareto-optimal routes")
    return pareto_routes


# ─────────────────────────────────────────────
# STEP 6: SAVE AND REPORT RESULTS
# ─────────────────────────────────────────────

def save_routes(
    pareto_routes : list,
    G             : nx.MultiDiGraph,
    origin_str    : str,
    dest_str      : str,
) -> pd.DataFrame:
    """Save Pareto routes to parquet and print summary table."""

    records = []
    for rank, (path, objs) in enumerate(pareto_routes):
        # extract node coordinates for the path
        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in path]
        records.append({
            "rank"            : rank + 1,
            "label"           : objs["label"],
            "travel_time_min" : round(objs["travel_time_min"], 2),
            "distance_km"     : round(objs["distance_km"], 3),
            "Te_mean"         : round(objs["Te_mean"], 4),
            "Ue_mean"         : round(objs["Ue_mean"], 4),
            "Re_mean"         : round(objs["Re_mean"], 4),
            "Pe_mean"         : round(objs["Pe_mean"], 4),
            "composite"       : round(objs["composite"], 4),
            "n_nodes"         : objs["n_nodes"],
            #"path_nodes"      : str(path),   # full path for map
            "path_nodes"      : json.dumps([int(n) for n in path]),  # JSON serialised for map
            "origin"          : origin_str,
            "destination"     : dest_str,
        })

    routes_df = pd.DataFrame(records)
    # sort by ETA for intuitive reading — fastest first in table
    # recommended route is still the lowest composite score
    routes_df = routes_df.sort_values("travel_time_min").reset_index(drop=True)
    routes_df["rank"] = range(1, len(routes_df) + 1)
    routes_df.to_parquet(PROCESSED_DIR / "pareto_routes.parquet", index=False)
    routes_df.iloc[[0]].to_parquet(PROCESSED_DIR / "best_route.parquet", index=False)

    print("\n" + "="*70)
    print("  PARETO-OPTIMAL ROUTES — SUMMARY")
    print("="*70)
    print(f"  Origin      : {origin_str}")
    print(f"  Destination : {dest_str}")
    print(f"  Routes found: {len(pareto_routes)}")
    print()
    header = f"  {'Rank':<5} {'Label':<18} {'ETA(min)':<10} {'Dist(km)':<10} {'Te':>6} {'Ue':>6} {'Re':>6} {'Pe':>6}"
    print(header)
    print("  " + "-"*65)
    for _, row in routes_df.iterrows():
        print(f"  {int(row['rank']):<5} {row['label']:<18} "
              f"{row['travel_time_min']:<10.1f} {row['distance_km']:<10.2f} "
              f"{row['Te_mean']:>6.3f} {row['Ue_mean']:>6.3f} "
              f"{row['Re_mean']:>6.3f} {row['Pe_mean']:>6.3f}")

    print()
    best = routes_df.iloc[0]
    # recommended = lowest composite score (best overall tradeoff)
    recommended = routes_df.loc[routes_df["composite"].idxmin()]
    print(f"  RECOMMENDED ROUTE: '{recommended['label']}'")
    print(f"  ETA      : {recommended['travel_time_min']:.1f} minutes")
    print(f"  Distance : {recommended['distance_km']:.2f} km")
    print(f"  Composite score  : {recommended['composite']:.4f} (lower = better)")
    print(f"\n  NOTE: 'low-Te (speed priority)' has shorter Te score but longer")
    print(f"  physical distance because it routes through faster but longer roads.")
    print(f"  The recommended route balances all four objectives including Re and Pe.")
    print("="*70)

    print(f"\n  [SAVED] pareto_routes.parquet")
    print(f"  [SAVED] best_route.parquet")
    return routes_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_routing(
    origin_str  : str,
    dest_str    : str,
    alpha       : float = DEFAULT_ALPHA,
    beta        : float = DEFAULT_BETA,
    gamma       : float = DEFAULT_GAMMA,
    delta       : float = DEFAULT_DELTA,
    force_download : bool = False,
) -> tuple:

    # validate weights
    total = alpha + beta + gamma + delta
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Weights must sum to 1.0 (got {total:.3f}). "
            f"Adjust alpha, beta, gamma, delta."
        )

    print("\n" + "="*60)
    print("  STAGE 4 — Multi-Objective Emergency Routing")
    print("  NIT Karnataka — Emergency Routing Framework")
    print("="*60)
    print(f"\n  Origin      : {origin_str}")
    print(f"  Destination : {dest_str}")
    print(f"  Weights     : alpha={alpha} beta={beta} gamma={gamma} delta={delta}")

    # step 1: OSM graph
    print("\n>>> Step 1: Loading road network...")
    G = get_osm_graph(force_download)

    # step 2: load stage outputs
    print("\n>>> Step 2: Loading Stage 2/3 outputs...")
    forecasts  = load_tft_forecasts()
    ate_lookup = load_ate_lookup()

    # step 3: assign edge weights
    G, edges_df = assign_edge_weights(
        G, forecasts, ate_lookup, alpha, beta, gamma, delta
    )

    # step 4: geocode locations
    print("\n>>> Step 3: Resolving locations...")
    origin_lat, origin_lon = parse_location(origin_str)
    dest_lat,   dest_lon   = parse_location(dest_str)

    origin_node = get_nearest_node(G, origin_lat, origin_lon)
    dest_node   = get_nearest_node(G, dest_lat,   dest_lon)

    if origin_node == dest_node:
        raise ValueError("Origin and destination resolve to the same node. Use more specific locations.")

    # step 5: Pareto routing
    pareto_routes = find_pareto_routes(
        G, origin_node, dest_node, alpha, beta, gamma, delta
    )

    # step 6: save results
    routes_df = save_routes(pareto_routes, G, origin_str, dest_str)

    print("\n" + "="*60)
    print("  STAGE 4 COMPLETE")
    print(f"  Results -> {PROCESSED_DIR / 'pareto_routes.parquet'}")
    print("  Next    -> python stage4_evaluate.py")
    print("="*60)

    return G, pareto_routes, routes_df


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-objective emergency vehicle routing for Bangalore"
    )
    parser.add_argument(
        "--origin", type=str,
        default="Manipal Hospital, Bangalore",
        help="Origin location (place name or 'lat,lon')"
    )
    parser.add_argument(
        "--dest", type=str,
        default="Victoria Hospital, Bangalore",
        help="Destination location (place name or 'lat,lon')"
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Weight for Te (travel time)")
    parser.add_argument("--beta",  type=float, default=DEFAULT_BETA,
                        help="Weight for Ue (uncertainty)")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help="Weight for Re (incident risk)")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA,
                        help="Weight for Pe (pollution)")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download OSM graph even if cached")

    args = parser.parse_args()

    G, pareto_routes, routes_df = run_routing(
        origin_str     = args.origin,
        dest_str       = args.dest,
        alpha          = args.alpha,
        beta           = args.beta,
        gamma          = args.gamma,
        delta          = args.delta,
        force_download = args.force_download,
    )
