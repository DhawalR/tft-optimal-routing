"""
Stage 3: Causal Inference Evaluation & Visualisation
======================================================
Multi-Factor Optimal Routing using Temporal Fusion Transformer
NIT Karnataka, Surathkal

Loads ATE results from stage3_causal.py and produces
publication-ready plots for the paper and presentation.

Plots produced (saved to C:\\Dhawal\\plots\\):
  1. ate_distribution.png       distribution of ATEs across road segments
  2. ate_per_segment.png        bar chart of ATE per road segment with CIs
  3. ate_vs_congestion.png      scatter: ATE vs mean congestion level
  4. confounder_balance.png     propensity score overlap check
  5. causal_graph.png           visual of the causal DAG
  6. ate_heatmap.png            spatial heatmap of ATE on Bangalore map

Usage:
    python stage3_evaluate.py
    (Run after stage3_causal.py has completed)
"""

import warnings
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
    "positive"  : "#E74C3C",
    "negative"  : "#2ECC71",
    "neutral"   : "#3498DB",
    "ci"        : "#AED6F1",
    "confounder": "#9B59B6",
    "grid"      : "#ECF0F1",
    "text"      : "#2C3E50",
}

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
# LOAD RESULTS
# ─────────────────────────────────────────────

def load_results() -> tuple:
    seg_path  = PROCESSED_DIR / "ate_results.parquet"
    glob_path = PROCESSED_DIR / "ate_global.parquet"
    meta_path = PROCESSED_DIR / "nf_meta.parquet"

    if not seg_path.exists():
        raise FileNotFoundError(
            f"ATE results not found at {seg_path}\n"
            f"Run: python stage3_causal.py first"
        )

    seg_df  = pd.read_parquet(seg_path)
    glob_df = pd.read_parquet(glob_path)
    meta_df = pd.read_parquet(meta_path) if meta_path.exists() else None

    # merge lat/lon for spatial plots
    if meta_df is not None and "segment" in seg_df.columns:
        seg_df = seg_df.merge(
            meta_df[["unique_id", "Latitude", "Longitude"]].rename(
                columns={"unique_id": "segment"}
            ),
            on="segment", how="left"
        )

    valid = seg_df.dropna(subset=["ate"])
    print(f"  [OK] Loaded {len(valid)} valid segment ATE estimates")
    print(f"  [OK] Global ATE: {glob_df['ate'].iloc[0]:+.4f}")
    return valid, glob_df


# ─────────────────────────────────────────────
# PLOT 1: ATE DISTRIBUTION
# ─────────────────────────────────────────────

def plot_ate_distribution(seg_df: pd.DataFrame, global_ate: float):
    """Histogram of ATE values across all road segments."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [COLORS["positive"] if v > 0 else COLORS["negative"]
              for v in seg_df["ate"]]

    ax.hist(seg_df["ate"], bins=max(10, len(seg_df)//2),
            color=COLORS["neutral"], alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", lw=1.5, linestyle="--", label="Zero effect")
    ax.axvline(global_ate, color=COLORS["positive"], lw=2,
               linestyle="-", label=f"Global ATE = {global_ate:+.4f}")
    ax.axvline(seg_df["ate"].mean(), color=COLORS["confounder"], lw=2,
               linestyle=":", label=f"Mean segment ATE = {seg_df['ate'].mean():+.4f}")

    ax.set_xlabel("ATE (Travel Time Index units caused by incident)")
    ax.set_ylabel("Number of road segments")
    ax.set_title("Distribution of Causal Incident Effects Across Road Segments",
                 fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9)

    # annotation
    pct_positive = (seg_df["ate"] > 0).mean() * 100
    ax.text(0.97, 0.95,
            f"{pct_positive:.0f}% of segments show\npositive ATE (delay increase)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    out = PLOTS_DIR / "ate_distribution.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 2: ATE PER SEGMENT WITH CONFIDENCE INTERVALS
# ─────────────────────────────────────────────

def plot_ate_per_segment(seg_df: pd.DataFrame):
    """
    Horizontal bar chart of ATE per segment with 95% CI error bars.
    This is the key plot showing segment-level causal heterogeneity.
    """
    df = seg_df.dropna(subset=["ate", "ci_lower", "ci_upper"]).copy()
    df = df.sort_values("ate", ascending=True).reset_index(drop=True)

    # shorten segment labels for readability
    df["label"] = df["segment"].apply(
        lambda x: str(x).split("__")[-1][:30] if "__" in str(x) else str(x)[:30]
    )

    fig_h = max(6, len(df) * 0.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    bar_colors = [COLORS["positive"] if v > 0 else COLORS["negative"]
                  for v in df["ate"]]
    xerr_low  = (df["ate"] - df["ci_lower"]).clip(lower=0).values
    xerr_high = (df["ci_upper"] - df["ate"]).clip(lower=0).values

    ax.barh(range(len(df)), df["ate"], color=bar_colors, alpha=0.8,
            xerr=[xerr_low, xerr_high],
            error_kw=dict(ecolor="gray", capsize=3, lw=1))

    ax.axvline(0, color="black", lw=1.5, linestyle="--")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_xlabel("ATE (Travel Time Index units caused by incident)")
    ax.set_title("Per-Segment Causal Effect of Incidents on Travel Time\n"
                 "with 95% Confidence Intervals",
                 fontsize=12, fontweight="bold")

    pos_patch = mpatches.Patch(color=COLORS["positive"], alpha=0.8, label="Delay increase")
    neg_patch = mpatches.Patch(color=COLORS["negative"], alpha=0.8, label="Delay decrease / no effect")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right")

    out = PLOTS_DIR / "ate_per_segment.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 3: ATE vs CONGESTION LEVEL
# ─────────────────────────────────────────────

def plot_ate_vs_congestion(seg_df: pd.DataFrame, raw_df: pd.DataFrame):
    """
    Scatter plot: ATE vs mean congestion level per segment.
    Shows whether incidents cause MORE delay on already-congested roads.
    This is a key insight for the routing engine.
    """
    # compute mean congestion per segment
    if "unique_id" not in raw_df.columns:
        raw_df["unique_id"] = (
            raw_df["Area Name"].str.strip() + "__" +
            raw_df["Road/Intersection Name"].str.strip()
        )

    cong_mean = (
        raw_df.groupby("unique_id")["Congestion Level"]
              .mean().reset_index()
              .rename(columns={"unique_id": "segment",
                                "Congestion Level": "mean_congestion"})
    )
    df = seg_df.dropna(subset=["ate"]).merge(cong_mean, on="segment", how="left")
    df = df.dropna(subset=["mean_congestion"])

    if len(df) < 3:
        print("  [SKIP] Not enough segments for ATE vs congestion plot")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    sc = ax.scatter(df["mean_congestion"], df["ate"],
                    c=df["ate"], cmap="RdYlGn_r",
                    s=120, alpha=0.85, edgecolors="white", lw=0.5)
    plt.colorbar(sc, ax=ax, label="ATE value")

    # add segment labels
    for _, row in df.iterrows():
        label = str(row["segment"]).split("__")[-1][:15]
        ax.annotate(label, (row["mean_congestion"], row["ate"]),
                    fontsize=8, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points")

    # fit trend line if enough points
    if len(df) >= 4:
        z    = np.polyfit(df["mean_congestion"], df["ate"], 1)
        p    = np.poly1d(z)
        xfit = np.linspace(df["mean_congestion"].min(),
                           df["mean_congestion"].max(), 100)
        ax.plot(xfit, p(xfit), color=COLORS["neutral"],
                lw=2, linestyle="--", alpha=0.8, label="Trend")
        ax.legend()

    ax.axhline(0, color="gray", lw=1, linestyle=":")
    ax.set_xlabel("Mean Congestion Level (0–1)")
    ax.set_ylabel("ATE — Causal delay caused by incident")
    ax.set_title("Do Incidents Cause More Delay on Congested Roads?",
                 fontsize=12, fontweight="bold")

    out = PLOTS_DIR / "ate_vs_congestion.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 4: CAUSAL GRAPH (DAG)
# ─────────────────────────────────────────────

def plot_causal_dag():
    """
    Draw the causal DAG used in the analysis.
    Shows treatment, outcome, and confounders with arrows.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # node positions
    nodes = {
        "Incident\nReports\n(Treatment)"      : (2.0, 3.5),
        "Travel Time\nIndex\n(Outcome)"        : (8.0, 3.5),
        "Congestion\nLevel"                    : (5.0, 6.0),
        "Time of Day\n& Weekend"               : (5.0, 1.0),
        "Weather\n(Rain / Fog)"                : (1.0, 6.0),
        "Festival\nDay"                        : (1.0, 1.0),
    }

    node_colors = {
        "Incident\nReports\n(Treatment)"  : "#E8D5F5",
        "Travel Time\nIndex\n(Outcome)"   : "#D5E8F5",
        "Congestion\nLevel"               : "#F5E8D5",
        "Time of Day\n& Weekend"          : "#F5E8D5",
        "Weather\n(Rain / Fog)"           : "#F5E8D5",
        "Festival\nDay"                   : "#F5E8D5",
    }

    # draw edges first (so nodes appear on top)
    arrows = [
        # confounders → treatment
        ("Congestion\nLevel",      "Incident\nReports\n(Treatment)"),
        ("Time of Day\n& Weekend", "Incident\nReports\n(Treatment)"),
        ("Weather\n(Rain / Fog)",  "Incident\nReports\n(Treatment)"),
        ("Festival\nDay",          "Incident\nReports\n(Treatment)"),
        # confounders → outcome
        ("Congestion\nLevel",      "Travel Time\nIndex\n(Outcome)"),
        ("Time of Day\n& Weekend", "Travel Time\nIndex\n(Outcome)"),
        ("Weather\n(Rain / Fog)",  "Travel Time\nIndex\n(Outcome)"),
        ("Festival\nDay",          "Travel Time\nIndex\n(Outcome)"),
        # treatment → outcome (the causal path)
        ("Incident\nReports\n(Treatment)", "Travel Time\nIndex\n(Outcome)"),
    ]

    for src, dst in arrows:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        is_causal = (src == "Incident\nReports\n(Treatment)")
        color = COLORS["positive"] if is_causal else "#AAAAAA"
        lw    = 2.5 if is_causal else 1.2
        ax.annotate("",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->", color=color,
                lw=lw, connectionstyle="arc3,rad=0.1"
            )
        )

    # draw nodes
    for label, (x, y) in nodes.items():
        is_treatment = "Treatment" in label
        is_outcome   = "Outcome" in label
        bbox_props   = dict(
            boxstyle  = "round,pad=0.4",
            facecolor = node_colors[label],
            edgecolor = COLORS["positive"] if is_treatment else
                        COLORS["neutral"] if is_outcome else "#AAAAAA",
            linewidth = 2.5 if (is_treatment or is_outcome) else 1.2,
        )
        ax.text(x, y, label, ha="center", va="center",
                fontsize=10, fontweight="bold" if (is_treatment or is_outcome) else "normal",
                bbox=bbox_props)

    # legend
    treatment_patch = mpatches.Patch(
        facecolor="#E8D5F5", edgecolor=COLORS["positive"],
        linewidth=2, label="Treatment variable"
    )
    outcome_patch = mpatches.Patch(
        facecolor="#D5E8F5", edgecolor=COLORS["neutral"],
        linewidth=2, label="Outcome variable"
    )
    confounder_patch = mpatches.Patch(
        facecolor="#F5E8D5", edgecolor="#AAAAAA",
        linewidth=1.2, label="Confounders (controlled)"
    )
    causal_line = mpatches.Patch(
        facecolor=COLORS["positive"],
        label="Causal path of interest"
    )
    ax.legend(handles=[treatment_patch, outcome_patch,
                        confounder_patch, causal_line],
              loc="upper right", fontsize=9)

    ax.set_title("Causal DAG — Incident Effect on Travel Time",
                 fontsize=13, fontweight="bold", pad=15)

    out = PLOTS_DIR / "causal_graph.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 5: SPATIAL HEATMAP (lat/lon scatter)
# ─────────────────────────────────────────────

def plot_spatial_ate(seg_df: pd.DataFrame):
    """
    Scatter plot of road segments coloured by ATE magnitude.
    Shows which areas of Bangalore have highest incident impact.
    """
    df = seg_df.dropna(subset=["ate", "Latitude", "Longitude"])
    if len(df) < 2:
        print("  [SKIP] Not enough segments with lat/lon for spatial plot")
        return

    fig, ax = plt.subplots(figsize=(9, 8))

    sc = ax.scatter(
        df["Longitude"], df["Latitude"],
        c=df["ate"], cmap="RdYlGn_r",
        s=200, alpha=0.9, edgecolors="white", lw=0.8,
        vmin=df["ate"].quantile(0.05),
        vmax=df["ate"].quantile(0.95),
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("ATE — Incident delay (Travel Time Index units)", fontsize=10)

    # label each point with shortened road name
    for _, row in df.iterrows():
        label = str(row["segment"]).split("__")[-1][:20]
        ax.annotate(label,
                    (row["Longitude"], row["Latitude"]),
                    fontsize=7, alpha=0.8,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Distribution of Causal Incident Effects\nBangalore Road Network",
                 fontsize=12, fontweight="bold")

    out = PLOTS_DIR / "ate_spatial.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 6: REFUTATION SUMMARY
# ─────────────────────────────────────────────

def plot_refutation_summary(seg_df: pd.DataFrame, global_ate: float):
    """
    Bar chart showing how many segments passed refutation tests.
    This validates the robustness of our causal estimates.
    """
    df = seg_df.dropna(subset=["ate"])

    passed  = (df["refute_pass"] == True).sum()
    failed  = (df["refute_pass"] == False).sum()
    unknown = df["refute_pass"].isna().sum()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Causal Estimate Robustness", fontsize=13, fontweight="bold")

    # refutation results pie
    labels  = ["Passed", "Failed", "Unknown"]
    sizes   = [passed, failed, unknown]
    colors  = [COLORS["negative"], COLORS["positive"], "#BDC3C7"]
    sizes_f = [s for s, l in zip(sizes, labels) if s > 0]
    labels_f= [l for s, l in zip(sizes, labels) if s > 0]
    colors_f= [c for s, c in zip(sizes, colors) if s > 0]

    if sum(sizes_f) > 0:
        axes[0].pie(sizes_f, labels=labels_f, colors=colors_f,
                    autopct="%1.0f%%", startangle=90,
                    wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[0].set_title("Refutation Test Results\n(% of road segments)")

    # CI width distribution
    ci_widths = (df["ci_upper"] - df["ci_lower"]).dropna()
    if len(ci_widths) > 0:
        axes[1].hist(ci_widths, bins=max(5, len(ci_widths)//2),
                     color=COLORS["neutral"], alpha=0.8, edgecolor="white")
        axes[1].axvline(ci_widths.median(), color=COLORS["positive"],
                        lw=2, linestyle="--",
                        label=f"Median CI width = {ci_widths.median():.3f}")
        axes[1].set_xlabel("95% CI width")
        axes[1].set_ylabel("Number of segments")
        axes[1].set_title("Confidence Interval Width Distribution")
        axes[1].legend()

    out = PLOTS_DIR / "refutation_summary.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def evaluate():
    print("\n" + "="*60)
    print("  STAGE 3 — Evaluation & Visualisation")
    print("="*60)

    # load ATE results
    print("\n>>> Loading ATE results...")
    seg_df, glob_df = load_results()
    global_ate = float(glob_df["ate"].iloc[0])

    # load raw data for contextual plots
    raw_df = pd.read_parquet(
        Path(r"C:\Dhawal\Datasets\processed\stage1_merged.parquet")
    )

    print("\n>>> Generating plots...")

    plot_causal_dag()
    plot_ate_distribution(seg_df, global_ate)
    plot_ate_per_segment(seg_df)
    plot_ate_vs_congestion(seg_df, raw_df)
    plot_spatial_ate(seg_df)
    plot_refutation_summary(seg_df, global_ate)

    # print final summary
    valid = seg_df.dropna(subset=["ate"])
    print("\n" + "="*60)
    print("  STAGE 3 EVALUATION COMPLETE")
    print("="*60)
    print(f"  Global ATE        : {global_ate:+.4f}")
    print(f"  Mean segment ATE  : {valid['ate'].mean():+.4f}")
    print(f"  Segments evaluated: {len(valid)}")
    print(f"  Refutation passed : {(valid['refute_pass']==True).sum()} / {len(valid)}")
    print(f"  All plots saved to: {PLOTS_DIR}")
    print("="*60)

    return seg_df, global_ate


if __name__ == "__main__":
    seg_df, global_ate = evaluate()
