"""
Stage 2: Evaluation & Visualisation
=====================================
Multi-Factor Optimal Routing using Temporal Fusion Transformer
NIT Karnataka, Surathkal

Loads trained TFT checkpoints, runs inference on the validation set,
computes all metrics, and produces publication-ready plots.

Metrics computed:
  - MAE    : Mean Absolute Error (median forecast)
  - RMSE   : Root Mean Squared Error (median forecast)
  - R2     : Coefficient of determination
  - CRPS   : Continuous Ranked Probability Score (full distribution)
  - QCov   : Quantile Coverage (calibration check)

Plots produced (saved to C:\\Dhawal\\plots\\):
  1. val_loss_curve.png         training convergence
  2. forecast_speed.png         predicted vs actual speed (time series)
  3. forecast_congestion.png    predicted vs actual congestion
  4. quantile_bands_speed.png   uncertainty bands on speed
  5. quantile_bands_congestion  uncertainty bands on congestion
  6. feature_importance.png     input gradient-based importance
  7. residual_dist.png          error distribution histograms
  8. metrics_summary.png        bar chart of all metrics

Usage:
    python stage2_evaluate.py
"""

import warnings
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from stage2_dataset import (
    build_datasets, TARGET_COLS,
    ENCODER_LENGTH, DECODER_LENGTH,
)
from stage2_train import TFTModel, TFTLightning, QUANTILES, HIDDEN_DIM, N_HEADS, DROPOUT

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CKPT_DIR   = Path(r"C:\Dhawal\checkpoints")
LOG_DIR    = Path(r"C:\Dhawal\logs")
PLOTS_DIR  = Path(r"C:\Dhawal\plots")
PLOTS_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = Path(r"C:\Dhawal\Datasets\processed")

# number of road segments to plot in time-series panels
N_SEGMENTS_TO_PLOT = 3

# style
COLORS = {
    "actual"     : "#2C3E50",
    "median"     : "#E74C3C",
    "q10"        : "#AED6F1",
    "q90"        : "#AED6F1",
    "band"       : "#D6EAF8",
    "mae"        : "#3498DB",
    "rmse"       : "#E67E22",
    "r2"         : "#2ECC71",
    "crps"       : "#9B59B6",
    "qcov"       : "#1ABC9C",
    "grid"       : "#ECF0F1",
}

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.color"      : COLORS["grid"],
    "grid.linewidth"  : 0.8,
    "figure.dpi"      : 150,
})


# ─────────────────────────────────────────────
# CRPS IMPLEMENTATION
# ─────────────────────────────────────────────

def crps_quantile(
    quantile_preds : np.ndarray,
    actuals        : np.ndarray,
    quantiles      : list = QUANTILES,
) -> float:
    """
    Compute CRPS from quantile forecasts using the pinball loss identity:

        CRPS = 2 * mean_q [ L_q(y, q_pred) ]

    where L_q is the pinball (quantile) loss at level q.

    This is the standard approach when you have quantile forecasts
    rather than a full parametric distribution.

    Args:
        quantile_preds : [N, n_quantiles]  predicted quantiles
        actuals        : [N]               observed values
        quantiles      : list of quantile levels matching last dim of preds

    Returns:
        scalar CRPS (lower is better)
    """
    n_q    = len(quantiles)
    losses = []
    for i, q in enumerate(quantiles):
        preds_q = quantile_preds[:, i]
        err     = actuals - preds_q
        loss_q  = np.where(err >= 0, q * err, (q - 1) * err)
        losses.append(loss_q)
    # CRPS = 2 * mean over quantiles and samples
    crps = 2.0 * np.mean(losses)
    return float(crps)


def quantile_coverage(
    q_low    : np.ndarray,
    q_high   : np.ndarray,
    actuals  : np.ndarray,
) -> float:
    """
    Fraction of actuals that fall within [q_low, q_high].
    For 10th–90th interval, ideal coverage = 80%.
    """
    covered = ((actuals >= q_low) & (actuals <= q_high)).mean()
    return float(covered)


# ─────────────────────────────────────────────
# LOAD CHECKPOINT
# ─────────────────────────────────────────────

def load_model(target_key: str, meta: dict) -> TFTLightning:
    """Load best checkpoint for a given target."""
    # find best checkpoint (lowest val_loss in filename)
    pattern = f"tft_{target_key}-epoch=*.ckpt"
    ckpts   = sorted(CKPT_DIR.glob(pattern))

    if not ckpts:
        # fall back to last checkpoint
        last = CKPT_DIR / f"tft_{target_key}_last.ckpt"
        if not last.exists():
            raise FileNotFoundError(
                f"No checkpoint found for '{target_key}' in {CKPT_DIR}\n"
                f"Run: python stage2_train.py --epochs 10"
            )
        ckpt_path = last
    else:
        # pick checkpoint with lowest val_loss from filename
        def _val_loss(p):
            try:
                return float(str(p).split("val_loss=")[-1].replace(".ckpt",""))
            except:
                return 999.0
        ckpt_path = min(ckpts, key=_val_loss)

    print(f"  [CKPT] Loading: {ckpt_path.name}")

    model = TFTModel(
        enc_in_dim  = meta["enc_in_dim"],
        futr_dim    = meta["futr_dim"],
        stat_dim    = meta["stat_dim"],
        hidden_dim  = HIDDEN_DIM,
        n_heads     = N_HEADS,
        dropout     = DROPOUT,
        encoder_len = meta["encoder_length"],
        decoder_len = meta["decoder_length"],
    )
    lit = TFTLightning.load_from_checkpoint(
        str(ckpt_path), model=model, lr=1e-3
    )
    lit.eval()
    return lit


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    lit      : TFTLightning,
    loader,
    device   : str = "cuda" if torch.cuda.is_available() else "cpu",
    max_batches: int = 200,   # limit for speed; set None for full val set
) -> dict:
    """
    Run model on val loader, collect predictions and actuals.

    Returns dict with:
      preds   : [N, decoder_length, n_quantiles]
      actuals : [N, decoder_length]
    """
    lit   = lit.to(device)
    preds_list, actuals_list = [], []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        out = lit({
            "enc_input": batch["enc_input"].to(device),
            "enc_futr" : batch["enc_futr"].to(device),
            "dec_futr" : batch["dec_futr"].to(device),
            "static"   : batch["static"].to(device),
            "target"   : batch["target"].to(device),
        })
        preds_list.append(out.cpu().numpy())
        actuals_list.append(batch["target"].numpy())

    preds   = np.concatenate(preds_list,   axis=0)   # [N, H, Q]
    actuals = np.concatenate(actuals_list, axis=0)   # [N, H]
    return {"preds": preds, "actuals": actuals}


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(results: dict, target_name: str) -> dict:
    """Compute all metrics from inference results."""
    preds   = results["preds"]     # [N, H, Q]
    actuals = results["actuals"]   # [N, H]

    # flatten horizon dimension: treat each (sample, step) as one prediction
    N, H, Q = preds.shape
    preds_flat   = preds.reshape(N * H, Q)
    actuals_flat = actuals.reshape(N * H)

    q10 = preds_flat[:, 0]   # 10th percentile
    q50 = preds_flat[:, 1]   # median
    q90 = preds_flat[:, 2]   # 90th percentile

    mae  = mean_absolute_error(actuals_flat, q50)
    rmse = np.sqrt(mean_squared_error(actuals_flat, q50))
    r2   = r2_score(actuals_flat, q50)
    crps = crps_quantile(preds_flat, actuals_flat, QUANTILES)
    qcov = quantile_coverage(q10, q90, actuals_flat)

    metrics = {
        "MAE"   : mae,
        "RMSE"  : rmse,
        "R2"    : r2,
        "CRPS"  : crps,
        "QCov"  : qcov,
    }

    print(f"\n  ── {target_name} ──")
    print(f"  MAE               : {mae:.4f}")
    print(f"  RMSE              : {rmse:.4f}")
    print(f"  R²                : {r2:.4f}")
    print(f"  CRPS              : {crps:.4f}  (lower = better distribution)")
    print(f"  Quantile Coverage : {qcov*100:.1f}%  (ideal = 80% for 10th–90th interval)")

    return metrics


# ─────────────────────────────────────────────
# PLOT 1: VALIDATION LOSS CURVE
# ─────────────────────────────────────────────

def plot_loss_curves():
    """Read CSVLogger output and plot train/val loss per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training convergence", fontsize=14, fontweight="bold", y=1.01)

    for ax, target_key, title in zip(
        axes,
        ["average_speed", "congestion_level"],
        ["Average Speed", "Congestion Level"]
    ):
        log_dirs = sorted((LOG_DIR / f"tft_{target_key}").glob("version_*/metrics.csv"))
        if not log_dirs:
            ax.text(0.5, 0.5, "No log found\nRun training first",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            continue

        df = pd.read_csv(log_dirs[-1])   # latest version

        if "train_loss_epoch" in df.columns:
            train = df.dropna(subset=["train_loss_epoch"])
            ax.plot(train["epoch"], train["train_loss_epoch"],
                    color=COLORS["mae"], lw=2, label="Train loss")

        if "val_loss" in df.columns:
            val = df.dropna(subset=["val_loss"])
            ax.plot(val["epoch"], val["val_loss"],
                    color=COLORS["rmse"], lw=2, linestyle="--", label="Val loss")

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Quantile Loss")
        ax.legend()

    plt.tight_layout()
    out = PLOTS_DIR / "val_loss_curve.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 2 & 3: FORECAST vs ACTUAL (TIME SERIES)
# ─────────────────────────────────────────────

def plot_forecast_timeseries(
    results     : dict,
    target_name : str,
    filename    : str,
    n_steps     : int = 288,   # ~1 day of 5-min intervals
):
    """Plot median forecast vs actual for a contiguous sequence."""
    preds   = results["preds"][:n_steps, 0, 1]    # first step, median
    actuals = results["actuals"][:n_steps, 0]
    time_ax = np.arange(len(preds)) * 5            # minutes

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time_ax, actuals, color=COLORS["actual"], lw=1.5,
            label="Actual", alpha=0.9)
    ax.plot(time_ax, preds,   color=COLORS["median"], lw=1.5,
            linestyle="--", label="Predicted (median)", alpha=0.9)

    ax.set_title(f"{target_name} — Forecast vs Actual",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (minutes from start of validation period)")
    ax.set_ylabel(target_name)
    ax.legend(framealpha=0.9)

    out = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 4 & 5: QUANTILE BANDS (UNCERTAINTY)
# ─────────────────────────────────────────────

def plot_quantile_bands(
    results     : dict,
    target_name : str,
    filename    : str,
    n_steps     : int = 144,   # ~12 hours
):
    """
    Plot the 10th–90th percentile band around the median forecast.
    This is the key uncertainty visualisation that maps to the Ue
    routing term in Stage 4.
    """
    preds   = results["preds"][:n_steps, 0, :]    # [n_steps, 3]
    actuals = results["actuals"][:n_steps, 0]
    time_ax = np.arange(n_steps) * 5

    q10 = preds[:, 0]
    q50 = preds[:, 1]
    q90 = preds[:, 2]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(time_ax, q10, q90,
                    color=COLORS["band"], alpha=0.7,
                    label="10th–90th percentile (uncertainty band)")
    ax.plot(time_ax, q10, color=COLORS["q10"], lw=0.8, linestyle=":")
    ax.plot(time_ax, q90, color=COLORS["q90"], lw=0.8, linestyle=":")
    ax.plot(time_ax, q50, color=COLORS["median"], lw=2,
            linestyle="--", label="Median forecast (50th pct)")
    ax.plot(time_ax, actuals, color=COLORS["actual"], lw=1.5,
            label="Actual", alpha=0.9)

    ax.set_title(f"{target_name} — Uncertainty Bands",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (minutes from start of validation period)")
    ax.set_ylabel(target_name)
    ax.legend(framealpha=0.9)

    # annotate coverage
    covered = ((actuals >= q10) & (actuals <= q90)).mean() * 100
    ax.text(0.98, 0.04,
            f"Coverage: {covered:.1f}% (ideal 80%)",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    out = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 6: FEATURE IMPORTANCE (INPUT GRADIENTS)
# ─────────────────────────────────────────────

def plot_feature_importance(
    lit         : TFTLightning,
    loader,
    meta        : dict,
    target_name : str,
    filename    : str,
    device      : str = "cuda" if torch.cuda.is_available() else "cpu",
    n_batches   : int = 10,
):
    """
    Approximate feature importance via input gradient magnitudes.
    For each input feature, compute mean |grad| w.r.t. loss.
    This shows which features the TFT relies on most.
    """
    lit   = lit.to(device)
    # lit.eval()
    lit.train()  # must be train mode for LSTM backward pass
    grads = None
    count = 0

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        enc  = batch["enc_input"].float().to(device).requires_grad_(True)
        futr = batch["enc_futr"].float().to(device)
        dfut = batch["dec_futr"].float().to(device)
        stat = batch["static"].float().to(device)
        tgt  = batch["target"].float().to(device)

        out  = lit.model(enc, futr, dfut, stat)
        loss = lit.loss_fn(out, tgt)
        loss.backward()

        g = enc.grad.abs().mean(dim=(0, 1)).cpu().detach().numpy()
        grads = g if grads is None else grads + g
        count += 1

    grads = grads / count

    # column names for encoder input: [target] + hist_cols
    target_key = target_name.lower().replace(" ", "_")
    enc_cols   = [target_name] + meta["hist_cols"]
    enc_cols   = enc_cols[:len(grads)]   # match actual dim

    # sort by importance
    idx    = np.argsort(grads)[::-1]
    top_n  = min(15, len(idx))
    labels = [enc_cols[i] if i < len(enc_cols) else f"feat_{i}" for i in idx[:top_n]]
    values = grads[idx[:top_n]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), values[::-1],
                   color=COLORS["crps"], alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=10)
    ax.set_xlabel("Mean |gradient| — proxy for feature importance")
    ax.set_title(f"{target_name} — Top {top_n} Input Features",
                 fontsize=13, fontweight="bold")

    out = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 7: RESIDUAL DISTRIBUTION
# ─────────────────────────────────────────────

def plot_residuals(
    results     : dict,
    target_name : str,
    filename    : str,
):
    """Histogram of residuals (actual - median forecast)."""
    preds   = results["preds"][:, :, 1].flatten()
    actuals = results["actuals"].flatten()
    resid   = actuals - preds

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{target_name} — Residual Analysis",
                 fontsize=13, fontweight="bold")

    # histogram
    axes[0].hist(resid, bins=60, color=COLORS["mae"], alpha=0.8, edgecolor="white")
    axes[0].axvline(0, color="black", lw=1.5, linestyle="--")
    axes[0].set_xlabel("Residual (actual − predicted)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Residual distribution")
    mu, std = resid.mean(), resid.std()
    axes[0].text(0.97, 0.95,
                 f"μ = {mu:.3f}\nσ = {std:.3f}",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # scatter actual vs predicted
    sample_n = min(2000, len(actuals))
    idx      = np.random.choice(len(actuals), sample_n, replace=False)
    axes[1].scatter(actuals[idx], preds[idx],
                    alpha=0.3, s=8, color=COLORS["crps"])
    mn = min(actuals[idx].min(), preds[idx].min())
    mx = max(actuals[idx].max(), preds[idx].max())
    axes[1].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect forecast")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted (median)")
    axes[1].set_title("Actual vs Predicted")
    axes[1].legend()

    out = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# PLOT 8: METRICS SUMMARY BAR CHART
# ─────────────────────────────────────────────

def plot_metrics_summary(all_metrics: dict):
    """
    Side-by-side bar chart comparing metrics across both targets.
    Uses a dual-axis layout since MAE/RMSE and R2/QCov are on different scales.
    """
    targets = list(all_metrics.keys())
    metrics = ["MAE", "RMSE", "CRPS"]   # error metrics (lower = better)
    good    = ["R2", "QCov"]             # quality metrics (higher = better)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Performance Summary", fontsize=14, fontweight="bold")

    x     = np.arange(len(targets))
    width = 0.25
    color_map = [COLORS["mae"], COLORS["rmse"], COLORS["crps"]]

    # error metrics
    for i, (m, c) in enumerate(zip(metrics, color_map)):
        vals = [all_metrics[t][m] for t in targets]
        axes[0].bar(x + i * width, vals, width, label=m, color=c, alpha=0.85)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([t.replace("_", " ").title() for t in targets])
    axes[0].set_ylabel("Error (lower = better)")
    axes[0].set_title("Error Metrics")
    axes[0].legend()

    # quality metrics
    color_map2 = [COLORS["r2"], COLORS["qcov"]]
    for i, (m, c) in enumerate(zip(good, color_map2)):
        vals = [all_metrics[t][m] for t in targets]
        axes[1].bar(x + i * width * 1.5, vals, width * 1.5, label=m,
                    color=c, alpha=0.85)
    axes[1].axhline(0.8, color="gray", linestyle="--", lw=1,
                    label="80% ideal coverage")
    axes[1].set_xticks(x + width * 1.5 / 2)
    axes[1].set_xticklabels([t.replace("_", " ").title() for t in targets])
    axes[1].set_ylabel("Score (higher = better)")
    axes[1].set_title("Quality Metrics")
    axes[1].legend()

    out = PLOTS_DIR / "metrics_summary.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {out}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def evaluate():
    print("\n" + "="*60)
    print("  STAGE 2 — Evaluation & Visualisation")
    print("="*60)

    # ── load datasets (needed for val loaders and meta) ──
    print("\n>>> Building datasets...")
    datasets, loaders, meta, scalers = build_datasets()

    all_metrics = {}

    target_map = {
        "average_speed"   : ("Average Speed",    "speed"),
        "congestion_level": ("Congestion Level",  "congestion"),
    }

    for target_key, (target_name, short) in target_map.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {target_name}")
        print("="*60)

        # load model
        try:
            lit = load_model(target_key, meta)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        val_loader = loaders[target_key]["val"]

        # inference
        print("\n>>> Running inference on validation set...")
        results = run_inference(lit, val_loader)
        print(f"  [OK] Predictions: {results['preds'].shape}")

        # metrics
        print("\n>>> Computing metrics...")
        metrics = compute_metrics(results, target_name)
        all_metrics[target_key] = metrics

        # plots
        print("\n>>> Generating plots...")
        plot_forecast_timeseries(
            results, target_name,
            filename=f"forecast_{short}.png"
        )
        plot_quantile_bands(
            results, target_name,
            filename=f"quantile_bands_{short}.png"
        )
        plot_feature_importance(
            lit, val_loader, meta, target_name,
            filename=f"feature_importance_{short}.png"
        )
        plot_residuals(
            results, target_name,
            filename=f"residual_dist_{short}.png"
        )

    # loss curves (from logs)
    print("\n>>> Plotting training loss curves...")
    plot_loss_curves()

    # summary bar chart
    if all_metrics:
        print("\n>>> Plotting metrics summary...")
        plot_metrics_summary(all_metrics)

    # print final summary table
    print("\n" + "="*60)
    print("  FINAL METRICS SUMMARY")
    print("="*60)
    header = f"  {'Metric':<10}  {'Average Speed':>16}  {'Congestion Level':>16}"
    print(header)
    print("  " + "-"*50)
    for m in ["MAE", "RMSE", "R2", "CRPS", "QCov"]:
        row = f"  {m:<10}"
        for tk in ["average_speed", "congestion_level"]:
            val = all_metrics.get(tk, {}).get(m, float("nan"))
            row += f"  {val:>16.4f}"
        print(row)

    print(f"\n  All plots saved to: {PLOTS_DIR}")
    print("="*60)

    return all_metrics


if __name__ == "__main__":
    all_metrics = evaluate()