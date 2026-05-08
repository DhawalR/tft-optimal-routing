"""
Stage 2a: Dataset Preparation (pure PyTorch)
=============================================
Multi-Factor Optimal Routing using Temporal Fusion Transformer
NIT Karnataka, Surathkal

Prepares windowed sequence datasets for the TFT model.
No external forecasting library required.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DATA_DIR      = Path(r"C:\Dhawal\Datasets")
PROCESSED_DIR = DATA_DIR / "processed"
PARQUET_PATH  = PROCESSED_DIR / "stage1_merged.parquet"

ENCODER_LENGTH = 12
DECODER_LENGTH = 6
BATCH_SIZE     = 64
NUM_WORKERS    = 0

TARGET_COLS = ["Average Speed", "Congestion Level"]

HIST_EXOG_COLS = [
    "Traffic Volume", "Travel Time Index", "Road Capacity Utilization",
    "Incident Reports", "Environmental Impact", "Public Transport Usage",
    "Traffic Signal Compliance", "Parking Usage", "Pedestrian and Cyclist Count",
    "pm2_5_mean", "flood_risk_score",
    "weather_clear", "weather_fog", "weather_overcast",
    "weather_rain", "weather_windy", "roadwork_yes",
]

FUTR_EXOG_COLS = [
    "hour_of_day", "minute_of_day", "day_of_week", "is_weekend", "month",
    "is_festival_day", "fest_major_holiday", "fest_religious_large",
    "fest_religious_small", "fest_harvest_cultural",
]

STAT_EXOG_COLS = ["Latitude", "Longitude"]


# ─────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────

class TrafficSequenceDataset(Dataset):
    """
    Sliding window dataset for TFT training.
    Each sample contains:
      enc_input : [encoder_length, 1 + n_hist]  past target + hist exog
      enc_futr  : [encoder_length, n_futr]       known future in encoder window
      dec_futr  : [decoder_length, n_futr]       known future in decoder window
      static    : [n_stat]                       road-level static features
      target    : [decoder_length]               ground truth future target
    """

    def __init__(
        self,
        df             : pd.DataFrame,
        target_col     : str,
        hist_cols      : list,
        futr_cols      : list,
        stat_cols      : list,
        encoder_length : int = ENCODER_LENGTH,
        decoder_length : int = DECODER_LENGTH,
        scaler         : StandardScaler = None,
        fit_scaler     : bool = False,
    ):
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.samples        = []

        scale_cols = [c for c in [target_col] + hist_cols if c in df.columns]

        if fit_scaler:
            self.scaler = StandardScaler()
            df = df.copy()
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols].values)
        elif scaler is not None:
            self.scaler = scaler
            df = df.copy()
            df[scale_cols] = self.scaler.transform(df[scale_cols].values)
        else:
            self.scaler = None

        win = encoder_length + decoder_length
        h_avail = [c for c in hist_cols if c in df.columns]
        f_avail = [c for c in futr_cols if c in df.columns]
        s_avail = [c for c in stat_cols if c in df.columns]
        n_hist  = len(h_avail)
        n_futr  = len(f_avail)

        for uid, grp in df.groupby("unique_id"):
            grp = grp.sort_values("ds").reset_index(drop=True)
            if len(grp) < win:
                continue

            stat_vec = grp[s_avail].iloc[0].values.astype(np.float32) if s_avail else np.zeros(1, dtype=np.float32)

            tgt_arr  = grp[target_col].values.astype(np.float32)
            hist_arr = grp[h_avail].values.astype(np.float32) if h_avail else np.zeros((len(grp), 1), dtype=np.float32)
            futr_arr = grp[f_avail].values.astype(np.float32) if f_avail else np.zeros((len(grp), 1), dtype=np.float32)

            for i in range(len(grp) - win + 1):
                enc_tgt  = tgt_arr[i : i + encoder_length].reshape(-1, 1)
                enc_hist = hist_arr[i : i + encoder_length]
                enc_inp  = np.concatenate([enc_tgt, enc_hist], axis=1)

                enc_futr = futr_arr[i : i + encoder_length]
                dec_futr = futr_arr[i + encoder_length : i + win]
                target   = tgt_arr[i + encoder_length : i + win]

                self.samples.append((enc_inp, enc_futr, dec_futr, stat_vec, target))

        print(f"  [DATASET] '{target_col}': {len(self.samples):,} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc_inp, enc_futr, dec_futr, static, target = self.samples[idx]
        return {
            "enc_input" : torch.from_numpy(enc_inp),
            "enc_futr"  : torch.from_numpy(enc_futr),
            "dec_futr"  : torch.from_numpy(dec_futr),
            "static"    : torch.from_numpy(static),
            "target"    : torch.from_numpy(target),
        }


# ─────────────────────────────────────────────
# BUILD FUNCTION
# ─────────────────────────────────────────────

def build_datasets(gnn_emb_cols: list = None) -> tuple:
    """
    Returns datasets, loaders, meta, scalers for both targets.
    """
    print("\n" + "="*60)
    print("  STAGE 2a - Dataset Preparation (pure PyTorch)")
    print("="*60)

    print(f"\n>>> Loading {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  [OK] Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    if "date" in df.columns:
        df = df.drop(columns=["date"])

    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype("float32")

    df["unique_id"] = (
        df["Area Name"].str.strip() + "__" +
        df["Road/Intersection Name"].str.strip()
    )
    df["ds"] = pd.to_datetime(df["timestamp"])
    print(f"  [ID]   {df['unique_id'].nunique()} unique road segments")

    stat_cols = STAT_EXOG_COLS.copy()
    if gnn_emb_cols:
        stat_cols += [c for c in gnn_emb_cols if c in df.columns]

    hist_cols = [c for c in HIST_EXOG_COLS if c in df.columns]
    futr_cols = [c for c in FUTR_EXOG_COLS if c in df.columns]
    stat_cols = [c for c in stat_cols       if c in df.columns]
    print(f"  [COLS] hist={len(hist_cols)}  futr={len(futr_cols)}  stat={len(stat_cols)}")

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    all_ts   = sorted(df["ds"].unique())
    cut_ts   = all_ts[int(len(all_ts) * 0.85)]
    train_df = df[df["ds"] <  cut_ts].copy()
    val_df   = df[df["ds"] >= cut_ts].copy()
    print(f"  [SPLIT] Train: {len(train_df):,}  Val: {len(val_df):,}")

    id_map = df[["unique_id", "Area Name", "Road/Intersection Name",
                 "Latitude", "Longitude"]].drop_duplicates()
    id_map.to_parquet(PROCESSED_DIR / "nf_meta.parquet", index=False)

    datasets, loaders, scalers = {}, {}, {}

    for target in TARGET_COLS:
        key = target.lower().replace(" ", "_")
        print(f"\n>>> Building dataset for: {target}")

        train_ds = TrafficSequenceDataset(
            train_df, target, hist_cols, futr_cols, stat_cols, fit_scaler=True
        )
        val_ds = TrafficSequenceDataset(
            val_df, target, hist_cols, futr_cols, stat_cols,
            scaler=train_ds.scaler, fit_scaler=False
        )

        scaler_path = PROCESSED_DIR / f"scaler_{key}.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(train_ds.scaler, f)

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
        val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        print(f"  [OK] Train batches: {len(train_dl):,}  Val batches: {len(val_dl):,}")

        datasets[key] = {"train": train_ds, "val": val_ds}
        loaders[key]  = {"train": train_dl, "val": val_dl}
        scalers[key]  = train_ds.scaler

    sample    = datasets["average_speed"]["train"][0]
    enc_in_dim = sample["enc_input"].shape[1]
    futr_dim   = sample["dec_futr"].shape[1]
    stat_dim   = sample["static"].shape[0]

    meta = {
        "hist_cols"     : hist_cols,
        "futr_cols"     : futr_cols,
        "stat_cols"     : stat_cols,
        "encoder_length": ENCODER_LENGTH,
        "decoder_length": DECODER_LENGTH,
        "enc_in_dim"    : enc_in_dim,
        "futr_dim"      : futr_dim,
        "stat_dim"      : stat_dim,
        "n_segments"    : df["unique_id"].nunique(),
        "train_df"      : train_df,
        "val_df"        : val_df,
    }

    print(f"\n  [DIMS] enc_in={enc_in_dim}  futr={futr_dim}  stat={stat_dim}")
    print("\n  STAGE 2a COMPLETE")
    return datasets, loaders, meta, scalers


if __name__ == "__main__":
    datasets, loaders, meta, scalers = build_datasets()
    batch = next(iter(loaders["average_speed"]["train"]))
    print(f"\n>>> Batch shapes:")
    for k, v in batch.items():
        print(f"  {k:12s}: {tuple(v.shape)}")
