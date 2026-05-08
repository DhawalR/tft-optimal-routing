"""
Stage 1: Heterogeneous Data Ingestion and Preprocessing
=======================================================
Multi-Factor Optimal Routing using Temporal Fusion Transformer
NIT Karnataka, Surathkal

Datasets consumed:
  - Bangalore_Traffic_Dataset.csv       (backbone, 8936 rows x 18 cols)
  - Bangalore_Urban_Flood_Dataset.csv   (flood risk per road segment)
  - RideSafety_Dataset.csv              (Mumbai/Delhi → domain-adapted to Bangalore)
  - Air_Pollution_India.csv             (PM2.5 per city per date)
  - Festival_Calendar_India.csv         (known future event flags)

Outputs  (written to  DATA_DIR/processed/):
  - master_traffic.parquet              (aligned 5-min backbone)
  - flood_risk_lookup.parquet           (lat/lon → flood score)
  - air_quality_lookup.parquet          (date → PM2.5 for Bangalore)
  - festival_flags.parquet              (date → one-hot festival flags)
  - ridesafety_adapted.parquet          (domain-adapted incident/congestion rows)
  - stage1_merged.parquet               (single merged frame ready for Stage 2)

Usage:
  python stage1_preprocess.py

Author : Dhawal Ramdham / Madhusudhan R
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  CONFIGURATION  — edit these paths only
# ─────────────────────────────────────────────

DATA_DIR = Path(r"C:\Dhawal\Datasets")          # root folder with all raw CSVs
PROCESSED_DIR = DATA_DIR / "processed"          # outputs land here
PROCESSED_DIR.mkdir(exist_ok=True)

# ── raw file names (adjust if yours differ slightly) ──
TRAFFIC_FILE    = DATA_DIR / "Bangalore_Traffic_Dataset.csv"
FLOOD_FILE      = DATA_DIR / "Bangalore_Urban_Flood_Dataset.csv"
RIDESAFETY_FILE = DATA_DIR / "RideSafety_Dataset.csv"
AIR_FILE        = DATA_DIR / "Air_Pollution_India.csv"
FESTIVAL_FILE   = DATA_DIR / "Festival_Calendar_India.csv"

# ── temporal resolution ──
FREQ = "5min"          # resample everything to 5-minute intervals

# ── domain adaptation: columns shared between RideSafety and Traffic ──
#    left = RideSafety column name,  right = Traffic column name
SHARED_FEATURE_MAP = {
    "Traffic_Congestion_Level" : "Congestion Level",
    "Traffic_Avg_Speed_KMH"    : "Average Speed",
    "Road_Accidents_Reported"  : "Incident Reports",
    "Weather_Temperature_C"    : None,              # no direct match, kept as-is
    "Weather_Rainfall_mm"      : None,
    "Weather_Humidity_%"       : None,
    "Weather_Wind_Speed_KMH"   : None,
}

# ── air quality: city name as it appears in air pollution CSV ──
BANGALORE_CITY_NAMES = ["Bengaluru", "Bangalore", "bengaluru", "bangalore"]


# ─────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────

def _assert_shape(df: pd.DataFrame, name: str, min_rows: int = 10) -> None:
    """Hard-stop assertion so silent shape errors are caught early."""
    assert len(df) >= min_rows, (
        f"[ASSERT] {name} has only {len(df)} rows — expected at least {min_rows}. "
        f"Check the file path or column names."
    )
    print(f"  [OK] {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")


def _safe_read(path: Path, name: str) -> pd.DataFrame:
    """Read CSV with informative error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Cannot find '{name}' at:\n  {path}\n"
            f"Please check the filename and that it is inside {DATA_DIR}"
        )
    df = pd.read_csv(path)
    print(f"\n>>> Loaded {name}")
    _assert_shape(df, name)
    return df


def _percentile_rank(series: pd.Series) -> np.ndarray:
    """Convert a series to its percentile rank in [0, 1]."""
    return rankdata(series.fillna(series.median()), method="average") / len(series)


def _percentile_map(source: pd.Series, target: pd.Series) -> pd.Series:
    """
    Domain adaptation via percentile matching.
    Each value in `source` is replaced by the value at the same
    percentile position in `target`.
    """
    src_ranks  = _percentile_rank(source)                    # rank in source
    tgt_sorted = np.sort(target.dropna().values)             # sorted target values
    indices    = (src_ranks * (len(tgt_sorted) - 1)).astype(int)
    indices    = np.clip(indices, 0, len(tgt_sorted) - 1)
    return pd.Series(tgt_sorted[indices], index=source.index)


# ─────────────────────────────────────────────
# STEP 1 — BANGALORE TRAFFIC (backbone)
# ─────────────────────────────────────────────

def process_traffic(path: Path) -> pd.DataFrame:
    """
    Load and clean the Bangalore traffic backbone.

    Key operations:
      - Parse 'Date' column → datetime
      - Synthesize 5-min sub-timestamps so the backbone has a proper
        DatetimeIndex (the raw data has daily granularity; we expand to
        5-min intervals and forward-fill numeric columns so Stage 2 has
        a continuous time axis as required by TFT)
      - One-hot encode: Weather Conditions, Roadwork and Construction Activity
      - Normalise lat/lon: drop rows where both are NaN (4446 rows have NaN),
        fill remaining NaN with area centroid
    """
    print("\n" + "="*60)
    print("STEP 1 — Bangalore Traffic Dataset")
    print("="*60)

    df = _safe_read(path, "Bangalore Traffic")

    # ── 1a. parse date ──
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    n_bad_dates = df["Date"].isna().sum()
    if n_bad_dates:
        print(f"  [WARN] {n_bad_dates} unparseable dates — dropped")
    df = df.dropna(subset=["Date"])

    # ── 1b. handle missing lat/lon (4490 non-null out of 8936) ──
    #   fill NaN lat/lon with the mean of what we do have
    lat_mean = df["Latitude"].mean()
    lon_mean = df["Longitude"].mean()
    df["Latitude"]  = df["Latitude"].fillna(lat_mean)
    df["Longitude"] = df["Longitude"].fillna(lon_mean)
    print(f"  [INFO] Lat/Lon NaN filled with centroid ({lat_mean:.4f}, {lon_mean:.4f})")

    # ── 1c. one-hot encode categorical columns ──
    weather_dummies = pd.get_dummies(
        df["Weather Conditions"].str.lower().str.strip(),
        prefix="weather"
    )
    roadwork_dummies = pd.get_dummies(
        df["Roadwork and Construction Activity"].str.lower().str.strip(),
        prefix="roadwork"
    )
    df = pd.concat([df, weather_dummies, roadwork_dummies], axis=1)
    df = df.drop(columns=["Weather Conditions", "Roadwork and Construction Activity"])
    print(f"  [INFO] One-hot cols added: {list(weather_dummies.columns) + list(roadwork_dummies.columns)}")

    # ── 1d. expand to 5-minute intervals ──
    #   For each unique (Area Name, Road/Intersection Name, Date) group,
    #   generate 288 timestamps (24h × 12 per hour) and forward-fill.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols    = [c for c in df.columns if df[c].dtype == bool]
    expand_cols  = numeric_cols + bool_cols

    print("  [INFO] Expanding to 5-min intervals (this may take ~30 seconds)...")
    frames = []
    for (area, road), grp in df.groupby(["Area Name", "Road/Intersection Name"]):
        # one row per date in this group
        for _, row in grp.iterrows():
            day_start = row["Date"].normalize()
            day_end   = day_start + pd.Timedelta(hours=23, minutes=55)
            ts_range  = pd.date_range(day_start, day_end, freq=FREQ)
            expanded  = pd.DataFrame({"timestamp": ts_range})
            expanded["Area Name"]              = area
            expanded["Road/Intersection Name"] = road
            expanded["Latitude"]               = row["Latitude"]
            expanded["Longitude"]              = row["Longitude"]
            for col in expand_cols:
                if col in row.index:
                    expanded[col] = row[col]
            frames.append(expanded)

    traffic_5min = pd.concat(frames, ignore_index=True)
    traffic_5min = traffic_5min.sort_values(["Area Name", "Road/Intersection Name", "timestamp"])
    traffic_5min = traffic_5min.reset_index(drop=True)

    # ── 1e. add time-of-day features (known future inputs for TFT) ──
    traffic_5min["hour_of_day"]    = traffic_5min["timestamp"].dt.hour
    traffic_5min["minute_of_day"]  = traffic_5min["timestamp"].dt.minute
    traffic_5min["day_of_week"]    = traffic_5min["timestamp"].dt.dayofweek   # 0=Mon
    traffic_5min["is_weekend"]     = (traffic_5min["day_of_week"] >= 5).astype(int)
    traffic_5min["month"]          = traffic_5min["timestamp"].dt.month

    _assert_shape(traffic_5min, "Traffic 5-min expanded", min_rows=100_000)
    out = PROCESSED_DIR / "master_traffic.parquet"
    traffic_5min.to_parquet(out, index=False)
    print(f"  [SAVED] {out}")
    return traffic_5min


# ─────────────────────────────────────────────
# STEP 2 — URBAN FLOOD DATASET
# ─────────────────────────────────────────────

def process_flood(path: Path) -> pd.DataFrame:
    """
    Build a flood-risk lookup table keyed on (Latitude, Longitude).

    The 'flood' column (0/1) is converted to a continuous risk score
    by incorporating Rainfall_Intensity and Drainage_System_Condition:
        flood_risk = P(flood=1) weighted by rainfall percentile
    This gives the Stage 4 router a soft edge weight rather than a
    binary flag.
    """
    print("\n" + "="*60)
    print("STEP 2 — Urban Flood Dataset")
    print("="*60)

    df = _safe_read(path, "Urban Flood")

    # ── normalise rainfall to [0,1] ──
    df["rainfall_norm"] = (
        df["Rainfall_Intensity"] - df["Rainfall_Intensity"].min()
    ) / (df["Rainfall_Intensity"].max() - df["Rainfall_Intensity"].min() + 1e-9)

    # ── drainage condition: lower value = worse drainage ──
    #   column is int64; invert so that 0 = worst (highest risk)
    df["drainage_risk"] = 1.0 - (
        df["Drainage_System_Condition"] / (df["Drainage_System_Condition"].max() + 1e-9)
    )

    # ── composite flood risk score ──
    df["flood_risk_score"] = (
        0.5 * df["flood"].astype(float)
        + 0.3 * df["rainfall_norm"]
        + 0.2 * df["drainage_risk"]
    )
    df["flood_risk_score"] = df["flood_risk_score"].clip(0, 1)

    lookup = df[["Latitude", "Longitude", "flood_risk_score", "Rainfall_Intensity",
                 "River_Level", "Drainage_System_Condition"]].copy()

    _assert_shape(lookup, "Flood risk lookup")
    out = PROCESSED_DIR / "flood_risk_lookup.parquet"
    lookup.to_parquet(out, index=False)
    print(f"  [SAVED] {out}")
    return lookup


# ─────────────────────────────────────────────
# STEP 3 — RIDESAFETY DOMAIN ADAPTATION
# ─────────────────────────────────────────────

def process_ridesafety(path: Path, traffic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain adaptation: Mumbai/Delhi → Bangalore.

    Strategy — percentile matching (distribution alignment):
      For every shared feature, each Mumbai/Delhi value is mapped to
      the value at the *same percentile rank* in the Bangalore distribution.
      This preserves the relative structure (peak hours, incident clusters)
      while making the absolute scale Bangalore-compatible.

    The adapted rows are labelled with domain_source = 'mumbai_delhi_adapted'
    so they can be weighted separately during TFT training if desired.
    """
    print("\n" + "="*60)
    print("STEP 3 — RideSafety Domain Adaptation")
    print("="*60)

    df = _safe_read(path, "RideSafety (Mumbai/Delhi)")

    # ── 3a. parse date + time → single timestamp ──
    df["timestamp"] = pd.to_datetime(
        df["Ride_Date"].astype(str) + " " + df["Ride_Time"].astype(str),
        errors="coerce"
    )
    n_bad = df["timestamp"].isna().sum()
    if n_bad:
        print(f"  [WARN] {n_bad} rows with unparseable timestamps dropped")
    df = df.dropna(subset=["timestamp"])

    # ── 3b. round to nearest 5-min ──
    df["timestamp"] = df["timestamp"].dt.round(FREQ)

    # ── 3c. map city names to Bangalore-compatible area labels ──
    df["Area Name"] = df["City"].apply(
        lambda c: f"adapted_{str(c).lower().strip()}"
    )

    # ── 3d. select and rename shared columns ──
    keep = {
        "Traffic_Congestion_Level" : "Congestion Level",
        "Traffic_Avg_Speed_KMH"    : "Average Speed",
        "Road_Accidents_Reported"  : "Incident Reports",
        "Weather_Temperature_C"    : "weather_temperature_c",
        "Weather_Rainfall_mm"      : "weather_rainfall_mm",
        "Weather_Humidity_%"       : "weather_humidity_pct",
        "Weather_Wind_Speed_KMH"   : "weather_wind_speed_kmh",
        "Weather_Condition"        : "Weather Conditions",
        "Latitude"                 : "Latitude",
        "Longitude"                : "Longitude",
    }
    available = {k: v for k, v in keep.items() if k in df.columns}
    adapted   = df[list(available.keys()) + ["timestamp", "Area Name"]].rename(
        columns=available
    )

    # ── 3e. percentile matching for numeric shared features ──
    numeric_shared = ["Congestion Level", "Average Speed", "Incident Reports"]
    for col in numeric_shared:
        if col not in adapted.columns:
            continue
        if col not in traffic_df.columns:
            print(f"  [SKIP] '{col}' not found in traffic backbone — skipping match")
            continue
        print(f"  [ADAPT] Percentile-matching '{col}' (Mumbai/Delhi → Bangalore)")
        adapted[col] = _percentile_map(adapted[col], traffic_df[col])

    # ── 3f. one-hot encode adapted weather conditions ──
    if "Weather Conditions" in adapted.columns:
        w_dummies = pd.get_dummies(
            adapted["Weather Conditions"].str.lower().str.strip(),
            prefix="weather"
        )
        adapted = pd.concat([adapted, w_dummies], axis=1)
        adapted = adapted.drop(columns=["Weather Conditions"])

    adapted["domain_source"] = "mumbai_delhi_adapted"
    adapted["Road/Intersection Name"] = "adapted_road"

    # ── 3g. add time features ──
    adapted["hour_of_day"]   = adapted["timestamp"].dt.hour
    adapted["minute_of_day"] = adapted["timestamp"].dt.minute
    adapted["day_of_week"]   = adapted["timestamp"].dt.dayofweek
    adapted["is_weekend"]    = (adapted["day_of_week"] >= 5).astype(int)
    adapted["month"]         = adapted["timestamp"].dt.month

    _assert_shape(adapted, "RideSafety adapted")
    out = PROCESSED_DIR / "ridesafety_adapted.parquet"
    adapted.to_parquet(out, index=False)
    print(f"  [SAVED] {out}")
    return adapted


# ─────────────────────────────────────────────
# STEP 4 — AIR QUALITY (PM2.5)
# ─────────────────────────────────────────────

def process_air_quality(path: Path) -> pd.DataFrame:
    """
    Extract Bangalore-specific PM2.5 readings.
    Produces a date-keyed lookup (Stage 4 uses Pe = PM2.5 edge weight).
    """
    print("\n" + "="*60)
    print("STEP 4 — Air Quality Dataset")
    print("="*60)

    df = _safe_read(path, "Air Pollution India")

    # ── filter for Bangalore ──
    mask = df["city"].str.lower().isin([c.lower() for c in BANGALORE_CITY_NAMES])
    blr  = df[mask].copy()
    if len(blr) == 0:
        print(f"  [WARN] No rows matched for Bangalore cities {BANGALORE_CITY_NAMES}")
        print(f"         Unique city values: {df['city'].unique()[:10]}")
        print(f"         Using ALL cities as fallback and taking daily mean")
        blr = df.copy()

    blr["date"] = pd.to_datetime(blr["date"], errors="coerce")
    blr = blr.dropna(subset=["date", "pm2_5"])

    # ── daily aggregation ──
    aqi_daily = (
        blr.groupby("date")
           .agg(pm2_5_mean=("pm2_5", "mean"),
                pm2_5_max =("pm2_5", "max"),
                aqi_mean  =("aqi",   "mean"))
           .reset_index()
    )

    # ── normalise PM2.5 to [0,1] for use as routing edge weight ──
    pm_max = aqi_daily["pm2_5_max"].max()
    aqi_daily["pm2_5_norm"] = (aqi_daily["pm2_5_mean"] / pm_max).clip(0, 1)

    _assert_shape(aqi_daily, "Air quality lookup")
    out = PROCESSED_DIR / "air_quality_lookup.parquet"
    aqi_daily.to_parquet(out, index=False)
    print(f"  [SAVED] {out}")
    return aqi_daily


# ─────────────────────────────────────────────
# STEP 5 — FESTIVAL CALENDAR
# ─────────────────────────────────────────────

def process_festivals(path: Path) -> pd.DataFrame:
    """
    Build a date-keyed binary feature table for festivals.
    Each unique festival becomes a one-hot column (known future input for TFT).
    """
    print("\n" + "="*60)
    print("STEP 5 — Festival Calendar")
    print("="*60)

    df = _safe_read(path, "Festival Calendar India")

    #df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    #df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Year"].astype(str),
        format="%B %d %Y",
        errors="coerce"
    )
    df = df.dropna(subset=["Date"])

    # the following commented out snippet created 59 individual one hot 
    # columns viz not ideal since the TFT gets almost no signal from them individually
    # # ── one-hot per festival ──
    # fest_dummies = pd.get_dummies(
    #     df["Festival name"].str.lower().str.strip().str.replace(" ", "_"),
    #     prefix="festival"
    # )
    # festivals = pd.concat([df[["Date"]], fest_dummies], axis=1)

    # # ── if multiple festivals fall on the same date, OR the rows ──
    # festivals = festivals.groupby("Date").max().reset_index()
    # festivals = festivals.rename(columns={"Date": "date"})

    # # ── is_festival_day: any festival active ──
    # festival_cols = [c for c in festivals.columns if c.startswith("festival_")]
    # festivals["is_festival_day"] = festivals[festival_cols].max(axis=1)
#HERE is the meaningfull way to group them into 3-4 meaningful colums
    # ── categorise festivals by traffic impact type ──
    # major_holiday    : public holidays — offices/schools closed, high leisure traffic
    # religious_large  : large gatherings, processions, road closures
    # religious_small  : smaller observances, moderate impact
    # harvest_cultural : regional festivals, moderate-high impact

    major_holidays = [
        "new year's day", "republic day", "independence day",
        "mahatma gandhi jayanti", "christmas", "good friday", "easter day"
    ]
    religious_large = [
        "diwali/deepavali", "holi", "eid-ul-fitar", "ramzan id/eid-ul-fitar",
        "bakr id/eid ul-adha", "dussehra", "ganesh chaturthi/vinayaka chaturthi",
        "janmashtami", "maha shivaratri/shivaratri", "guru nanak jayanti",
        "muharram/ashura", "milad un-nabi/id-e-milad", "buddha purnima/vesak",
        "mahavir jayanti", "rama navami", "rath yatra",
        "first day of durga puja festivities", "first day of sharad navratri",
        "maha saptami", "maha ashtami", "maha navami", "govardhan puja",
        "bhai duj", "holika dahana", "naraka chaturdasi", "chhat puja (pratihar sashthi/surya sashthi)"
    ]
    harvest_cultural = [
        "makar sankranti", "pongal", "lohri", "onam", "ugadi",
        "gudi padwa", "vaisakhi", "chaitra sukhladi", "mesadi / vaisakhadi",
        "dolyatra", "vasant panchami", "parsi new year"
    ]

    def categorise(name):
        n = str(name).lower().strip()
        if any(h in n for h in major_holidays):
            return "major_holiday"
        elif any(h in n for h in religious_large):
            return "religious_large"
        elif any(h in n for h in harvest_cultural):
            return "harvest_cultural"
        else:
            return "religious_small"

    df["festival_category"] = df["Festival name"].apply(categorise)

    cat_dummies = pd.get_dummies(df["festival_category"], prefix="fest")
    festivals = pd.concat([df[["Date"]], cat_dummies], axis=1)
    festivals = festivals.groupby("Date").max().reset_index()
    festivals = festivals.rename(columns={"Date": "date"})
    festival_cols = [c for c in festivals.columns if c.startswith("fest_")]
    festivals["is_festival_day"] = festivals[festival_cols].max(axis=1)
#---------------------------------------------------------------------------

    _assert_shape(festivals, "Festival flags")
    out = PROCESSED_DIR / "festival_flags.parquet"
    festivals.to_parquet(out, index=False)
    print(f"  [INFO] Festival one-hot cols: {festival_cols}")
    print(f"  [SAVED] {out}")
    return festivals


# ─────────────────────────────────────────────
# STEP 6 — FINAL MERGE
# ─────────────────────────────────────────────

def merge_all(
    traffic_df  : pd.DataFrame,
    air_df      : pd.DataFrame,
    festival_df : pd.DataFrame,
    flood_df    : pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all processed frames into one master frame ready for Stage 2.

    Join keys:
      - Air quality  : date (from timestamp)
      - Festivals    : date (from timestamp)
      - Flood risk   : nearest lat/lon match (within 0.05° ≈ 5 km)

    The flood join uses a nearest-neighbour approach on lat/lon because
    the flood dataset has different spatial granularity than the traffic data.
    """
    print("\n" + "="*60)
    print("STEP 6 — Final Merge")
    print("="*60)

    df = traffic_df.copy()
    df["date"] = df["timestamp"].dt.normalize()

    # ── join air quality ──
    df = df.merge(
        air_df[["date", "pm2_5_mean", "pm2_5_norm", "aqi_mean"]],
        on="date", how="left"
    )
    print(f"  [JOIN] Air quality: {df['pm2_5_mean'].notna().sum():,} rows matched")

    # ── join festivals ──
    festival_cols = [c for c in festival_df.columns if c != "date"]
    df = df.merge(festival_df[["date"] + festival_cols], on="date", how="left")
    # fill missing festival flags with 0 (no festival)
    for col in festival_cols:
        df[col] = df[col].fillna(0).astype(int)
    print(f"  [JOIN] Festival flags: {len(festival_cols)} columns merged")

    # ── join flood risk: nearest lat/lon ──
    print("  [JOIN] Flood risk: nearest-neighbour lat/lon match...")
    from scipy.spatial import cKDTree

    flood_coords  = flood_df[["Latitude", "Longitude"]].values
    traffic_coords = df[["Latitude", "Longitude"]].values

    tree = cKDTree(flood_coords)
    dist, idx = tree.query(traffic_coords, k=1)

    # only accept matches within ~5 km (0.05 degrees)
    MAX_DIST = 0.05
    matched = dist <= MAX_DIST
    df["flood_risk_score"] = np.where(
        matched,
        flood_df["flood_risk_score"].values[idx],
        0.0   # no flood risk if no nearby flood data point
    )
    df["flood_rainfall_mm"] = np.where(
        matched,
        flood_df["Rainfall_Intensity"].values[idx],
        0.0
    )
    print(f"  [JOIN] Flood risk: {matched.sum():,} / {len(df):,} rows matched within {MAX_DIST}°")

    # ── fill remaining NaNs ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # ── final column audit ──
    print(f"\n  [SUMMARY] Final merged frame: {df.shape[0]:,} rows × {df.shape[1]} cols")
    tft_static   = ["Latitude", "Longitude", "Area Name", "Road/Intersection Name"]
    tft_known    = ["hour_of_day", "minute_of_day", "day_of_week", "is_weekend",
                    "month", "is_festival_day"] + \
                   [c for c in df.columns if c.startswith("festival_")]
    tft_observed = ["Traffic Volume", "Average Speed", "Congestion Level",
                    "Travel Time Index", "Incident Reports",
                    "pm2_5_mean", "flood_risk_score"]
    tft_target   = ["Average Speed", "Congestion Level"]

    print("\n  ── TFT input categorisation ──")
    print(f"  Static covariates    : {[c for c in tft_static    if c in df.columns]}")
    print(f"  Known future inputs  : {[c for c in tft_known     if c in df.columns]}")
    print(f"  Observed past inputs : {[c for c in tft_observed  if c in df.columns]}")
    print(f"  Target variables     : {[c for c in tft_target    if c in df.columns]}")

    _assert_shape(df, "Final merged frame", min_rows=100_000)
    out = PROCESSED_DIR / "stage1_merged.parquet"
    df.to_parquet(out, index=False)
    print(f"\n  [SAVED] {out}")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  STAGE 1 — Data Ingestion and Preprocessing")
    print("  NIT Karnataka — Emergency Routing Framework")
    print("="*60)

    # run all steps
    traffic_df  = process_traffic(TRAFFIC_FILE)
    flood_df    = process_flood(FLOOD_FILE)
    adapted_df  = process_ridesafety(RIDESAFETY_FILE, traffic_df)
    air_df      = process_air_quality(AIR_FILE)
    festival_df = process_festivals(FESTIVAL_FILE)
    merged_df   = merge_all(traffic_df, air_df, festival_df, flood_df)

    print("\n" + "="*60)
    print("  STAGE 1 COMPLETE")
    print(f"  All outputs saved to: {PROCESSED_DIR}")
    print("="*60)

    # quick sanity print
    print("\nColumn list of stage1_merged.parquet:")
    for i, col in enumerate(merged_df.columns):
        print(f"  {i:3d}  {col}  [{merged_df[col].dtype}]")

    return merged_df


# if __name__ == "__main__":
#     df = main()
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "festivals":
        # run only the festival step
        festival_df = process_festivals(FESTIVAL_FILE)
    elif len(sys.argv) > 1 and sys.argv[1] == "merge":
        # run only the final merge (all parquets already exist)
        traffic_df  = pd.read_parquet(PROCESSED_DIR / "master_traffic.parquet")
        air_df      = pd.read_parquet(PROCESSED_DIR / "air_quality_lookup.parquet")
        festival_df = pd.read_parquet(PROCESSED_DIR / "festival_flags.parquet")
        flood_df    = pd.read_parquet(PROCESSED_DIR / "flood_risk_lookup.parquet")
        merge_all(traffic_df, air_df, festival_df, flood_df)
    else:
        df = main()