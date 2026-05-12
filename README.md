# Multi-Factor Optimal Routing using Temporal Fusion Transformer

> **AINA 2026** — NIT Karnataka, Surathkal  
> Authors: Madhusudhan R & Dhawal Ramdham

A research framework for emergency vehicle routing in Bangalore using a Temporal Fusion Transformer (TFT) for multi-factor traffic forecasting, GNN-based road embeddings, DoWhy causal inference, and Pareto-optimal path selection.

---

## Overview

Emergency vehicles need routes that balance multiple competing objectives simultaneously:

| Symbol | Objective | Source |
|--------|-----------|--------|
| **Tₑ** | Expected travel time | TFT median forecast |
| **Uₑ** | Forecast uncertainty | TFT quantile band width |
| **Rₑ** | Incident risk | DoWhy causal ATE per segment |
| **Pₑ** | PM2.5 pollution exposure | Air quality lookup |

The composite edge weight is:

```
Wₑ = α·Tₑ + β·Uₑ + γ·Rₑ + δ·Pₑ     (α + β + γ + δ = 1)
```

Default: α=0.40, β=0.20, γ=0.25, δ=0.15

---

## Pipeline

```
stage1_preprocess.py 
       ↓
stage2_dataset.py / stage2_gnn.py / stage2_train.py
       ↓                                 
stage3_causal.py       
       ↓
stage4_routing.py   →   stage4_map.py / stage4_maps.py / stage4_evaluate.py
       ↓
stage5_dashboard.py
```

### Stage 1 — Data Ingestion & Preprocessing
Merges five heterogeneous datasets into a single 5-minute time-series frame:
- **Bangalore Traffic Dataset** (backbone, 8936 rows × 18 cols)
- **Urban Flood Dataset** → composite flood risk score per lat/lon
- **RideSafety Dataset** → domain-adapted to Bangalore via percentile matching
- **Air Pollution India** → daily PM2.5 lookup for Bangalore
- **Festival Calendar India** → one-hot festival flags (known future inputs for TFT)

Output: `Datasets/processed/stage1_merged.parquet`

### Stage 2 — TFT + GNN Training
- **`stage2_dataset.py`** — builds encoder/decoder windows for TFT training
- **`stage2_gnn.py`** — GNN encoder that produces 32-dim road segment embeddings from the OSM graph
- **`stage2_train.py`** — trains separate TFT models for `Average Speed` and `Congestion Level` using quantile loss (P10/P50/P90)

Output: `checkpoints/tft_average_speed-*.ckpt`, `checkpoints/tft_congestion_level-*.ckpt`

### Stage 3 — Causal Inference
Uses **DoWhy** (4-step process: model → identify → estimate → refute) to estimate the Average Treatment Effect (ATE) of traffic incidents on Travel Time Index.

- Treatment: `incident_flag` (binary, ≥90th percentile of incident count)
- Confounders: congestion level, time of day, weather, festival flags
- Method: Propensity Score Weighting (IPW)
- Refutation: placebo treatment + random common cause tests

Output: `Datasets/processed/routing_ate_lookup.parquet`

### Stage 4 — Pareto-Optimal Routing
Downloads the Bangalore OSM road network via OSMnx, assigns composite edge weights from TFT + ATE + PM2.5 outputs, then finds Pareto-optimal emergency routes between any two locations.

```bash
# by place name
python stage4_routing.py --origin "Manipal Hospital, Bangalore" --dest "Victoria Hospital, Bangalore"

# by coordinates
python stage4_routing.py --origin "12.9716,77.5946" --dest "12.9352,77.6245"

# custom weights
python stage4_routing.py --origin "..." --dest "..." --alpha 0.5 --beta 0.2 --gamma 0.2 --delta 0.1
```

Output: `Datasets/processed/pareto_routes.parquet`, `best_route.parquet`

### Stage 5 — Streamlit Dashboard
Interactive dashboard with:
- Origin/destination picker with live routing
- Adjustable α/β/γ/δ weight sliders
- Folium map with all Pareto-optimal routes overlaid on congestion heatmap
- Radar chart + stacked bar chart comparing route objectives
- Per-segment ATE heatmap

```bash
streamlit run stage5_dashboard.py
```

---

## Installation

### Requirements

```bash
# GPU (used — RTX 4000 Ada / CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# All other dependencies
pip install -r Files/requirements.txt
```

### Key dependencies
- `pytorch-lightning` — TFT training loop
- `pytorch-forecasting` — TFT architecture
- `torch-geometric` — GNN encoder
- `dowhy`, `econml` — causal inference
- `osmnx`, `networkx` — road network & routing
- `streamlit`, `folium`, `plotly` — dashboard

See `Files/requirements.txt` for pinned versions and GPU install notes.

---

## Datasets

Place all raw CSVs in your data directory (configured as `DATA_DIR` in each stage script):

| File | Description |
|------|-------------|
| `Bangalore_Traffic_Dataset.csv` | Backbone traffic data — speed, congestion, incidents |
| `Bangalore_Urban_Flood_Dataset.csv` | Flood risk per road segment |
| `RideSafety_Dataset.csv` | Incident/safety data |
| `Air_Pollution_India.csv` | City-level daily PM2.5 readings |
| `Festival_Calendar_India.csv` | Indian festival dates with traffic impact categories |

---

## Running the Full Pipeline

```bash
python stage1_preprocess.py
python stage2_train.py --epochs 50
python stage2_evaluate.py
python stage3_causal.py
python stage3_evaluate.py
python stage4_routing.py --origin "Manipal Hospital, Bangalore" --dest "Victoria Hospital, Bangalore"
python stage4_evaluate.py
streamlit run stage5_dashboard.py
```

---

## Project Structure

```
TFT Project/
├── stage1_preprocess.py     # Data ingestion and merging
├── stage2_dataset.py        # TFT dataset builder
├── stage2_gnn.py            # GNN encoder
├── stage2_train.py          # TFT training loop
├── stage2_evaluate.py       # Stage 2 evaluation & plots
├── stage3_causal.py         # DoWhy causal ATE estimation
├── stage3_evaluate.py       # Stage 3 evaluation & plots
├── stage4_routing.py        # Pareto-optimal routing engine
├── stage4_maps.py            # Route map generator
├── stage4_evaluate.py       # Stage 4 evaluation & plots
├── stage5_dashboard.py      # Streamlit dashboard
├── Files/
│   ├── requirements.txt
│   └── INSTALL_GUIDE.txt
└── Datasets/                # Raw datasets 
       ├──Air_Pollution_India.csv
       ├──Bangalore_Traffic_Dataset.csv
       ├──Bangalore_Urban_Flood_Dataset.csv
       ├──Bangalore_traffic_Dataset_wo_cors.csv
       ├──Festival_Calendar_India.csv
       └──RideSafety_Dataset.csv               
```

---

## Citation

If you use this work, please cite:

> Madhusudhan R, Dhawal Ramdham. *Multifactor Optimal Routing using Temporal Fusion Transformer*. AINA 2026, NIT Karnataka, Surathkal.
