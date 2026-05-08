"""
Stage 2c: TFT Model + Training Loop (pure PyTorch)
===================================================
Multi-Factor Optimal Routing using Temporal Fusion Transformer
NIT Karnataka, Surathkal

Self-contained TFT implementation using only:
  - torch 2.1.2
  - pytorch-lightning 2.1.4
  - torch-geometric 2.4.0

No neuralforecast, pytorch-forecasting, or darts required.

Usage:
    python stage2_train.py --epochs 10
    python stage2_train.py --epochs 50 --resume
"""

import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from pytorch_lightning.loggers import CSVLogger

from stage2_dataset import build_datasets, ENCODER_LENGTH, DECODER_LENGTH
from stage2_gnn import GNNEncoder, build_graph, get_node_embeddings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CKPT_DIR = Path(r"C:\Dhawal\checkpoints")
LOG_DIR  = Path(r"C:\Dhawal\logs")
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

HIDDEN_DIM      = 64
N_HEADS         = 4
DROPOUT         = 0.1
LEARNING_RATE   = 1e-3
MAX_EPOCHS      = 10
PATIENCE        = 5
QUANTILES       = [0.1, 0.5, 0.9]   # for uncertainty-aware forecasting
GNN_EMBED_DIM   = 32
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# QUANTILE LOSS
# ─────────────────────────────────────────────

class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    Trains the model to predict the 10th, 50th, and 90th percentile.
    The 50th percentile = median = Te (expected travel time).
    The width (90th - 10th) = Ue (uncertainty) used in Stage 4 routing.
    """
    def __init__(self, quantiles=QUANTILES):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds  : [B, horizon, n_quantiles]
        # target : [B, horizon]
        losses = []
        for i, q in enumerate(self.quantiles):
            err = target - preds[:, :, i]
            losses.append(torch.max(q * err, (q - 1) * err))
        return torch.stack(losses, dim=-1).mean()


# ─────────────────────────────────────────────
# TFT BUILDING BLOCKS
# ─────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """
    Core GRN block from the TFT paper (Lim et al., 2020).
    Applies a gated skip connection with optional context input.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int, context_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, output_dim)
        self.gate    = nn.Linear(hidden_dim, output_dim)
        self.skip    = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm    = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.ctx_fc  = nn.Linear(context_dim, hidden_dim) if context_dim else None

    def forward(self, x: torch.Tensor,
                context: torch.Tensor = None) -> torch.Tensor:
        h = self.fc1(x)
        if context is not None and self.ctx_fc is not None:
            h = h + self.ctx_fc(context)
        h = F.elu(h)
        h = self.dropout(h)
        gate = torch.sigmoid(self.gate(h))
        out  = gate * self.fc2(h)
        return self.norm(out + self.skip(x))


class VariableSelectionNetwork(nn.Module):
    """
    VSN from TFT: learns which input features matter most at each timestep.
    Returns weighted sum of feature embeddings + importance weights.
    """
    def __init__(self, input_dim: int, n_vars: int,
                 hidden_dim: int, context_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.grns   = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim,
                                 context_dim, dropout)
            for _ in range(n_vars)
        ])
        self.softmax_grn = GatedResidualNetwork(
            n_vars * input_dim, hidden_dim, n_vars, context_dim, dropout
        )

    def forward(self, x: torch.Tensor,
                context: torch.Tensor = None) -> tuple:
        # x: [B, T, n_vars, input_dim] or [B, n_vars, input_dim]
        B = x.shape[0]
        flat = x.reshape(B, -1) if x.dim() == 3 else x.reshape(B, x.shape[1], -1)
        weights = torch.softmax(self.softmax_grn(flat, context), dim=-1)

        processed = []
        for i, grn in enumerate(self.grns):
            xi = x[..., i, :] if x.dim() == 4 else x[:, i, :]
            processed.append(grn(xi, context))
        processed = torch.stack(processed, dim=-2)

        if weights.dim() == 2:
            out = (weights.unsqueeze(-1) * processed).sum(dim=-2)
        else:
            out = (weights.unsqueeze(-1) * processed).sum(dim=-2)
        return out, weights


class TFTModel(nn.Module):
    """
    Temporal Fusion Transformer (Lim et al., 2020).

    Architecture:
      1. Static covariate encoders  → context vectors for VSN and LSTM
      2. Variable Selection Networks for encoder and decoder inputs
      3. LSTM encoder + decoder for local temporal processing
      4. Multi-head self-attention for long-range dependencies
      5. Position-wise feed-forward with gating
      6. Quantile output head

    Inputs per forward pass:
      enc_input : [B, encoder_length, enc_in_dim]   past target + hist exog
      enc_futr  : [B, encoder_length, futr_dim]      known future in enc window
      dec_futr  : [B, decoder_length, futr_dim]      known future in dec window
      static    : [B, stat_dim]                      road-level static features
    """

    def __init__(
        self,
        enc_in_dim  : int,
        futr_dim    : int,
        stat_dim    : int,
        hidden_dim  : int = HIDDEN_DIM,
        n_heads     : int = N_HEADS,
        dropout     : float = DROPOUT,
        encoder_len : int = ENCODER_LENGTH,
        decoder_len : int = DECODER_LENGTH,
        n_quantiles : int = len(QUANTILES),
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.n_quantiles = n_quantiles

        # ── static encoders ──
        self.static_encoder = GatedResidualNetwork(
            stat_dim, hidden_dim, hidden_dim, dropout=dropout
        )
        # 4 context vectors: for VSN_enc, VSN_dec, LSTM init h, LSTM init c
        self.static_ctx = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(4)
        ])

        # ── input projections ──
        self.enc_proj  = nn.Linear(enc_in_dim, hidden_dim)
        self.futr_proj = nn.Linear(futr_dim,   hidden_dim)

        # ── LSTM encoder/decoder ──
        self.lstm_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_gate    = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)

        # ── multi-head self-attention ──
        self.attn      = nn.MultiheadAttention(hidden_dim, n_heads,
                                                dropout=dropout, batch_first=True)
        self.attn_gate = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # ── position-wise feed-forward ──
        self.ff        = GatedResidualNetwork(hidden_dim, hidden_dim * 4, hidden_dim, dropout=dropout)
        self.ff_norm   = nn.LayerNorm(hidden_dim)

        # ── output head: predict n_quantiles per horizon step ──
        self.output    = nn.Linear(hidden_dim, n_quantiles)

    def _static_context(self, static: torch.Tensor) -> list:
        s = self.static_encoder(static)
        return [ctx(s) for ctx in self.static_ctx]

    def forward(
        self,
        enc_input : torch.Tensor,   # [B, enc_len, enc_in_dim]
        enc_futr  : torch.Tensor,   # [B, enc_len, futr_dim]
        dec_futr  : torch.Tensor,   # [B, dec_len, futr_dim]
        static    : torch.Tensor,   # [B, stat_dim]
    ) -> torch.Tensor:

        B = enc_input.shape[0]

        # static context
        ctx = self._static_context(static)   # 4 vectors each [B, hidden]

        # project inputs
        enc = self.enc_proj(enc_input) + self.futr_proj(enc_futr)  # [B, enc_len, H]
        dec = self.futr_proj(dec_futr)                              # [B, dec_len, H]

        # LSTM initial state from static context
        h0 = ctx[2].unsqueeze(0)   # [1, B, H]
        c0 = ctx[3].unsqueeze(0)   # [1, B, H]

        # LSTM encode + decode
        enc_out, (hn, cn)  = self.lstm_encoder(enc, (h0, c0))
        dec_out, _         = self.lstm_decoder(dec, (hn, cn))

        # gate LSTM output
        enc_out = self.lstm_gate(enc_out)
        dec_out = self.lstm_gate(dec_out)

        # concatenate for attention: [B, enc+dec, H]
        seq = torch.cat([enc_out, dec_out], dim=1)

        # multi-head self-attention
        attn_out, _ = self.attn(seq, seq, seq)
        attn_out    = self.attn_gate(attn_out)
        seq         = self.attn_norm(seq + attn_out)

        # position-wise FF
        ff_out = self.ff(seq)
        seq    = self.ff_norm(seq + ff_out)

        # take only decoder positions
        dec_seq = seq[:, self.encoder_len:, :]   # [B, dec_len, H]

        # quantile output
        out = self.output(dec_seq)               # [B, dec_len, n_quantiles]
        return out


# ─────────────────────────────────────────────
# LIGHTNING MODULE
# ─────────────────────────────────────────────

class TFTLightning(pl.LightningModule):
    """
    pytorch-lightning wrapper for TFTModel.
    Handles train/val loops, logging, and optimiser.
    """

    def __init__(self, model: TFTModel, lr: float = LEARNING_RATE):
        super().__init__()
        self.model    = model
        self.lr       = lr
        self.loss_fn  = QuantileLoss(QUANTILES)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(
            batch["enc_input"].float(),
            batch["enc_futr"].float(),
            batch["dec_futr"].float(),
            batch["static"].float(),
        )

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        preds  = self(batch)
        target = batch["target"].float()
        loss   = self.loss_fn(preds, target)
        self.log(f"{stage}_loss", loss, prog_bar=True,
                 on_epoch=True, on_step=(stage == "train"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=2, factor=0.5, verbose=True
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


# ─────────────────────────────────────────────
# GNN INJECTION
# ─────────────────────────────────────────────

def inject_gnn_embeddings(meta: dict) -> tuple:
    """Extract GNN embeddings and add them to train/val DataFrames."""
    import pandas as pd
    from scipy.spatial import cKDTree
    PROCESSED_DIR = Path(r"C:\Dhawal\Datasets\processed")

    graph_data = build_graph(meta)
    gnn        = GNNEncoder(in_channels=4, hidden=64, out_channels=GNN_EMBED_DIM)
    embeddings = get_node_embeddings(gnn, graph_data, save=True)

    emb_cols = [f"gnn_emb_{i}" for i in range(GNN_EMBED_DIM)]
    node_df  = graph_data.node_df

    for split in ["train_df", "val_df"]:
        df   = meta[split]
        coords = df[["Latitude", "Longitude"]].values
        tree   = cKDTree(node_df[["lat", "lon"]].values)
        _, idx = tree.query(coords, k=1)
        emb_matched = embeddings[idx]
        for j, col in enumerate(emb_cols):
            df[col] = emb_matched[:, j]
        meta[split] = df

    print(f"  [GNN]  {GNN_EMBED_DIM} embedding dims added to train/val DataFrames")
    return meta, emb_cols


# ─────────────────────────────────────────────
# TRAIN ONE MODEL
# ─────────────────────────────────────────────

def train_one(
    name       : str,
    loaders    : dict,
    meta       : dict,
    max_epochs : int,
    resume     : bool,
) -> TFTLightning:

    ckpt_path = CKPT_DIR / f"tft_{name}_last.ckpt"

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
    lit = TFTLightning(model, lr=LEARNING_RATE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [TFT:{name}] Parameters: {n_params:,}")

    callbacks = [
        ModelCheckpoint(
            dirpath   = CKPT_DIR,
            filename  = f"tft_{name}" + "-{epoch:02d}-{val_loss:.4f}",
            monitor   = "val_loss",
            save_top_k= 3,
            save_last = True,
            mode      = "min",
        ),
        EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs        = max_epochs,
        accelerator       = DEVICE,
        devices           = 1,
        callbacks         = callbacks,
        logger            = CSVLogger(LOG_DIR, name=f"tft_{name}"),
        gradient_clip_val = 0.1,
        precision         = "16-mixed" if DEVICE == "cuda" else 32,
        log_every_n_steps = 20,
        enable_progress_bar = True,
    )

    resume_ckpt = str(ckpt_path) if (resume and ckpt_path.exists()) else None
    if resume_ckpt:
        print(f"  [RESUME] Loading {resume_ckpt}")

    trainer.fit(
        lit,
        train_dataloaders = loaders["train"],
        val_dataloaders   = loaders["val"],
        ckpt_path         = resume_ckpt,
    )

    best = trainer.checkpoint_callback.best_model_path
    print(f"  [BEST] {best}")
    return lit, trainer


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def train(max_epochs: int = MAX_EPOCHS, resume: bool = False):
    print("\n" + "="*60)
    print("  STAGE 2c - TFT Training (pure PyTorch)")
    print(f"  Device: {DEVICE.upper()}  |  Epochs: {max_epochs}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu.name} ({gpu.total_memory//1024**2:,} MB)")
    print("="*60)

    # step 1: inject GNN embeddings into train/val DataFrames
    # we need a temporary meta to build the graph first
    print("\n>>> Pre-loading data for GNN graph construction...")
    import pandas as pd
    PROCESSED_DIR = Path(r"C:\Dhawal\Datasets\processed")
    df_tmp = pd.read_parquet(PROCESSED_DIR / "stage1_merged.parquet")
    df_tmp["unique_id"] = (df_tmp["Area Name"].str.strip() + "__" +
                           df_tmp["Road/Intersection Name"].str.strip())
    df_tmp["ds"] = pd.to_datetime(df_tmp["timestamp"])
    all_ts   = sorted(df_tmp["ds"].unique())
    cut_ts   = all_ts[int(len(all_ts) * 0.85)]
    pre_meta = {
        "train_df": df_tmp[df_tmp["ds"] < cut_ts].copy(),
        "val_df"  : df_tmp[df_tmp["ds"] >= cut_ts].copy(),
    }
    pre_meta, emb_cols = inject_gnn_embeddings(pre_meta)

    # step 2: build full datasets with GNN embeddings
    datasets, loaders, meta, scalers = build_datasets(gnn_emb_cols=emb_cols)

    results = {}
    for target_key in ["average_speed", "congestion_level"]:
        print(f"\n{'='*60}")
        print(f"  Training: {target_key.replace('_', ' ').title()}")
        print("="*60)
        lit, trainer = train_one(
            name       = target_key,
            loaders    = loaders[target_key],
            meta       = meta,
            max_epochs = max_epochs,
            resume     = resume,
        )
        results[target_key] = (lit, trainer)

    print("\n" + "="*60)
    print("  STAGE 2 COMPLETE")
    print(f"  Checkpoints -> {CKPT_DIR}")
    print(f"  Logs        -> {LOG_DIR}")
    print("  Next -> run stage3_causal.py")
    print("="*60)
    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(max_epochs=args.epochs, resume=args.resume)