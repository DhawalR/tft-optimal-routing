"""
Stage 2b: GNN Spatial Encoder
==============================
Multi-Factor Optimal Routing using Temporal Fusion Transformer
NIT Karnataka, Surathkal

GraphSAGE encoder producing 32-dim spatial embeddings per road segment.
Zero dependency on any forecasting library.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import NearestNeighbors

PROCESSED_DIR  = Path(r"C:\Dhawal\Datasets\processed")
NODE_EMBED_DIM = 32
GNN_HIDDEN     = 64
K_NEIGHBOURS   = 4
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_graph(meta: dict) -> Data:
    print("\n" + "="*60)
    print("  STAGE 2b - GNN Graph Construction")
    print("="*60)

    train_df = meta["train_df"]

    node_df = (
        train_df.groupby("unique_id")
                .agg(lat=("Latitude","mean"), lon=("Longitude","mean"),
                     avg_speed=("Average Speed","mean"),
                     congestion=("Congestion Level","mean"))
                .reset_index().sort_values("unique_id").reset_index(drop=True)
    )
    n_nodes = len(node_df)
    print(f"  [GRAPH] Nodes: {n_nodes}")

    def _norm(col):
        mn, mx = col.min(), col.max()
        return (col - mn) / (mx - mn + 1e-9)

    node_features = torch.tensor(np.column_stack([
        _norm(node_df["lat"]).values,
        _norm(node_df["lon"]).values,
        _norm(node_df["avg_speed"]).values,
        _norm(node_df["congestion"]).values,
    ]), dtype=torch.float32)

    coords = node_df[["lat","lon"]].values
    k      = min(K_NEIGHBOURS, n_nodes - 1)
    nbrs   = NearestNeighbors(n_neighbors=k+1, algorithm="ball_tree",
                               metric="haversine").fit(np.radians(coords))
    _, indices = nbrs.kneighbors(np.radians(coords))

    src, dst = [], []
    for i, nbrs_i in enumerate(indices):
        for j in nbrs_i[1:]:
            src += [i, j]
            dst += [j, i]

    edge_index = torch.unique(
        torch.tensor([src, dst], dtype=torch.long), dim=1
    )
    print(f"  [GRAPH] Edges (K={k}, undirected): {edge_index.shape[1]}")

    graph_data         = Data(x=node_features, edge_index=edge_index)
    graph_data.node_df = node_df
    node_df.to_parquet(PROCESSED_DIR / "graph_nodes.parquet", index=False)
    print(f"  [SAVED] graph_nodes.parquet")
    return graph_data


class GNNEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden=GNN_HIDDEN,
                 out_channels=NODE_EMBED_DIM, dropout=0.2):
        super().__init__()
        self.conv1   = SAGEConv(in_channels, hidden)
        self.conv2   = SAGEConv(hidden, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.ELU()

    def forward(self, x, edge_index):
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        return self.conv2(x, edge_index)


def get_node_embeddings(gnn: GNNEncoder, graph_data: Data,
                        save: bool = True) -> np.ndarray:
    gnn.eval().to(DEVICE)
    with torch.no_grad():
        emb = gnn(graph_data.x.to(DEVICE),
                  graph_data.edge_index.to(DEVICE)).cpu().numpy()

    print(f"  [GNN]  Embeddings: {emb.shape}")
    if save:
        node_df  = graph_data.node_df.copy()
        emb_cols = [f"gnn_emb_{i}" for i in range(emb.shape[1])]
        emb_df   = pd.concat([node_df[["unique_id"]],
                               pd.DataFrame(emb, columns=emb_cols)], axis=1)
        emb_df.to_parquet(PROCESSED_DIR / "gnn_embeddings.parquet", index=False)
        print(f"  [SAVED] gnn_embeddings.parquet")
    return emb


if __name__ == "__main__":
    import pandas as pd, numpy as np
    dummy_meta = {"train_df": pd.DataFrame({
        "unique_id"       : [f"road_{i}" for i in np.repeat(range(5), 100)],
        "Latitude"        : np.random.uniform(12.8, 13.1, 500),
        "Longitude"       : np.random.uniform(77.5, 77.7, 500),
        "Average Speed"   : np.random.uniform(20, 60, 500),
        "Congestion Level": np.random.uniform(0.2, 0.9, 500),
    })}
    gd  = build_graph(dummy_meta)
    gnn = GNNEncoder()
    emb = get_node_embeddings(gnn, gd, save=False)
    print(f"[SMOKE TEST PASSED] {emb.shape}")
