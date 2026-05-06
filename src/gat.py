"""GAT (Graph Attention Network) for graph-level binary classification.

Same overall scaffold as the GCN — input projection, n stacked attention layers
with optional BatchNorm + residual, global mean pool, linear head — but with
GATConv instead of GCNConv. Multi-head attention is supported via --heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 128,
        n_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.4,
        use_batch_norm: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        # All layers concat heads back to hidden_dim, so per-head dim = hidden_dim // heads.
        per_head = max(hidden_dim // heads, 1)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, per_head, heads=heads, dropout=dropout, concat=True)
            for _ in range(n_layers)
        ])
        # If hidden_dim is not divisible by heads, GAT output is per_head*heads which can != hidden_dim.
        # We project back to hidden_dim after each conv to keep residual sizes aligned.
        self.proj = nn.ModuleList(
            [nn.Linear(per_head * heads, hidden_dim) for _ in range(n_layers)]
        )
        self.bns = (
            nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
            if use_batch_norm else None
        )
        self.use_residual = use_residual
        self.dropout = dropout
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        for i, (conv, proj) in enumerate(zip(self.convs, self.proj)):
            h = conv(x, edge_index)
            h = proj(h)
            if self.bns is not None:
                h = self.bns[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h + x if self.use_residual else h
        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)
