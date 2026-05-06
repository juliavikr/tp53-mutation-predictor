"""Configurable GCN for graph-level binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.5,
        use_batch_norm: bool = False,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.bns = (
            nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
            if use_batch_norm
            else None
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
        for i, conv in enumerate(self.convs):
            h = conv(x, edge_index)
            if self.bns is not None:
                h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h + x if self.use_residual else h
        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)
