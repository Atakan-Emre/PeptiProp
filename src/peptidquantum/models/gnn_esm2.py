"""
GATv2 + ESM-2 dual-encoder model for protein–peptide interaction scoring.

Architecture:
  Protein branch:  GATv2 (N layers) → Global Attention Pooling → protein_vec
  Peptide branch:  GATv2 (N layers) → Global Attention Pooling → peptide_vec
  Interaction:     [prot; pep; prot*pep; |prot-pep|] → MLP → sigmoid → score
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch


class AttentionPooling(nn.Module):
    """Learnable attention-weighted global pooling."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        gate_scores = self.gate(x)  # (N, 1)
        # Softmax per graph
        from torch_geometric.utils import softmax as pyg_softmax
        attn = pyg_softmax(gate_scores, batch, dim=0)  # (N, 1)
        weighted = x * attn  # (N, D)
        return global_mean_pool(weighted, batch)  # (B, D) — sum via mean*N approx


class GATv2Encoder(nn.Module):
    """Multi-layer GATv2 graph encoder."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            conv = GATv2Conv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=True,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.pool = AttentionPooling(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Batch) -> torch.Tensor:
        x = self.input_proj(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, "edge_attr") and data.edge_attr is not None else None

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr=edge_attr)
            h = norm(h)
            h = torch.nn.functional.silu(h)
            h = self.dropout(h)
            x = x + h  # residual

        return self.pool(x, data.batch)


class PeptiPropGNN(nn.Module):
    """
    Dual-encoder GATv2 model for protein–peptide interaction scoring.
    """

    def __init__(
        self,
        node_feat_dim: int = 326,
        hidden_dim: int = 128,
        num_gnn_layers: int = 4,
        heads: int = 4,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        edge_dim: int = 4,
    ):
        super().__init__()
        self.protein_encoder = GATv2Encoder(
            node_feat_dim, hidden_dim, num_gnn_layers, heads, dropout, edge_dim
        )
        self.peptide_encoder = GATv2Encoder(
            node_feat_dim, hidden_dim, num_gnn_layers, heads, dropout, edge_dim
        )

        interaction_dim = hidden_dim * 4  # [prot; pep; prot*pep; |prot-pep|]
        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(
        self, protein_batch: Batch, peptide_batch: Batch
    ) -> torch.Tensor:
        prot_vec = self.protein_encoder(protein_batch)  # (B, hidden)
        pep_vec = self.peptide_encoder(peptide_batch)  # (B, hidden)

        interaction = torch.cat(
            [prot_vec, pep_vec, prot_vec * pep_vec, torch.abs(prot_vec - pep_vec)],
            dim=-1,
        )
        logits = self.classifier(interaction).squeeze(-1)  # (B,)
        return logits

    def predict_proba(
        self, protein_batch: Batch, peptide_batch: Batch
    ) -> torch.Tensor:
        logits = self.forward(protein_batch, peptide_batch)
        return torch.sigmoid(logits)
