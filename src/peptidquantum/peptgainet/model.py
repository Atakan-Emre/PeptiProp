from __future__ import annotations

import math

import torch
import torch.nn as nn


class DenseGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads

        self.proj = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim * heads)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], adj: [B, N, N], mask: [B, N]
        b, n, _ = x.shape
        h = self.proj(x).view(b, n, self.heads, self.out_dim)

        src = (h * self.attn_src.view(1, 1, self.heads, self.out_dim)).sum(-1)
        dst = (h * self.attn_dst.view(1, 1, self.heads, self.out_dim)).sum(-1)

        e = src[:, :, None, :] + dst[:, None, :, :]  # [B, N, N, H]
        e = self.leaky_relu(e).permute(0, 3, 1, 2)  # [B, H, N, N]

        valid = (adj > 0).unsqueeze(1)
        valid = valid & (mask[:, None, :, None] > 0) & (mask[:, None, None, :] > 0)

        e = e.masked_fill(~valid, -1e9)
        alpha = torch.softmax(e, dim=-1)
        alpha = self.dropout(alpha)

        h_heads = h.permute(0, 2, 1, 3)  # [B, H, N, D]
        out = torch.matmul(alpha, h_heads)  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).reshape(b, n, self.heads * self.out_dim)

        out = out * mask.unsqueeze(-1)
        out = self.norm(out)
        return out


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        blocks = []
        cur_dim = in_dim
        for _ in range(layers):
            blocks.append(DenseGATLayer(cur_dim, hidden_dim, heads=heads, dropout=dropout))
            cur_dim = hidden_dim * heads
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h, adj, mask)
        return h


class CoAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj_pep = nn.Linear(dim, dim, bias=False)
        self.proj_prot = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        pep_h: torch.Tensor,
        prot_h: torch.Tensor,
        pep_mask: torch.Tensor,
        prot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pep_h: [B, Np, D], prot_h: [B, Nt, D]
        q = self.proj_pep(pep_h)
        k = self.proj_prot(prot_h)
        d = pep_h.shape[-1]
        score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d)  # [B, Np, Nt]

        valid = (pep_mask[:, :, None] > 0) & (prot_mask[:, None, :] > 0)
        score = score.masked_fill(~valid, -1e9)

        attn_p2t = torch.softmax(score, dim=-1)
        attn_t2p = torch.softmax(score.transpose(1, 2), dim=-1)

        pep_ctx = torch.matmul(attn_p2t, prot_h)
        prot_ctx = torch.matmul(attn_t2p, pep_h)

        pep_out = torch.cat([pep_h, pep_ctx], dim=-1)
        prot_out = torch.cat([prot_h, prot_ctx], dim=-1)
        return pep_out, prot_out, attn_p2t


class RescalScorer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.relation = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.relation)

        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, 1),
        )

    def forward(self, pep_vec: torch.Tensor, prot_vec: torch.Tensor) -> torch.Tensor:
        bilinear = ((pep_vec @ self.relation) * prot_vec).sum(-1, keepdim=True)
        mix = torch.cat([pep_vec, prot_vec, torch.abs(pep_vec - prot_vec), pep_vec * prot_vec], dim=-1)
        return bilinear + self.mlp(mix)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    den = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / den


class PeptGAINET(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.peptide_encoder = GraphEncoder(node_dim, hidden_dim, heads=heads, layers=layers)
        self.protein_encoder = GraphEncoder(node_dim, hidden_dim, heads=heads, layers=layers)

        enc_dim = hidden_dim * heads
        self.co_attention = CoAttention(enc_dim)

        fused_dim = enc_dim * 2
        self.pep_gate = nn.Linear(fused_dim, fused_dim)
        self.prot_gate = nn.Linear(fused_dim, fused_dim)

        self.scorer = RescalScorer(fused_dim)

    def forward(self, batch: dict[str, torch.Tensor], return_attention: bool = False) -> dict[str, torch.Tensor]:
        pep_h = self.peptide_encoder(batch["peptide_x"], batch["peptide_adj"], batch["peptide_mask"])
        prot_h = self.protein_encoder(batch["protein_x"], batch["protein_adj"], batch["protein_mask"])

        pep_h2, prot_h2, attn = self.co_attention(pep_h, prot_h, batch["peptide_mask"], batch["protein_mask"])

        pep_h2 = pep_h2 * torch.sigmoid(self.pep_gate(pep_h2))
        prot_h2 = prot_h2 * torch.sigmoid(self.prot_gate(prot_h2))

        pep_vec = masked_mean(pep_h2, batch["peptide_mask"])
        prot_vec = masked_mean(prot_h2, batch["protein_mask"])

        logit = self.scorer(pep_vec, prot_vec).squeeze(-1)
        out = {"logit": logit, "prob": torch.sigmoid(logit)}
        if return_attention:
            out["attention"] = attn
        return out
