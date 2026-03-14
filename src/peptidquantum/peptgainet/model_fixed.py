from __future__ import annotations

import math
import torch
import torch.nn as nn


class MessagePassingLayer(nn.Module):
    """MPNN-inspired message passing for residue graphs"""
    
    def __init__(self, node_dim: int, hidden_dim: int, message_steps: int = 6):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps
        
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        self.update_net = nn.GRUCell(node_dim, node_dim)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, F = x.shape
        h = x
        
        for step in range(self.message_steps):
            h_src = h.unsqueeze(2).expand(-1, -1, N, -1)
            h_tgt = h.unsqueeze(1).expand(-1, N, -1, -1)
            
            messages = torch.cat([h_src, h_tgt], dim=-1)
            messages = self.message_net(messages)
            
            valid_edges = (adj > 0).unsqueeze(-1) & mask.unsqueeze(1).unsqueeze(-1) & mask.unsqueeze(2).unsqueeze(-1)
            messages = messages * valid_edges
            
            agg_messages = messages.sum(dim=2)
            
            h_flat = h.view(B * N, F)
            messages_flat = agg_messages.view(B * N, F)
            h = self.update_net(messages_flat, h_flat).view(B, N, F)
            h = h * mask.unsqueeze(-1)
            
        return h


class TransformerReadout(nn.Module):
    """Transformer-based readout with proper dimension handling"""
    
    def __init__(self, node_dim: int, num_heads: int = 8, dense_dim: int = 512):
        super().__init__()
        
        self.node_dim = node_dim
        adjusted_dim = (node_dim // num_heads) * num_heads
        if adjusted_dim == 0:
            adjusted_dim = num_heads
        
        self.input_proj = nn.Linear(node_dim, adjusted_dim) if node_dim != adjusted_dim else nn.Identity()
        self.attention = nn.MultiheadAttention(adjusted_dim, num_heads, batch_first=True)
        self.dense_proj = nn.Sequential(
            nn.Linear(adjusted_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dense_dim, adjusted_dim)
        )
        self.norm1 = nn.LayerNorm(adjusted_dim)
        self.norm2 = nn.LayerNorm(adjusted_dim)
        self.dropout = nn.Dropout(0.1)
        self.output_dim = adjusted_dim
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        attn_mask = ~mask.bool()
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.dense_proj(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        mask_expanded = mask.unsqueeze(-1)
        x_masked = x.masked_fill(~mask_expanded.bool(), -float('inf'))
        return x_masked.max(dim=1)[0]


class ImprovedPeptGAINET(nn.Module):
    """Fixed MPNN-inspired PeptGAINET"""
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 64,
        message_steps: int = 6,
        num_attention_heads: int = 8,
        dense_units: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.peptide_encoder = MessagePassingLayer(node_dim, hidden_dim, message_steps)
        self.protein_encoder = MessagePassingLayer(node_dim, hidden_dim, message_steps)
        
        self.peptide_readout = TransformerReadout(node_dim, num_attention_heads, dense_units)
        self.protein_readout = TransformerReadout(node_dim, num_attention_heads, dense_units)
        
        output_dim = self.peptide_readout.output_dim
        
        self.interaction_net = nn.Sequential(
            nn.Linear(output_dim * 2, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pep_h = self.peptide_encoder(
            batch["peptide_x"], 
            batch["peptide_adj"], 
            batch["peptide_mask"]
        )
        prot_h = self.protein_encoder(
            batch["protein_x"], 
            batch["protein_adj"], 
            batch["protein_mask"]
        )
        
        pep_vec = self.peptide_readout(pep_h, batch["peptide_mask"])
        prot_vec = self.protein_readout(prot_h, batch["protein_mask"])
        
        combined = torch.cat([pep_vec, prot_vec], dim=-1)
        prob = self.interaction_net(combined).squeeze(-1)
        
        return {"prob": prob, "peptide_vec": pep_vec, "protein_vec": prot_vec}
