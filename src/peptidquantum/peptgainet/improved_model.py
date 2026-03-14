from __future__ import annotations

import math

import torch
import torch.nn as nn


class MessagePassingLayer(nn.Module):
    """MPNN-inspired message passing layer for residue graphs"""
    
    def __init__(self, node_dim: int, hidden_dim: int, message_steps: int = 6):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps
        
        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update network (GRU-like)
        self.update_net = nn.GRUCell(hidden_dim, node_dim)
        
        # Edge feature embedding (distance-based)
        self.edge_embedding = nn.Linear(1, 16)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], adj: [B, N, N] with distance weights, mask: [B, N]
        B, N, F = x.shape
        
        # Convert adjacency to edge features
        edge_features = self.edge_embedding(adj.unsqueeze(-1))  # [B, N, N, 16]
        
        h = x
        for step in range(self.message_steps):
            # Prepare messages
            h_expanded = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, F]
            h_t = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, F]
            
            # Concatenate source, target, and edge features
            messages = torch.cat([
                h_expanded,  # source
                h_t,  # target
                edge_features  # edge
            ], dim=-1)  # [B, N, N, F*2+16]
            
            # Apply message network
            messages = self.message_net(messages)  # [B, N, N, hidden_dim]
            
            # Mask invalid edges
            valid_edges = (adj > 0).unsqueeze(-1) & (mask.unsqueeze(1) > 0) & (mask.unsqueeze(2) > 0)
            messages = messages * valid_edges
            
            # Aggregate messages (sum)
            agg_messages = messages.sum(dim=2)  # [B, N, hidden_dim]
            
            # Update hidden states
            h_flat = h.view(B * N, F)
            messages_flat = agg_messages.view(B * N, -1)
            h = self.update_net(messages_flat, h_flat).view(B, N, F)
            
            # Apply mask
            h = h * mask.unsqueeze(-1)
            
        return h


class TransformerEncoderReadout(nn.Module):
    """Transformer encoder for readout as in MPNN"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dense_dim: int = 512):
        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            # Adjust embed_dim to be divisible
            embed_dim = (embed_dim // num_heads) * num_heads
            if embed_dim == 0:
                embed_dim = num_heads
        
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D], mask: [B, N]
        
        # Adjust input dimension if needed
        input_dim = x.shape[-1]
        if input_dim != self.attention.embed_dim:
            # Add a linear projection to match embed_dim
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Linear(input_dim, self.attention.embed_dim).to(x.device)
            x = self.input_proj(x)
        
        # Create attention mask (True means keep)
        attn_mask = ~mask.bool()  # Invert for PyTorch convention
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.dense_proj(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        # Global max pooling
        mask_expanded = mask.unsqueeze(-1)
        x_masked = x.masked_fill(~mask_expanded.bool(), -float('inf'))
        return x_masked.max(dim=1)[0]  # [B, D]


class ImprovedPeptGAINET(nn.Module):
    """Improved PeptGAINET with MPNN-inspired architecture"""
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 72,
        message_steps: int = 6,
        num_attention_heads: int = 10,
        dense_units: int = 576,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Message passing encoders
        self.peptide_encoder = MessagePassingLayer(node_dim, hidden_dim, message_steps)
        self.protein_encoder = MessagePassingLayer(node_dim, hidden_dim, message_steps)
        
        # Transformer readout - ensure output dimension matches node_dim
        self.peptide_readout = TransformerEncoderReadout(
            embed_dim=node_dim,
            num_heads=num_attention_heads,
            dense_dim=dense_units
        )
        self.protein_readout = TransformerEncoderReadout(
            embed_dim=node_dim,
            num_heads=num_attention_heads,
            dense_dim=dense_units
        )
        
        # Store the actual output dimension (might be adjusted by transformer)
        self.output_dim = node_dim
        if node_dim % num_attention_heads != 0:
            self.output_dim = (node_dim // num_attention_heads) * num_attention_heads
            if self.output_dim == 0:
                self.output_dim = num_attention_heads
        
        # Interaction layers
        self.interaction_net = nn.Sequential(
            nn.Linear(self.output_dim * 2, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Message passing
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
        
        # Transformer readout
        pep_vec = self.peptide_readout(pep_h, batch["peptide_mask"])
        prot_vec = self.protein_readout(prot_h, batch["protein_mask"])
        
        # Interaction prediction
        combined = torch.cat([pep_vec, prot_vec], dim=-1)
        prob = self.interaction_net(combined).squeeze(-1)
        
        return {"prob": prob, "peptide_vec": pep_vec, "protein_vec": prot_vec}


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean"""
    den = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / den


class PeptGAINETV2(nn.Module):
    """Alternative version with co-attention + MPNN hybrid"""
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 72,
        message_steps: int = 6,
        num_attention_heads: int = 8,
        dense_units: int = 512
    ):
        super().__init__()
        
        # Message passing encoders
        self.peptide_encoder = MessagePassingLayer(node_dim, hidden_dim, message_steps)
        self.protein_encoder = MessagePassingLayer(node_dim, hidden_dim, message_steps)
        
        # Co-attention
        enc_dim = node_dim
        self.co_attention = nn.MultiheadAttention(
            enc_dim, num_attention_heads, batch_first=True
        )
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(enc_dim * 4, dense_units),  # pep + prot + attention contexts
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dense_units, dense_units // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dense_units // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Message passing
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
        
        # Cross-attention
        pep_attn, _ = self.co_attention(
            pep_h, prot_h, prot_h,
            key_padding_mask=~batch["peptide_mask"].bool()
        )
        prot_attn, _ = self.co_attention(
            prot_h, pep_h, pep_h,
            key_padding_mask=~batch["protein_mask"].bool()
        )
        
        # Pooling
        pep_vec = masked_mean(pep_h, batch["peptide_mask"])
        prot_vec = masked_mean(prot_h, batch["protein_mask"])
        pep_attn_vec = masked_mean(pep_attn, batch["peptide_mask"])
        prot_attn_vec = masked_mean(prot_attn, batch["protein_mask"])
        
        # Predict
        combined = torch.cat([
            pep_vec, prot_vec, pep_attn_vec, prot_attn_vec
        ], dim=-1)
        prob = self.predictor(combined).squeeze(-1)
        
        return {"prob": prob}
