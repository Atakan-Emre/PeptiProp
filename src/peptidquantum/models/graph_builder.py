"""
Build PyG Data objects from canonical residues and ESM-2 embeddings.

Each protein–peptide pair becomes two separate graphs (dual encoder)
with node features = ESM-2 embedding + structural annotations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

SS_ONEHOT = {"helix": 0, "sheet": 1, "coil": 2, "unknown": 3}
STRUCT_FEAT_DIM = 1 + 1 + len(SS_ONEHOT)  # is_interface, is_pocket, SS one-hot = 6


def normalize_ss(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "unknown"
    t = str(val).strip().lower()
    if not t or t == "none":
        return "unknown"
    if t in {"h", "g", "i", "helix", "alpha_helix", "310_helix", "pi_helix"}:
        return "helix"
    if t in {"e", "b", "sheet", "beta_sheet", "strand"}:
        return "sheet"
    if t in {"c", "coil", "loop", "turn", "bend", "s"}:
        return "coil"
    return "unknown"


def build_structural_features(residues_df) -> np.ndarray:
    """Build per-residue structural feature vector (6-d)."""
    n = len(residues_df)
    feats = np.zeros((n, STRUCT_FEAT_DIM), dtype=np.float32)
    feats[:, 0] = residues_df["is_interface"].fillna(False).astype(np.float32).values
    feats[:, 1] = residues_df["is_pocket"].fillna(False).astype(np.float32).values

    for i, ss_raw in enumerate(residues_df["secondary_structure"]):
        ss = normalize_ss(ss_raw)
        idx = SS_ONEHOT.get(ss, 3)
        feats[i, 2 + idx] = 1.0
    return feats


def build_edges_radius(coords: np.ndarray, cutoff: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    """Build edge index and edge features using distance cutoff on Cα coords."""
    n = len(coords)
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    # Pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    dists = np.linalg.norm(diff, axis=-1)  # (N, N)

    mask = (dists < cutoff) & (dists > 0)
    src, dst = np.where(mask)
    edge_index = np.stack([src, dst], axis=0)

    edge_dists = dists[src, dst].reshape(-1, 1)
    edge_dirs = diff[src, dst]  # (E, 3)
    norms = np.linalg.norm(edge_dirs, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    edge_dirs = edge_dirs / norms

    edge_attr = np.concatenate([edge_dists, edge_dirs], axis=-1).astype(np.float32)
    return edge_index, edge_attr


class PairGraphBuilder:
    """Builds PyG graph pairs from canonical data + ESM-2 embeddings."""

    def __init__(
        self,
        embedding_dir: Path,
        lookup: Dict[str, str],
        cutoff: float = 8.0,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.lookup = lookup
        self.cutoff = cutoff
        self._emb_cache: Dict[str, np.ndarray] = {}

    def _load_embedding(self, complex_id: str, chain_id: str) -> Optional[np.ndarray]:
        key = f"{complex_id}::{chain_id}"
        npz_name = self.lookup.get(key)
        if npz_name is None:
            return None
        if npz_name in self._emb_cache:
            return self._emb_cache[npz_name]
        path = self.embedding_dir / npz_name
        if not path.exists():
            return None
        emb = np.load(path)["embedding"].astype(np.float32)
        self._emb_cache[npz_name] = emb
        return emb

    def build_chain_graph(
        self,
        residues_df,
        complex_id: str,
        chain_id: str,
    ) -> Optional[Data]:
        """Build a single-chain PyG Data object."""
        residues_df = residues_df.sort_values("residue_number_auth").reset_index(drop=True)
        n_res = len(residues_df)
        if n_res == 0:
            return None

        esm_emb = self._load_embedding(complex_id, chain_id)
        struct_feats = build_structural_features(residues_df)

        if esm_emb is not None and esm_emb.shape[0] == n_res:
            node_feats = np.concatenate([esm_emb, struct_feats], axis=-1)
        elif esm_emb is not None and esm_emb.shape[0] != n_res:
            # Length mismatch: truncate or pad
            min_len = min(esm_emb.shape[0], n_res)
            pad_emb = np.zeros((n_res, esm_emb.shape[1]), dtype=np.float32)
            pad_emb[:min_len] = esm_emb[:min_len]
            node_feats = np.concatenate([pad_emb, struct_feats], axis=-1)
        else:
            # No ESM-2 embedding available: use zeros
            node_feats = np.concatenate(
                [np.zeros((n_res, 320), dtype=np.float32), struct_feats], axis=-1
            )

        coords = residues_df[["x", "y", "z"]].values.astype(np.float32)
        edge_index, edge_attr = build_edges_radius(coords, self.cutoff)

        return Data(
            x=torch.from_numpy(node_feats),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(edge_attr),
            pos=torch.from_numpy(coords),
            num_nodes=n_res,
        )

    def build_pair(
        self,
        protein_residues_df,
        peptide_residues_df,
        protein_complex_id: str,
        protein_chain_id: str,
        peptide_complex_id: str,
        peptide_chain_id: str,
    ) -> Optional[Tuple[Data, Data]]:
        """Build (protein_graph, peptide_graph) pair."""
        prot_graph = self.build_chain_graph(
            protein_residues_df, protein_complex_id, protein_chain_id
        )
        pep_graph = self.build_chain_graph(
            peptide_residues_df, peptide_complex_id, peptide_chain_id
        )
        if prot_graph is None or pep_graph is None:
            return None
        return prot_graph, pep_graph
