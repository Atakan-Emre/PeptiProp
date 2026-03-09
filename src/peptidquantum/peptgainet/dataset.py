from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .graph import ResidueGraph, build_residue_graph


@dataclass
class PairRecord:
    pair_id: str
    protein_seq: str
    peptide_seq: str
    label: int


@dataclass
class PairGraphSample:
    pair_id: str
    protein_graph: ResidueGraph
    peptide_graph: ResidueGraph
    label: int


def load_pairs_jsonl(path: str | Path) -> list[PairRecord]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    items: list[PairRecord] = []
    with p.open() as f:
        for i, ln in enumerate(f, start=1):
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            items.append(
                PairRecord(
                    pair_id=str(obj.get("pair_id", f"pair_{i}")),
                    protein_seq=str(obj["protein_seq"]).upper(),
                    peptide_seq=str(obj["peptide_seq"]).upper(),
                    label=int(obj["label"]),
                )
            )
    return items


class PairGraphDataset(Dataset):
    def __init__(self, records: list[PairRecord], seq_radius: int = 1, spatial_k: int = 0):
        self.samples = [
            PairGraphSample(
                pair_id=r.pair_id,
                protein_graph=build_residue_graph(r.protein_seq, seq_radius=seq_radius, spatial_k=spatial_k),
                peptide_graph=build_residue_graph(r.peptide_seq, seq_radius=1, spatial_k=0),
                label=r.label,
            )
            for r in records
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PairGraphSample:
        return self.samples[idx]


def _pad_graph_batch(graphs: list[ResidueGraph]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = len(graphs)
    nmax = max(g.x.shape[0] for g in graphs)
    fdim = graphs[0].x.shape[1]

    x = np.zeros((b, nmax, fdim), dtype=np.float32)
    adj = np.zeros((b, nmax, nmax), dtype=np.float32)
    mask = np.zeros((b, nmax), dtype=np.float32)

    for i, g in enumerate(graphs):
        n = g.x.shape[0]
        x[i, :n] = g.x
        adj[i, :n, :n] = g.adj
        mask[i, :n] = 1.0

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(adj, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
    )


def collate_pair_graphs(batch: list[PairGraphSample]) -> dict[str, Any]:
    prot_graphs = [x.protein_graph for x in batch]
    pep_graphs = [x.peptide_graph for x in batch]
    labels = torch.tensor([x.label for x in batch], dtype=torch.float32)
    pair_ids = [x.pair_id for x in batch]

    p_x, p_adj, p_mask = _pad_graph_batch(prot_graphs)
    q_x, q_adj, q_mask = _pad_graph_batch(pep_graphs)

    return {
        "pair_ids": pair_ids,
        "protein_x": p_x,
        "protein_adj": p_adj,
        "protein_mask": p_mask,
        "peptide_x": q_x,
        "peptide_adj": q_adj,
        "peptide_mask": q_mask,
        "labels": labels,
    }
