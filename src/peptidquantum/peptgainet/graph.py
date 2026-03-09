from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

AA_PROPS = {
    "A": [0.62, 0.50, 0.11], "C": [0.29, 0.46, 0.13], "D": [-0.90, 0.30, 0.54],
    "E": [-0.74, 0.36, 0.50], "F": [1.19, 0.79, 0.07], "G": [0.48, 0.00, 0.15],
    "H": [-0.40, 0.62, 0.23], "I": [1.38, 0.73, 0.05], "K": [-1.50, 0.68, 0.75],
    "L": [1.06, 0.73, 0.06], "M": [0.64, 0.72, 0.22], "N": [-0.78, 0.40, 0.39],
    "P": [0.12, 0.57, 0.18], "Q": [-0.85, 0.44, 0.37], "R": [-2.53, 0.71, 0.72],
    "S": [-0.18, 0.26, 0.30], "T": [-0.05, 0.38, 0.24], "V": [1.08, 0.64, 0.06],
    "W": [0.81, 1.00, 0.11], "Y": [0.26, 0.86, 0.21], "X": [0.0, 0.5, 0.25],
}


@dataclass
class ResidueGraph:
    x: np.ndarray  # [N, F]
    adj: np.ndarray  # [N, N] binary matrix


def residue_features(sequence: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    n = len(sequence)
    for i, aa in enumerate(sequence):
        aa = aa if aa in AA_INDEX else "X"
        one_hot = np.zeros(len(AA_ORDER), dtype=np.float32)
        one_hot[AA_INDEX[aa]] = 1.0
        props = np.asarray(AA_PROPS[aa], dtype=np.float32)
        pos = np.asarray([
            i / max(n - 1, 1),
            np.sin(2 * np.pi * i / max(n, 1)),
            np.cos(2 * np.pi * i / max(n, 1)),
        ], dtype=np.float32)
        rows.append(np.concatenate([one_hot, props, pos], axis=0))
    return np.vstack(rows)


def _add_seq_edges(adj: np.ndarray, radius: int) -> None:
    n = adj.shape[0]
    for i in range(n):
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            adj[i, j] = 1.0
            adj[j, i] = 1.0


def _add_spatial_knn_edges(adj: np.ndarray, coords: np.ndarray, k: int) -> None:
    if k <= 0:
        return
    n = coords.shape[0]
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    for i in range(n):
        nn_idx = np.argsort(d2[i])[1 : k + 1]
        for j in nn_idx:
            adj[i, j] = 1.0
            adj[j, i] = 1.0


def build_residue_graph(
    sequence: str,
    coords: np.ndarray | None = None,
    seq_radius: int = 1,
    spatial_k: int = 0,
) -> ResidueGraph:
    x = residue_features(sequence)
    n = x.shape[0]
    adj = np.eye(n, dtype=np.float32)
    _add_seq_edges(adj, radius=seq_radius)

    if coords is not None and len(coords) == n and spatial_k > 0:
        _add_spatial_knn_edges(adj, coords=np.asarray(coords, dtype=np.float32), k=spatial_k)

    return ResidueGraph(x=x, adj=adj)
