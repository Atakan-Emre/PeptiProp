"""
Train GATv2 + ESM-2 dual-encoder model for protein–peptide interaction scoring.

Reads canonical pairs, builds residue-level graphs on-the-fly using pre-computed
ESM-2 embeddings, and trains with BCE + pairwise ranking loss.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peptidquantum.models.gnn_esm2 import PeptiPropGNN


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairGraphDataset(Dataset):
    """Loads pre-built PyG graphs from disk for each protein–peptide pair."""

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        graph_dir: Path,
    ):
        self.pairs = pairs_df.reset_index(drop=True)
        self.graph_dir = Path(graph_dir)
        self._cache: Dict[str, Optional[Data]] = {}

    def __len__(self):
        return len(self.pairs)

    def _load_graph(self, complex_id: str, chain_id: str) -> Optional[Data]:
        key = f"{complex_id}__{chain_id}"
        if key in self._cache:
            return self._cache[key]
        path = self.graph_dir / f"{key}.pt"
        if not path.exists():
            self._cache[key] = None
            return None
        g = torch.load(path, weights_only=False)
        self._cache[key] = g
        return g

    def __getitem__(self, idx) -> Optional[Tuple[Data, Data, float, int]]:
        row = self.pairs.iloc[idx]
        prot_cid = str(row["protein_complex_id"])
        prot_ch = str(row["protein_chain_id"])
        pep_cid = str(row["peptide_complex_id"])
        pep_ch = str(row["peptide_chain_id"])
        label = float(row["label"])
        group = int(row.get("group_idx", idx)) if "group_idx" in row.index else idx

        prot_g = self._load_graph(prot_cid, prot_ch)
        pep_g = self._load_graph(pep_cid, pep_ch)

        if prot_g is None or pep_g is None:
            return None

        return prot_g, pep_g, label, group


def collate_pairs(batch):
    """Custom collate that filters None and builds PyG batches."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    prot_list, pep_list, labels, groups = zip(*batch)
    return (
        Batch.from_data_list(list(prot_list)),
        Batch.from_data_list(list(pep_list)),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(groups, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    bce_alpha: float = 0.5,
    margin: float = 0.2,
    use_ranking: bool = True,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

    if not use_ranking:
        return bce

    probs = torch.sigmoid(logits)
    rank_loss = torch.tensor(0.0, device=logits.device)
    n_pairs = 0
    for gid in groups.unique():
        mask = groups == gid
        g_probs = probs[mask]
        g_labels = labels[mask]
        pos_mask = g_labels > 0.5
        neg_mask = g_labels <= 0.5
        if not pos_mask.any() or not neg_mask.any():
            continue
        pos_scores = g_probs[pos_mask]
        neg_scores = g_probs[neg_mask]
        for ps in pos_scores:
            diff = margin - (ps - neg_scores)
            rank_loss = rank_loss + torch.clamp(diff, min=0).sum()
            n_pairs += len(neg_scores)

    if n_pairs > 0:
        rank_loss = rank_loss / n_pairs
    return bce_alpha * bce + (1 - bce_alpha) * rank_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: np.ndarray, scores: np.ndarray, groups: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    preds = (scores >= threshold).astype(int)
    metrics = {
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0,
        "auprc": float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }

    # Ranking metrics
    group_to_idx: Dict[int, List[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_idx[int(g)].append(i)

    mrrs, hit1s, hit3s, hit5s = [], [], [], []
    for idxs in group_to_idx.values():
        g_lab = labels[idxs]
        g_sc = scores[idxs]
        if not g_lab.any() or g_lab.all():
            continue
        order = np.argsort(-g_sc)
        ranked_labels = g_lab[order]
        pos_positions = np.where(ranked_labels > 0.5)[0]
        if len(pos_positions) == 0:
            continue
        rank = pos_positions[0] + 1
        mrrs.append(1.0 / rank)
        hit1s.append(1.0 if rank <= 1 else 0.0)
        hit3s.append(1.0 if rank <= 3 else 0.0)
        hit5s.append(1.0 if rank <= 5 else 0.0)

    metrics["mrr"] = float(np.mean(mrrs)) if mrrs else 0.0
    metrics["hit@1"] = float(np.mean(hit1s)) if hit1s else 0.0
    metrics["hit@3"] = float(np.mean(hit3s)) if hit3s else 0.0
    metrics["hit@5"] = float(np.mean(hit5s)) if hit5s else 0.0
    return metrics


def find_best_threshold(labels: np.ndarray, scores: np.ndarray, metric: str = "mcc") -> float:
    best_t, best_v = 0.5, -1.0
    for t in np.arange(0.1, 0.95, 0.01):
        preds = (scores >= t).astype(int)
        if metric == "mcc":
            v = float(matthews_corrcoef(labels, preds))
        else:
            v = float(f1_score(labels, preds, zero_division=0))
        if v > best_v:
            best_v, best_t = v, t
    return float(best_t)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def assign_group_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Assign contiguous group indices based on protein identity."""
    protein_group = df["protein_complex_id"].astype(str) + "::" + df["protein_chain_id"].astype(str)
    groups = protein_group.unique()
    g_map = {g: i for i, g in enumerate(groups)}
    df = df.copy()
    df["group_idx"] = protein_group.map(g_map).astype(int)
    return df


@torch.no_grad()
def evaluate(
    model: PeptiPropGNN,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_scores, all_groups = [], [], []
    for batch in loader:
        if batch is None:
            continue
        prot_b, pep_b, labels, groups = batch
        prot_b = prot_b.to(device)
        pep_b = pep_b.to(device)
        logits = model(prot_b, pep_b)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(labels.numpy())
        all_scores.append(probs)
        all_groups.append(groups.numpy())
    return np.concatenate(all_labels), np.concatenate(all_scores), np.concatenate(all_groups)


def train(cfg: dict):
    seed = cfg["training"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    print(f"Device: {device}")

    # ---- Data ----
    pairs_dir = Path(cfg["data"]["pairs_dir"])
    graph_dir = Path(cfg["data"]["graph_dir"])

    print("Veri yükleniyor...")
    train_pairs = assign_group_indices(pd.read_parquet(pairs_dir / "train_pairs.parquet"))
    val_pairs = assign_group_indices(pd.read_parquet(pairs_dir / "val_pairs.parquet"))
    test_pairs = assign_group_indices(pd.read_parquet(pairs_dir / "test_pairs.parquet"))

    smoke_limit = cfg.get("_smoke_max_pairs")
    if smoke_limit:
        train_pairs = train_pairs.head(smoke_limit)
        val_pairs = val_pairs.head(smoke_limit)
        test_pairs = test_pairs.head(smoke_limit)

    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    train_ds = PairGraphDataset(train_pairs, graph_dir)
    val_ds = PairGraphDataset(val_pairs, graph_dir)
    test_ds = PairGraphDataset(test_pairs, graph_dir)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_pairs, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_pairs, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate_pairs, num_workers=0)

    # ---- Model ----
    mcfg = cfg["model"]
    model = PeptiPropGNN(
        node_feat_dim=mcfg["node_feat_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_gnn_layers=mcfg["num_gnn_layers"],
        heads=mcfg["heads"],
        mlp_hidden=mcfg["mlp_hidden"],
        dropout=mcfg["dropout"],
        edge_dim=mcfg["edge_dim"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametreleri: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training ----
    best_val_metric = -1.0
    patience_counter = 0
    patience = cfg["training"]["early_stopping_patience"]
    monitor = cfg["evaluation"]["monitor_metric"]
    lcfg = cfg["loss"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        total_batches = len(train_loader)
        for bi, batch in enumerate(train_loader):
            if batch is None:
                continue
            prot_b, pep_b, labels, groups = batch
            prot_b = prot_b.to(device)
            pep_b = pep_b.to(device)
            labels = labels.to(device)
            groups = groups.to(device)

            optimizer.zero_grad()
            logits = model(prot_b, pep_b)
            loss = combined_loss(
                logits, labels, groups,
                bce_alpha=lcfg["bce_alpha"],
                margin=lcfg["margin"],
                use_ranking=lcfg["use_ranking"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            if n_batches % 200 == 0:
                print(f"  batch {n_batches}/{total_batches}  loss={loss.item():.4f}", flush=True)

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # Validation
        val_labels, val_scores, val_groups = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_scores, val_groups)

        val_monitor = val_metrics.get(monitor.replace("val_", ""), val_metrics.get("mrr", 0))
        improved = val_monitor > best_val_metric

        print(
            f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
            f"val_auroc={val_metrics['auroc']:.4f} val_mrr={val_metrics['mrr']:.4f} "
            f"val_hit@3={val_metrics['hit@3']:.4f} | {elapsed:.0f}s"
            f"{' *' if improved else ''}"
        )

        if improved:
            best_val_metric = val_monitor
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Erken durdurma (patience={patience})")
                break

    # ---- Test ----
    print("\nTest değerlendirmesi...")
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))
    test_labels, test_scores, test_groups = evaluate(model, test_loader, device)

    threshold = find_best_threshold(val_labels, val_scores, cfg["evaluation"]["threshold_selection_metric"])
    test_metrics = compute_metrics(test_labels, test_scores, test_groups, threshold)

    print(f"\n{'='*50}")
    print(f"Test Sonuçları (threshold={threshold:.2f}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*50}")

    # Save
    results = {
        "test_metrics": test_metrics,
        "selected_threshold": threshold,
        "epochs_completed": epoch,
        "best_val_monitor_metric": best_val_metric,
        "model_params": total_params,
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSonuçlar: {save_dir / 'metrics.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test with subset")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.smoke:
        cfg["training"]["epochs"] = 2
        cfg["training"]["batch_size"] = 8
        cfg["_smoke_max_pairs"] = 100

    train(cfg)


if __name__ == "__main__":
    main()
