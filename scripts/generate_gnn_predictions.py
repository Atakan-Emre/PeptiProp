"""
Generate top-ranked predictions and plots from the trained GNN+ESM-2 model.
Outputs: top_ranked_examples.json, ROC/PR curves, histograms, confusion matrix.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from peptidquantum.models.gnn_esm2 import PeptiPropGNN


class PairGraphDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_df, graph_dir):
        self.pairs = pairs_df.reset_index(drop=True)
        self.graph_dir = Path(graph_dir)
        self._cache = {}

    def __len__(self):
        return len(self.pairs)

    def _load_graph(self, cid, ch):
        key = f"{cid}__{ch}"
        if key in self._cache:
            return self._cache[key]
        path = self.graph_dir / f"{key}.pt"
        if not path.exists():
            self._cache[key] = None
            return None
        g = torch.load(path, weights_only=False)
        self._cache[key] = g
        return g

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        prot_g = self._load_graph(str(row["protein_complex_id"]), str(row["protein_chain_id"]))
        pep_g = self._load_graph(str(row["peptide_complex_id"]), str(row["peptide_chain_id"]))
        if prot_g is None or pep_g is None:
            return None
        return prot_g, pep_g, idx


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    prot_list, pep_list, idxs = zip(*batch)
    return Batch.from_data_list(list(prot_list)), Batch.from_data_list(list(pep_list)), list(idxs)


def main():
    out_dir = Path("outputs/training/peptiprop_v0_2_gnn_esm2")
    graph_dir = Path("data/graphs")
    pairs_dir = Path("data/canonical/pairs")

    device = torch.device("cpu")
    model = PeptiPropGNN(node_feat_dim=326, hidden_dim=128, num_gnn_layers=4, heads=4,
                         mlp_hidden=256, dropout=0.1, edge_dim=4)
    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device, weights_only=True))
    model.eval()
    print("Model yüklendi.")

    test_pairs = pd.read_parquet(pairs_dir / "test_pairs.parquet")
    ds = PairGraphDataset(test_pairs, graph_dir)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0)

    all_scores = np.full(len(test_pairs), np.nan)
    print(f"Test çiftleri: {len(test_pairs)}")

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            prot_b, pep_b, idxs = batch
            logits = model(prot_b, pep_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i, idx in enumerate(idxs):
                all_scores[idx] = probs[i]

    valid_mask = ~np.isnan(all_scores)
    print(f"Geçerli tahmin: {valid_mask.sum()}/{len(all_scores)}")

    labels = test_pairs["label"].values[valid_mask]
    scores = all_scores[valid_mask]

    # --- Plots ---
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"GNN+ESM-2 (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test Set"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "roc_curve.png", dpi=150); plt.close()

    # PR curve
    prec, rec, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, linewidth=2, label=f"GNN+ESM-2 (AP={ap:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Curve — Test Set"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "pr_curve.png", dpi=150); plt.close()

    # Score histogram
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(scores[labels == 1], bins=50, alpha=0.7, label="Pozitif", color="#2196F3")
    ax.hist(scores[labels == 0], bins=50, alpha=0.7, label="Negatif", color="#f44336")
    ax.set_xlabel("Skor"); ax.set_ylabel("Frekans")
    ax.set_title("Test Skor Dağılımı"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "score_histogram_pos_neg.png", dpi=150); plt.close()

    # Confusion matrix
    threshold = 0.21
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val:,}", ha="center", va="center",
                color="white" if val > cm.max() * 0.5 else "black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negatif", "Pozitif"]); ax.set_yticklabels(["Negatif", "Pozitif"])
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek"); ax.set_title("Confusion Matrix")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=150); plt.close()

    # --- Top-ranked examples ---
    test_pairs_scored = test_pairs.copy()
    test_pairs_scored["score"] = all_scores

    protein_group = test_pairs_scored["protein_complex_id"].astype(str) + "::" + test_pairs_scored["protein_chain_id"].astype(str)
    test_pairs_scored["protein_group"] = protein_group

    top_ranked = []
    best_tp, worst_fp = None, None

    for gname, gdf in test_pairs_scored.groupby("protein_group"):
        gdf = gdf.dropna(subset=["score"]).sort_values("score", ascending=False).reset_index(drop=True)
        if gdf.empty:
            continue
        for rank, (_, row) in enumerate(gdf.iterrows(), 1):
            entry = {
                "pair_id": str(row.get("pair_id", "")),
                "protein_complex_id": str(row["protein_complex_id"]),
                "protein_chain_id": str(row["protein_chain_id"]),
                "peptide_complex_id": str(row["peptide_complex_id"]),
                "peptide_chain_id": str(row["peptide_chain_id"]),
                "score": float(row["score"]),
                "label_eval": int(row["label"]),
                "label": int(row["label"]),
                "rank": rank,
                "candidate_size": len(gdf),
            }
            if rank == 1:
                top_ranked.append(entry)
            if entry["label_eval"] == 1 and entry["rank"] == 1:
                if best_tp is None or entry["score"] > best_tp["score"]:
                    best_tp = entry
            if entry["label_eval"] == 0 and entry["rank"] == 1:
                if worst_fp is None or entry["score"] > worst_fp["score"]:
                    worst_fp = entry

    result = {
        "best_true_positive": best_tp,
        "worst_false_positive": worst_fp,
        "top_ranked_candidates_preview": top_ranked[:200],
    }
    with open(out_dir / "top_ranked_examples.json", "w") as f:
        json.dump(result, f, indent=2)

    # --- Ranking metrics ---
    group_metrics = defaultdict(list)
    for gname, gdf in test_pairs_scored.groupby("protein_group"):
        gdf = gdf.dropna(subset=["score"]).sort_values("score", ascending=False)
        ranked_labels = gdf["label"].values
        pos_positions = np.where(ranked_labels > 0.5)[0]
        if len(pos_positions) == 0:
            continue
        rank = pos_positions[0] + 1
        group_metrics["mrr"].append(1.0 / rank)
        group_metrics["hit@1"].append(1.0 if rank <= 1 else 0.0)
        group_metrics["hit@3"].append(1.0 if rank <= 3 else 0.0)
        group_metrics["hit@5"].append(1.0 if rank <= 5 else 0.0)

    ranking = {k: float(np.mean(v)) for k, v in group_metrics.items()}
    ranking["groups_evaluated"] = len(group_metrics["mrr"])
    with open(out_dir / "ranking_metrics.json", "w") as f:
        json.dump(ranking, f, indent=2)

    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"AP: {ap:.4f}")
    print(f"MRR: {ranking['mrr']:.4f}")
    print(f"Hit@1: {ranking['hit@1']:.4f}")
    print(f"Hit@3: {ranking['hit@3']:.4f}")
    print(f"Hit@5: {ranking['hit@5']:.4f}")
    print(f"\nÇıktılar: {out_dir}")


if __name__ == "__main__":
    main()
