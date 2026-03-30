"""
Post-process a trained GNN+ESM-2 model into the repo's final evaluation surface.

Artifacts written under outputs/training/peptiprop_v0_2_gnn_esm2/:
  - metrics.json
  - ranking_metrics.json
  - best_thresholds.json
  - pair_data_report.json
  - candidate_set_report.json
  - calibration_metrics.json
  - threshold_vs_f1_table.csv
  - test_summary.txt
  - test_topk_candidates.csv
  - test_topk_positive_hits.csv
  - top_ranked_examples.json
  - roc_curve.png
  - pr_curve.png
  - confusion_matrix.png
  - score_histogram_pos_neg.png
  - validation_score_histogram_pos_neg.png
  - validation_threshold_sweep.png
  - calibration_curve.png

Also writes a visualization sample list:
  - data/reports/audit_gallery_propedia/sample_list_top_ranked_gnn_v0_2.txt
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from peptidquantum.models.gnn_esm2 import PeptiPropGNN


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "training" / "peptiprop_v0_2_gnn_esm2"
GRAPH_DIR = ROOT / "data" / "graphs"
PAIRS_DIR = ROOT / "data" / "canonical" / "pairs"
AUDIT_DIR = ROOT / "data" / "reports" / "audit_gallery_propedia"
MODEL_PATH = OUT_DIR / "best_model.pt"
EXISTING_METRICS_PATH = OUT_DIR / "metrics.json"


class PairGraphDataset(torch.utils.data.Dataset):
    def __init__(self, pairs_df: pd.DataFrame, graph_dir: Path):
        self.pairs = pairs_df.reset_index(drop=True)
        self.graph_dir = Path(graph_dir)
        self._cache: Dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_graph(self, complex_id: str, chain_id: str):
        key = f"{complex_id}__{chain_id}"
        if key in self._cache:
            return self._cache[key]
        path = self.graph_dir / f"{key}.pt"
        if not path.exists():
            self._cache[key] = None
            return None
        graph = torch.load(path, weights_only=False)
        self._cache[key] = graph
        return graph

    def __getitem__(self, idx: int):
        row = self.pairs.iloc[idx]
        prot_g = self._load_graph(str(row["protein_complex_id"]), str(row["protein_chain_id"]))
        pep_g = self._load_graph(str(row["peptide_complex_id"]), str(row["peptide_chain_id"]))
        if prot_g is None or pep_g is None:
            return None
        return prot_g, pep_g, idx


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    prot_list, pep_list, idxs = zip(*batch)
    return Batch.from_data_list(list(prot_list)), Batch.from_data_list(list(pep_list)), list(idxs)


def assign_group_indices(df: pd.DataFrame) -> pd.DataFrame:
    protein_group = df["protein_complex_id"].astype(str) + "::" + df["protein_chain_id"].astype(str)
    group_ids = {name: idx for idx, name in enumerate(protein_group.unique())}
    out = df.copy()
    out["protein_group"] = protein_group
    out["group_idx"] = protein_group.map(group_ids).astype(int)
    return out


def evaluate_scores(model: PeptiPropGNN, pairs_df: pd.DataFrame, graph_dir: Path, batch_size: int = 64) -> np.ndarray:
    dataset = PairGraphDataset(pairs_df, graph_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    scores = np.full(len(pairs_df), np.nan, dtype=np.float32)
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            prot_b, pep_b, idxs = batch
            logits = model(prot_b, pep_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i, idx in enumerate(idxs):
                scores[idx] = float(probs[i])
    return scores


def compute_classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0,
        "auprc": float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "positive_predictions": int(preds.sum()),
        "negative_predictions": int((preds == 0).sum()),
    }


def compute_ranking_metrics(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    group_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, group_id in enumerate(groups):
        group_to_indices[int(group_id)].append(idx)

    mrr_values: List[float] = []
    hit1_values: List[float] = []
    hit3_values: List[float] = []
    hit5_values: List[float] = []

    for idxs in group_to_indices.values():
        group_labels = labels[idxs]
        group_scores = scores[idxs]
        if not (group_labels > 0.5).any() or (group_labels > 0.5).all():
            continue
        order = np.argsort(-group_scores)
        ranked_labels = group_labels[order]
        pos_positions = np.where(ranked_labels > 0.5)[0]
        if len(pos_positions) == 0:
            continue
        rank = int(pos_positions[0]) + 1
        mrr_values.append(1.0 / rank)
        hit1_values.append(1.0 if rank <= 1 else 0.0)
        hit3_values.append(1.0 if rank <= 3 else 0.0)
        hit5_values.append(1.0 if rank <= 5 else 0.0)

    return {
        "mrr": float(np.mean(mrr_values)) if mrr_values else 0.0,
        "hit@1": float(np.mean(hit1_values)) if hit1_values else 0.0,
        "hit@3": float(np.mean(hit3_values)) if hit3_values else 0.0,
        "hit@5": float(np.mean(hit5_values)) if hit5_values else 0.0,
        "groups_evaluated": int(len(mrr_values)),
    }


def compute_candidate_group_integrity(df: pd.DataFrame) -> Dict[str, int]:
    grouped = df.groupby(["protein_complex_id", "protein_chain_id"])
    groups_without_positive = 0
    groups_without_negative = 0
    valid_groups = 0
    for _, group_df in grouped:
        has_positive = bool((group_df["label"] == 1).any())
        has_negative = bool((group_df["label"] == 0).any())
        if has_positive and has_negative:
            valid_groups += 1
        if not has_positive:
            groups_without_positive += 1
        if not has_negative:
            groups_without_negative += 1
    return {
        "total_groups": int(len(grouped)),
        "valid_groups": int(valid_groups),
        "groups_without_positive": int(groups_without_positive),
        "groups_without_negative": int(groups_without_negative),
    }


def compute_threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    best_f1 = (-1.0, 0.5)
    best_mcc = (-2.0, 0.5)

    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        preds = (scores >= threshold).astype(int)
        f1 = float(f1_score(labels, preds, zero_division=0))
        mcc = float(matthews_corrcoef(labels, preds))
        precision = float(precision_score(labels, preds, zero_division=0))
        recall = float(recall_score(labels, preds, zero_division=0))
        positive_predictions = int(preds.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "f1": f1,
                "mcc": mcc,
                "precision": precision,
                "recall": recall,
                "positive_predictions": positive_predictions,
            }
        )
        if f1 > best_f1[0]:
            best_f1 = (f1, float(threshold))
        if mcc > best_mcc[0]:
            best_mcc = (mcc, float(threshold))

    sweep_df = pd.DataFrame(rows)
    thresholds = {
        "best_f1_threshold": float(best_f1[1]),
        "best_f1": float(best_f1[0]),
        "best_mcc_threshold": float(best_mcc[1]),
        "best_mcc": float(best_mcc[0]),
        "selected_threshold": float(best_mcc[1]),
        "selection_metric": "mcc",
    }
    return sweep_df, thresholds


def compute_calibration_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "brier_score": float(brier_score_loss(labels, scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "positive_rate": float(np.mean(labels)),
    }


def plot_roc(labels: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"GNN+ESM-2 (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Test Set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr(labels: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, label=f"GNN+ESM-2 (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve - Test Set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_histogram(labels: np.ndarray, scores: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(scores[labels == 1], bins=50, alpha=0.7, label="Positive", color="#2196F3")
    ax.hist(scores[labels == 0], bins=50, alpha=0.7, label="Negative", color="#f44336")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(labels: np.ndarray, scores: np.ndarray, threshold: float, out_path: Path) -> None:
    preds = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    for (i, j), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() * 0.5 else "black"
        ax.text(j, i, f"{value:,}", ha="center", va="center", color=color, fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix @ threshold={threshold:.2f}")
    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_threshold_sweep(sweep_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1", linewidth=2)
    ax.plot(sweep_df["threshold"], sweep_df["mcc"], label="MCC", linewidth=2)
    ax.set_xlabel("Validation threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Validation Threshold Sweep")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration(val_labels: np.ndarray, val_scores: np.ndarray, test_labels: np.ndarray, test_scores: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, labels, scores, color in [
        ("Validation", val_labels, val_scores, "#1f77b4"),
        ("Test", test_labels, test_scores, "#d62728"),
    ]:
        prob_true, prob_pred = calibration_curve(labels, scores, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=name, color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration Curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_topk_tables(scored_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    topk_rows: List[Dict[str, object]] = []
    preview_rows: List[Dict[str, object]] = []
    best_true_positive = None
    worst_false_positive = None

    for _, group_df in scored_df.groupby("protein_group"):
        ordered = group_df.dropna(subset=["score"]).sort_values("score", ascending=False).reset_index(drop=True)
        if ordered.empty:
            continue
        candidate_size = int(len(ordered))
        for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
            entry = {
                "pair_id": str(row["pair_id"]),
                "protein_complex_id": str(row["protein_complex_id"]),
                "protein_chain_id": str(row["protein_chain_id"]),
                "peptide_complex_id": str(row["peptide_complex_id"]),
                "peptide_chain_id": str(row["peptide_chain_id"]),
                "score": float(row["score"]),
                "label": int(row["label"]),
                "rank": int(rank),
                "candidate_size": candidate_size,
            }
            if rank <= 5:
                topk_rows.append(entry)
            if rank == 1:
                preview_rows.append(entry)
                if entry["label"] == 1 and (best_true_positive is None or entry["score"] > best_true_positive["score"]):
                    best_true_positive = entry
                if entry["label"] == 0 and (worst_false_positive is None or entry["score"] > worst_false_positive["score"]):
                    worst_false_positive = entry

    topk_df = pd.DataFrame(topk_rows)
    topk_positive_hits = topk_df[topk_df["label"] == 1].copy()
    summary = {
        "best_true_positive": best_true_positive,
        "worst_false_positive": worst_false_positive,
        "top_ranked_candidates_preview": preview_rows[:200],
    }
    return topk_df, topk_positive_hits, summary


def write_test_summary(
    out_path: Path,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    val_rank: Dict[str, float],
    test_rank: Dict[str, float],
    thresholds: Dict[str, float],
    calibration: Dict[str, Dict[str, float]],
) -> None:
    lines = [
        "PeptiProp GNN+ESM-2 Final Evaluation Summary",
        "=" * 60,
        "",
        f"selected_threshold: {thresholds['selected_threshold']:.2f}",
        f"best_f1_threshold: {thresholds['best_f1_threshold']:.2f}",
        f"best_mcc_threshold: {thresholds['best_mcc_threshold']:.2f}",
        "",
        "[validation_metrics_at_selected_threshold]",
    ]
    for key in ("auroc", "auprc", "f1", "mcc", "precision", "recall"):
        lines.append(f"{key}: {val_metrics[key]:.6f}")
    lines.extend(["", "[validation_ranking_metrics]"])
    for key in ("mrr", "hit@1", "hit@3", "hit@5", "groups_evaluated"):
        value = val_rank[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    lines.extend(["", "[test_metrics]"])
    for key in ("auroc", "auprc", "f1", "mcc", "precision", "recall"):
        lines.append(f"{key}: {test_metrics[key]:.6f}")
    lines.extend(["", "[test_ranking_metrics]"])
    for key in ("mrr", "hit@1", "hit@3", "hit@5", "groups_evaluated"):
        value = test_rank[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    lines.extend(["", "[calibration_test]"])
    for key in ("brier_score", "score_min", "score_max", "score_mean", "score_std", "positive_rate"):
        lines.append(f"{key}: {calibration['test'][key]:.6f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_visualization_sample_list(topk_positive_hits: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = (
        topk_positive_hits[topk_positive_hits["rank"] == 1]
        .sort_values("score", ascending=False)["protein_complex_id"]
        .drop_duplicates()
        .head(100)
        .tolist()
    )
    out_path.write_text("\n".join(sample_ids) + ("\n" if sample_ids else ""), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.is_file():
        raise SystemExit(f"[FAIL] Model file not found: {MODEL_PATH}")

    existing_metrics = {}
    if EXISTING_METRICS_PATH.is_file():
        existing_metrics = json.loads(EXISTING_METRICS_PATH.read_text(encoding="utf-8"))

    device = torch.device("cpu")
    model = PeptiPropGNN(
        node_feat_dim=326,
        hidden_dim=128,
        num_gnn_layers=4,
        heads=4,
        mlp_hidden=256,
        dropout=0.1,
        edge_dim=4,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"[OK] Loaded model: {MODEL_PATH}")

    train_pairs = assign_group_indices(pd.read_parquet(PAIRS_DIR / "train_pairs.parquet"))
    val_pairs = assign_group_indices(pd.read_parquet(PAIRS_DIR / "val_pairs.parquet"))
    test_pairs = assign_group_indices(pd.read_parquet(PAIRS_DIR / "test_pairs.parquet"))

    print(f"[INFO] train_pairs={len(train_pairs)} val_pairs={len(val_pairs)} test_pairs={len(test_pairs)}")
    val_scores = evaluate_scores(model, val_pairs, GRAPH_DIR)
    test_scores = evaluate_scores(model, test_pairs, GRAPH_DIR)

    val_mask = ~np.isnan(val_scores)
    test_mask = ~np.isnan(test_scores)
    print(f"[INFO] valid_predictions val={int(val_mask.sum())}/{len(val_pairs)} test={int(test_mask.sum())}/{len(test_pairs)}")

    val_valid = val_pairs.loc[val_mask].copy()
    test_valid = test_pairs.loc[test_mask].copy()
    val_valid["score"] = val_scores[val_mask]
    test_valid["score"] = test_scores[test_mask]

    val_labels = val_valid["label"].to_numpy(dtype=np.int32)
    val_scores_valid = val_valid["score"].to_numpy(dtype=np.float32)
    val_groups = val_valid["group_idx"].to_numpy(dtype=np.int32)

    test_labels = test_valid["label"].to_numpy(dtype=np.int32)
    test_scores_valid = test_valid["score"].to_numpy(dtype=np.float32)
    test_groups = test_valid["group_idx"].to_numpy(dtype=np.int32)

    sweep_df, thresholds = compute_threshold_sweep(val_labels, val_scores_valid)
    selected_threshold = float(thresholds["selected_threshold"])

    val_metrics = compute_classification_metrics(val_labels, val_scores_valid, selected_threshold)
    test_metrics = compute_classification_metrics(test_labels, test_scores_valid, selected_threshold)
    val_rank = compute_ranking_metrics(val_labels, val_scores_valid, val_groups)
    test_rank = compute_ranking_metrics(test_labels, test_scores_valid, test_groups)

    candidate_group_integrity = {
        "train": compute_candidate_group_integrity(train_pairs),
        "val": compute_candidate_group_integrity(val_pairs),
        "test": compute_candidate_group_integrity(test_pairs),
    }

    calibration_metrics = {
        "validation": compute_calibration_metrics(val_labels, val_scores_valid),
        "test": compute_calibration_metrics(test_labels, test_scores_valid),
    }

    plot_roc(test_labels, test_scores_valid, OUT_DIR / "roc_curve.png")
    plot_pr(test_labels, test_scores_valid, OUT_DIR / "pr_curve.png")
    plot_histogram(test_labels, test_scores_valid, OUT_DIR / "score_histogram_pos_neg.png", "Test Score Distribution")
    plot_histogram(
        val_labels,
        val_scores_valid,
        OUT_DIR / "validation_score_histogram_pos_neg.png",
        "Validation Score Distribution",
    )
    plot_confusion(test_labels, test_scores_valid, selected_threshold, OUT_DIR / "confusion_matrix.png")
    plot_threshold_sweep(sweep_df, OUT_DIR / "validation_threshold_sweep.png")
    plot_calibration(
        val_labels,
        val_scores_valid,
        test_labels,
        test_scores_valid,
        OUT_DIR / "calibration_curve.png",
    )
    sweep_df.to_csv(OUT_DIR / "threshold_vs_f1_table.csv", index=False)

    best_thresholds_payload = {
        "validation": thresholds,
        "selected_threshold": selected_threshold,
        "selection_metric": "mcc",
    }
    (OUT_DIR / "best_thresholds.json").write_text(
        json.dumps(best_thresholds_payload, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "calibration_metrics.json").write_text(
        json.dumps(calibration_metrics, indent=2), encoding="utf-8"
    )

    shutil.copy2(PAIRS_DIR / "pair_data_report.json", OUT_DIR / "pair_data_report.json")
    shutil.copy2(PAIRS_DIR / "candidate_set_report.json", OUT_DIR / "candidate_set_report.json")

    test_topk_candidates, test_topk_positive_hits, top_ranked_examples = build_topk_tables(test_valid)
    test_topk_candidates.to_csv(OUT_DIR / "test_topk_candidates.csv", index=False)
    test_topk_positive_hits.to_csv(OUT_DIR / "test_topk_positive_hits.csv", index=False)
    (OUT_DIR / "top_ranked_examples.json").write_text(
        json.dumps(top_ranked_examples, indent=2), encoding="utf-8"
    )

    sample_list_path = AUDIT_DIR / "sample_list_top_ranked_gnn_v0_2.txt"
    write_visualization_sample_list(test_topk_positive_hits, sample_list_path)

    write_test_summary(
        OUT_DIR / "test_summary.txt",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_rank=val_rank,
        test_rank=test_rank,
        thresholds=thresholds,
        calibration=calibration_metrics,
    )

    metrics_payload = {
        "validation_metrics_at_selected_threshold": val_metrics,
        "test_metrics": test_metrics,
        "val_ranking_metrics": val_rank,
        "test_ranking_metrics": test_rank,
        "selected_threshold": selected_threshold,
        "threshold_selection_metric": "mcc",
        "best_val_monitor_metric": float(existing_metrics.get("best_val_monitor_metric", val_rank["mrr"])),
        "epochs_completed": int(existing_metrics.get("epochs_completed", 0) or 0),
        "candidate_group_integrity": candidate_group_integrity,
        "model_params": int(existing_metrics.get("model_params", 0) or 0),
        "test_brier": calibration_metrics["test"]["brier_score"],
        "calibration": calibration_metrics["test"],
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    ranking_payload = {"validation": val_rank, "test": test_rank}
    (OUT_DIR / "ranking_metrics.json").write_text(json.dumps(ranking_payload, indent=2), encoding="utf-8")

    # Keep the legacy figures/ directory populated for Pages fallback code paths.
    figures_dir = OUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name in ("roc_curve.png", "pr_curve.png", "confusion_matrix.png", "score_histogram_pos_neg.png"):
        shutil.copy2(OUT_DIR / name, figures_dir / name)

    print("[OK] GNN evaluation surface refreshed")
    print(f"      output_dir={OUT_DIR}")
    print(f"      sample_list={sample_list_path}")
    print(
        "[METRICS] "
        f"val_mrr={val_rank['mrr']:.4f} "
        f"test_mrr={test_rank['mrr']:.4f} "
        f"test_auroc={test_metrics['auroc']:.4f} "
        f"test_f1={test_metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
