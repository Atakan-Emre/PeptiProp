"""
Train leakage-free post-hoc heads (classical + quantum) on frozen baseline representations.

Purpose:
- Keep baseline architecture unchanged.
- Compare whether a quantum kernel head improves decision quality on the same representation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import train_baseline as tb  # noqa: E402


def compute_threshold_free_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    try:
        auroc = float(roc_auc_score(labels, scores))
    except Exception:
        auroc = 0.0
    try:
        auprc = float(average_precision_score(labels, scores))
    except Exception:
        auprc = 0.0
    try:
        brier = float(brier_score_loss(labels, scores))
    except Exception:
        brier = 0.0
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier_score": brier,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_range": float(np.max(scores) - np.min(scores)),
    }


def compute_binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "positive_predictions": int(preds.sum()),
    }


def compute_ranking_metrics(scores: np.ndarray, labels: np.ndarray, protein_group_ids, top_k=(1, 3, 5)) -> Dict[str, float]:
    by_group = defaultdict(list)
    for idx, gid in enumerate(protein_group_ids):
        by_group[gid].append((float(scores[idx]), int(labels[idx])))

    reciprocal_ranks = []
    hits = {k: [] for k in top_k}
    for entries in by_group.values():
        entries_sorted = sorted(entries, key=lambda item: item[0], reverse=True)
        labels_sorted = [label for _, label in entries_sorted]
        if sum(labels_sorted) <= 0:
            continue
        rank = next((i + 1 for i, label in enumerate(labels_sorted) if label == 1), None)
        if rank is None:
            continue
        reciprocal_ranks.append(1.0 / rank)
        for k in top_k:
            hits[k].append(1.0 if rank <= k else 0.0)

    metrics = {
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }
    for k in top_k:
        metrics[f"hit@{k}"] = float(np.mean(hits[k])) if hits[k] else 0.0
    return metrics


def select_threshold(scores: np.ndarray, labels: np.ndarray, metric: str = "mcc") -> Dict[str, float]:
    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for thr in thresholds:
        row = compute_binary_metrics(scores, labels, float(thr))
        rows.append(row)
    best_f1 = max(rows, key=lambda x: (x["f1"], x["mcc"]))
    best_mcc = max(rows, key=lambda x: (x["mcc"], x["f1"]))
    selected = best_mcc if metric == "mcc" else best_f1
    return {
        "selected_metric": metric,
        "selected_threshold": float(selected["threshold"]),
        "best_f1_threshold": float(best_f1["threshold"]),
        "best_f1": float(best_f1["f1"]),
        "best_mcc_threshold": float(best_mcc["threshold"]),
        "best_mcc": float(best_mcc["mcc"]),
        "sweep": rows,
    }


def stratified_subsample(features: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(labels) <= max_samples:
        return features, labels
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = min(len(pos_idx), max_samples // 2)
    n_neg = min(len(neg_idx), max_samples - n_pos)
    chosen = np.concatenate(
        [
            rng.choice(pos_idx, size=n_pos, replace=False),
            rng.choice(neg_idx, size=n_neg, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return features[chosen], labels[chosen]


def downsample_dataset_pairs(dataset: tb.PeptideProteinDataset, max_samples: int, seed: int) -> None:
    """In-place stratified downsample on dataset.pairs before graph encoding."""
    if len(dataset.pairs) <= max_samples:
        return
    rng = np.random.default_rng(seed)
    positives = dataset.pairs[dataset.pairs["label"] == 1]
    negatives = dataset.pairs[dataset.pairs["label"] == 0]
    n_pos = min(len(positives), max_samples // 2)
    n_neg = min(len(negatives), max_samples - n_pos)
    pos_idx = rng.choice(positives.index.to_numpy(), size=n_pos, replace=False)
    neg_idx = rng.choice(negatives.index.to_numpy(), size=n_neg, replace=False)
    chosen = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(chosen)
    dataset.pairs = dataset.pairs.loc[chosen].reset_index(drop=True)
    dataset.full_pairs = dataset.pairs.copy().reset_index(drop=True)
    dataset.summary = dataset._build_summary()
    dataset._log_summary()


def to_unit_interval(raw_scores: np.ndarray) -> np.ndarray:
    # Smooth monotonic squashing keeps ranking while enabling threshold sweep in [0,1]
    return 1.0 / (1.0 + np.exp(-raw_scores))


def extract_representations(
    model: tb.PeptideProteinModel,
    dataset: tb.PeptideProteinDataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, list]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=tb.collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    model.eval()
    all_repr = []
    all_labels = []
    all_group_ids = []
    with torch.no_grad():
        for batch in loader:
            protein_batch = batch["protein_batch"].to(device)
            peptide_batch = batch["peptide_batch"].to(device)
            pair_features = batch["pair_features"].to(device)
            labels = batch["labels"].cpu().numpy().astype(np.int64)
            group_ids = batch["protein_group_ids"]
            _, rep = model(
                protein_batch,
                peptide_batch,
                pair_features,
                return_repr=True,
            )
            all_repr.append(rep.cpu().numpy())
            all_labels.append(labels)
            all_group_ids.extend(group_ids)
    return np.concatenate(all_repr, axis=0), np.concatenate(all_labels, axis=0), all_group_ids


def evaluate_model(name: str, model, x_val, y_val, g_val, x_test, y_test, g_test) -> Dict[str, object]:
    if hasattr(model, "decision_function"):
        val_raw = model.decision_function(x_val)
        test_raw = model.decision_function(x_test)
    else:
        val_raw = model.predict_proba(x_val)[:, 1]
        test_raw = model.predict_proba(x_test)[:, 1]
    val_scores = to_unit_interval(np.asarray(val_raw, dtype=np.float64))
    test_scores = to_unit_interval(np.asarray(test_raw, dtype=np.float64))

    threshold_info = select_threshold(val_scores, y_val, metric="mcc")
    selected_thr = threshold_info["selected_threshold"]

    val_metrics = compute_threshold_free_metrics(val_scores, y_val)
    val_metrics.update(compute_binary_metrics(val_scores, y_val, selected_thr))
    val_metrics.update(compute_ranking_metrics(val_scores, y_val, g_val))
    test_metrics = compute_threshold_free_metrics(test_scores, y_test)
    test_metrics.update(compute_binary_metrics(test_scores, y_test, selected_thr))
    test_metrics.update(compute_ranking_metrics(test_scores, y_test, g_test))

    return {
        "name": name,
        "threshold_info": threshold_info,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "selected_threshold": float(selected_thr),
        "val_scores": [float(v) for v in val_scores],
        "test_scores": [float(v) for v in test_scores],
        "val_labels": [int(v) for v in y_val],
        "test_labels": [int(v) for v in y_test],
    }


def main():
    parser = argparse.ArgumentParser(description="Train post-hoc quantum head on frozen baseline representations.")
    parser.add_argument("--config", required=True, help="Baseline config used to build datasets/model.")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing baseline checkpoints.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--train-max-samples", type=int, default=1200)
    parser.add_argument("--eval-max-samples", type=int, default=3000)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs_dir = Path(config["data"]["canonical_dir"]) / "pairs"
    canonical_dir = Path(config["data"]["canonical_dir"])
    tb.validate_split_metadata(
        canonical_dir=canonical_dir,
        pairs_dir=pairs_dir,
        splits_dir=config["data"]["splits_dir"],
    )

    train_ds = tb.PeptideProteinDataset(pairs_dir / "train_pairs.parquet", canonical_dir, config, "train")
    val_ds = tb.PeptideProteinDataset(pairs_dir / "val_pairs.parquet", canonical_dir, config, "val")
    test_ds = tb.PeptideProteinDataset(pairs_dir / "test_pairs.parquet", canonical_dir, config, "test")
    downsample_dataset_pairs(train_ds, args.train_max_samples, args.seed)
    downsample_dataset_pairs(val_ds, args.eval_max_samples, args.seed + 1)
    downsample_dataset_pairs(test_ds, args.eval_max_samples, args.seed + 2)

    model = tb.PeptideProteinModel(config).to(device)
    ckpt_dir = Path(args.checkpoint_dir)
    checkpoint = ckpt_dir / "best_model.pt"
    if (ckpt_dir / "metrics.json").exists():
        with open(ckpt_dir / "metrics.json", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("final_checkpoint"):
            checkpoint = ckpt_dir / payload["final_checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))

    batch_size = int(config["training"]["batch_size"])
    x_train, y_train, g_train = extract_representations(model, train_ds, batch_size, device)
    x_val, y_val, g_val = extract_representations(model, val_ds, batch_size, device)
    x_test, y_test, g_test = extract_representations(model, test_ds, batch_size, device)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train)
    x_val_std = scaler.transform(x_val)
    x_test_std = scaler.transform(x_test)

    n_components = int(min(args.n_qubits, x_train_std.shape[1]))
    pca = PCA(n_components=n_components, random_state=args.seed)
    x_train_pca = pca.fit_transform(x_train_std)
    x_val_pca = pca.transform(x_val_std)
    x_test_pca = pca.transform(x_test_std)

    results = []

    lr = LogisticRegression(max_iter=2000, random_state=args.seed)
    lr.fit(x_train_pca, y_train)
    results.append(evaluate_model("logistic_head", lr, x_val_pca, y_val, g_val, x_test_pca, y_test, g_test))

    svm = SVC(kernel="rbf", probability=False, random_state=args.seed)
    svm.fit(x_train_pca, y_train)
    results.append(evaluate_model("rbf_svm_head", svm, x_val_pca, y_val, g_val, x_test_pca, y_test, g_test))

    feature_map = ZZFeatureMap(feature_dimension=n_components, reps=2, entanglement="linear")
    qkernel = FidelityQuantumKernel(feature_map=feature_map)
    qsvc = QSVC(quantum_kernel=qkernel)
    qsvc.fit(x_train_pca, y_train)
    results.append(evaluate_model("quantum_qsvc_head", qsvc, x_val_pca, y_val, g_val, x_test_pca, y_test, g_test))

    best = max(
        results,
        key=lambda row: (
            row["val_metrics"]["mcc"],
            row["val_metrics"]["auroc"],
            row["test_metrics"]["mcc"],
        ),
    )

    output = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "n_qubits": n_components,
        "sample_counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "results": results,
        "best_model": best["name"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved quantum head comparison to {output_path}")
    print(f"Best head: {best['name']}")
    print(f"Best validation MCC: {best['val_metrics']['mcc']:.4f}")
    print(f"Best test MCC: {best['test_metrics']['mcc']:.4f}")


if __name__ == "__main__":
    main()
