"""
Train interaction scoring model with Apple MLX on pre-exported dense features.

Designed for Apple Silicon (M-series) as a parallel backend to the PyTorch
graph pipeline, without modifying the active CUDA training path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
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
import matplotlib.pyplot as plt


def import_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        return mx, nn, optim
    except Exception as exc:
        raise RuntimeError(
            "MLX is required for this script. On Mac M4 install with: pip install -r mlx/requirements-m4.txt"
        ) from exc


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    np.random.seed(seed)


def load_split_npz(feature_dir: Path, split: str) -> Dict[str, np.ndarray]:
    payload = np.load(feature_dir / f"{split}_mlx_features.npz", allow_pickle=True)
    return {k: payload[k] for k in payload.files}


def standardize_features(
    train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std, mean, std


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ranking_metrics(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    group_to_indices: Dict[int, np.ndarray] = {}
    for gid in np.unique(groups):
        group_to_indices[int(gid)] = np.where(groups == gid)[0]

    reciprocal_ranks = []
    hit1 = []
    hit3 = []
    hit5 = []

    for idxs in group_to_indices.values():
        g_labels = labels[idxs]
        g_scores = scores[idxs]
        pos = np.where(g_labels > 0.5)[0]
        if len(pos) == 0:
            continue
        # Candidate set spec expects one positive, but we robustly take best positive rank.
        order = np.argsort(-g_scores)
        ranked_pos = [int(np.where(order == p)[0][0]) + 1 for p in pos]
        rank = min(ranked_pos)
        reciprocal_ranks.append(1.0 / rank)
        hit1.append(1.0 if rank <= 1 else 0.0)
        hit3.append(1.0 if rank <= 3 else 0.0)
        hit5.append(1.0 if rank <= 5 else 0.0)

    if not reciprocal_ranks:
        return {"mrr": 0.0, "hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0}
    return {
        "mrr": float(np.mean(reciprocal_ranks)),
        "hit@1": float(np.mean(hit1)),
        "hit@3": float(np.mean(hit3)),
        "hit@5": float(np.mean(hit5)),
    }


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(np.int32)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> dict:
    rows = []
    best_f1 = {"threshold": 0.5, "value": -1.0}
    best_mcc = {"threshold": 0.5, "value": -2.0}

    for thr in np.arange(0.05, 0.951, 0.01):
        preds = (scores >= thr).astype(np.int32)
        f1 = float(f1_score(labels, preds, zero_division=0))
        mcc = float(matthews_corrcoef(labels, preds))
        rows.append({"threshold": float(thr), "f1": f1, "mcc": mcc})
        if f1 > best_f1["value"]:
            best_f1 = {"threshold": float(thr), "value": f1}
        if mcc > best_mcc["value"]:
            best_mcc = {"threshold": float(thr), "value": mcc}

    return {"sweep": rows, "best_f1": best_f1, "best_mcc": best_mcc}


def save_threshold_plot(sweep_rows: list[dict], out_path: Path):
    df = pd.DataFrame(sweep_rows)
    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.plot(df["threshold"], df["mcc"], label="MCC")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Validation Threshold Sweep")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_score_hist(labels: np.ndarray, scores: np.ndarray, out_path: Path, title: str):
    pos_scores = scores[labels > 0.5]
    neg_scores = scores[labels <= 0.5]
    plt.figure(figsize=(8, 5))
    plt.hist(neg_scores, bins=40, alpha=0.6, label="negative")
    plt.hist(pos_scores, bins=40, alpha=0.6, label="positive")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_roc(labels: np.ndarray, scores: np.ndarray, out_path: Path):
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_pr(labels: np.ndarray, scores: np.ndarray, out_path: Path):
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_confusion(labels: np.ndarray, scores: np.ndarray, threshold: float, out_path: Path):
    preds = (scores >= threshold).astype(np.int32)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix @ {threshold:.2f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xticks([0, 1], ["neg", "pos"])
    plt.yticks([0, 1], ["neg", "pos"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_calibration(labels: np.ndarray, scores: np.ndarray, curve_path: Path, metrics_path: Path):
    prob_true, prob_pred = calibration_curve(labels, scores, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=180)
    plt.close()

    payload = {
        "brier_score": float(brier_score_loss(labels, scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class MLXScoringMLP:
    def __init__(self, nn, input_dim: int, hidden_dim: int, num_layers: int):
        class _Model(nn.Module):
            def __init__(self, in_dim: int, hid_dim: int, layers: int):
                super().__init__()
                self.layers = []
                prev = in_dim
                for _ in range(max(1, layers)):
                    self.layers.append(nn.Linear(prev, hid_dim))
                    prev = hid_dim
                self.out = nn.Linear(prev, 1)

            def __call__(self, x):
                h = x
                for layer in self.layers:
                    h = nn.silu(layer(h))
                return self.out(h)

        self.model = _Model(input_dim, hidden_dim, num_layers)


def pairwise_ranking_loss(mx, logits, labels, groups, margin: float):
    logits = logits.reshape((-1,))
    labels = labels.reshape((-1,))
    groups = groups.reshape((-1,))

    pos_mask = labels > 0.5
    neg_mask = labels < 0.5
    group_eq = mx.equal(mx.expand_dims(groups, 1), mx.expand_dims(groups, 0))
    pair_mask = mx.logical_and(mx.expand_dims(pos_mask, 1), mx.expand_dims(neg_mask, 0))
    pair_mask = mx.logical_and(pair_mask, group_eq)
    pair_mask_f = pair_mask.astype(mx.float32)

    diff = mx.expand_dims(logits, 1) - mx.expand_dims(logits, 0)
    rank_losses = mx.maximum(0.0, margin - diff) * pair_mask_f
    denom = mx.maximum(mx.sum(pair_mask_f), 1.0)
    return mx.sum(rank_losses) / denom


def main():
    parser = argparse.ArgumentParser(description="Train MLX scoring model from exported NPZ features")
    parser.add_argument("--config", type=Path, required=True, help="Path to MLX config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["training"].get("seed", 42)))
    mx, nn, optim = import_mlx()

    feature_dir = Path(cfg["data"]["feature_dir"])
    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    train = load_split_npz(feature_dir, "train")
    val = load_split_npz(feature_dir, "val")
    test = load_split_npz(feature_dir, "test")

    x_train, x_val, x_test, scaler_mean, scaler_std = standardize_features(train["x"], val["x"], test["x"])
    y_train = train["y"].astype(np.float32)
    y_val = val["y"].astype(np.float32)
    y_test = test["y"].astype(np.float32)
    g_train = train["group_idx"].astype(np.int32)
    g_val = val["group_idx"].astype(np.int32)
    g_test = test["group_idx"].astype(np.int32)

    input_dim = x_train.shape[1]
    model_wrapper = MLXScoringMLP(
        nn=nn,
        input_dim=input_dim,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
    )
    model = model_wrapper.model
    mx.eval(model.parameters())

    optimizer = optim.AdamW(
        learning_rate=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )

    batch_size = int(cfg["training"]["batch_size"])
    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"]["early_stopping_patience"])
    use_ranking = bool(cfg["loss"].get("use_ranking", True))
    margin = float(cfg["loss"].get("margin", 0.2))
    alpha = float(cfg["loss"].get("bce_alpha", 0.5))
    monitor_metric = cfg["evaluation"].get("monitor_metric", "val_mrr")
    threshold_metric = cfg["evaluation"].get("threshold_selection_metric", "mcc")

    pos_count = float(np.sum(y_train > 0.5))
    neg_count = float(np.sum(y_train <= 0.5))
    pos_weight = neg_count / max(pos_count, 1.0)

    def total_loss_fn(model_obj, xb, yb, gb):
        logits = model_obj(xb).reshape((-1,))
        probs = nn.sigmoid(logits)
        probs = mx.clip(probs, 1e-6, 1.0 - 1e-6)

        bce = -(
            pos_weight * yb * mx.log(probs)
            + (1.0 - yb) * mx.log(1.0 - probs)
        )
        bce = mx.mean(bce)
        if use_ranking:
            rank = pairwise_ranking_loss(mx, logits, yb, gb, margin=margin)
            return rank + alpha * bce
        return bce

    loss_and_grad_fn = nn.value_and_grad(model, total_loss_fn)

    def predict_scores(x_np: np.ndarray) -> np.ndarray:
        scores = []
        for start in range(0, len(x_np), 4096):
            batch = mx.array(x_np[start : start + 4096], dtype=mx.float32)
            logits = model(batch).reshape((-1,))
            probs = nn.sigmoid(logits)
            mx.eval(probs)
            scores.append(np.array(probs, dtype=np.float32))
        if not scores:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(scores, axis=0)

    train_log = []
    best_val_metric = -1e9
    patience_counter = 0
    best_weight_path = save_dir / "best_model_mlx.npz"

    for epoch in range(1, epochs + 1):
        order = np.random.permutation(len(x_train))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = mx.array(x_train[idx], dtype=mx.float32)
            yb = mx.array(y_train[idx], dtype=mx.float32)
            gb = mx.array(g_train[idx], dtype=mx.int32)

            loss, grads = loss_and_grad_fn(model, xb, yb, gb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            epoch_losses.append(float(np.array(loss)))

        val_scores = predict_scores(x_val)
        sweep = threshold_sweep(y_val, val_scores)
        selected_threshold = (
            sweep["best_mcc"]["threshold"] if threshold_metric == "mcc" else sweep["best_f1"]["threshold"]
        )
        val_cls = classification_metrics(y_val, val_scores, selected_threshold)
        val_rank = ranking_metrics(y_val, val_scores, g_val)
        val_brier = float(brier_score_loss(y_val, val_scores))

        if monitor_metric == "val_mrr":
            monitor_value = val_rank["mrr"]
        elif monitor_metric == "val_auroc":
            monitor_value = val_cls["auroc"]
        else:
            monitor_value = val_rank["mrr"]

        train_log.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses) if epoch_losses else 0.0),
                "val_auroc": val_cls["auroc"],
                "val_auprc": val_cls["auprc"],
                "val_f1": val_cls["f1"],
                "val_mcc": val_cls["mcc"],
                "val_mrr": val_rank["mrr"],
                "val_hit@1": val_rank["hit@1"],
                "val_hit@3": val_rank["hit@3"],
                "val_hit@5": val_rank["hit@5"],
                "val_brier": val_brier,
                "selected_threshold": selected_threshold,
            }
        )
        print(
            f"Epoch {epoch:03d}/{epochs} | loss={train_log[-1]['train_loss']:.4f} | "
            f"val_auroc={val_cls['auroc']:.4f} | val_mrr={val_rank['mrr']:.4f} | "
            f"val_hit@3={val_rank['hit@3']:.4f} | val_mcc={val_cls['mcc']:.4f}"
        )

        if monitor_value > best_val_metric:
            best_val_metric = monitor_value
            patience_counter = 0
            model.save_weights(str(best_weight_path))
            print(f"[BEST] {monitor_metric} -> {monitor_value:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    if best_weight_path.exists():
        model.load_weights(str(best_weight_path))

    val_scores = predict_scores(x_val)
    test_scores = predict_scores(x_test)
    sweep = threshold_sweep(y_val, val_scores)
    selected_threshold = (
        sweep["best_mcc"]["threshold"] if threshold_metric == "mcc" else sweep["best_f1"]["threshold"]
    )

    val_cls = classification_metrics(y_val, val_scores, selected_threshold)
    test_cls = classification_metrics(y_test, test_scores, selected_threshold)
    val_rank = ranking_metrics(y_val, val_scores, g_val)
    test_rank = ranking_metrics(y_test, test_scores, g_test)

    metrics = {
        "validation_metrics_at_selected_threshold": val_cls,
        "test_metrics": test_cls,
        "val_ranking_metrics": val_rank,
        "test_ranking_metrics": test_rank,
        "selected_threshold": float(selected_threshold),
        "threshold_selection_metric": threshold_metric,
        "best_val_monitor_metric": float(best_val_metric),
    }
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(save_dir / "ranking_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"validation": val_rank, "test": test_rank}, f, indent=2)

    with open(save_dir / "best_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_f1_threshold": sweep["best_f1"]["threshold"],
                "best_f1": sweep["best_f1"]["value"],
                "best_mcc_threshold": sweep["best_mcc"]["threshold"],
                "best_mcc": sweep["best_mcc"]["value"],
                "selected_metric": threshold_metric,
                "selected_threshold": selected_threshold,
            },
            f,
            indent=2,
        )

    pd.DataFrame(train_log).to_csv(save_dir / "train_log.csv", index=False)
    pd.DataFrame(sweep["sweep"]).to_csv(save_dir / "threshold_vs_f1_table.csv", index=False)

    save_threshold_plot(sweep["sweep"], save_dir / "validation_threshold_sweep.png")
    save_score_hist(y_val, val_scores, save_dir / "validation_score_histogram_pos_neg.png", "Validation Score Histogram")
    save_score_hist(y_test, test_scores, save_dir / "score_histogram_pos_neg.png", "Test Score Histogram")
    save_roc(y_test, test_scores, save_dir / "roc_curve.png")
    save_pr(y_test, test_scores, save_dir / "pr_curve.png")
    save_confusion(y_test, test_scores, selected_threshold, save_dir / "confusion_matrix.png")
    save_calibration(y_test, test_scores, save_dir / "calibration_curve.png", save_dir / "calibration_metrics.json")

    # Copy pair/candidate reports from feature export dir for parity with main pipeline.
    for report_name in ("pair_data_report.json", "candidate_set_report.json"):
        src = feature_dir / report_name
        if src.exists():
            (save_dir / report_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    with open(save_dir / "test_summary.txt", "w", encoding="utf-8") as f:
        f.write("MLX scoring run summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"selected_threshold: {selected_threshold:.4f}\n")
        f.write(
            f"val -> AUROC={val_cls['auroc']:.4f}, AUPRC={val_cls['auprc']:.4f}, "
            f"F1={val_cls['f1']:.4f}, MCC={val_cls['mcc']:.4f}, MRR={val_rank['mrr']:.4f}\n"
        )
        f.write(
            f"test -> AUROC={test_cls['auroc']:.4f}, AUPRC={test_cls['auprc']:.4f}, "
            f"F1={test_cls['f1']:.4f}, MCC={test_cls['mcc']:.4f}, MRR={test_rank['mrr']:.4f}\n"
        )

    np.savez_compressed(save_dir / "feature_scaler_stats.npz", mean=scaler_mean, std=scaler_std)
    print(f"Training complete. Outputs: {save_dir}")


if __name__ == "__main__":
    main()

