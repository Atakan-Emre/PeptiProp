"""
Run frozen-embedding quantum kernel pilot from a YAML config.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def save_quantum_artifacts(output_dir: Path, output_json: Path, checkpoint_dir: Path):
    with open(output_json, encoding="utf-8") as handle:
        payload = json.load(handle)

    best_name = payload["best_model"]
    best_row = next(row for row in payload["results"] if row["name"] == best_name)

    val_scores = np.asarray(best_row.get("val_scores", []), dtype=np.float64)
    val_labels = np.asarray(best_row.get("val_labels", []), dtype=np.int64)
    test_scores = np.asarray(best_row.get("test_scores", []), dtype=np.float64)
    test_labels = np.asarray(best_row.get("test_labels", []), dtype=np.int64)

    threshold_info = best_row["threshold_info"]
    selected_threshold = float(threshold_info["selected_threshold"])

    metrics_payload = {
        "best_model": best_name,
        "sample_counts": payload.get("sample_counts", {}),
        "n_qubits": payload.get("n_qubits"),
        "best_thresholds": {
            "best_f1_threshold": float(threshold_info["best_f1_threshold"]),
            "best_mcc_threshold": float(threshold_info["best_mcc_threshold"]),
            "selected_threshold": selected_threshold,
            "selected_metric": str(threshold_info["selected_metric"]),
        },
        "validation_metrics_at_selected_threshold": best_row["val_metrics"],
        "test_metrics": best_row["test_metrics"],
        "test_metrics_best_f1_threshold": best_row["test_metrics"],
        "test_metrics_best_mcc_threshold": best_row["test_metrics"],
        "val_ranking_metrics": {
            key: best_row["val_metrics"].get(key)
            for key in ("mrr", "hit@1", "hit@3", "hit@5")
        },
        "test_ranking_metrics": {
            key: best_row["test_metrics"].get(key)
            for key in ("mrr", "hit@1", "hit@3", "hit@5")
        },
        "note": "Quantum kernel head pilot on downsampled frozen embeddings.",
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    with open(output_dir / "ranking_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_model": best_name,
                "validation": metrics_payload["val_ranking_metrics"],
                "test": metrics_payload["test_ranking_metrics"],
            },
            handle,
            indent=2,
        )

    with open(output_dir / "best_thresholds.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                **metrics_payload["best_thresholds"],
                "sweep": threshold_info["sweep"],
            },
            handle,
            indent=2,
        )

    with open(output_dir / "calibration_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "validation": {
                    "brier_score": best_row["val_metrics"].get("brier_score"),
                    "mean_score": best_row["val_metrics"].get("score_mean"),
                    "std_score": best_row["val_metrics"].get("score_std"),
                    "min_score": best_row["val_metrics"].get("score_min"),
                    "max_score": best_row["val_metrics"].get("score_max"),
                    "score_range": best_row["val_metrics"].get("score_range"),
                },
                "test": {
                    "brier_score": best_row["test_metrics"].get("brier_score"),
                    "mean_score": best_row["test_metrics"].get("score_mean"),
                    "std_score": best_row["test_metrics"].get("score_std"),
                    "min_score": best_row["test_metrics"].get("score_min"),
                    "max_score": best_row["test_metrics"].get("score_max"),
                    "score_range": best_row["test_metrics"].get("score_range"),
                },
            },
            handle,
            indent=2,
        )

    sweep_df = pd.DataFrame(threshold_info["sweep"])
    sweep_df.to_csv(output_dir / "threshold_vs_f1_table.csv", index=False)

    if len(val_scores) > 0 and len(test_scores) > 0:
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUROC={best_row['test_metrics']['auroc']:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Quantum Pilot)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=200)
        plt.close()

        precision, recall, _ = precision_recall_curve(test_labels, test_scores)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"AUPRC={best_row['test_metrics']['auprc']:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Quantum Pilot)")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(output_dir / "pr_curve.png", dpi=200)
        plt.close()

        test_pred = (test_scores >= selected_threshold).astype(np.int64)
        cm = confusion_matrix(test_labels, test_pred, labels=[0, 1])
        plt.figure(figsize=(5, 5))
        plt.imshow(cm, cmap="Blues")
        plt.xticks([0, 1], ["Pred 0", "Pred 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.title("Confusion Matrix (Quantum Pilot)")
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
        plt.close()

        frac_val, mean_val = calibration_curve(val_labels, val_scores, n_bins=10, strategy="uniform")
        frac_test, mean_test = calibration_curve(test_labels, test_scores, n_bins=10, strategy="uniform")
        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect")
        plt.plot(mean_val, frac_val, marker="o", label="Validation")
        plt.plot(mean_test, frac_test, marker="s", label="Test")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction Positives")
        plt.title("Calibration Curve (Quantum Pilot)")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(output_dir / "calibration_curve.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.hist(val_scores[val_labels == 1], bins=30, alpha=0.6, label="Positive", density=True)
        plt.hist(val_scores[val_labels == 0], bins=30, alpha=0.6, label="Negative", density=True)
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.title("Validation Score Histogram (Quantum Pilot)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(output_dir / "validation_score_histogram_pos_neg.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.hist(test_scores[test_labels == 1], bins=30, alpha=0.6, label="Positive", density=True)
        plt.hist(test_scores[test_labels == 0], bins=30, alpha=0.6, label="Negative", density=True)
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.title("Test Score Histogram (Quantum Pilot)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(output_dir / "score_histogram_pos_neg.png", dpi=200)
        plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    plt.plot(sweep_df["threshold"], sweep_df["mcc"], label="MCC")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title("Validation Threshold Sweep (Quantum Pilot)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "validation_threshold_sweep.png", dpi=200)
    plt.close()

    for report_name in ("pair_data_report.json", "candidate_set_report.json"):
        src = checkpoint_dir / report_name
        if src.exists():
            shutil.copy2(src, output_dir / report_name)

    summary_lines = [
        "Quantum Pilot Test Summary",
        "=" * 60,
        f"Best model: {best_name}",
        f"Selected threshold: {selected_threshold:.4f}",
        f"Validation AUROC: {best_row['val_metrics']['auroc']:.4f}",
        f"Validation AUPRC: {best_row['val_metrics']['auprc']:.4f}",
        f"Validation MCC: {best_row['val_metrics']['mcc']:.4f}",
        f"Validation MRR: {best_row['val_metrics'].get('mrr', 0.0):.4f}",
        f"Test AUROC: {best_row['test_metrics']['auroc']:.4f}",
        f"Test AUPRC: {best_row['test_metrics']['auprc']:.4f}",
        f"Test MCC: {best_row['test_metrics']['mcc']:.4f}",
        f"Test MRR: {best_row['test_metrics'].get('mrr', 0.0):.4f}",
        "",
        "NOTE: Quantum pilot uses downsampled frozen embeddings.",
    ]
    with open(output_dir / "test_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")


def main(config_path: Path):
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    base_config = Path(cfg["base_config"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / cfg.get("output_file", "quantum_pilot_results.json")

    cmd = [
        "python",
        "scripts/train_quantum_head.py",
        "--config",
        str(base_config),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--output",
        str(output_json),
        "--train-max-samples",
        str(int(cfg.get("train_max_samples", 1200))),
        "--eval-max-samples",
        str(int(cfg.get("eval_max_samples", 3000))),
        "--n-qubits",
        str(int(cfg.get("n_qubits", 4))),
        "--seed",
        str(int(cfg.get("seed", 42))),
    ]

    print("Running quantum pilot:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Quantum pilot output: {output_json}")
    save_quantum_artifacts(output_dir=output_dir, output_json=output_json, checkpoint_dir=checkpoint_dir)
    print(f"Quantum pilot artefacts saved under: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quantum pilot from YAML config")
    parser.add_argument("--config", required=True, type=Path, help="Path to quantum pilot config")
    args = parser.parse_args()
    main(args.config)
