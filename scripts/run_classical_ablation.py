"""
Run classical graph-model ablation for interaction scoring + reranking.

Grid:
- model family: mpnn, gatv2, gin
- feature set: F1, F2
- objective: L1 (BCE), L2 (ranking + BCE)
- curriculum: C0 (off), C1 (on)

Workflow:
1) Run 12 smoke combinations (4 per family)
2) Select top 1-2 per family
3) Run selected full configs
4) Pick best overall classical run
5) Sync best run to outputs/training/peptidquantum_v0_1_final_best_classical
6) Write ablation summary + plots
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_TEMPLATE = ROOT / "configs" / "train_v0_1_scoring_b.yaml"
GEN_CONFIG_DIR = ROOT / "configs" / "ablation_generated"
OUTPUT_ROOT = ROOT / "outputs" / "training"
FINAL_BEST_DIR = OUTPUT_ROOT / "peptidquantum_v0_1_final_best_classical"


def load_template() -> dict:
    with open(CONFIG_TEMPLATE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_cfg(family: str) -> Dict[str, object]:
    family = family.lower()
    if family == "mpnn":
        return {"type": "MPNN", "heads": 1, "hidden_dim": 96, "num_layers": 3, "dropout": 0.1}
    if family == "gatv2":
        return {"type": "GATv2", "heads": 4, "hidden_dim": 128, "num_layers": 3, "dropout": 0.1}
    if family == "gin":
        return {"type": "GIN", "heads": 1, "hidden_dim": 128, "num_layers": 3, "dropout": 0.1}
    raise ValueError(f"Unknown family: {family}")


def feature_cfg(feature_level: str) -> Dict[str, object]:
    if feature_level == "F1":
        return {
            "use_residue_type_onehot": True,
            "use_position_index": True,
            "use_chain_flags": True,
            "use_interface_flag": False,
            "use_pocket_flag": False,
            "use_local_density": True,
            "use_secondary_structure": False,
            "use_distance_edge_features": True,
            "edge_dim": 8,
            "normalize_distances": True,
            "distance_threshold": 8.0,
            "local_density_radius": 8.0,
        }
    if feature_level == "F2":
        return {
            "use_residue_type_onehot": True,
            "use_position_index": True,
            "use_chain_flags": True,
            "use_interface_flag": True,
            "use_pocket_flag": True,
            "use_local_density": True,
            "use_secondary_structure": True,
            "use_distance_edge_features": True,
            "edge_dim": 8,
            "normalize_distances": True,
            "distance_threshold": 8.0,
            "local_density_radius": 8.0,
            "pocket_distance_threshold": 8.0,
            "interface_distance_threshold": 5.0,
        }
    raise ValueError(f"Unknown feature level: {feature_level}")


def objective_cfg(loss_level: str) -> Dict[str, object]:
    if loss_level == "L1":
        return {
            "auto_pos_weight": True,
            "bce_alpha": 1.0,
            "ranking": {"enabled": False, "margin": 0.2},
        }
    if loss_level == "L2":
        return {
            "auto_pos_weight": True,
            "bce_alpha": 0.3,
            "ranking": {"enabled": True, "margin": 0.2},
        }
    raise ValueError(f"Unknown loss level: {loss_level}")


def curriculum_cfg(curriculum: str) -> Dict[str, object]:
    if curriculum == "C0":
        return {"enabled": False}
    if curriculum == "C1":
        return {
            "enabled": True,
            "seed": 42,
            "stages": [
                {"name": "easy_only", "end_epoch": 5, "ratios": {"easy": 1.0, "hard": 0.0, "structure_hard": 0.0}},
                {"name": "easy_hard", "end_epoch": 15, "ratios": {"easy": 0.7, "hard": 0.3, "structure_hard": 0.0}},
                {"name": "full_mix", "end_epoch": None, "ratios": {"easy": 0.7, "hard": 0.3, "structure_hard": 0.0}},
            ],
        }
    raise ValueError(f"Unknown curriculum: {curriculum}")


def build_config(
    template: dict,
    family: str,
    feature_level: str,
    loss_level: str,
    curriculum: str,
    stage: str,
    smoke_epochs: int,
    smoke_patience: int,
    full_epochs: int,
    full_patience: int,
    full_subset_train: int,
    full_subset_val: int,
    full_subset_test: int,
) -> dict:
    cfg = deepcopy(template)
    run_name = f"ablation_{family}_{feature_level}_{loss_level}_{curriculum}_{stage}".lower()
    save_dir = OUTPUT_ROOT / run_name

    model_encoder_cfg = model_cfg(family)
    cfg["experiment_name"] = run_name
    cfg["model"]["protein_encoder"].update(model_encoder_cfg)
    cfg["model"]["peptide_encoder"].update(model_encoder_cfg)
    cfg["model"]["quantum_head"] = {"enabled": False}
    cfg["features"].update(feature_cfg(feature_level))
    cfg["loss"].update(objective_cfg(loss_level))
    cfg["training"]["negative_curriculum"] = curriculum_cfg(curriculum)
    cfg["training"]["balanced_sampling"] = {"enabled": True}
    cfg["evaluation"]["monitor_metric"] = "val_mrr"
    cfg["evaluation"]["threshold_selection_metric"] = "mcc"
    cfg["evaluation"].setdefault("calibration", {})
    cfg["evaluation"]["calibration"].update(
        {
            "enabled": True,
            "temperature_scaling": True,
            "selection_metric": "nll",
            "temperature_min": 0.35,
            "temperature_max": 6.0,
            "temperature_steps": 120,
            "min_improvement": 1e-5,
        }
    )
    cfg["logging"]["save_dir"] = str(save_dir).replace("\\", "/")

    if stage == "smoke":
        cfg["training"]["epochs"] = int(smoke_epochs)
        cfg["training"]["early_stopping_patience"] = int(smoke_patience)
        cfg["training"]["subset_max_pairs"] = {"train": 18000, "val": 6000, "test": 6000}
    elif stage == "full":
        cfg["training"]["epochs"] = int(full_epochs)
        cfg["training"]["early_stopping_patience"] = int(full_patience)
        cfg["training"]["subset_max_pairs"] = {
            "train": int(full_subset_train),
            "val": int(full_subset_val),
            "test": int(full_subset_test),
        }
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if family.lower() == "mpnn":
        if stage == "smoke":
            cfg["training"]["batch_size"] = 16
            cfg["training"]["subset_max_pairs"] = {"train": 12000, "val": 4000, "test": 4000}
        else:
            cfg["training"]["batch_size"] = 24
            cfg["training"]["subset_max_pairs"] = {"train": 48000, "val": 16000, "test": 16000}

    return cfg


def write_config(cfg: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def run_training(config_path: Path):
    cmd = ["python", "scripts/train_scoring_model.py", "--config", str(config_path)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_run_metrics(save_dir: Path) -> dict:
    with open(save_dir / "metrics.json", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with open(save_dir / "calibration_metrics.json", encoding="utf-8") as handle:
        calibration = json.load(handle)
    train_log = pd.read_csv(save_dir / "train_log.csv")

    val = metrics["validation_metrics_at_selected_threshold"]
    test = metrics["test_metrics"]
    val_rank = metrics["val_ranking_metrics"]
    test_rank = metrics["test_ranking_metrics"]
    best_thr = metrics["best_thresholds"]

    return {
        "val_auroc": float(val["auroc"]),
        "val_auprc": float(val["auprc"]),
        "val_f1": float(val["f1"]),
        "val_mcc": float(val["mcc"]),
        "val_mrr": float(val_rank["mrr"]),
        "val_hit1": float(val_rank["hit@1"]),
        "val_hit3": float(val_rank["hit@3"]),
        "val_hit5": float(val_rank["hit@5"]),
        "test_auroc": float(test["auroc"]),
        "test_auprc": float(test["auprc"]),
        "test_f1": float(test["f1"]),
        "test_mcc": float(test["mcc"]),
        "test_mrr": float(test_rank["mrr"]),
        "test_hit1": float(test_rank["hit@1"]),
        "test_hit3": float(test_rank["hit@3"]),
        "test_hit5": float(test_rank["hit@5"]),
        "best_f1_threshold": float(best_thr["best_f1_threshold"]),
        "best_mcc_threshold": float(best_thr["best_mcc_threshold"]),
        "selected_threshold": float(best_thr["selected_threshold"]),
        "val_brier": float(calibration["validation"]["brier_score"]),
        "test_brier": float(calibration["test"]["brier_score"]),
        "val_score_range": float(calibration["validation"]["score_range"]),
        "test_score_range": float(calibration["test"]["score_range"]),
        "balanced_sampler_enabled": bool(train_log["balanced_sampler_enabled"].iloc[-1]),
    }


def smoke_pass(row: dict) -> bool:
    random_hit3 = 0.5  # K=5 negatives + 1 positive => 6 candidates
    random_mrr = np.mean([1.0 / i for i in range(1, 7)])
    return (
        row["val_mrr"] > random_mrr
        and row["val_hit3"] > random_hit3
        and row["val_score_range"] > 1e-4
    )


def choose_family_finalists(smoke_df: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    finalists = []
    for family, group in smoke_df.groupby("family"):
        group = group.copy()
        group["smoke_pass"] = group.apply(lambda r: smoke_pass(r.to_dict()), axis=1)
        passing = group[group["smoke_pass"]]
        source = passing if len(passing) > 0 else group
        source = source.sort_values(
            by=["val_mrr", "val_hit3", "val_auroc", "val_mcc"],
            ascending=[False, False, False, False],
        )
        finalists.append(source.head(min(top_k, len(source))))
    return pd.concat(finalists, ignore_index=True)


def plot_heatmap(df: pd.DataFrame, out_path: Path):
    pivot = df.pivot_table(index="family", columns="combo", values="val_mrr", aggfunc="max")
    families = list(pivot.index)
    combos = list(pivot.columns)
    values = pivot.values

    plt.figure(figsize=(max(10, len(combos) * 0.9), 4))
    im = plt.imshow(values, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Val MRR")
    plt.xticks(range(len(combos)), combos, rotation=45, ha="right")
    plt.yticks(range(len(families)), families)
    plt.title("Ablation Heatmap (Val MRR)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_family_comparison(full_df: pd.DataFrame, out_path: Path):
    best_by_family = (
        full_df.sort_values(by=["val_mrr", "val_hit3", "val_auroc"], ascending=False)
        .groupby("family", as_index=False)
        .first()
    )

    x = np.arange(len(best_by_family))
    width = 0.2
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, best_by_family["test_mrr"], width=width, label="Test MRR")
    plt.bar(x, best_by_family["test_hit3"], width=width, label="Test Hit@3")
    plt.bar(x + width, best_by_family["test_auroc"], width=width, label="Test AUROC")
    plt.xticks(x, best_by_family["family"].tolist())
    plt.ylim(0, 1)
    plt.title("Model Family Comparison (Best Full per Family)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def sync_best_run(best_dir: Path):
    if FINAL_BEST_DIR.exists():
        shutil.rmtree(FINAL_BEST_DIR)
    shutil.copytree(best_dir, FINAL_BEST_DIR)


def build_top_ranked_sample_list(best_dir: Path, out_path: Path, limit: int = 10):
    top_hits_path = best_dir / "test_topk_positive_hits.csv"
    if not top_hits_path.exists():
        return
    df = pd.read_csv(top_hits_path)
    if "protein_complex_id" not in df.columns:
        return
    ids = df["protein_complex_id"].dropna().astype(str).drop_duplicates().head(limit).tolist()
    if not ids:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ids) + "\n", encoding="utf-8")


def run_visualization(sample_list: Path, output_dir: Path):
    cmd = [
        "python",
        "scripts/run_visualization_sanity.py",
        "--canonical",
        "data/canonical",
        "--sample-list",
        str(sample_list),
        "--output",
        str(output_dir),
        "--limit",
        "10",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run classical model-selection ablation.")
    parser.add_argument("--smoke-only", action="store_true", help="Only run smoke ablation stage.")
    parser.add_argument("--smoke-epochs", type=int, default=6, help="Smoke stage max epochs.")
    parser.add_argument("--smoke-patience", type=int, default=4, help="Smoke stage early stopping patience.")
    parser.add_argument("--full-epochs", type=int, default=200, help="Full stage max epochs.")
    parser.add_argument("--full-patience", type=int, default=20, help="Full stage early stopping patience.")
    parser.add_argument("--finalists-per-family", type=int, default=1, help="Top-k finalists per family for full runs.")
    parser.add_argument("--full-subset-train", type=int, default=60000, help="Full stage train subset max pairs.")
    parser.add_argument("--full-subset-val", type=int, default=20000, help="Full stage val subset max pairs.")
    parser.add_argument("--full-subset-test", type=int, default=20000, help="Full stage test subset max pairs.")
    args = parser.parse_args()

    template = load_template()

    families = ["mpnn", "gatv2", "gin"]
    smoke_combos = [("F1", "L1", "C0"), ("F1", "L2", "C0"), ("F2", "L2", "C0"), ("F2", "L2", "C1")]

    smoke_rows: List[dict] = []
    full_rows: List[dict] = []

    for family in families:
        for feature_level, loss_level, curriculum in smoke_combos:
            cfg = build_config(
                template,
                family,
                feature_level,
                loss_level,
                curriculum,
                stage="smoke",
                smoke_epochs=args.smoke_epochs,
                smoke_patience=args.smoke_patience,
                full_epochs=args.full_epochs,
                full_patience=args.full_patience,
                full_subset_train=args.full_subset_train,
                full_subset_val=args.full_subset_val,
                full_subset_test=args.full_subset_test,
            )
            config_path = GEN_CONFIG_DIR / f"{cfg['experiment_name']}.yaml"
            write_config(cfg, config_path)
            run_training(config_path)
            save_dir = Path(cfg["logging"]["save_dir"])
            metrics = load_run_metrics(save_dir)
            row = {
                "stage": "smoke",
                "family": family,
                "feature_level": feature_level,
                "loss_level": loss_level,
                "curriculum": curriculum,
                "combo": f"{feature_level}_{loss_level}_{curriculum}",
                "save_dir": str(save_dir).replace("\\", "/"),
            }
            row.update(metrics)
            row["smoke_pass"] = smoke_pass(row)
            smoke_rows.append(row)

    smoke_df = pd.DataFrame(smoke_rows)

    if not args.smoke_only:
        finalists = choose_family_finalists(smoke_df, top_k=max(1, int(args.finalists_per_family)))
        for _, finalist in finalists.iterrows():
            family = finalist["family"]
            feature_level = finalist["feature_level"]
            loss_level = finalist["loss_level"]
            curriculum = finalist["curriculum"]
            cfg = build_config(
                template,
                family,
                feature_level,
                loss_level,
                curriculum,
                stage="full",
                smoke_epochs=args.smoke_epochs,
                smoke_patience=args.smoke_patience,
                full_epochs=args.full_epochs,
                full_patience=args.full_patience,
                full_subset_train=args.full_subset_train,
                full_subset_val=args.full_subset_val,
                full_subset_test=args.full_subset_test,
            )
            config_path = GEN_CONFIG_DIR / f"{cfg['experiment_name']}.yaml"
            write_config(cfg, config_path)
            run_training(config_path)
            save_dir = Path(cfg["logging"]["save_dir"])
            metrics = load_run_metrics(save_dir)
            row = {
                "stage": "full",
                "family": family,
                "feature_level": feature_level,
                "loss_level": loss_level,
                "curriculum": curriculum,
                "combo": f"{feature_level}_{loss_level}_{curriculum}",
                "save_dir": str(save_dir).replace("\\", "/"),
            }
            row.update(metrics)
            row["smoke_pass"] = None
            full_rows.append(row)

    all_df = pd.concat(
        [smoke_df, pd.DataFrame(full_rows)],
        ignore_index=True,
    )
    summary_csv = FINAL_BEST_DIR / "ablation_summary.csv"
    if len(full_rows) > 0:
        full_df = pd.DataFrame(full_rows)
        best_idx = full_df.sort_values(
            by=["val_mrr", "val_hit3", "val_auroc", "test_mrr", "test_auroc"],
            ascending=[False, False, False, False, False],
        ).index[0]
        best_row = full_df.loc[best_idx].to_dict()
        best_dir = Path(best_row["save_dir"])
        sync_best_run(best_dir)

        all_df.to_csv(summary_csv, index=False)
        plot_heatmap(smoke_df, FINAL_BEST_DIR / "ablation_heatmap.png")
        plot_family_comparison(full_df, FINAL_BEST_DIR / "model_family_comparison.png")

        with open(FINAL_BEST_DIR / "selection_summary.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "best_run": best_row,
                    "full_finalists": full_df.to_dict(orient="records"),
                },
                handle,
                indent=2,
            )

        sample_list_path = ROOT / "data" / "reports" / "audit_gallery_propedia" / "sample_list_final_best_model.txt"
        build_top_ranked_sample_list(FINAL_BEST_DIR, sample_list_path, limit=10)
        if sample_list_path.exists():
            run_visualization(sample_list_path, ROOT / "outputs" / "analysis_propedia_batch")
            run_visualization(sample_list_path, ROOT / "outputs" / "analysis_propedia_top_ranked_batch")
    else:
        FINAL_BEST_DIR.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(summary_csv, index=False)
        plot_heatmap(smoke_df, FINAL_BEST_DIR / "ablation_heatmap.png")

    print(f"Ablation summary: {summary_csv}")
    print(f"Final best dir: {FINAL_BEST_DIR}")


if __name__ == "__main__":
    main()
