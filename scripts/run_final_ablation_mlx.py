"""
Run MLX ablation with the same smoke/full selection structure as classical ablation.

Grid:
- model family: mlp_s, mlp_m, mlp_l
- feature set: F1, F2
- objective: L1 (BCE), L2 (ranking + BCE)
- schedule mode: C0 (normal early stop), C1 (full-epoch style)

Workflow:
1) Run 12 smoke combinations (4 per family)
2) Select top 1-2 per family
3) Run selected full configs
4) Pick best overall MLX run
5) Sync best run to outputs/training/peptidquantum_v0_1_final_best_mlx_ablation
6) Write ablation summary + plots
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_TEMPLATE = ROOT / "configs" / "train_v0_1_scoring_mlx_m4.yaml"
GEN_CONFIG_DIR = ROOT / "configs" / "ablation_generated_mlx"
FEATURE_ROOT = ROOT / "data" / "mlx" / "ablation_features"
OUTPUT_ROOT = ROOT / "outputs" / "training"
FINAL_BEST_DIR = OUTPUT_ROOT / "peptidquantum_v0_1_final_best_mlx_ablation"


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_template() -> dict:
    return load_yaml(CONFIG_TEMPLATE)


def model_cfg(family: str) -> Dict[str, object]:
    family = family.lower()
    if family == "mlp_s":
        return {"hidden_dim": 128, "num_layers": 2}
    if family == "mlp_m":
        return {"hidden_dim": 192, "num_layers": 3}
    if family == "mlp_l":
        return {"hidden_dim": 256, "num_layers": 4}
    raise ValueError(f"Unknown family: {family}")


def feature_cfg(feature_level: str) -> Dict[str, object]:
    if feature_level == "F1":
        return {
            "use_local_density": False,
            "local_density_radius": 8.0,
        }
    if feature_level == "F2":
        return {
            "use_local_density": True,
            "local_density_radius": 8.0,
        }
    raise ValueError(f"Unknown feature level: {feature_level}")


def objective_cfg(loss_level: str) -> Dict[str, object]:
    if loss_level == "L1":
        return {
            "use_ranking": False,
            "margin": 0.2,
            "bce_alpha": 1.0,
        }
    if loss_level == "L2":
        return {
            "use_ranking": True,
            "margin": 0.2,
            "bce_alpha": 0.5,
        }
    raise ValueError(f"Unknown loss level: {loss_level}")


def build_config(
    template: dict,
    family: str,
    feature_level: str,
    loss_level: str,
    curriculum: str,
    stage: str,
    stage_settings: Dict[str, int],
) -> dict:
    cfg = deepcopy(template)
    run_name = f"ablation_mlx_{family}_{feature_level}_{loss_level}_{curriculum}_{stage}".lower()

    cfg["experiment_name"] = run_name
    cfg["model"].update(model_cfg(family))
    cfg["features"].update(feature_cfg(feature_level))
    cfg["loss"].update(objective_cfg(loss_level))
    cfg["evaluation"]["monitor_metric"] = "val_mrr"
    cfg["evaluation"]["threshold_selection_metric"] = "mcc"
    cfg["logging"]["save_dir"] = str((OUTPUT_ROOT / run_name).as_posix())

    # Share feature exports by feature profile; training outputs stay run-specific.
    feature_bucket = f"{feature_level.lower()}_ld{int(bool(cfg['features']['use_local_density']))}"
    cfg["data"]["feature_dir"] = str((FEATURE_ROOT / feature_bucket).as_posix())

    if stage == "smoke":
        cfg["training"]["epochs"] = int(stage_settings["smoke_epochs"])
        cfg["training"]["early_stopping_patience"] = int(stage_settings["smoke_patience"])
        smoke_subset = {
            "train": int(stage_settings["smoke_subset_train"]),
            "val": int(stage_settings["smoke_subset_val"]),
            "test": int(stage_settings["smoke_subset_test"]),
        }
        if any(value > 0 for value in smoke_subset.values()):
            cfg["training"]["subset_max_pairs"] = smoke_subset
        else:
            cfg["training"].pop("subset_max_pairs", None)
    elif stage == "full":
        cfg["training"]["epochs"] = int(stage_settings["full_epochs"])
        cfg["training"]["early_stopping_patience"] = int(stage_settings["full_patience"])
        full_subset = {
            "train": int(stage_settings["full_subset_train"]),
            "val": int(stage_settings["full_subset_val"]),
            "test": int(stage_settings["full_subset_test"]),
        }
        if any(value > 0 for value in full_subset.values()):
            cfg["training"]["subset_max_pairs"] = full_subset
        else:
            cfg["training"].pop("subset_max_pairs", None)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # C1 mimics the "long/strict" track by delaying early-stop checks.
    if curriculum == "C1":
        cfg["training"]["min_epochs_before_early_stop"] = int(cfg["training"]["epochs"])
    elif curriculum == "C0":
        cfg["training"]["min_epochs_before_early_stop"] = 0
    else:
        raise ValueError(f"Unknown curriculum: {curriculum}")

    if family.lower() == "mlp_l":
        cfg["training"]["batch_size"] = 384
    return cfg


def write_config(cfg: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def run_command(cmd: List[str]):
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def ensure_features(config_path: Path, refresh: bool = False):
    cfg = load_yaml(config_path)
    feature_dir = Path(cfg["data"]["feature_dir"])
    required_npz = [
        feature_dir / "train_mlx_features.npz",
        feature_dir / "val_mlx_features.npz",
        feature_dir / "test_mlx_features.npz",
    ]
    required_reports = [
        feature_dir / "pair_data_report.json",
        feature_dir / "candidate_set_report.json",
        feature_dir / "feature_export_meta.json",
    ]

    if refresh:
        run_command([sys.executable, "scripts/export_mlx_features.py", "--config", str(config_path)])
        return

    has_outputs = all(path.exists() for path in (required_npz + required_reports))
    if not has_outputs:
        run_command([sys.executable, "scripts/export_mlx_features.py", "--config", str(config_path)])
        return

    pairs_dir = Path(cfg["data"]["pairs_dir"])
    pair_inputs = [
        pairs_dir / "train_pairs.parquet",
        pairs_dir / "val_pairs.parquet",
        pairs_dir / "test_pairs.parquet",
        pairs_dir / "pair_data_report.json",
        pairs_dir / "candidate_set_report.json",
    ]
    newest_input = max(_mtime(path) for path in pair_inputs)
    oldest_output = min(_mtime(path) for path in (required_npz + required_reports))
    if oldest_output < newest_input:
        run_command([sys.executable, "scripts/export_mlx_features.py", "--config", str(config_path)])


def run_training(config_path: Path, skip_existing: bool, refresh_features: bool = False):
    cfg = load_yaml(config_path)
    ensure_features(config_path, refresh=refresh_features)
    save_dir = Path(cfg["logging"]["save_dir"])
    if skip_existing and (save_dir / "metrics.json").exists():
        return
    run_command([sys.executable, "scripts/train_scoring_mlx.py", "--config", str(config_path)])


def load_run_metrics(save_dir: Path) -> dict:
    with open(save_dir / "metrics.json", encoding="utf-8") as handle:
        metrics = json.load(handle)

    best_thresholds_path = save_dir / "best_thresholds.json"
    if best_thresholds_path.exists():
        with open(best_thresholds_path, encoding="utf-8") as handle:
            best_thr = json.load(handle)
    else:
        best_thr = {
            "best_f1_threshold": metrics["selected_threshold"],
            "best_mcc_threshold": metrics["selected_threshold"],
            "selected_threshold": metrics["selected_threshold"],
        }

    calibration_path = save_dir / "calibration_metrics.json"
    calibration = {}
    if calibration_path.exists():
        with open(calibration_path, encoding="utf-8") as handle:
            calibration = json.load(handle)

    train_log = pd.read_csv(save_dir / "train_log.csv")
    val = metrics["validation_metrics_at_selected_threshold"]
    test = metrics["test_metrics"]
    val_rank = metrics["val_ranking_metrics"]
    test_rank = metrics["test_ranking_metrics"]

    score_min = float(calibration.get("score_min", np.nan))
    score_max = float(calibration.get("score_max", np.nan))
    score_range = float(score_max - score_min) if np.isfinite(score_min) and np.isfinite(score_max) else np.nan

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
        "val_brier": float(train_log["val_brier"].iloc[-1]) if "val_brier" in train_log.columns else np.nan,
        "test_brier": float(calibration.get("brier_score", np.nan)),
        "test_score_range": score_range,
        "epochs_completed": int(len(train_log)),
    }


def smoke_pass(row: dict) -> bool:
    random_hit3 = 0.5  # K=5 negatives + 1 positive => 6 candidates
    random_mrr = np.mean([1.0 / i for i in range(1, 7)])
    return (
        row["val_mrr"] > random_mrr
        and row["val_hit3"] > random_hit3
        and row["val_auroc"] > 0.55
    )


def choose_family_finalists(smoke_df: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    """
    Pick smoke finalists per model family using validation metrics only.

    Ensures diversity: when top_k >= 2, prefer including at least one L2 config if any
    L2 smoke run exists for that family (still ranked by val metrics within L2).
    """
    finalists = []
    for family, group in smoke_df.groupby("family"):
        group = group.copy()
        group["smoke_pass"] = group.apply(lambda r: smoke_pass(r.to_dict()), axis=1)
        passing = group[group["smoke_pass"]]
        source = passing if len(passing) > 0 else group
        source = source.sort_values(
            by=["val_mrr", "val_hit3", "val_auroc", "val_mcc"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        if source.empty:
            continue
        k = min(top_k, len(source))
        if k <= 1:
            finalists.append(source.head(1))
            continue

        picked_rows = [source.iloc[0]]
        used_combos = {picked_rows[0]["combo"]}
        second = source.iloc[1]
        if picked_rows[0]["loss_level"] != "L2" and second["loss_level"] != "L2":
            l2_ranked = source[source["loss_level"] == "L2"]
            if len(l2_ranked) > 0:
                for _, row in l2_ranked.iterrows():
                    if row["combo"] not in used_combos:
                        second = row
                        break
        if second["combo"] in used_combos:
            for i in range(1, len(source)):
                cand = source.iloc[i]
                if cand["combo"] not in used_combos:
                    second = cand
                    break
        picked_rows.append(second)
        used_combos.add(second["combo"])

        idx = 2
        while len(picked_rows) < k and idx < len(source):
            cand = source.iloc[idx]
            if cand["combo"] not in used_combos:
                picked_rows.append(cand)
                used_combos.add(cand["combo"])
            idx += 1

        finalists.append(pd.DataFrame(picked_rows))
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
    plt.title("MLX Ablation Heatmap (Val MRR)")
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
    plt.title("MLX Family Comparison (Best Full per Family)")
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


def run_visualization(sample_list: Path, output_dir: Path, relax_tool_fraction_check: bool = False):
    cmd = [
        sys.executable,
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
    if relax_tool_fraction_check:
        cmd.append("--no-enforce-tool-fraction")
    try:
        run_command(cmd)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] Visualization sanity failed for {output_dir}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run MLX final ablation with smoke/full stages.")
    parser.add_argument("--smoke-only", action="store_true", help="Only run smoke stage.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have metrics.json.")
    parser.add_argument("--refresh-features", action="store_true", help="Force re-export MLX features before each run.")
    parser.add_argument(
        "--finalists-per-family",
        type=int,
        default=2,
        help="How many smoke finalists per family go to full stage (>=2 helps retain L2 ablations).",
    )
    parser.add_argument("--smoke-epochs", type=int, default=8, help="Epochs for smoke stage.")
    parser.add_argument("--smoke-patience", type=int, default=4, help="Early-stop patience for smoke stage.")
    parser.add_argument("--full-epochs", type=int, default=200, help="Epochs for full stage.")
    parser.add_argument("--full-patience", type=int, default=20, help="Early-stop patience for full stage.")
    parser.add_argument("--smoke-subset-train", type=int, default=25000, help="Smoke subset size for train split (0 disables subsetting).")
    parser.add_argument("--smoke-subset-val", type=int, default=8000, help="Smoke subset size for val split (0 disables subsetting).")
    parser.add_argument("--smoke-subset-test", type=int, default=8000, help="Smoke subset size for test split (0 disables subsetting).")
    parser.add_argument("--full-subset-train", type=int, default=0, help="Full subset size for train split (0 disables subsetting).")
    parser.add_argument("--full-subset-val", type=int, default=0, help="Full subset size for val split (0 disables subsetting).")
    parser.add_argument("--full-subset-test", type=int, default=0, help="Full subset size for test split (0 disables subsetting).")
    parser.add_argument(
        "--vis-relax-tool-fraction",
        action="store_true",
        help="Do not enforce min PLIP/Arpeggio interaction fraction in post-ablation visualization sanity.",
    )
    args = parser.parse_args()

    template = load_template()
    families = ["mlp_s", "mlp_m", "mlp_l"]
    smoke_combos = [("F1", "L1", "C0"), ("F1", "L2", "C0"), ("F2", "L2", "C0"), ("F2", "L2", "C1")]
    stage_settings = {
        "smoke_epochs": args.smoke_epochs,
        "smoke_patience": args.smoke_patience,
        "full_epochs": args.full_epochs,
        "full_patience": args.full_patience,
        "smoke_subset_train": args.smoke_subset_train,
        "smoke_subset_val": args.smoke_subset_val,
        "smoke_subset_test": args.smoke_subset_test,
        "full_subset_train": args.full_subset_train,
        "full_subset_val": args.full_subset_val,
        "full_subset_test": args.full_subset_test,
    }

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
                stage_settings=stage_settings,
            )
            config_path = GEN_CONFIG_DIR / f"{cfg['experiment_name']}.yaml"
            write_config(cfg, config_path)
            run_training(
                config_path,
                skip_existing=args.skip_existing,
                refresh_features=args.refresh_features,
            )
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
        finalists = choose_family_finalists(smoke_df, top_k=max(1, args.finalists_per_family))
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
                stage_settings=stage_settings,
            )
            config_path = GEN_CONFIG_DIR / f"{cfg['experiment_name']}.yaml"
            write_config(cfg, config_path)
            run_training(
                config_path,
                skip_existing=args.skip_existing,
                refresh_features=args.refresh_features,
            )
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

    all_df = pd.concat([smoke_df, pd.DataFrame(full_rows)], ignore_index=True)
    summary_csv = FINAL_BEST_DIR / "ablation_summary.csv"

    if len(full_rows) > 0:
        full_df = pd.DataFrame(full_rows)
        # Selection must be validation-only; test metrics are never used for model choice.
        best_idx = full_df.sort_values(
            by=["val_mrr", "val_hit3", "val_auroc", "val_mcc", "val_auprc"],
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
                    "selection_policy": "validation_only",
                    "selection_sort_keys": ["val_mrr", "val_hit3", "val_auroc", "val_mcc", "val_auprc"],
                    "best_run": best_row,
                    "full_finalists": full_df.to_dict(orient="records"),
                },
                handle,
                indent=2,
            )

        sample_list_path = ROOT / "data" / "reports" / "audit_gallery_propedia" / "sample_list_final_best_mlx_model.txt"
        build_top_ranked_sample_list(FINAL_BEST_DIR, sample_list_path, limit=10)
        if sample_list_path.exists():
            run_visualization(
                sample_list_path,
                ROOT / "outputs" / "analysis_propedia_batch_mlx",
                relax_tool_fraction_check=args.vis_relax_tool_fraction,
            )
            run_visualization(
                sample_list_path,
                ROOT / "outputs" / "analysis_propedia_top_ranked_batch_mlx",
                relax_tool_fraction_check=args.vis_relax_tool_fraction,
            )
    else:
        FINAL_BEST_DIR.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(summary_csv, index=False)
        plot_heatmap(smoke_df, FINAL_BEST_DIR / "ablation_heatmap.png")

    print(f"Ablation summary: {summary_csv}")
    print(f"Final best dir: {FINAL_BEST_DIR}")


if __name__ == "__main__":
    main()
