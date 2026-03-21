"""
Run seed-stability checks for a scoring config and summarize variance.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = ROOT / "configs" / "seed_runs"


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def run_training(config_path: Path):
    cmd = ["python", "scripts/train_scoring_model.py", "--config", str(config_path)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def collect_metrics(run_dir: Path) -> dict:
    with open(run_dir / "metrics.json", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with open(run_dir / "calibration_metrics.json", encoding="utf-8") as handle:
        calibration = json.load(handle)
    test = metrics["test_metrics"]
    val = metrics["validation_metrics_at_selected_threshold"]
    return {
        "val_auroc": float(val["auroc"]),
        "val_auprc": float(val["auprc"]),
        "val_f1": float(val["f1"]),
        "val_mcc": float(val["mcc"]),
        "val_mrr": float(val["mrr"]),
        "val_hit1": float(val["hit@1"]),
        "val_hit3": float(val["hit@3"]),
        "val_hit5": float(val["hit@5"]),
        "test_auroc": float(test["auroc"]),
        "test_auprc": float(test["auprc"]),
        "test_f1": float(test["f1"]),
        "test_mcc": float(test["mcc"]),
        "test_mrr": float(test["mrr"]),
        "test_hit1": float(test["hit@1"]),
        "test_hit3": float(test["hit@3"]),
        "test_hit5": float(test["hit@5"]),
        "test_brier": float(calibration["test"]["brier_score"]),
        "selected_threshold": float(metrics["best_thresholds"]["selected_threshold"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Run seed-stability training suite.")
    parser.add_argument("--base-config", type=Path, required=True, help="Base YAML config.")
    parser.add_argument("--seeds", type=str, default="42,1337,2026", help="Comma-separated seeds.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/training/seed_stability"), help="Output root.")
    parser.add_argument("--dry-run", action="store_true", help="Only write generated configs.")
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    if not seeds:
        raise ValueError("No valid seeds provided")

    output_root = (ROOT / args.output_root).resolve() if not str(args.output_root).startswith(("/", "\\")) else args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        cfg = deepcopy(base_cfg)
        base_name = str(cfg.get("experiment_name", "seed_run"))
        run_name = f"{base_name}_seed{seed}"
        cfg["experiment_name"] = run_name
        cfg.setdefault("training", {})["seed"] = seed
        run_dir = output_root / run_name
        cfg.setdefault("logging", {})["save_dir"] = str(run_dir).replace("\\", "/")

        cfg_path = GEN_DIR / f"{run_name}.yaml"
        dump_yaml(cfg, cfg_path)
        print(f"[SEED] {seed} -> {cfg_path}")

        if args.dry_run:
            continue

        run_training(cfg_path)
        metric_row = {"seed": seed, "run_name": run_name, "run_dir": str(run_dir).replace("\\", "/")}
        metric_row.update(collect_metrics(run_dir))
        rows.append(metric_row)

    if args.dry_run:
        print("[DRY RUN] Configs generated only.")
        return

    if not rows:
        raise RuntimeError("No training results collected.")

    df = pd.DataFrame(rows)
    df.to_csv(output_root / "seed_stability_runs.csv", index=False)

    summary = {}
    for column in df.columns:
        if column in {"seed", "run_name", "run_dir"}:
            continue
        values = df[column].astype(float).values
        summary[column] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    with open(output_root / "seed_stability_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_config": str(args.base_config).replace("\\", "/"),
                "seeds": seeds,
                "num_runs": len(rows),
                "metrics": summary,
            },
            handle,
            indent=2,
        )
    print(f"[OK] Saved: {output_root / 'seed_stability_runs.csv'}")
    print(f"[OK] Saved: {output_root / 'seed_stability_summary.json'}")


if __name__ == "__main__":
    main()
