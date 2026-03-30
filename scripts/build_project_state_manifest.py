#!/usr/bin/env python3
"""Generate a machine-readable snapshot of the repo's active data and result surfaces."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_PATH = DATA_DIR / "reports" / "project_state_manifest.json"
CANONICAL_DIR = DATA_DIR / "canonical"
PAIRS_DIR = CANONICAL_DIR / "pairs"
TRAINING_DIRS = {
    "gnn_final": ROOT / "outputs" / "training" / "peptiprop_v0_2_gnn_esm2",
    "mlx_final": ROOT / "outputs" / "training" / "peptidquantum_v0_1_final_mlx_m4",
}
VIS_DIRS = {
    "gnn_batch": ROOT / "outputs" / "analysis_propedia_batch_gnn",
    "gnn_top_ranked": ROOT / "outputs" / "analysis_propedia_top_ranked_batch_gnn",
    "mlx_batch": ROOT / "outputs" / "analysis_propedia_batch_mlx",
    "mlx_top_ranked": ROOT / "outputs" / "analysis_propedia_top_ranked_batch_mlx",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def _dir_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    files = sorted(p.name for p in path.iterdir() if p.is_file())
    return {
        "exists": True,
        "file_count": len(files),
        "files": files,
    }


def _training_summary(path: Path) -> Dict[str, Any]:
    summary = _dir_summary(path)
    if not path.exists():
        return summary
    summary["metrics"] = _load_json(path / "metrics.json")
    summary["ranking_metrics"] = _load_json(path / "ranking_metrics.json")
    summary["calibration_metrics"] = _load_json(path / "calibration_metrics.json")
    return summary


def _visual_summary(path: Path) -> Dict[str, Any]:
    summary = _dir_summary(path)
    sanity = _load_json(path / "visualization_sanity_summary.json")
    summary["sanity_summary"] = sanity
    if isinstance(sanity, list) and sanity:
        summary["sanity_stats"] = {
            "sample_count": len(sanity),
            "success_count": sum(1 for item in sanity if item.get("status") == "success"),
            "mode_distribution": {
                str(mode): sum(1 for item in sanity if str(item.get("extraction_mode")) == str(mode))
                for mode in sorted({str(item.get("extraction_mode")) for item in sanity})
            },
            "mean_tool_fraction": float(sum(float(item.get("tool_based_interaction_fraction", 0.0)) for item in sanity) / len(sanity)),
        }
    return summary


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    complexes = pd.read_parquet(CANONICAL_DIR / "complexes.parquet")
    chains = pd.read_parquet(CANONICAL_DIR / "chains.parquet")
    residues = pd.read_parquet(CANONICAL_DIR / "residues.parquet")
    pair_report = _load_json(PAIRS_DIR / "pair_data_report.json")
    candidate_report = _load_json(PAIRS_DIR / "candidate_set_report.json")

    gnn_cfg = yaml.safe_load((ROOT / "configs" / "train_v0_2_gnn_esm2.yaml").read_text(encoding="utf-8"))
    mlx_cfg = yaml.safe_load((ROOT / "configs" / "train_v0_1_scoring_mlx_m4.yaml").read_text(encoding="utf-8"))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": "PeptiProp",
            "active_final_model": "GATv2 + ESM-2 v0.2",
            "baseline_model": "MLX MLP v0.1",
            "dataset_policy": "PROPEDIA-only",
            "reported_visual_interaction_source": "geometric residue-contact fallback",
            "external_tool_extractors_used_in_reported_results": False,
        },
        "data": {
            "raw_propedia": {
                "complex_cif_count": _count_files(DATA_DIR / "raw" / "propedia" / "complexes", "*.cif"),
                "interface_file_count": _count_files(DATA_DIR / "raw" / "propedia" / "interfaces", "*"),
            },
            "canonical": {
                "complex_rows": int(len(complexes)),
                "chain_rows": int(len(chains)),
                "residue_rows": int(len(residues)),
                "unique_pdb_ids": int(complexes["pdb_id"].nunique()),
                "quality_distribution": {
                    str(k): int(v) for k, v in complexes["quality_flag"].value_counts().to_dict().items()
                },
                "split_distribution": {
                    str(k): int(v) for k, v in complexes["split_tag"].value_counts().to_dict().items()
                },
            },
            "pairs": {
                "pair_report": pair_report,
                "candidate_set_report": candidate_report,
            },
        },
        "processing": {
            "split_strategy": "sequence-cluster-aware leakage-free split (MMseqs2 30% identity, exact fallback)",
            "negative_sampling": "split-local easy + hard negatives, 1 positive + 5 negatives per protein",
            "embedding_surface": {
                "esm2_npz_count": _count_files(DATA_DIR / "embeddings" / "esm2_residue", "*.npz"),
                "graph_pt_count": _count_files(DATA_DIR / "graphs", "*.pt"),
                "mlx_feature_npz_count": _count_files(DATA_DIR / "mlx" / "features_v0_1_m4", "*.npz"),
            },
        },
        "methods": {
            "gnn_config": {
                "model": gnn_cfg.get("model", {}),
                "training": gnn_cfg.get("training", {}),
                "loss": gnn_cfg.get("loss", {}),
                "evaluation": gnn_cfg.get("evaluation", {}),
            },
            "mlx_config": {
                "model": mlx_cfg.get("model", {}),
                "training": mlx_cfg.get("training", {}),
                "loss": mlx_cfg.get("loss", {}),
                "evaluation": mlx_cfg.get("evaluation", {}),
            },
        },
        "results": {
            "gnn_final": _training_summary(TRAINING_DIRS["gnn_final"]),
            "mlx_final": _training_summary(TRAINING_DIRS["mlx_final"]),
        },
        "outputs": {
            "training_dirs": {name: _training_summary(path) for name, path in TRAINING_DIRS.items()},
            "visualization_dirs": {name: _visual_summary(path) for name, path in VIS_DIRS.items()},
        },
    }

    REPORT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] project state manifest -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
