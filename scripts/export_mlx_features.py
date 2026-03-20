"""
Export leakage-free pair features for MLX training on Apple Silicon.

This script builds per-pair dense features from canonical residues/chains and
pair parquet files, then writes compact NPZ files for train/val/test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml


AA_VOCAB = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,
}

HYDROPHOBIC_AA_IDS = [
    AA_VOCAB["ALA"],
    AA_VOCAB["VAL"],
    AA_VOCAB["ILE"],
    AA_VOCAB["LEU"],
    AA_VOCAB["MET"],
    AA_VOCAB["PHE"],
    AA_VOCAB["TRP"],
    AA_VOCAB["TYR"],
    AA_VOCAB["CYS"],
    AA_VOCAB["PRO"],
]
POLAR_AA_IDS = [AA_VOCAB["SER"], AA_VOCAB["THR"], AA_VOCAB["ASN"], AA_VOCAB["GLN"]]
POSITIVE_AA_IDS = [AA_VOCAB["LYS"], AA_VOCAB["ARG"], AA_VOCAB["HIS"]]
NEGATIVE_AA_IDS = [AA_VOCAB["ASP"], AA_VOCAB["GLU"]]

CHAIN_SUMMARY_BASE_DIM = 11
CHAIN_SUMMARY_AA_DIM = len(AA_VOCAB)
CHAIN_SUMMARY_DIM = CHAIN_SUMMARY_BASE_DIM + CHAIN_SUMMARY_AA_DIM
PAIR_COMPAT_DIM = 3
PAIR_FEATURE_DIM = CHAIN_SUMMARY_DIM * 4 + PAIR_COMPAT_DIM

RESIDUE_COLUMNS = [
    "complex_id",
    "chain_id",
    "resname",
    "is_interface",
    "is_pocket",
    "x",
    "y",
    "z",
    "secondary_structure",
]


def normalize_secondary_structure(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"

    text = str(value).strip().lower()
    if not text or text == "none":
        return "unknown"
    if text in {"h", "g", "i", "helix", "alpha_helix", "310_helix", "pi_helix"}:
        return "helix"
    if text in {"e", "b", "sheet", "beta_sheet", "strand"}:
        return "sheet"
    if text in {"c", "coil", "loop", "turn", "bend", "s"}:
        return "coil"
    return "unknown"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_chain_summary(
    residues_df: pd.DataFrame,
    chain_type: str,
    use_local_density: bool,
    local_density_radius: float,
) -> np.ndarray:
    residues_df = residues_df.reset_index(drop=True)
    n_residues = len(residues_df)

    if n_residues == 0:
        raise ValueError("Chain has zero residues; cannot build summary")

    # Optional local density (disabled by default on MLX export for speed).
    local_density_mean = 0.0
    if use_local_density and n_residues > 1:
        coords = residues_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        # O(N^2) per chain; keep optional.
        dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        density = (dists <= local_density_radius).sum(axis=1) - 1.0
        density = density / float(n_residues - 1)
        local_density_mean = float(np.mean(density))

    ss = residues_df["secondary_structure"].map(normalize_secondary_structure)
    helix_fraction = float((ss == "helix").mean())
    sheet_fraction = float((ss == "sheet").mean())
    coil_fraction = float((ss == "coil").mean())

    aa_ids = residues_df["resname"].map(lambda v: AA_VOCAB.get(v, AA_VOCAB["UNK"])).to_numpy(dtype=np.int64)
    aa_counts = np.bincount(aa_ids, minlength=len(AA_VOCAB)).astype(np.float32)
    aa_freq = aa_counts / max(float(n_residues), 1.0)

    hydrophobic_fraction = float(np.sum(aa_freq[HYDROPHOBIC_AA_IDS]))
    polar_fraction = float(np.sum(aa_freq[POLAR_AA_IDS]))
    positive_fraction = float(np.sum(aa_freq[POSITIVE_AA_IDS]))
    negative_fraction = float(np.sum(aa_freq[NEGATIVE_AA_IDS]))

    length_scale = 500.0 if chain_type == "protein" else 50.0

    base = np.array(
        [
            min(float(n_residues) / length_scale, 1.0),
            float(residues_df["is_interface"].fillna(False).astype(float).mean()),
            float(residues_df["is_pocket"].fillna(False).astype(float).mean()),
            local_density_mean,
            helix_fraction,
            sheet_fraction,
            coil_fraction,
            hydrophobic_fraction,
            polar_fraction,
            positive_fraction,
            negative_fraction,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, aa_freq.astype(np.float32)], axis=0).astype(np.float32)


def _pair_features(protein_summary: np.ndarray, peptide_summary: np.ndarray) -> np.ndarray:
    eps = 1e-8
    cosine = float(
        np.dot(protein_summary, peptide_summary)
        / (np.linalg.norm(protein_summary) * np.linalg.norm(peptide_summary) + eps)
    )
    l2 = float(np.linalg.norm(protein_summary - peptide_summary))
    dot = float(np.dot(protein_summary, peptide_summary))

    return np.concatenate(
        [
            protein_summary,
            peptide_summary,
            np.abs(protein_summary - peptide_summary),
            protein_summary * peptide_summary,
            np.array([cosine, l2, dot], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _candidate_set_distribution(df: pd.DataFrame) -> Dict[str, int]:
    protein_group = df["protein_complex_id"].astype(str) + "::" + df["protein_chain_id"].astype(str)
    return {
        str(k): int(v)
        for k, v in protein_group.value_counts().sort_index().value_counts().sort_index().to_dict().items()
    }


def _build_split_report(df: pd.DataFrame) -> dict:
    protein_group = df["protein_complex_id"].astype(str) + "::" + df["protein_chain_id"].astype(str)
    split_values = sorted(df["split"].dropna().astype(str).unique().tolist()) if "split" in df.columns else []
    dup_cols = ["protein_complex_id", "protein_chain_id", "peptide_complex_id", "peptide_chain_id"]
    return {
        "total_pairs": int(len(df)),
        "positive_pairs": int((df["label"] == 1).sum()),
        "negative_pairs": int((df["label"] == 0).sum()),
        "negative_type_distribution": {
            str(k): int(v)
            for k, v in df["negative_type"].fillna("unknown").value_counts().to_dict().items()
        },
        "quality_flag_distribution": {
            str(k): int(v)
            for k, v in df["pair_quality_flag"].fillna("unknown").value_counts().to_dict().items()
        },
        "duplicate_pair_count": int(df[dup_cols].duplicated().sum()),
        "split_column_consistent": split_values == [str(df["split"].iloc[0])] if len(df) > 0 and "split" in df.columns else True,
        "candidate_set_size_distribution": _candidate_set_distribution(df),
        "candidate_set_avg_size": float(protein_group.value_counts().mean()) if len(df) else 0.0,
        "unique_protein_groups": int(protein_group.nunique()),
    }


def _iter_pairs(df: pd.DataFrame) -> Iterable[Tuple[Tuple[str, str, str], Tuple[str, str, str]]]:
    for row in df.itertuples(index=False):
        yield (
            (str(row.protein_complex_id), str(row.protein_chain_id), "protein"),
            (str(row.peptide_complex_id), str(row.peptide_chain_id), "peptide"),
        )


def build_chain_summaries(
    split_dfs: Dict[str, pd.DataFrame],
    canonical_dir: Path,
    use_local_density: bool,
    local_density_radius: float,
) -> Dict[Tuple[str, str, str], np.ndarray]:
    required_typed_keys: Dict[Tuple[str, str, str], str] = {}
    required_chain_keys: set[Tuple[str, str]] = set()

    for df in split_dfs.values():
        for protein_key, peptide_key in _iter_pairs(df):
            required_typed_keys[protein_key] = "protein"
            required_typed_keys[peptide_key] = "peptide"
            required_chain_keys.add((protein_key[0], protein_key[1]))
            required_chain_keys.add((peptide_key[0], peptide_key[1]))

    key_df = pd.DataFrame(sorted(required_chain_keys), columns=["complex_id", "chain_id"])
    residues = pd.read_parquet(canonical_dir / "residues.parquet", columns=RESIDUE_COLUMNS)
    residues = residues.merge(key_df, on=["complex_id", "chain_id"], how="inner")

    grouped = residues.groupby(["complex_id", "chain_id"], sort=False)
    per_chain_summary: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for (complex_id, chain_id), group in grouped:
        per_chain_summary[(str(complex_id), str(chain_id))] = {}

    summaries: Dict[Tuple[str, str, str], np.ndarray] = {}
    for typed_key, chain_type in required_typed_keys.items():
        chain_key = (typed_key[0], typed_key[1])
        if chain_key not in per_chain_summary:
            raise KeyError(f"Missing residues for key={typed_key}")
        group = grouped.get_group(chain_key)
        summaries[typed_key] = _build_chain_summary(
            group,
            chain_type=chain_type,
            use_local_density=use_local_density,
            local_density_radius=local_density_radius,
        )
    return summaries


def build_pair_matrix(df: pd.DataFrame, summaries: Dict[Tuple[str, str, str], np.ndarray]) -> dict:
    n = len(df)
    x = np.zeros((n, PAIR_FEATURE_DIM), dtype=np.float32)
    y = df["label"].astype(np.float32).to_numpy()
    pair_id = df["pair_id"].astype(str).to_numpy()
    neg_type = df["negative_type"].fillna("unknown").astype(str).to_numpy()

    protein_group = df["protein_complex_id"].astype(str) + "::" + df["protein_chain_id"].astype(str)
    group_idx, group_names = pd.factorize(protein_group)

    for i, row in enumerate(df.itertuples(index=False)):
        p_key = (str(row.protein_complex_id), str(row.protein_chain_id), "protein")
        pep_key = (str(row.peptide_complex_id), str(row.peptide_chain_id), "peptide")
        p_summary = summaries[p_key]
        pep_summary = summaries[pep_key]
        x[i] = _pair_features(p_summary, pep_summary)

    return {
        "x": x,
        "y": y.astype(np.float32),
        "group_idx": group_idx.astype(np.int32),
        "group_names": group_names.astype(str).to_numpy(),
        "pair_id": pair_id,
        "negative_type": neg_type,
    }


def main():
    parser = argparse.ArgumentParser(description="Export leakage-free NPZ features for MLX training")
    parser.add_argument("--config", type=Path, required=True, help="MLX config yaml")
    parser.add_argument("--max-pairs-per-split", type=int, default=None, help="Optional smoke cap per split")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    canonical_dir = Path(cfg["data"]["canonical_dir"])
    pairs_dir = Path(cfg["data"]["pairs_dir"])
    output_dir = Path(cfg["data"]["feature_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_filter = cfg["data"].get("quality_filter", "clean")
    use_local_density = bool(cfg["features"].get("use_local_density", False))
    local_density_radius = float(cfg["features"].get("local_density_radius", 8.0))

    split_dfs: Dict[str, pd.DataFrame] = {}
    pair_report = {}
    for split in ("train", "val", "test"):
        df = pd.read_parquet(pairs_dir / f"{split}_pairs.parquet")
        if quality_filter:
            df = df[df["pair_quality_flag"] == quality_filter].reset_index(drop=True)
        if args.max_pairs_per_split and len(df) > args.max_pairs_per_split:
            df = df.sample(n=args.max_pairs_per_split, random_state=42).reset_index(drop=True)
        split_dfs[split] = df
        pair_report[split] = _build_split_report(df)

    summaries = build_chain_summaries(
        split_dfs=split_dfs,
        canonical_dir=canonical_dir,
        use_local_density=use_local_density,
        local_density_radius=local_density_radius,
    )

    for split, df in split_dfs.items():
        matrix = build_pair_matrix(df, summaries)
        np.savez_compressed(
            output_dir / f"{split}_mlx_features.npz",
            x=matrix["x"],
            y=matrix["y"],
            group_idx=matrix["group_idx"],
            group_names=matrix["group_names"],
            pair_id=matrix["pair_id"],
            negative_type=matrix["negative_type"],
            feature_dim=np.array([PAIR_FEATURE_DIM], dtype=np.int32),
        )

    with open(output_dir / "pair_data_report.json", "w", encoding="utf-8") as f:
        json.dump(pair_report, f, indent=2)

    candidate_report = {
        split: {
            "candidate_set_size_distribution": pair_report[split]["candidate_set_size_distribution"],
            "candidate_set_avg_size": pair_report[split]["candidate_set_avg_size"],
            "unique_protein_groups": pair_report[split]["unique_protein_groups"],
            "positive_pairs": pair_report[split]["positive_pairs"],
            "negative_pairs": pair_report[split]["negative_pairs"],
        }
        for split in ("train", "val", "test")
    }
    with open(output_dir / "candidate_set_report.json", "w", encoding="utf-8") as f:
        json.dump(candidate_report, f, indent=2)

    with open(output_dir / "feature_export_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "quality_filter": quality_filter,
                "pair_feature_dim": PAIR_FEATURE_DIM,
                "chain_summary_dim": CHAIN_SUMMARY_DIM,
                "use_local_density": use_local_density,
                "local_density_radius": local_density_radius,
            },
            f,
            indent=2,
        )

    print("=" * 60)
    print("MLX feature export complete")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(split_dfs[split])} pairs -> {output_dir / f'{split}_mlx_features.npz'}")
    print(f"Pair feature dim: {PAIR_FEATURE_DIM}")


if __name__ == "__main__":
    main()

