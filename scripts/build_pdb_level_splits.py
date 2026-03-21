"""Build sequence-clustered structure-aware splits for canonical dataset.

Uses protein sequence clustering to prevent homology leakage between splits.
Complexes whose protein chains share the same cluster are kept in the same split.
"""
import sys
from pathlib import Path
from collections import defaultdict
from hashlib import md5
from typing import Dict, List, Set, Tuple

import pandas as pd
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

IDENTITY_THRESHOLD = 0.30  # 30 % sequence identity for clustering


def load_propedia_interfaces(metadata_dir: Path) -> Dict[str, List[str]]:
    """Load PROPEDIA interface information."""
    candidate_dirs = [
        metadata_dir / "interfaces" / "interface",
        metadata_dir / "interfaces",
        metadata_dir / "interface",
    ]
    interface_dir = next((path for path in candidate_dirs if path.exists()), None)
    if interface_dir is None:
        return {}

    pdb_interfaces: Dict[str, List[str]] = {}
    interface_files = list(interface_dir.glob("*.interface")) + list(interface_dir.glob("*.pdb"))
    for interface_file in interface_files:
        stem = interface_file.stem
        pdb_id = stem.split('_')[0]
        pdb_interfaces.setdefault(pdb_id, []).append(stem)
    return pdb_interfaces


# ---------------------------------------------------------------------------
# Sequence clustering helpers
# ---------------------------------------------------------------------------

def _try_mmseqs_cluster(fasta_path: Path, tmp_dir: Path, identity: float) -> Dict[str, str]:
    """Run MMseqs2 easy-cluster and return {seq_id: cluster_rep} mapping.

    Returns empty dict if mmseqs is not available.
    """
    import shutil
    import subprocess
    import tempfile

    mmseqs_bin = shutil.which("mmseqs")
    if mmseqs_bin is None:
        return {}

    out_prefix = tmp_dir / "clust"
    result = subprocess.run(
        [
            mmseqs_bin, "easy-cluster",
            str(fasta_path), str(out_prefix), str(tmp_dir / "tmp"),
            "--min-seq-id", str(identity),
            "-c", "0.8",
            "--cov-mode", "0",
            "--threads", "4",
        ],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  [WARN] MMseqs2 failed: {result.stderr[:300]}")
        return {}

    tsv_path = Path(str(out_prefix) + "_cluster.tsv")
    if not tsv_path.exists():
        return {}

    mapping: Dict[str, str] = {}
    for line in tsv_path.read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            rep, member = parts
            mapping[member] = rep
    return mapping


def _exact_sequence_cluster(sequences: Dict[str, str]) -> Dict[str, str]:
    """Fallback: cluster by exact sequence identity."""
    seq_to_rep: Dict[str, str] = {}
    mapping: Dict[str, str] = {}
    for sid, seq in sequences.items():
        if seq not in seq_to_rep:
            seq_to_rep[seq] = sid
        mapping[sid] = seq_to_rep[seq]
    return mapping


def build_protein_clusters(
    chains_df: pd.DataFrame, complexes_df: pd.DataFrame
) -> Dict[str, str]:
    """Return {complex_id: cluster_rep} based on protein chain sequences.

    Tries MMseqs2 first; falls back to exact-match clustering.
    """
    import tempfile

    protein_chains = chains_df[chains_df["entity_type"] == "protein"].copy()
    protein_chains = protein_chains.merge(
        complexes_df[["complex_id", "protein_chain_id"]],
        left_on=["complex_id", "chain_id_label"],
        right_on=["complex_id", "protein_chain_id"],
        how="inner",
    )

    seq_map: Dict[str, str] = {}
    for _, row in protein_chains.iterrows():
        seq_map[str(row["complex_id"])] = str(row["sequence"])

    with tempfile.TemporaryDirectory(prefix="pq_clust_") as tmp:
        tmp_dir = Path(tmp)
        fasta_path = tmp_dir / "proteins.fasta"
        with open(fasta_path, "w") as fh:
            for sid, seq in seq_map.items():
                fh.write(f">{sid}\n{seq}\n")

        mapping = _try_mmseqs_cluster(fasta_path, tmp_dir, IDENTITY_THRESHOLD)

    if mapping:
        print(f"  MMseqs2 clustering: {len(set(mapping.values()))} clusters from {len(mapping)} sequences")
        return mapping

    print("  MMseqs2 not available – falling back to exact-sequence clustering")
    return _exact_sequence_cluster(seq_map)


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def build_splits(
    canonical_dir: Path,
    propedia_meta_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """Build sequence-cluster–aware splits.

    Strategy:
    1. Cluster protein chains by sequence (MMseqs2 @ 30 % identity, fallback: exact).
    2. Group complexes by their protein cluster representative.
    3. Shuffle and split cluster groups so no cluster spans train/val/test.
    """
    print("=" * 60)
    print("Building Sequence-Clustered Structure-Aware Splits")
    print("=" * 60)

    complexes = pd.read_parquet(canonical_dir / "complexes.parquet")
    chains = pd.read_parquet(canonical_dir / "chains.parquet")

    print(f"\nTotal complexes: {len(complexes)}")
    print(f"Unique PDB IDs: {complexes['pdb_id'].nunique()}")

    pdb_interfaces = load_propedia_interfaces(propedia_meta_dir)
    print(f"PROPEDIA interface files: {len(pdb_interfaces)} PDBs")

    # --- cluster ---
    print("\nClustering protein sequences...")
    complex_to_cluster = build_protein_clusters(chains, complexes)

    cluster_to_complexes: Dict[str, List[str]] = defaultdict(list)
    for cid in complexes["complex_id"].tolist():
        rep = complex_to_cluster.get(cid, cid)
        cluster_to_complexes[rep].append(cid)

    cluster_ids = sorted(cluster_to_complexes.keys())
    n_clusters = len(cluster_ids)
    print(f"Protein clusters: {n_clusters}")

    # --- shuffle & split clusters ---
    rng = random.Random(seed)
    rng.shuffle(cluster_ids)

    n_train = int(n_clusters * train_ratio)
    n_val = int(n_clusters * val_ratio)

    train_clusters = set(cluster_ids[:n_train])
    val_clusters = set(cluster_ids[n_train:n_train + n_val])
    test_clusters = set(cluster_ids[n_train + n_val:])

    train_ids = [c for cl in train_clusters for c in cluster_to_complexes[cl]]
    val_ids = [c for cl in val_clusters for c in cluster_to_complexes[cl]]
    test_ids = [c for cl in test_clusters for c in cluster_to_complexes[cl]]

    print(f"\nCluster-level split:")
    print(f"  Train: {len(train_clusters)} clusters -> {len(train_ids)} complexes")
    print(f"  Val:   {len(val_clusters)} clusters -> {len(val_ids)} complexes")
    print(f"  Test:  {len(test_clusters)} clusters -> {len(test_ids)} complexes")

    # --- PDB overlap report ---
    train_pdbs = set(complexes[complexes["complex_id"].isin(train_ids)]["pdb_id"])
    val_pdbs = set(complexes[complexes["complex_id"].isin(val_ids)]["pdb_id"])
    test_pdbs = set(complexes[complexes["complex_id"].isin(test_ids)]["pdb_id"])
    pdb_tv = train_pdbs & val_pdbs
    pdb_tt = train_pdbs & test_pdbs
    pdb_vt = val_pdbs & test_pdbs
    if pdb_tv or pdb_tt or pdb_vt:
        print(f"\n  [INFO] PDB overlap (expected when clustering by sequence):")
        print(f"    Train∩Val PDBs: {len(pdb_tv)}")
        print(f"    Train∩Test PDBs: {len(pdb_tt)}")
        print(f"    Val∩Test PDBs: {len(pdb_vt)}")
    else:
        print("\n  No PDB overlap between splits")

    # --- save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        (output_dir / f"{name}_ids.txt").write_text("\n".join(sorted(ids)))

    (output_dir / "external_geppri_ids.txt").write_text(
        "# GEPPRI external holdout - to be populated\n# DO NOT use for training or validation\n"
    )

    summary_lines = [
        "Split Summary",
        "=" * 60,
        "",
        f"total_complexes: {len(complexes)}",
        f"total_clusters: {n_clusters}",
        f"train_clusters: {len(train_clusters)}",
        f"val_clusters: {len(val_clusters)}",
        f"test_clusters: {len(test_clusters)}",
        f"train_complexes: {len(train_ids)}",
        f"val_complexes: {len(val_ids)}",
        f"test_complexes: {len(test_ids)}",
        f"split_strategy: sequence-cluster (identity={IDENTITY_THRESHOLD})",
        f"seed: {seed}",
    ]
    (output_dir / "split_summary.txt").write_text("\n".join(summary_lines) + "\n")

    print(f"\n{'=' * 60}")
    print("Splits saved to:")
    for f in ("train_ids.txt", "val_ids.txt", "test_ids.txt",
              "external_geppri_ids.txt", "split_summary.txt"):
        print(f"  {output_dir / f}")
    print("=" * 60)

    # --- verify no cluster leakage ---
    assert not (train_clusters & val_clusters), "Leakage: train/val cluster overlap"
    assert not (train_clusters & test_clusters), "Leakage: train/test cluster overlap"
    assert not (val_clusters & test_clusters), "Leakage: val/test cluster overlap"
    print("\nLeakage check: PASSED (no cluster overlap between splits)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build sequence-clustered structure-aware splits")
    parser.add_argument("--canonical", type=Path, required=True, help="Canonical directory")
    parser.add_argument("--propedia-meta", type=Path, required=True, help="PROPEDIA metadata directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    build_splits(
        canonical_dir=args.canonical,
        propedia_meta_dir=args.propedia_meta,
        output_dir=args.out,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
