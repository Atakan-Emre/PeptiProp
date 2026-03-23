"""
Pre-build all residue-level PyG graphs for protein and peptide chains
referenced in the pair data. Saves graphs to data/graphs/ as individual .pt files,
keyed by complex_id::chain_id.

This avoids expensive on-the-fly graph construction during training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peptidquantum.models.graph_builder import PairGraphBuilder


def collect_required_chains(pairs_dir: Path) -> Set[Tuple[str, str]]:
    """Collect all unique (complex_id, chain_id) pairs from train/val/test."""
    chains: Set[Tuple[str, str]] = set()
    for split in ["train_pairs.parquet", "val_pairs.parquet", "test_pairs.parquet"]:
        path = pairs_dir / split
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            chains.add((str(row["protein_complex_id"]), str(row["protein_chain_id"])))
            chains.add((str(row["peptide_complex_id"]), str(row["peptide_chain_id"])))
    return chains


def main():
    parser = argparse.ArgumentParser(description="Pre-build residue graphs")
    parser.add_argument("--canonical-dir", type=Path, default=Path("data/canonical"))
    parser.add_argument("--pairs-dir", type=Path, default=Path("data/canonical/pairs"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/embeddings/esm2_residue"))
    parser.add_argument("--embedding-lookup", type=Path, default=Path("data/embeddings/esm2_chain_lookup.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max-chains", type=int, default=None, help="Limit for smoke testing")
    args = parser.parse_args()

    with open(args.embedding_lookup) as f:
        lookup = json.load(f)
    builder = PairGraphBuilder(args.embedding_dir, lookup, cutoff=args.cutoff)

    print("Gerekli zincirler toplanıyor...")
    required = collect_required_chains(args.pairs_dir)
    print(f"Toplam benzersiz zincir: {len(required)}")

    if args.max_chains:
        required = set(list(required)[:args.max_chains])
        print(f"Smoke test: ilk {args.max_chains} zincir")

    print("Rezidüler yükleniyor...")
    residues = pd.read_parquet(
        args.canonical_dir / "residues.parquet",
        columns=["complex_id", "chain_id", "residue_number_auth", "resname",
                 "is_interface", "is_pocket", "x", "y", "z", "secondary_structure"],
    )

    # Pre-partition residues into a dict for fast lookup
    print("Rezidüler indeksleniyor...")
    res_dict: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (cid, ch), group in residues.groupby(["complex_id", "chain_id"]):
        key = (str(cid), str(ch))
        if key in required:
            res_dict[key] = group

    del residues
    print(f"İndekslenen zincir: {len(res_dict)}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    built = 0
    skipped = 0
    for i, (cid, ch) in enumerate(required):
        safe_key = f"{cid}__{ch}"
        out_path = out_dir / f"{safe_key}.pt"
        if out_path.exists():
            skipped += 1
            continue

        res_df = res_dict.get((cid, ch))
        if res_df is None or res_df.empty:
            continue

        graph = builder.build_chain_graph(res_df, cid, ch)
        if graph is None:
            continue

        torch.save(graph, out_path)
        built += 1

        if (built + skipped) % 500 == 0 or (i + 1) == len(required):
            elapsed = time.time() - t0
            total = built + skipped
            print(f"  [{total}/{len(required)}] {elapsed:.0f}s — built={built}, skipped={skipped}")

    elapsed = time.time() - t0
    print(f"\nToplam: {built} graf oluşturuldu, {skipped} atlandı")
    print(f"Süre: {elapsed:.0f}s")
    print(f"Çıktı: {out_dir}")


if __name__ == "__main__":
    main()
