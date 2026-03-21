"""Re-annotate is_interface and is_pocket in residues.parquet using distance-based criteria.

Run after canonical build to patch the residue data in-place.
Usage:
    python scripts/annotate_interface_pocket.py --canonical data/canonical
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

INTERFACE_CUTOFF = 5.0   # Å – residue centroid-to-centroid between chains
POCKET_CUTOFF = 8.0      # Å – protein residues near any peptide residue centroid


def annotate(canonical_dir: Path) -> None:
    complexes = pd.read_parquet(canonical_dir / "complexes.parquet")
    residues = pd.read_parquet(canonical_dir / "residues.parquet")

    residues["is_interface"] = False
    residues["is_pocket"] = False

    complex_map = {
        row.complex_id: (row.protein_chain_id, row.peptide_chain_id)
        for row in complexes.itertuples(index=False)
    }

    grouped = residues.groupby("complex_id", sort=False)
    updated = 0
    total_interface = 0
    total_pocket = 0

    for cid, group in grouped:
        if cid not in complex_map:
            continue
        prot_ch, pep_ch = complex_map[cid]

        prot_mask = group["chain_id"] == prot_ch
        pep_mask = group["chain_id"] == pep_ch

        prot_idx = group.index[prot_mask]
        pep_idx = group.index[pep_mask]

        if len(prot_idx) == 0 or len(pep_idx) == 0:
            continue

        prot_coords = group.loc[prot_idx, ["x", "y", "z"]].values.astype(np.float64)
        pep_coords = group.loc[pep_idx, ["x", "y", "z"]].values.astype(np.float64)

        diff = prot_coords[:, None, :] - pep_coords[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))

        min_per_prot = dists.min(axis=1)
        min_per_pep = dists.min(axis=0)

        prot_intf = min_per_prot <= INTERFACE_CUTOFF
        prot_pocket = min_per_prot <= POCKET_CUTOFF
        pep_intf = min_per_pep <= INTERFACE_CUTOFF

        residues.loc[prot_idx[prot_intf], "is_interface"] = True
        residues.loc[prot_idx[prot_pocket], "is_pocket"] = True
        residues.loc[pep_idx[pep_intf], "is_interface"] = True

        total_interface += int(prot_intf.sum()) + int(pep_intf.sum())
        total_pocket += int(prot_pocket.sum())
        updated += 1

    print(f"Annotated {updated} complexes")
    print(f"  Interface residues: {total_interface}")
    print(f"  Pocket residues:    {total_pocket}")
    print(f"  Total residues:     {len(residues)}")
    print(f"  Interface %:        {100 * total_interface / len(residues):.2f}%")
    print(f"  Pocket %:           {100 * total_pocket / len(residues):.2f}%")

    out = canonical_dir / "residues.parquet"
    residues.to_parquet(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate interface/pocket residues")
    parser.add_argument("--canonical", type=Path, required=True)
    args = parser.parse_args()
    annotate(args.canonical)
