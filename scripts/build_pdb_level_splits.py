"""Build PDB-level structure-aware splits for canonical dataset."""
import sys
from pathlib import Path
import pandas as pd
import random
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_propedia_interfaces(metadata_dir: Path) -> Dict[str, List[str]]:
    """
    Load PROPEDIA interface information
    
    Returns dict: {pdb_id: [interface_files]}
    """
    candidate_dirs = [
        metadata_dir / "interfaces" / "interface",
        metadata_dir / "interfaces",
        metadata_dir / "interface",
    ]
    interface_dir = next((path for path in candidate_dirs if path.exists()), None)

    if interface_dir is None:
        return {}
    
    pdb_interfaces = {}
    interface_files = list(interface_dir.glob("*.interface")) + list(interface_dir.glob("*.pdb"))

    for interface_file in interface_files:
        # Format: {pdb_id}_{chain1}_{chain2}.{interface|pdb}
        stem = interface_file.stem
        pdb_id = stem.split('_')[0]
        
        if pdb_id not in pdb_interfaces:
            pdb_interfaces[pdb_id] = []
        pdb_interfaces[pdb_id].append(stem)
    
    return pdb_interfaces


def build_splits(
    canonical_dir: Path,
    propedia_meta_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Build PDB-level structure-aware splits.
    
    Strategy:
    - Group by PDB ID (structure-level split to prevent leakage)
    - Random split at PDB level (cluster info not available in current metadata)
    - GEPPRI kept separate as external holdout
    """
    
    print("="*60)
    print("Building PDB-Level Structure-Aware Splits")
    print("="*60)
    
    # Load canonical complexes
    complexes = pd.read_parquet(canonical_dir / "complexes.parquet")
    
    print(f"\nTotal complexes: {len(complexes)}")
    print(f"Unique PDB IDs: {complexes['pdb_id'].nunique()}")
    
    # Load PROPEDIA interface info
    pdb_interfaces = load_propedia_interfaces(propedia_meta_dir)
    print(f"PROPEDIA interface files: {len(pdb_interfaces)} PDBs")
    
    # Get unique PDB IDs from canonical
    pdb_ids = complexes['pdb_id'].unique().tolist()
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(pdb_ids)
    
    # Calculate split sizes
    n_total = len(pdb_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split at PDB level
    train_pdbs = set(pdb_ids[:n_train])
    val_pdbs = set(pdb_ids[n_train:n_train + n_val])
    test_pdbs = set(pdb_ids[n_train + n_val:])
    
    print(f"\nPDB-level split:")
    print(f"  Train: {len(train_pdbs)} PDBs")
    print(f"  Val: {len(val_pdbs)} PDBs")
    print(f"  Test: {len(test_pdbs)} PDBs")
    
    # Get complex IDs for each split
    train_ids = complexes[complexes['pdb_id'].isin(train_pdbs)]['complex_id'].tolist()
    val_ids = complexes[complexes['pdb_id'].isin(val_pdbs)]['complex_id'].tolist()
    test_ids = complexes[complexes['pdb_id'].isin(test_pdbs)]['complex_id'].tolist()
    
    print(f"\nComplex-level split:")
    print(f"  Train: {len(train_ids)} complexes")
    print(f"  Val: {len(val_ids)} complexes")
    print(f"  Test: {len(test_ids)} complexes")
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "train_ids.txt", 'w') as f:
        f.write('\n'.join(train_ids))
    
    with open(output_dir / "val_ids.txt", 'w') as f:
        f.write('\n'.join(val_ids))
    
    with open(output_dir / "test_ids.txt", 'w') as f:
        f.write('\n'.join(test_ids))
    
    # GEPPRI external holdout (placeholder - will be populated when GEPPRI is processed)
    with open(output_dir / "external_geppri_ids.txt", 'w') as f:
        f.write("# GEPPRI external holdout - to be populated\n")
        f.write("# DO NOT use for training or validation\n")
    
    # Generate split summary
    summary = {
        'total_complexes': len(complexes),
        'total_pdbs': n_total,
        'train_pdbs': len(train_pdbs),
        'val_pdbs': len(val_pdbs),
        'test_pdbs': len(test_pdbs),
        'train_complexes': len(train_ids),
        'val_complexes': len(val_ids),
        'test_complexes': len(test_ids),
        'split_strategy': 'PDB-level structure-aware',
        'seed': seed
    }
    
    summary_file = output_dir / "split_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Split Summary\n")
        f.write("="*60 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n" + "="*60)
    print("Splits saved to:")
    print(f"  {output_dir / 'train_ids.txt'}")
    print(f"  {output_dir / 'val_ids.txt'}")
    print(f"  {output_dir / 'test_ids.txt'}")
    print(f"  {output_dir / 'external_geppri_ids.txt'}")
    print(f"  {output_dir / 'split_summary.txt'}")
    print("="*60)
    
    # Verify no leakage
    train_set = set(train_pdbs)
    val_set = set(val_pdbs)
    test_set = set(test_pdbs)
    
    assert len(train_set & val_set) == 0, "Leakage: train/val overlap"
    assert len(train_set & test_set) == 0, "Leakage: train/test overlap"
    assert len(val_set & test_set) == 0, "Leakage: val/test overlap"
    
    print("\nLeakage check: PASSED (no PDB overlap between splits)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build PDB-level structure-aware splits")
    parser.add_argument("--canonical", type=Path, required=True,
                       help="Canonical directory")
    parser.add_argument("--propedia-meta", type=Path, required=True,
                       help="PROPEDIA metadata directory")
    parser.add_argument("--out", type=Path, required=True,
                       help="Output directory for splits")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Train ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Val ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    build_splits(
        canonical_dir=args.canonical,
        propedia_meta_dir=args.propedia_meta,
        output_dir=args.out,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
