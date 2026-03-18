"""Check complex uniqueness and pair explosion"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_uniqueness(canonical_dir: Path, output_file: Path):
    """Check for duplicate complex_key and pair explosion"""
    
    # Load complexes
    complexes = pd.read_parquet(canonical_dir / "complexes.parquet")
    
    print("="*60)
    print("Complex Uniqueness Analysis")
    print("="*60)
    print(f"\nTotal complexes: {len(complexes)}")
    
    # Extract PDB ID from complex_id
    complexes['pdb_id_extracted'] = complexes['complex_id'].str.split('_').str[0]
    
    # Create complex_key
    complexes['complex_key'] = (
        complexes['pdb_id_extracted'] + '_' + 
        complexes['protein_chain_id'] + '_' + 
        complexes['peptide_chain_id']
    )
    
    # Check duplicates
    duplicates = complexes[complexes.duplicated(subset=['complex_key'], keep=False)]
    
    print(f"Unique complex_key: {complexes['complex_key'].nunique()}")
    print(f"Duplicate complex_key: {len(duplicates)}")
    
    if len(duplicates) > 0:
        print("\nWARNING: Duplicates found!")
        print("\nTop 10 duplicated complex_key:")
        dup_counts = duplicates['complex_key'].value_counts().head(10)
        for key, count in dup_counts.items():
            print(f"  {key}: {count} times")
    
    # Pair explosion analysis
    print("\n" + "="*60)
    print("Pair Explosion Analysis")
    print("="*60)
    
    pairs_per_pdb = complexes.groupby('pdb_id_extracted').size()
    
    print(f"\nPairs per PDB statistics:")
    print(f"  Mean: {pairs_per_pdb.mean():.1f}")
    print(f"  Median: {pairs_per_pdb.median():.1f}")
    print(f"  Max: {pairs_per_pdb.max()}")
    print(f"  Min: {pairs_per_pdb.min()}")
    
    print(f"\nTop 10 PDBs by pair count:")
    for pdb_id, count in pairs_per_pdb.nlargest(10).items():
        print(f"  {pdb_id}: {count} pairs")
    
    # Source analysis
    print("\n" + "="*60)
    print("Source Database Analysis")
    print("="*60)
    
    source_counts = complexes['source_db'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count} complexes")
    
    # Create detailed report
    report = complexes.groupby('complex_key').agg({
        'pdb_id_extracted': 'first',
        'protein_chain_id': 'first',
        'peptide_chain_id': 'first',
        'complex_id': 'count',
        'source_db': lambda x: ','.join(x.unique())
    }).rename(columns={'complex_id': 'count'})
    
    report['is_duplicate'] = report['count'] > 1
    report = report.sort_values('count', ascending=False)
    
    # Save report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_file)
    
    print(f"\n" + "="*60)
    print(f"Report saved to: {output_file}")
    print("="*60)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  Total records: {len(complexes)}")
    print(f"  Unique complex_key: {complexes['complex_key'].nunique()}")
    print(f"  Duplicates: {len(duplicates)} ({len(duplicates)/len(complexes)*100:.1f}%)")
    print(f"  Avg pairs per PDB: {pairs_per_pdb.mean():.1f}")
    
    if pairs_per_pdb.mean() > 100:
        print("\n  WARNING: High pair count per PDB suggests aggressive expansion!")
    
    if len(duplicates) > 0:
        print("\n  ACTION REQUIRED: Remove duplicates before splitting!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check complex uniqueness")
    parser.add_argument("--canonical", type=Path, required=True,
                       help="Canonical directory")
    parser.add_argument("--out", type=Path, required=True,
                       help="Output CSV file")
    
    args = parser.parse_args()
    
    check_uniqueness(args.canonical, args.out)
