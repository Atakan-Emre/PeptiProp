"""Rebuild canonical dataset with duplicate prevention and pair explosion limit"""
import sys
from pathlib import Path
import pandas as pd
import shutil

sys.path.insert(0, 'src')

from peptidquantum.data.processors.canonical_builder import CanonicalBuilder
from peptidquantum.data.canonical.schema import SourceDatabase

# Clean old canonical data
canonical_dir = Path('data/canonical')
if canonical_dir.exists():
    print("Removing old canonical data...")
    shutil.rmtree(canonical_dir)

# Read validation results to get passed complexes
validation_results = pd.read_csv('data/staging/validation/validation_results.csv')
passed = validation_results[validation_results['quarantine_reason'].isna()]

# Get mmCIF files
mmcif_files = []
for _, row in passed.iterrows():
    mmcif_path = Path(row['mmcif_path'])
    if mmcif_path.exists():
        mmcif_files.append(mmcif_path)

print(f"Building canonical dataset from {len(mmcif_files)} mmCIF files")
print(f"Max pairs per structure: 50")
print(f"Duplicate prevention: ENABLED\n")

# Initialize builder with limits
builder = CanonicalBuilder(
    staging_dir=Path('data/staging'),
    canonical_dir=Path('data/canonical'),
    chain_id_mode='auth',
    residue_number_mode='auth',
    max_pairs_per_structure=50  # Prevent explosion
)

# Build
builder.build(
    source_files=mmcif_files,
    source_db=SourceDatabase.PROPEDIA,
    batch_size=10
)

# Check results
print("\n" + "="*60)
print("Build Complete - Checking Results")
print("="*60)

parquet_files = list(canonical_dir.glob('*.parquet'))
if parquet_files:
    print("\nGenerated files:")
    for f in parquet_files:
        df = pd.read_parquet(f)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:25s} {len(df):6d} rows ({size_mb:.2f} MB)")
    
    # Check uniqueness
    complexes = pd.read_parquet(canonical_dir / 'complexes.parquet')
    complexes['complex_key'] = (
        complexes['pdb_id'] + '_' + 
        complexes['protein_chain_id'] + '_' + 
        complexes['peptide_chain_id']
    )
    
    unique_keys = complexes['complex_key'].nunique()
    total_records = len(complexes)
    
    print(f"\nUniqueness check:")
    print(f"  Total records: {total_records}")
    print(f"  Unique complex_key: {unique_keys}")
    print(f"  Duplicates: {total_records - unique_keys}")
    
    if unique_keys == total_records:
        print("  STATUS: CLEAN (no duplicates)")
    else:
        print("  WARNING: Duplicates still present!")

print("="*60)
