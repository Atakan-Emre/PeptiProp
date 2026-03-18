"""Build canonical dataset from validated samples"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, 'src')

from peptidquantum.data.processors.canonical_builder import CanonicalBuilder
from peptidquantum.data.canonical.schema import SourceDatabase

# Read validation results to get passed complexes
validation_results = pd.read_csv('data/staging/validation/validation_results.csv')

# Filter passed complexes
passed = validation_results[validation_results['quarantine_reason'].isna()]
print(f"Building canonical dataset from {len(passed)} passed complexes\n")

# Get mmCIF files for passed complexes
mmcif_files = []
for _, row in passed.iterrows():
    mmcif_path = Path(row['mmcif_path'])
    if mmcif_path.exists():
        mmcif_files.append(mmcif_path)

print(f"Found {len(mmcif_files)} mmCIF files\n")

# Initialize builder
builder = CanonicalBuilder(
    staging_dir=Path('data/staging'),
    canonical_dir=Path('data/canonical'),
    chain_id_mode='auth',
    residue_number_mode='auth'
)

# Build canonical dataset
# Use PROPEDIA as source (most samples are from PROPEDIA)
builder.build(
    source_files=mmcif_files,
    source_db=SourceDatabase.PROPEDIA,
    batch_size=10
)

print("\n" + "="*60)
print("Canonical dataset build complete!")
print("="*60)

# Check output files
canonical_dir = Path('data/canonical')
parquet_files = list(canonical_dir.glob('*.parquet'))

if parquet_files:
    print("\nGenerated parquet files:")
    for f in parquet_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        df = pd.read_parquet(f)
        print(f"  - {f.name:25s} {len(df):5d} rows ({size_mb:.2f} MB)")
else:
    print("\nWARNING: No parquet files generated!")

# Check VERSION.txt
version_file = canonical_dir / 'VERSION.txt'
if version_file.exists():
    print(f"\nVersion info:")
    with open(version_file) as f:
        for line in f:
            print(f"  {line.strip()}")

print("\n" + "="*60)
