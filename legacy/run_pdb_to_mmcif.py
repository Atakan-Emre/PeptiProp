"""Convert PROPEDIA PDB samples to mmCIF via RCSB backfill"""
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from peptidquantum.data.processors.pdb_to_mmcif import PDBToMMCIFConverter

# Read sample list
sample_list = Path('data/staging/validation_samples.txt')
pdb_files = []

with open(sample_list, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            path = Path(line)
            if path.suffix == '.pdb':
                pdb_files.append(path)

print(f"Found {len(pdb_files)} PDB files to convert\n")

# Initialize converter
converter = PDBToMMCIFConverter('data/staging/mmcif_cache')

# Convert with rate limiting
results = converter.batch_convert(pdb_files, rate_limit_delay=0.3)

# Summary
print("\n" + "="*60)
print("Conversion Summary")
print("="*60)
print(f"Success: {len(results['success'])}")
print(f"Cached: {len(results['cached'])}")
print(f"Failed: {len(results['failed'])}")

if results['failed']:
    print("\nFailed conversions (first 10):")
    for pdb_file, reason in results['failed'][:10]:
        print(f"  {pdb_file.name}: {reason}")

stats = converter.get_statistics()
print(f"\nSuccess rate: {stats['success_rate']:.1%}")
print("="*60)
