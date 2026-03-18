"""Select 100 samples for validation"""
import sys
from pathlib import Path
import random

sys.path.insert(0, 'src')

# Set seed for reproducibility
random.seed(42)

# Get PROPEDIA PDB files
propedia_dir = Path('data/raw/propedia/complexes/complex')
propedia_files = list(propedia_dir.glob('*.pdb'))

print(f"Total PROPEDIA PDB files: {len(propedia_files)}")

# Select 50 random PROPEDIA samples
propedia_samples = random.sample(propedia_files, min(50, len(propedia_files)))

# Get RCSB mmCIF files (already downloaded)
rcsb_dir = Path('data/staging/mmcif_cache')
rcsb_files = list(rcsb_dir.glob('*.cif'))

print(f"Total RCSB mmCIF files: {len(rcsb_files)}")

# Create sample list file
sample_list = Path('data/staging/validation_samples.txt')
sample_list.parent.mkdir(parents=True, exist_ok=True)

with open(sample_list, 'w') as f:
    f.write("# PROPEDIA samples (PDB format)\n")
    for pdb_file in propedia_samples:
        f.write(f"{pdb_file}\n")
    
    f.write("\n# RCSB samples (mmCIF format)\n")
    for cif_file in rcsb_files:
        f.write(f"{cif_file}\n")

total_samples = len(propedia_samples) + len(rcsb_files)
print(f"\nSelected {total_samples} samples:")
print(f"  PROPEDIA (PDB): {len(propedia_samples)}")
print(f"  RCSB (mmCIF): {len(rcsb_files)}")
print(f"\nSample list saved to: {sample_list}")
