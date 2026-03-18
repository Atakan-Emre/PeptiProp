"""Test RCSB mmCIF download"""
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from peptidquantum.data.processors.pdb_to_mmcif import PDBToMMCIFConverter

# Test with known protein-peptide complexes
test_ids = ['1A2K', '1SSH', '2P54', '1TP5', '1NX1']

converter = PDBToMMCIFConverter('data/staging/mmcif_cache')

print("Testing RCSB mmCIF download...\n")

for pdb_id in test_ids:
    output_path = Path(f'data/staging/mmcif_cache/{pdb_id.lower()}.cif')
    success, message = converter._download_mmcif(pdb_id, output_path)
    
    status = "OK" if success else "FAIL"
    print(f"{status} {pdb_id}: {message}")

print(f"\nStatistics: {converter.get_statistics()}")
