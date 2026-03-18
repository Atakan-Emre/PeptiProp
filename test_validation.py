"""Test validation pipeline on downloaded mmCIF files"""
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from peptidquantum.data.processors.validation_checklist import ValidationChecklist

# Test validation on downloaded files
mmcif_dir = Path('data/staging/mmcif_cache')
cif_files = list(mmcif_dir.glob('*.cif'))

print(f"Found {len(cif_files)} mmCIF files\n")

if not cif_files:
    print("No mmCIF files found. Run test_rcsb_download.py first.")
    sys.exit(1)

# Initialize validator
validator = ValidationChecklist(
    mmcif_cache_dir='data/staging/mmcif_cache',
    chain_id_mode='auth',
    residue_number_mode='auth'
)

# Validate
print("Running validation...\n")
results = validator.validate_batch(cif_files, source_db='test')

# Print results
print("\n" + "="*60)
print("Validation Results")
print("="*60)

for result in results:
    status = "PASS" if result.passed else "FAIL"
    print(f"\n{status}: {result.complex_id}")
    print(f"  Parse: {result.parse_success}")
    print(f"  Peptide chain: {result.peptide_chain_found}")
    print(f"  Protein chain: {result.protein_chain_found}")
    print(f"  Peptide length: {result.peptide_length} aa (valid: {result.peptide_length_valid})")
    print(f"  Protein length: {result.protein_length} aa (valid: {result.protein_length_valid})")
    print(f"  Confidence: {result.pair_extractor_confidence:.2f}")
    if result.quarantine_reason:
        print(f"  Quarantine: {result.quarantine_reason}")

# Summary
summary = validator.get_summary()
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Total: {summary['total']}")
print(f"Passed: {summary['passed']}")
print(f"Failed: {summary['failed']}")
print(f"Pass rate: {summary['pass_rate']:.1%}")

# Export
validator.export_results('data/staging/test_validation_results.csv')
validator.export_summary_report('data/staging/test_validation_summary.html')

print(f"\nResults exported to data/staging/")
