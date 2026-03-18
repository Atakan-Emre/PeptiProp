"""Run full validation on 55 samples (50 PROPEDIA + 5 RCSB)"""
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from peptidquantum.data.processors.validation_checklist import ValidationChecklist

# Get all mmCIF files from cache
mmcif_dir = Path('data/staging/mmcif_cache')
cif_files = list(mmcif_dir.glob('*.cif'))

print(f"Found {len(cif_files)} mmCIF files for validation\n")

# Initialize validator
validator = ValidationChecklist(
    mmcif_cache_dir='data/staging/mmcif_cache',
    chain_id_mode='auth',
    residue_number_mode='auth'
)

# Validate
print("Running validation on all samples...\n")
results = validator.validate_batch(cif_files, source_db='mixed')

# Print summary
summary = validator.get_summary()

print("\n" + "="*60)
print("Validation Summary")
print("="*60)
print(f"Total: {summary['total']}")
print(f"Passed: {summary['passed']}")
print(f"Failed: {summary['failed']}")
print(f"Pass rate: {summary['pass_rate']:.1%}")

print("\n10-Point Checklist:")
for item, count in summary['checklist_stats'].items():
    pct = (count / summary['total']) * 100
    status = "OK" if pct >= 80 else "WARN" if pct >= 50 else "FAIL"
    print(f"  [{status}] {item:30s} {count:3d}/{summary['total']:3d} ({pct:5.1f}%)")

if summary['quarantine_reasons']:
    print("\nTop Quarantine Reasons:")
    for reason, count in sorted(summary['quarantine_reasons'].items(), key=lambda x: -x[1])[:5]:
        pct = (count / summary['total']) * 100
        print(f"  - {reason:30s} {count:3d} ({pct:5.1f}%)")

# Export
validator.export_results('data/staging/validation/validation_results.csv')
validator.export_summary_report('data/staging/validation/validation_summary.html')

print("\n" + "="*60)
print("Results exported:")
print("  CSV: data/staging/validation/validation_results.csv")
print("  HTML: data/staging/validation/validation_summary.html")
print("="*60)

# Decision
if summary['pass_rate'] >= 0.8:
    print("\nSTATUS: PASS (>= 80%)")
    print("Next: Build canonical dataset")
elif summary['pass_rate'] >= 0.5:
    print("\nSTATUS: PARTIAL (50-80%)")
    print("Next: Review quarantine reasons and fix top issues")
else:
    print("\nSTATUS: FAIL (< 50%)")
    print("Next: Fix critical issues before proceeding")
