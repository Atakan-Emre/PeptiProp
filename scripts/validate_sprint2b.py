"""Sprint 2B validation script - 100 real complexes validation"""
import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peptidquantum.data.processors.validation_checklist import ValidationChecklist
from peptidquantum.data.processors.pdb_to_mmcif import PDBToMMCIFConverter


def setup_logging(verbose: bool = False, log_file: str = None):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )


def collect_sample_files(
    raw_dir: Path,
    propedia_count: int = 50,
    pepbdb_count: int = 25,
    biolip2_count: int = 25
) -> dict:
    """
    Collect sample files from each source
    
    Strategy:
    - PROPEDIA: 50 complexes (PDB format)
    - PepBDB: 25 complexes (mmCIF format)
    - BioLiP2: 25 complexes (mmCIF format)
    
    Mix:
    - Short peptides
    - Long peptides
    - Multi-chain structures
    - Low interaction structures
    """
    samples = {
        "propedia": [],
        "pepbdb": [],
        "biolip2": []
    }
    
    # PROPEDIA (PDB format)
    propedia_dir = raw_dir / "propedia" / "complexes"
    if propedia_dir.exists():
        pdb_files = list(propedia_dir.glob("**/*.pdb")) + list(propedia_dir.glob("**/*.ent"))
        samples["propedia"] = pdb_files[:propedia_count]
        logging.info(f"Found {len(samples['propedia'])} PROPEDIA samples")
    else:
        logging.warning(f"PROPEDIA directory not found: {propedia_dir}")
    
    # PepBDB (mmCIF format)
    pepbdb_dir = raw_dir / "pepbdb" / "structures"
    if pepbdb_dir.exists():
        cif_files = list(pepbdb_dir.glob("*.cif"))
        samples["pepbdb"] = cif_files[:pepbdb_count]
        logging.info(f"Found {len(samples['pepbdb'])} PepBDB samples")
    else:
        logging.warning(f"PepBDB directory not found: {pepbdb_dir}")
    
    # BioLiP2 (mmCIF format)
    biolip2_dir = raw_dir / "biolip2" / "peptide_subset" / "structures"
    if biolip2_dir.exists():
        cif_files = list(biolip2_dir.glob("*.cif"))
        samples["biolip2"] = cif_files[:biolip2_count]
        logging.info(f"Found {len(samples['biolip2'])} BioLiP2 samples")
    else:
        logging.warning(f"BioLiP2 directory not found: {biolip2_dir}")
    
    return samples


def run_validation(
    raw_dir: Path,
    staging_dir: Path,
    propedia_count: int = 50,
    pepbdb_count: int = 25,
    biolip2_count: int = 25,
    chain_id_mode: str = "auth",
    residue_number_mode: str = "auth"
):
    """Run Sprint 2B validation"""
    
    print("\n" + "="*70)
    print(" "*20 + "Sprint 2B Validation")
    print("="*70)
    print(f"\nTarget: 100 real complexes")
    print(f"  PROPEDIA: {propedia_count} (PDB format)")
    print(f"  PepBDB: {pepbdb_count} (mmCIF format)")
    print(f"  BioLiP2: {biolip2_count} (mmCIF format)")
    print(f"\nID Mode: {chain_id_mode} (chain), {residue_number_mode} (residue)")
    
    # Collect samples
    print("\n" + "-"*70)
    print("Collecting sample files...")
    print("-"*70)
    
    samples = collect_sample_files(
        raw_dir=raw_dir,
        propedia_count=propedia_count,
        pepbdb_count=pepbdb_count,
        biolip2_count=biolip2_count
    )
    
    total_samples = sum(len(files) for files in samples.values())
    
    if total_samples == 0:
        print("\n✗ No sample files found!")
        print("Please download data first:")
        print("  python scripts/download_all_datasets.py --output data/raw")
        return
    
    print(f"\nTotal samples collected: {total_samples}")
    
    # Initialize validator
    mmcif_cache_dir = staging_dir / "mmcif_cache"
    
    validator = ValidationChecklist(
        mmcif_cache_dir=mmcif_cache_dir,
        chain_id_mode=chain_id_mode,
        residue_number_mode=residue_number_mode
    )
    
    # Validate each source
    all_results = []
    
    for source, files in samples.items():
        if not files:
            continue
        
        print("\n" + "-"*70)
        print(f"Validating {source.upper()} ({len(files)} files)")
        print("-"*70)
        
        results = validator.validate_batch(files, source_db=source)
        all_results.extend(results)
    
    # Export results
    print("\n" + "-"*70)
    print("Exporting results...")
    print("-"*70)
    
    results_dir = staging_dir / "validation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV export
    csv_file = results_dir / "validation_results.csv"
    validator.export_results(csv_file)
    print(f"✓ Results CSV: {csv_file}")
    
    # HTML summary
    html_file = results_dir / "validation_summary.html"
    validator.export_summary_report(html_file)
    print(f"✓ Summary HTML: {html_file}")
    
    # Print summary
    summary = validator.get_summary()
    
    print("\n" + "="*70)
    print(" "*25 + "Summary")
    print("="*70)
    print(f"\nTotal complexes: {summary['total']}")
    print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
    print(f"Failed: {summary['failed']}")
    
    print("\n10-Point Checklist:")
    for item, count in summary['checklist_stats'].items():
        pct = (count / summary['total']) * 100
        status = "✓" if pct >= 80 else "⚠" if pct >= 50 else "✗"
        print(f"  {status} {item.replace('_', ' ').title():30s} {count:3d}/{summary['total']:3d} ({pct:5.1f}%)")
    
    if summary['quarantine_reasons']:
        print("\nTop Quarantine Reasons:")
        for reason, count in sorted(summary['quarantine_reasons'].items(), key=lambda x: -x[1])[:5]:
            pct = (count / summary['total']) * 100
            print(f"  • {reason.replace('_', ' ').title():30s} {count:3d} ({pct:5.1f}%)")
    
    print("\n" + "="*70)
    print("\nNext steps:")
    
    if summary['pass_rate'] >= 0.8:
        print("  ✓ Validation passed! (≥80% pass rate)")
        print("  1. Review validation_summary.html")
        print("  2. Build canonical dataset:")
        print("     python scripts/build_canonical_dataset.py \\")
        print("       --raw data/raw \\")
        print("       --staging data/staging \\")
        print("       --canonical data/canonical \\")
        print("       --all-sources")
        print("  3. Proceed to Sprint 3 (splits + QC)")
    elif summary['pass_rate'] >= 0.5:
        print("  ⚠ Validation partial (50-80% pass rate)")
        print("  1. Review quarantine reasons in validation_summary.html")
        print("  2. Fix common issues")
        print("  3. Re-run validation")
    else:
        print("  ✗ Validation failed (<50% pass rate)")
        print("  1. Review validation_results.csv for details")
        print("  2. Check quarantine reasons")
        print("  3. Fix critical issues before proceeding")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sprint 2B validation - 100 real complexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script validates the canonical dataset generation pipeline on 100 real complexes:
- 50 from PROPEDIA (PDB format → mmCIF backfill)
- 25 from PepBDB (mmCIF format)
- 25 from BioLiP2 peptide subset (mmCIF format)

10-Point Validation Checklist:
1. Parse successful
2. Auth chain found
3. Peptide chain found
4. Protein chain found
5. Peptide length policy passed (5-50 aa)
6. Protein length policy passed (≥30 aa)
7. Pair extractor confidence (≥0.5)
8. No quarantine reason
9. Parquet record writable
10. Visualization compatible

Example:
  python scripts/validate_sprint2b.py \\
    --raw data/raw \\
    --staging data/staging \\
    --verbose
        """
    )
    
    parser.add_argument("--raw", type=Path, required=True,
                       help="Raw data directory")
    parser.add_argument("--staging", type=Path, required=True,
                       help="Staging directory")
    
    # Sample counts
    parser.add_argument("--propedia-count", type=int, default=50,
                       help="Number of PROPEDIA samples (default: 50)")
    parser.add_argument("--pepbdb-count", type=int, default=25,
                       help="Number of PepBDB samples (default: 25)")
    parser.add_argument("--biolip2-count", type=int, default=25,
                       help="Number of BioLiP2 samples (default: 25)")
    
    # ID modes
    parser.add_argument("--chain-id-mode", type=str, default="auth",
                       choices=["auth", "label"],
                       help="Chain ID mode (default: auth)")
    parser.add_argument("--residue-number-mode", type=str, default="auth",
                       choices=["auth", "label"],
                       help="Residue number mode (default: auth)")
    
    # Logging
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--log-file", type=str,
                       help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    
    # Run validation
    run_validation(
        raw_dir=args.raw,
        staging_dir=args.staging,
        propedia_count=args.propedia_count,
        pepbdb_count=args.pepbdb_count,
        biolip2_count=args.biolip2_count,
        chain_id_mode=args.chain_id_mode,
        residue_number_mode=args.residue_number_mode
    )


if __name__ == "__main__":
    main()
