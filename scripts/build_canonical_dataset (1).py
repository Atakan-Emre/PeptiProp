"""Master script to build canonical dataset from raw data"""
import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peptidquantum.data.processors import CanonicalBuilder
from peptidquantum.data.canonical.schema import SourceDatabase


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
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


def find_cif_files(raw_dir: Path, source: str) -> list[Path]:
    """Find mmCIF files for source"""
    source_dir = raw_dir / source
    
    if not source_dir.exists():
        logging.error(f"Source directory not found: {source_dir}")
        return []
    
    # Different sources have different structures
    if source == "propedia":
        # PROPEDIA: complexes/, receptors/, peptides/
        cif_files = list((source_dir / "complexes").glob("**/*.cif"))
    elif source == "pepbdb":
        # PepBDB: structures/
        cif_files = list((source_dir / "structures").glob("*.cif"))
    elif source == "biolip2":
        # BioLiP2: peptide_subset/structures/
        cif_files = list((source_dir / "peptide_subset" / "structures").glob("*.cif"))
    else:
        # Generic: find all CIF files
        cif_files = list(source_dir.glob("**/*.cif"))
    
    return cif_files


def build_from_source(
    raw_dir: Path,
    staging_dir: Path,
    canonical_dir: Path,
    source: str,
    chain_id_mode: str,
    residue_number_mode: str,
    limit: Optional[int] = None,
    batch_size: int = 100
):
    """Build canonical dataset from single source"""
    
    print("\n" + "="*70)
    print(f"Building canonical dataset from {source.upper()}")
    print("="*70)
    
    # Find CIF files
    cif_files = find_cif_files(raw_dir, source)
    
    if not cif_files:
        print(f"✗ No CIF files found for {source}")
        return
    
    print(f"Found {len(cif_files)} CIF files")
    
    # Limit if specified
    if limit:
        cif_files = cif_files[:limit]
        print(f"Processing first {limit} files (limit specified)")
    
    # Map source to SourceDatabase enum
    source_db_map = {
        "propedia": SourceDatabase.PROPEDIA,
        "pepbdb": SourceDatabase.PEPBDB,
        "biolip2": SourceDatabase.BIOLIP2,
        "geppri": SourceDatabase.GEPPRI
    }
    
    source_db = source_db_map.get(source.lower())
    if not source_db:
        print(f"✗ Unknown source: {source}")
        return
    
    # Build canonical dataset
    builder = CanonicalBuilder(
        staging_dir=staging_dir,
        canonical_dir=canonical_dir,
        chain_id_mode=chain_id_mode,
        residue_number_mode=residue_number_mode
    )
    
    builder.build(
        source_files=cif_files,
        source_db=source_db,
        batch_size=batch_size
    )
    
    # Export quarantine report
    quarantine_report = staging_dir / "quarantine" / f"{source}_quarantine_report.html"
    builder.quarantine_manager.export_report(quarantine_report)
    print(f"\n✓ Quarantine report: {quarantine_report}")


def build_all(
    raw_dir: Path,
    staging_dir: Path,
    canonical_dir: Path,
    chain_id_mode: str,
    residue_number_mode: str,
    sources: list[str],
    limit_per_source: Optional[int] = None,
    batch_size: int = 100
):
    """Build canonical dataset from all sources"""
    
    print("\n" + "="*70)
    print(" "*20 + "Canonical Dataset Builder")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Raw directory: {raw_dir}")
    print(f"  Staging directory: {staging_dir}")
    print(f"  Canonical directory: {canonical_dir}")
    print(f"  Chain ID mode: {chain_id_mode}")
    print(f"  Residue number mode: {residue_number_mode}")
    print(f"  Sources: {', '.join(sources)}")
    if limit_per_source:
        print(f"  Limit per source: {limit_per_source}")
    
    # Process each source
    for source in sources:
        try:
            build_from_source(
                raw_dir=raw_dir,
                staging_dir=staging_dir,
                canonical_dir=canonical_dir,
                source=source,
                chain_id_mode=chain_id_mode,
                residue_number_mode=residue_number_mode,
                limit=limit_per_source,
                batch_size=batch_size
            )
        except Exception as e:
            logging.error(f"Failed to process {source}: {e}", exc_info=True)
            continue
    
    # Final summary
    print("\n" + "="*70)
    print(" "*25 + "Build Complete")
    print("="*70)
    
    # Check canonical files
    canonical_files = list(canonical_dir.glob("*.parquet"))
    if canonical_files:
        print("\n✓ Canonical files generated:")
        for f in canonical_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  • {f.name} ({size_mb:.1f} MB)")
    
    # Check version file
    version_file = canonical_dir / "VERSION.txt"
    if version_file.exists():
        print(f"\n✓ Version info: {version_file}")
        with open(version_file) as f:
            for line in f:
                print(f"  {line.strip()}")
    
    print("\n" + "="*70)
    print("\nNext steps:")
    print("  1. Review quarantine reports in staging/quarantine/")
    print("  2. Validate canonical dataset with QC dashboard")
    print("  3. Create cluster-aware splits")
    print("  4. Generate audit gallery")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build canonical dataset from raw data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from PROPEDIA (first 100 files)
  python scripts/build_canonical_dataset.py \\
    --raw data/raw \\
    --staging data/staging \\
    --canonical data/canonical \\
    --source propedia \\
    --limit 100
  
  # Build from all sources
  python scripts/build_canonical_dataset.py \\
    --raw data/raw \\
    --staging data/staging \\
    --canonical data/canonical \\
    --all-sources
  
  # Build with label IDs instead of auth
  python scripts/build_canonical_dataset.py \\
    --raw data/raw \\
    --staging data/staging \\
    --canonical data/canonical \\
    --source propedia \\
    --chain-id-mode label \\
    --residue-number-mode label
        """
    )
    
    parser.add_argument("--raw", type=Path, required=True,
                       help="Raw data directory")
    parser.add_argument("--staging", type=Path, required=True,
                       help="Staging directory")
    parser.add_argument("--canonical", type=Path, required=True,
                       help="Canonical output directory")
    
    # Source selection
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", type=str,
                             choices=["propedia", "pepbdb", "biolip2", "geppri"],
                             help="Single source to process")
    source_group.add_argument("--all-sources", action="store_true",
                             help="Process all sources")
    
    # ID modes
    parser.add_argument("--chain-id-mode", type=str, default="auth",
                       choices=["auth", "label"],
                       help="Chain ID mode (default: auth)")
    parser.add_argument("--residue-number-mode", type=str, default="auth",
                       choices=["auth", "label"],
                       help="Residue number mode (default: auth)")
    
    # Processing options
    parser.add_argument("--limit", type=int,
                       help="Limit number of files per source")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for progress reporting (default: 100)")
    
    # Logging
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--log-file", type=str,
                       help="Log file path")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    
    # Determine sources
    if args.all_sources:
        sources = ["propedia", "pepbdb", "biolip2"]  # Exclude GEPPRI (external holdout)
    else:
        sources = [args.source]
    
    # Build
    build_all(
        raw_dir=args.raw,
        staging_dir=args.staging,
        canonical_dir=args.canonical,
        chain_id_mode=args.chain_id_mode,
        residue_number_mode=args.residue_number_mode,
        sources=sources,
        limit_per_source=args.limit,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    from typing import Optional
    main()
