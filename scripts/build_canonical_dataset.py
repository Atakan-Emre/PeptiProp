"""Master script to build a canonical dataset from raw sources."""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peptidquantum.data.canonical.schema import SourceDatabase
from peptidquantum.data.processors import CanonicalBuilder


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Configure console/file logging."""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def find_structure_files(raw_dir: Path, source: str) -> list[Path]:
    """Find source structure files for a dataset."""
    source_dir = raw_dir / source
    if not source_dir.exists():
        logging.error("Source directory not found: %s", source_dir)
        return []

    if source == "propedia":
        return (
            list((source_dir / "complexes").glob("**/*.cif"))
            + list((source_dir / "complexes").glob("**/*.pdb"))
        )
    if source == "pepbdb":
        return list((source_dir / "structures").glob("*.cif"))
    if source == "biolip2":
        return list((source_dir / "peptide_subset" / "structures").glob("*.cif"))
    return list(source_dir.glob("**/*.cif"))


def build_from_source(
    raw_dir: Path,
    staging_dir: Path,
    canonical_dir: Path,
    source: str,
    chain_id_mode: str,
    residue_number_mode: str,
    limit: Optional[int] = None,
    batch_size: int = 100,
) -> None:
    """Build canonical parquet files from one source."""
    print("\n" + "=" * 70)
    print(f"Building canonical dataset from {source.upper()}")
    print("=" * 70)

    structure_files = find_structure_files(raw_dir, source)
    if not structure_files:
        print(f"[FAIL] No structure files found for {source}")
        return

    print(f"Found {len(structure_files)} structure files")
    if limit:
        structure_files = structure_files[:limit]
        print(f"Processing first {limit} files (limit specified)")

    source_db_map = {
        "propedia": SourceDatabase.PROPEDIA,
        "pepbdb": SourceDatabase.PEPBDB,
        "biolip2": SourceDatabase.BIOLIP2,
        "geppri": SourceDatabase.GEPPRI,
    }
    source_db = source_db_map.get(source.lower())
    if not source_db:
        print(f"[FAIL] Unknown source: {source}")
        return

    builder = CanonicalBuilder(
        staging_dir=staging_dir,
        canonical_dir=canonical_dir,
        chain_id_mode=chain_id_mode,
        residue_number_mode=residue_number_mode,
    )
    builder.build(
        source_files=structure_files,
        source_db=source_db,
        batch_size=batch_size,
    )

    quarantine_report = staging_dir / "quarantine" / f"{source}_quarantine_report.html"
    builder.quarantine_manager.export_report(quarantine_report)
    print(f"\n[OK] Quarantine report: {quarantine_report}")


def build_all(
    raw_dir: Path,
    staging_dir: Path,
    canonical_dir: Path,
    chain_id_mode: str,
    residue_number_mode: str,
    sources: list[str],
    limit_per_source: Optional[int] = None,
    batch_size: int = 100,
) -> None:
    """Build canonical parquet files from multiple sources."""
    print("\n" + "=" * 70)
    print(" " * 20 + "Canonical Dataset Builder")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Raw directory: {raw_dir}")
    print(f"  Staging directory: {staging_dir}")
    print(f"  Canonical directory: {canonical_dir}")
    print(f"  Chain ID mode: {chain_id_mode}")
    print(f"  Residue number mode: {residue_number_mode}")
    print(f"  Sources: {', '.join(sources)}")
    if limit_per_source:
        print(f"  Limit per source: {limit_per_source}")

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
                batch_size=batch_size,
            )
        except Exception as exc:
            logging.error("Failed to process %s: %s", source, exc, exc_info=True)
            continue

    print("\n" + "=" * 70)
    print(" " * 25 + "Build Complete")
    print("=" * 70)

    canonical_files = list(canonical_dir.glob("*.parquet"))
    if canonical_files:
        print("\n[OK] Canonical files generated:")
        for parquet_file in canonical_files:
            size_mb = parquet_file.stat().st_size / (1024 * 1024)
            print(f"  - {parquet_file.name} ({size_mb:.1f} MB)")

    version_file = canonical_dir / "VERSION.txt"
    if version_file.exists():
        print(f"\n[OK] Version info: {version_file}")
        with open(version_file, encoding="utf-8") as handle:
            for line in handle:
                print(f"  {line.strip()}")

    print("\n" + "=" * 70)
    print("\nNext steps:")
    print("  1. Review quarantine reports in staging/quarantine/")
    print("  2. Validate canonical dataset with QC dashboard")
    print("  3. Create PDB-level structure-aware splits")
    print("  4. Generate audit gallery")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical dataset from raw data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_canonical_dataset.py \
    --raw data/raw \
    --staging data/staging \
    --canonical data/canonical \
    --source propedia

  python scripts/build_canonical_dataset.py \
    --raw data/raw \
    --staging data/staging \
    --canonical data/canonical \
    --source propedia \
    --limit 100
        """,
    )

    parser.add_argument("--raw", type=Path, required=True, help="Raw data directory")
    parser.add_argument("--staging", type=Path, required=True, help="Staging directory")
    parser.add_argument("--canonical", type=Path, required=True, help="Canonical output directory")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        type=str,
        choices=["propedia", "pepbdb", "biolip2", "geppri"],
        help="Single source to process",
    )
    source_group.add_argument("--all-sources", action="store_true", help="Process all sources")

    parser.add_argument(
        "--chain-id-mode",
        type=str,
        default="auth",
        choices=["auth", "label"],
        help="Chain ID mode (default: auth)",
    )
    parser.add_argument(
        "--residue-number-mode",
        type=str,
        default="auth",
        choices=["auth", "label"],
        help="Residue number mode (default: auth)",
    )
    parser.add_argument("--limit", type=int, help="Limit files per source")
    parser.add_argument("--batch-size", type=int, default=100, help="Progress reporting batch size")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--log-file", type=str, help="Optional log file path")

    args = parser.parse_args()
    setup_logging(args.verbose, args.log_file)

    if args.all_sources:
        sources = ["propedia", "pepbdb", "biolip2"]
    else:
        sources = [args.source]

    build_all(
        raw_dir=args.raw,
        staging_dir=args.staging,
        canonical_dir=args.canonical,
        chain_id_mode=args.chain_id_mode,
        residue_number_mode=args.residue_number_mode,
        sources=sources,
        limit_per_source=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
