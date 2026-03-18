"""Master script to download all datasets for PeptidQuantum"""
import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peptidquantum.data.downloaders import (
    PROPEDIADownloader,
    PepBDBDownloader,
    BioLiP2Downloader,
    GEPPRIDownloader
)


def setup_logging(verbose: bool = False):
    """Setup logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def download_all(
    output_base: Path,
    skip_propedia: bool = False,
    skip_pepbdb: bool = False,
    skip_biolip2: bool = False,
    skip_geppri: bool = False,
    extract_archives: bool = False
):
    """Download all datasets"""
    
    print("\n" + "="*70)
    print(" "*20 + "PeptidQuantum Dataset Download")
    print("="*70)
    
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Track downloads
    downloaded = []
    failed = []
    
    # 1. PROPEDIA (Core structure corpus)
    if not skip_propedia:
        print("\n" + "-"*70)
        print("1/4: Downloading PROPEDIA v2.3 (Core Structure Corpus)")
        print("-"*70)
        try:
            propedia_dir = output_base / "propedia"
            downloader = PROPEDIADownloader(propedia_dir)
            downloader.download(
                download_structures=True,
                download_metadata=True,
                download_clusters=True
            )
            if extract_archives:
                downloader.extract_archives()
            downloaded.append("PROPEDIA")
        except Exception as e:
            logging.error(f"PROPEDIA download failed: {e}")
            failed.append("PROPEDIA")
    
    # 2. PepBDB (Clean structures)
    if not skip_pepbdb:
        print("\n" + "-"*70)
        print("2/4: Downloading PepBDB (Clean Structure Corpus)")
        print("-"*70)
        try:
            pepbdb_dir = output_base / "pepbdb"
            downloader = PepBDBDownloader(pepbdb_dir)
            downloader.download(
                download_structures=True,
                download_metadata=True,
                max_length=50  # Extension set
            )
            downloaded.append("PepBDB")
        except Exception as e:
            logging.error(f"PepBDB download failed: {e}")
            failed.append("PepBDB")
    
    # 3. BioLiP2 (Annotation enrichment)
    if not skip_biolip2:
        print("\n" + "-"*70)
        print("3/4: Downloading BioLiP2 Peptide Subset (Annotation Enrichment)")
        print("-"*70)
        try:
            biolip2_dir = output_base / "biolip2"
            downloader = BioLiP2Downloader(biolip2_dir)
            downloader.download(
                peptides_only=True,
                max_peptide_length=30,  # Core set
                download_structures=True,
                download_metadata=True
            )
            if extract_archives:
                downloader.extract_archives()
            downloaded.append("BioLiP2")
        except Exception as e:
            logging.error(f"BioLiP2 download failed: {e}")
            failed.append("BioLiP2")
    
    # 4. GEPPRI (External holdout)
    if not skip_geppri:
        print("\n" + "-"*70)
        print("4/4: Downloading GEPPRI (EXTERNAL HOLDOUT - DO NOT TOUCH)")
        print("-"*70)
        print("\nWARNING: This is the external holdout dataset.")
        print("Do NOT use for training, validation, or any development decisions.")
        print("Only use for final benchmark evaluation.\n")
        
        try:
            geppri_dir = output_base / "geppri"
            downloader = GEPPRIDownloader(geppri_dir)
            downloader.download(
                download_structures=True,
                download_metadata=True
            )
            downloaded.append("GEPPRI")
        except Exception as e:
            logging.error(f"GEPPRI download failed: {e}")
            failed.append("GEPPRI")
    
    # Summary
    print("\n" + "="*70)
    print(" "*25 + "Download Summary")
    print("="*70)
    
    if downloaded:
        print("\n✓ Successfully downloaded:")
        for dataset in downloaded:
            print(f"  • {dataset}")
    
    if failed:
        print("\n✗ Failed to download:")
        for dataset in failed:
            print(f"  • {dataset}")
    
    print("\n" + "="*70)
    print(f"\nOutput directory: {output_base.absolute()}")
    print("\nNext steps:")
    print("  1. Verify downloads in data/raw/")
    print("  2. Run staging processors to normalize data")
    print("  3. Generate canonical dataset")
    print("  4. Create cluster-aware splits")
    print("  5. Run QC dashboard")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download all PeptidQuantum datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset Tiers:
  1. PROPEDIA v2.3    - Core structure corpus (~49,300 complexes)
  2. PepBDB           - Clean structures (up to 50 aa peptides)
  3. BioLiP2          - Annotation enrichment (peptide subset)
  4. GEPPRI           - External holdout (FINAL BENCHMARK ONLY)

Example:
  # Download all datasets
  python scripts/download_all_datasets.py --output data/raw
  
  # Download only core datasets
  python scripts/download_all_datasets.py --output data/raw --skip-geppri
  
  # Download and extract archives
  python scripts/download_all_datasets.py --output data/raw --extract
        """
    )
    
    parser.add_argument("--output", type=Path, default="data/raw",
                       help="Output directory for raw data (default: data/raw)")
    parser.add_argument("--skip-propedia", action="store_true",
                       help="Skip PROPEDIA download")
    parser.add_argument("--skip-pepbdb", action="store_true",
                       help="Skip PepBDB download")
    parser.add_argument("--skip-biolip2", action="store_true",
                       help="Skip BioLiP2 download")
    parser.add_argument("--skip-geppri", action="store_true",
                       help="Skip GEPPRI download (external holdout)")
    parser.add_argument("--extract", action="store_true",
                       help="Extract archives after download")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Download
    download_all(
        output_base=args.output,
        skip_propedia=args.skip_propedia,
        skip_pepbdb=args.skip_pepbdb,
        skip_biolip2=args.skip_biolip2,
        skip_geppri=args.skip_geppri,
        extract_archives=args.extract
    )


if __name__ == "__main__":
    main()
