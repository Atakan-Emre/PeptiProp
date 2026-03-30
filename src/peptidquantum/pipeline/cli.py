"""Command-line interface for PeptidQuantum pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import PeptidQuantumPipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PeptidQuantum: Protein-Peptide Interaction Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze PDB structure
  peptidquantum run --pdb 1ABC --protein A --peptide B

  # Analyze local CIF file
  peptidquantum run --cif structure.cif --protein A --peptide B

  # Use config file
  peptidquantum run --config config.json

  # Quick analysis (skip PyMOL)
  peptidquantum run --pdb 1ABC --no-pymol
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    run_parser = subparsers.add_parser("run", help="Run analysis pipeline")

    input_group = run_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pdb", type=str, help="PDB ID to fetch from RCSB")
    input_group.add_argument("--cif", type=Path, help="Path to local CIF file")
    input_group.add_argument("--config", type=Path, help="Path to config JSON file")

    run_parser.add_argument("--protein", type=str, help="Protein chain ID")
    run_parser.add_argument("--peptide", type=str, help="Peptide chain ID")
    run_parser.add_argument(
        "--pocket-radius",
        type=float,
        default=8.0,
        help="Pocket extraction radius in Angstroms (default: 8.0)",
    )

    run_parser.add_argument(
        "--arpeggio",
        action="store_true",
        help="Enable experimental Arpeggio extraction (not used in final reported outputs)",
    )

    run_parser.add_argument(
        "--no-pymol",
        action="store_true",
        help="Disable PyMOL figure generation",
    )
    run_parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable HTML report generation",
    )
    run_parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Disable standalone 3Dmol.js viewer",
    )

    run_parser.add_argument(
        "--output", type=Path, default="outputs", help="Output directory (default: outputs)"
    )
    run_parser.add_argument(
        "--cache", type=Path, default="data/cache", help="Cache directory (default: data/cache)"
    )
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    config_parser = subparsers.add_parser("config", help="Generate config template")
    config_parser.add_argument(
        "--output", type=Path, default="config.json", help="Output config file (default: config.json)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    if args.command == "run":
        run_pipeline(args)
    elif args.command == "config":
        generate_config_template(args)


def run_pipeline(args):
    """Run the pipeline."""
    if args.config:
        config = PipelineConfig.from_json(args.config)
    else:
        config = PipelineConfig(
            complex_id=args.pdb,
            cif_path=args.cif,
            protein_chain=args.protein,
            peptide_chain=args.peptide,
            pocket_radius=args.pocket_radius,
            use_arpeggio=args.arpeggio,
            use_plip=False,
            generate_pymol=not args.no_pymol,
            generate_report=not args.no_report,
            generate_viewer=not args.no_viewer,
            output_dir=args.output,
            cache_dir=args.cache,
        )

    try:
        config.validate()
    except Exception as exc:
        logging.error(f"Configuration error: {exc}")
        sys.exit(1)

    pipeline = PeptidQuantumPipeline(
        output_base_dir=config.output_dir,
        cache_dir=config.cache_dir,
    )

    try:
        results = pipeline.run(
            complex_id=config.complex_id,
            cif_path=config.cif_path,
            protein_chain=config.protein_chain,
            peptide_chain=config.peptide_chain,
            pocket_radius=config.pocket_radius,
            use_arpeggio=config.use_arpeggio,
            use_plip=config.use_plip,
            generate_pymol=config.generate_pymol,
            generate_report=config.generate_report,
            generate_viewer=config.generate_viewer,
        )

        if results["status"] == "success":
            print("\n" + "=" * 60)
            print("Pipeline completed successfully!")
            print("=" * 60)
            print(f"Complex ID: {results['complex_id']}")
            print(f"Output directory: {results['output_dir']}")
            print(f"Interactions found: {results['num_interactions']}")
            print(f"Interaction types: {results['interaction_types']}")
            print("\nGenerated outputs:")
            if results.get("reports", {}).get("report"):
                print(f"  - HTML Report: {results['output_dir']}/report.html")
            if results.get("reports", {}).get("viewer"):
                print(f"  - 3D Viewer: {results['output_dir']}/viewer.html")
            print("=" * 60)
        else:
            print(f"\nPipeline failed at stage: {results.get('stage', 'unknown')}")
            sys.exit(1)

    except KeyboardInterrupt:
        logging.info("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logging.error(f"Pipeline error: {exc}", exc_info=True)
        sys.exit(1)


def generate_config_template(args):
    """Generate a config template file."""
    config = PipelineConfig(
        complex_id="1ABC",
        protein_chain="A",
        peptide_chain="B",
    )

    config.to_json(args.output)
    print(f"Config template saved to: {args.output}")
    print("\nEdit the config file and run:")
    print(f"  peptidquantum run --config {args.output}")


if __name__ == "__main__":
    main()
