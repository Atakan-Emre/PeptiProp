"""Run a small visualization sanity batch on selected canonical complexes."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peptidquantum.pipeline import PeptidQuantumPipeline


def parse_sample_list(sample_list_path: Path) -> list[str]:
    """Read non-comment complex ids from a sample list."""
    complex_ids = []
    for line in sample_list_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        complex_ids.append(text)
    return complex_ids


def main(
    canonical_dir: Path,
    sample_list: Path,
    output_dir: Path,
    limit: int,
    min_tool_fraction: float,
    enforce_tool_fraction: bool,
    use_arpeggio: bool,
) -> None:
    complexes = pd.read_parquet(canonical_dir / "complexes.parquet").set_index("complex_id")
    selected_ids = []
    seen = set()
    for complex_id in parse_sample_list(sample_list):
        if complex_id in complexes.index and complex_id not in seen:
            selected_ids.append(complex_id)
            seen.add(complex_id)
        if len(selected_ids) >= limit:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = PeptidQuantumPipeline(output_base_dir=output_dir, cache_dir=Path("data/cache"))
    summary = []

    for complex_id in selected_ids:
        row = complexes.loc[complex_id]
        pdb_id = str(row["pdb_id"])
        protein_chain = str(row["protein_chain_id"])
        peptide_chain = str(row["peptide_chain_id"])
        structure_file = Path(str(row["structure_file"]))
        cif_path = structure_file
        if structure_file.suffix.lower() not in {".cif", ".mmcif"}:
            fetched = pipeline.rcsb_fetcher.fetch_structure(pdb_id, format="cif")
            if fetched:
                cif_path = Path(fetched)
        print(f"[RUN] {complex_id} -> {pdb_id} {protein_chain}/{peptide_chain}")
        result = pipeline.run(
            complex_id=complex_id,
            cif_path=cif_path,
            protein_chain=protein_chain,
            peptide_chain=peptide_chain,
            use_arpeggio=use_arpeggio,
            use_plip=False,
            generate_pymol=False,
            generate_report=True,
            generate_viewer=True,
        )
        sample_output_dir = output_dir / complex_id
        prov = result.get("interaction_provenance") or {}
        tool_frac = float(prov.get("tool_based_interaction_fraction", 0.0))
        summary.append(
            {
                "complex_id": complex_id,
                "pdb_id": pdb_id,
                "protein_chain": protein_chain,
                "peptide_chain": peptide_chain,
                "status": result["status"],
                "extraction_mode": prov.get("extraction_mode"),
                "tool_based_interaction_fraction": tool_frac,
                "report_exists": (sample_output_dir / "report.html").exists(),
                "viewer_exists": (sample_output_dir / "viewer.html").exists(),
                "viewer_state_exists": (sample_output_dir / "data" / "viewer_state.json").exists(),
                "peptide_2d_exists": (sample_output_dir / "figures" / "peptide_2d.png").exists(),
            }
        )

    summary_path = output_dir / "visualization_sanity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    passed = sum(
        1
        for item in summary
        if item["status"] == "success"
        and item["report_exists"]
        and item["viewer_exists"]
        and item["viewer_state_exists"]
        and item["peptide_2d_exists"]
    )
    print(f"[OK] sanity batch complete: {passed}/{len(summary)} passed")
    print(f"[OK] summary: {summary_path}")

    if enforce_tool_fraction and summary:
        ok_samples = [
            item
            for item in summary
            if item["status"] == "success"
            and item["report_exists"]
            and item["viewer_exists"]
            and item["viewer_state_exists"]
            and item["peptide_2d_exists"]
        ]
        if ok_samples and use_arpeggio:
            low = [
                item
                for item in ok_samples
                if float(item.get("tool_based_interaction_fraction") or 0.0) + 1e-9 < min_tool_fraction
            ]
            if low:
                raise SystemExit(
                    f"[FAIL] {len(low)}/{len(ok_samples)} passed samples have "
                    f"tool_based_interaction_fraction < {min_tool_fraction}. "
                    "Experimental Arpeggio output is below --min-tool-fraction."
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run visualization sanity batch")
    parser.add_argument("--canonical", type=Path, required=True, help="Canonical directory")
    parser.add_argument("--sample-list", type=Path, required=True, help="Audit sample list path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to run")
    parser.add_argument(
        "--arpeggio",
        action="store_true",
        help="Enable experimental Arpeggio extraction during sanity generation",
    )
    parser.add_argument(
        "--min-tool-fraction",
        type=float,
        default=0.8,
        help="When --arpeggio is enabled, require this fraction of interactions from Arpeggio",
    )
    parser.add_argument(
        "--enforce-tool-fraction",
        action="store_true",
        help="Fail when experimental Arpeggio output is below --min-tool-fraction",
    )
    args = parser.parse_args()
    main(
        args.canonical,
        args.sample_list,
        args.output,
        args.limit,
        min_tool_fraction=args.min_tool_fraction,
        enforce_tool_fraction=args.enforce_tool_fraction,
        use_arpeggio=args.arpeggio,
    )
