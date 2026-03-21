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


def main(canonical_dir: Path, sample_list: Path, output_dir: Path, limit: int) -> None:
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
            use_arpeggio=True,
            use_plip=True,
            generate_pymol=False,
            generate_report=True,
            generate_viewer=True,
        )
        sample_output_dir = output_dir / complex_id
        summary.append(
            {
                "complex_id": complex_id,
                "pdb_id": pdb_id,
                "protein_chain": protein_chain,
                "peptide_chain": peptide_chain,
                "status": result["status"],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run visualization sanity batch")
    parser.add_argument("--canonical", type=Path, required=True, help="Canonical directory")
    parser.add_argument("--sample-list", type=Path, required=True, help="Audit sample list path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to run")
    args = parser.parse_args()
    main(args.canonical, args.sample_list, args.output, args.limit)
