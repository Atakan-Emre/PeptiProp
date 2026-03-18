"""Build audit gallery with top-k samples from each category"""
import sys
from pathlib import Path
import pandas as pd
import random
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def select_audit_samples(
    canonical_dir: Path,
    splits_dir: Path,
    top_k: int = 5
) -> dict:
    """
    Select audit samples across different categories
    
    Categories:
    - Highest pair count (per PDB)
    - Lowest pair count (per PDB)
    - Longest peptide
    - Shortest peptide
    - Random clean samples
    - Random borderline samples (low confidence)
    """
    
    complexes = pd.read_parquet(canonical_dir / "complexes.parquet")
    
    # Load splits
    with open(splits_dir / "train_ids.txt") as f:
        train_ids = set(f.read().strip().split('\n'))
    
    samples = {}
    
    # 1. Highest pair count per PDB
    pairs_per_pdb = complexes.groupby('pdb_id').size()
    top_pdbs = pairs_per_pdb.nlargest(top_k).index.tolist()
    samples['high_pair_count'] = []
    for pdb_id in top_pdbs:
        pdb_complexes = complexes[complexes['pdb_id'] == pdb_id]
        # Take first complex from this PDB
        samples['high_pair_count'].append(pdb_complexes.iloc[0]['complex_id'])
    
    # 2. Lowest pair count per PDB
    low_pdbs = pairs_per_pdb.nsmallest(top_k).index.tolist()
    samples['low_pair_count'] = []
    for pdb_id in low_pdbs:
        pdb_complexes = complexes[complexes['pdb_id'] == pdb_id]
        samples['low_pair_count'].append(pdb_complexes.iloc[0]['complex_id'])
    
    # 3. Longest peptides
    longest = complexes.nlargest(top_k, 'peptide_length')
    samples['longest_peptide'] = longest['complex_id'].tolist()
    
    # 4. Shortest peptides
    shortest = complexes.nsmallest(top_k, 'peptide_length')
    samples['shortest_peptide'] = shortest['complex_id'].tolist()
    
    # 5. Random clean samples (high quality)
    clean = complexes[complexes['quality_flag'] == 'clean']
    if len(clean) >= top_k:
        samples['random_clean'] = clean.sample(n=top_k, random_state=42)['complex_id'].tolist()
    else:
        samples['random_clean'] = clean['complex_id'].tolist()
    
    # 6. Borderline samples (warnings)
    borderline = complexes[complexes['quality_flag'] == 'warning']
    if len(borderline) >= top_k:
        samples['borderline'] = borderline.sample(n=top_k, random_state=42)['complex_id'].tolist()
    else:
        samples['borderline'] = borderline['complex_id'].tolist()
    
    return samples


def generate_audit_gallery(
    canonical_dir: Path,
    splits_dir: Path,
    output_dir: Path,
    top_k: int = 5
):
    """Generate audit gallery"""
    
    print("="*60)
    print("Building Audit Gallery")
    print("="*60)
    
    # Select samples
    samples = select_audit_samples(canonical_dir, splits_dir, top_k)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load complexes for metadata
    complexes = pd.read_parquet(canonical_dir / "complexes.parquet")
    
    # Generate HTML gallery
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Audit Gallery</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; }
        h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
        .category { margin: 40px 0; }
        .category h2 { color: #555; background: #f8f9fa; padding: 10px; border-left: 4px solid #007bff; }
        .samples { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .sample { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 15px; }
        .sample h3 { margin: 0 0 10px 0; color: #007bff; font-size: 1.1em; }
        .sample .meta { font-size: 0.9em; color: #666; }
        .sample .meta div { margin: 5px 0; }
        .note { background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 4px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Audit Gallery - Sample Selection</h1>
        
        <div class="note">
            <strong>Note:</strong> This gallery shows selected samples across different categories for quality control.
            Visualizations will be generated using the existing pipeline.
        </div>
"""
    
    total_samples = 0
    
    for category, complex_ids in samples.items():
        if not complex_ids:
            continue
        
        category_name = category.replace('_', ' ').title()
        html += f"""
        <div class="category">
            <h2>{category_name} ({len(complex_ids)} samples)</h2>
            <div class="samples">
"""
        
        for complex_id in complex_ids:
            complex_row = complexes[complexes['complex_id'] == complex_id].iloc[0]
            
            html += f"""
                <div class="sample">
                    <h3>{complex_id}</h3>
                    <div class="meta">
                        <div><strong>PDB:</strong> {complex_row['pdb_id']}</div>
                        <div><strong>Peptide:</strong> {complex_row['peptide_chain_id']} ({complex_row['peptide_length']} aa)</div>
                        <div><strong>Protein:</strong> {complex_row['protein_chain_id']} ({complex_row['protein_length']} aa)</div>
                        <div><strong>Quality:</strong> {complex_row['quality_flag']}</div>
                    </div>
                </div>
"""
            total_samples += 1
        
        html += """
            </div>
        </div>
"""
    
    html += f"""
        <div class="note">
            <strong>Total samples selected:</strong> {total_samples}
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML
    gallery_file = output_dir / "audit_gallery.html"
    with open(gallery_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    # Save sample list
    sample_list_file = output_dir / "sample_list.txt"
    with open(sample_list_file, 'w') as f:
        for category, complex_ids in samples.items():
            f.write(f"# {category}\n")
            for complex_id in complex_ids:
                f.write(f"{complex_id}\n")
            f.write("\n")
    
    print(f"\nSamples selected:")
    for category, complex_ids in samples.items():
        print(f"  {category}: {len(complex_ids)}")
    
    print(f"\nTotal samples: {total_samples}")
    
    print(f"\n" + "="*60)
    print("Audit gallery generated:")
    print(f"  HTML: {gallery_file}")
    print(f"  Sample list: {sample_list_file}")
    print("="*60)
    
    print("\nNote: Visualization generation (PyMOL, 3Dmol.js, contact maps)")
    print("will be integrated with existing pipeline in next phase.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build audit gallery")
    parser.add_argument("--canonical", type=Path, required=True,
                       help="Canonical directory")
    parser.add_argument("--splits", type=Path, required=True,
                       help="Splits directory")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output directory")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Top K samples per category (default: 5)")
    
    args = parser.parse_args()
    
    generate_audit_gallery(
        canonical_dir=args.canonical,
        splits_dir=args.splits,
        output_dir=args.output,
        top_k=args.top_k
    )
