"""Generate QC dashboard from canonical dataset"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

sys.path.insert(0, 'src')

# Load canonical data
canonical_dir = Path('data/canonical')
complexes = pd.read_parquet(canonical_dir / 'complexes.parquet')
chains = pd.read_parquet(canonical_dir / 'chains.parquet')
residues = pd.read_parquet(canonical_dir / 'residues.parquet')

print(f"Loaded canonical dataset:")
print(f"  Complexes: {len(complexes)}")
print(f"  Chains: {len(chains)}")
print(f"  Residues: {len(residues)}")

# Generate plots
sns.set_style("whitegrid")
plots = []

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# 1. Peptide length distribution
fig, ax = plt.subplots(figsize=(10, 6))
complexes['peptide_length'].hist(bins=50, ax=ax, edgecolor='black')
ax.set_xlabel('Peptide Length (aa)')
ax.set_ylabel('Count')
ax.set_title('Peptide Length Distribution')
ax.axvline(30, color='red', linestyle='--', label='Core/Extension boundary')
ax.legend()
plots.append(('Peptide Length Distribution', fig_to_base64(fig)))
plt.close(fig)

# 2. Source database breakdown
fig, ax = plt.subplots(figsize=(8, 6))
complexes['source_db'].value_counts().plot(kind='bar', ax=ax)
ax.set_xlabel('Source Database')
ax.set_ylabel('Count')
ax.set_title('Source Database Distribution')
plt.xticks(rotation=45)
plots.append(('Source Database', fig_to_base64(fig)))
plt.close(fig)

# 3. Protein length distribution
fig, ax = plt.subplots(figsize=(10, 6))
complexes['protein_length'].hist(bins=50, ax=ax, edgecolor='black')
ax.set_xlabel('Protein Length (aa)')
ax.set_ylabel('Count')
ax.set_title('Protein Length Distribution')
plots.append(('Protein Length Distribution', fig_to_base64(fig)))
plt.close(fig)

# 4. Quality flags
fig, ax = plt.subplots(figsize=(8, 6))
complexes['quality_flag'].value_counts().plot(kind='bar', ax=ax)
ax.set_xlabel('Quality Flag')
ax.set_ylabel('Count')
ax.set_title('Quality Flag Distribution')
plt.xticks(rotation=45)
plots.append(('Quality Flags', fig_to_base64(fig)))
plt.close(fig)

# Generate HTML
html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>QC Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background: #f8f9fa; padding: 20px; border-radius: 6px; text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .plot {{ margin: 30px 0; }}
        .plot img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Canonical Dataset QC Dashboard</h1>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{len(complexes):,}</div>
                <div>Complexes</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(chains):,}</div>
                <div>Chains</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(residues):,}</div>
                <div>Residues</div>
            </div>
        </div>
        
        <h2>Dataset Statistics</h2>
        <ul>
            <li>Peptide length range: {complexes['peptide_length'].min()}-{complexes['peptide_length'].max()} aa</li>
            <li>Protein length range: {complexes['protein_length'].min()}-{complexes['protein_length'].max()} aa</li>
            <li>Chain ID mode: {complexes['chain_id_mode'].iloc[0]}</li>
            <li>Residue number mode: {complexes['residue_number_mode'].iloc[0]}</li>
        </ul>
"""

for title, img_data in plots:
    html += f"""
        <div class="plot">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{img_data}" />
        </div>
"""

html += """
    </div>
</body>
</html>
"""

# Save
output_file = Path('data/reports/qc/qc_dashboard.html')
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html)

print("\n" + "="*60)
print("QC Dashboard generated successfully!")
print("="*60)
print(f"\nOutput: {output_file}")
print("="*60)
