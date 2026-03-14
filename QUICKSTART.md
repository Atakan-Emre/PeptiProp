# PeptidQuantum Quick Start Guide

Get started with PeptidQuantum in 5 minutes!

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/PeptidQuantum.git
cd PeptidQuantum
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install External Tools

For full functionality, install:
- **PyMOL** (3D figures)
- **Arpeggio** (interaction extraction)
- **PLIP** (interaction validation)

See [EXTERNAL_TOOLS.md](EXTERNAL_TOOLS.md) for detailed instructions.

**Note:** Pipeline works without these tools but with reduced functionality.

## Basic Usage

### Option 1: Command Line (Recommended)

Analyze a PDB structure:

```bash
python -m peptidquantum.pipeline.cli run --pdb 1A1M --protein A --peptide B
```

Analyze a local CIF file:

```bash
python -m peptidquantum.pipeline.cli run --cif structure.cif --protein A --peptide B
```

### Option 2: Python Script

```python
from peptidquantum.pipeline import PeptidQuantumPipeline

# Initialize pipeline
pipeline = PeptidQuantumPipeline()

# Run analysis
results = pipeline.run(
    complex_id="1A1M",
    protein_chain="A",
    peptide_chain="B"
)

print(f"Results saved to: {results['output_dir']}")
```

### Option 3: Configuration File

Create `config.json`:

```json
{
  "complex_id": "1A1M",
  "protein_chain": "A",
  "peptide_chain": "B",
  "pocket_radius": 8.0,
  "use_arpeggio": true,
  "use_plip": true,
  "generate_pymol": true,
  "generate_report": true,
  "output_dir": "outputs"
}
```

Run:

```bash
python -m peptidquantum.pipeline.cli run --config config.json
```

## Output Structure

After running, you'll get:

```
outputs/1A1M/
├── structures/
│   └── complex.cif              # Input structure
├── data/
│   ├── contacts.tsv             # Interaction table
│   ├── interaction_fingerprint.json
│   └── residue_residue_matrix.csv
├── figures/
│   ├── complex_overview.png     # 3D structure
│   ├── pocket_zoom.png          # Binding site
│   ├── contact_map.png          # Heatmap
│   └── peptide_2d.png           # 2D chemistry
├── viewer.html                  # Interactive 3D viewer
└── report.html                  # Complete report ⭐
```

## View Results

Open the HTML report in your browser:

```bash
# Windows
start outputs/1A1M/report.html

# macOS
open outputs/1A1M/report.html

# Linux
xdg-open outputs/1A1M/report.html
```

## Common Use Cases

### 1. Quick Analysis (No External Tools)

```bash
python -m peptidquantum.pipeline.cli run \
  --pdb 1A1M \
  --protein A \
  --peptide B \
  --no-arpeggio \
  --no-plip \
  --no-pymol
```

This generates:
- ✅ HTML report with 3Dmol.js viewer
- ✅ Contact maps
- ✅ 2D peptide structure
- ❌ No interaction extraction
- ❌ No PyMOL figures

### 2. Full Analysis (All Tools)

```bash
python -m peptidquantum.pipeline.cli run \
  --pdb 1A1M \
  --protein A \
  --peptide B
```

Requires: Arpeggio, PLIP, PyMOL installed.

### 3. Custom Pocket Radius

```bash
python -m peptidquantum.pipeline.cli run \
  --pdb 1A1M \
  --protein A \
  --peptide B \
  --pocket-radius 10.0
```

### 4. Auto-Detect Chains

```bash
python -m peptidquantum.pipeline.cli run --pdb 1A1M
```

Pipeline will classify chains based on length (peptide ≤ 50 residues).

## Examples

Run example scripts:

```bash
# All examples
python examples/run_example.py

# Specific example
python examples/run_example.py 1  # Basic usage
python examples/run_example.py 4  # Minimal (no external tools)
```

## Troubleshooting

### "Arpeggio not found"

Pipeline will continue with PLIP only. To install Arpeggio:

```bash
git clone https://github.com/harryjubb/arpeggio.git
cd arpeggio
export PATH=$PATH:$(pwd)
```

### "PyMOL not found"

Pipeline will skip 3D figures but generate contact maps. To install:

```bash
conda install -c conda-forge pymol-open-source
```

### "No interactions found"

Check:
1. Are Arpeggio and PLIP installed?
2. Are protein and peptide chains correct?
3. Is the structure valid?

Run with `--verbose` for detailed logs:

```bash
python -m peptidquantum.pipeline.cli run --pdb 1A1M --verbose
```

## Next Steps

- **Full Documentation:** See [README_NEW.md](README_NEW.md)
- **Architecture:** See [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)
- **External Tools:** See [EXTERNAL_TOOLS.md](EXTERNAL_TOOLS.md)
- **Examples:** Check `examples/` directory

## Getting Help

```bash
# Show help
python -m peptidquantum.pipeline.cli --help

# Show run command help
python -m peptidquantum.pipeline.cli run --help

# Generate config template
python -m peptidquantum.pipeline.cli config --output my_config.json
```

## Performance Tips

1. **Use local files** instead of fetching from RCSB for repeated analyses
2. **Disable PyMOL** (`--no-pymol`) for faster testing
3. **Use config files** for reproducible analyses
4. **Cache directory** is reused across runs (default: `data/cache`)

## Minimal Working Example

The absolute minimum to get started:

```python
from peptidquantum.pipeline import PeptidQuantumPipeline

pipeline = PeptidQuantumPipeline()
results = pipeline.run(complex_id="1A1M")
```

This will:
- ✅ Download structure from RCSB
- ✅ Auto-detect chains
- ✅ Generate HTML report
- ✅ Create interactive 3D viewer

No external tools required!

## What's Next?

After your first successful run:

1. **Explore the report:** Open `report.html` in your browser
2. **Check the data:** Review `contacts.tsv` and `interaction_fingerprint.json`
3. **Customize:** Edit config file for your specific needs
4. **Install tools:** Add Arpeggio, PLIP, PyMOL for full functionality

---

**Ready to go?** Run your first analysis:

```bash
python -m peptidquantum.pipeline.cli run --pdb 1A1M --protein A --peptide B
```

Then open `outputs/1A1M/report.html` in your browser! 🚀
