# External Tools Requirements

PeptidQuantum uses several external tools for interaction extraction and visualization. These tools are **optional** but highly recommended for full functionality.

## Required Python Packages

Install via pip:

```bash
pip install -r requirements.txt
```

## Optional External Tools

### 1. PyMOL (Publication-Quality Figures)

**Purpose:** Generate high-quality 3D structure figures for publications.

**Installation:**

- **Open-Source PyMOL:**
  ```bash
  # Conda (recommended)
  conda install -c conda-forge pymol-open-source
  
  # Or build from source
  # https://github.com/schrodinger/pymol-open-source
  ```

- **Commercial PyMOL:**
  - Download from https://pymol.org/

**Verification:**
```bash
pymol -c -q  # Should not error
```

**If not installed:** Pipeline will skip PyMOL figure generation but continue with other visualizations.

---

### 2. Arpeggio (Interaction Extraction)

**Purpose:** Comprehensive molecular interaction analysis (H-bonds, π-π, hydrophobic, etc.)

**Installation:**

```bash
# Clone repository
git clone https://github.com/harryjubb/arpeggio.git
cd arpeggio

# Install dependencies
pip install biopython numpy

# Add to PATH or specify path in pipeline
export PATH=$PATH:$(pwd)
```

**Verification:**
```bash
arpeggio --help
```

**Documentation:** https://github.com/harryjubb/arpeggio

**If not installed:** Pipeline will rely on PLIP only for interaction extraction.

---

### 3. PLIP 2025 (Interaction Validation)

**Purpose:** Protein-protein interaction detection and validation.

**Installation:**

```bash
# Via pip
pip install plip

# Or from source
git clone https://github.com/pharmai/plip.git
cd plip
python setup.py install
```

**Verification:**
```bash
plip --help
```

**Documentation:** https://github.com/pharmai/plip

**If not installed:** Pipeline will rely on Arpeggio only for interaction extraction.

---

## Tool Priority and Fallbacks

The pipeline uses a **graceful degradation** strategy:

| Tool | Status | Fallback Behavior |
|------|--------|-------------------|
| **Arpeggio** | Not available | Use PLIP only |
| **PLIP** | Not available | Use Arpeggio only |
| **Both** | Not available | ⚠️ No interaction extraction |
| **PyMOL** | Not available | Skip 3D figures, use contact maps only |
| **RDKit** | Not available | Skip 2D peptide chemistry |

## Minimal Working Setup

For basic functionality, you only need:

```bash
pip install biopython rdkit matplotlib seaborn pandas numpy
```

This enables:
- ✅ Structure parsing
- ✅ Contact map visualization
- ✅ 2D peptide rendering
- ✅ HTML report with 3Dmol.js viewer
- ❌ Interaction extraction (requires Arpeggio/PLIP)
- ❌ PyMOL figures (requires PyMOL)

## Recommended Full Setup

For complete functionality:

```bash
# Python packages
pip install -r requirements.txt

# PyMOL
conda install -c conda-forge pymol-open-source

# Arpeggio
git clone https://github.com/harryjubb/arpeggio.git
cd arpeggio && export PATH=$PATH:$(pwd)

# PLIP
pip install plip
```

## Web Visualization (No Installation Required)

- **3Dmol.js:** Loaded via CDN in HTML reports (no installation needed)
- **Mol\*:** Future optional export feature

## Checking Tool Availability

Run the pipeline with `--verbose` to see which tools are available:

```bash
python -m peptidquantum.pipeline.cli run --pdb 1ABC --verbose
```

Output will show:
```
INFO - Arpeggio: Available ✓
INFO - PLIP: Available ✓
INFO - PyMOL: Available ✓
INFO - RDKit: Available ✓
```

## Platform-Specific Notes

### Windows
- PyMOL: Use conda installation
- Arpeggio: May require WSL or manual path configuration
- PLIP: Works via pip

### macOS
- PyMOL: Conda or Homebrew
- Arpeggio: Works natively
- PLIP: Works via pip

### Linux
- All tools work natively
- Recommended for production use

## Troubleshooting

**PyMOL not found:**
```bash
# Check if pymol is in PATH
which pymol

# Or specify path in pipeline
export PYMOL_PATH=/path/to/pymol
```

**Arpeggio not found:**
```bash
# Add to PATH
export PATH=$PATH:/path/to/arpeggio

# Or specify in pipeline config
```

**PLIP not found:**
```bash
# Reinstall
pip install --upgrade plip
```

## Citation

If you use these tools, please cite:

- **PyMOL:** The PyMOL Molecular Graphics System, Version 2.0 Schrödinger, LLC.
- **Arpeggio:** Jubb et al. (2017) J Mol Biol. 429(3):365-371
- **PLIP:** Adasme et al. (2021) Nucleic Acids Res. 49(W1):W530-W534
