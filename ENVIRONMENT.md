# Environment Lock and Reproducibility

This document specifies the exact environment configuration for reproducible results.

> **Güncel giriş:** Apple Silicon MLX için `.venv-mlx` ve `mlx/requirements-m4.txt` kullanımı [mlx/README.md](mlx/README.md) ve kök [README.md](README.md) ile hizalanmıştır; Python sürümü burada kilitli aralıktan genişleyebilir.

## Python Version

**Required:** Python 3.9 - 3.11

**Tested on:** Python 3.10.12

**Not supported:** Python 3.12+ (RDKit compatibility issues)

## Core Dependencies (Locked Versions)

### Critical for Parsing and Chain/Residue ID Handling

```
# Biopython - CRITICAL: auth_chains and auth_residues handling
biopython==1.81

# Gemmi - Alternative mmCIF parser
gemmi==0.6.4
```

**Important:** Biopython's `MMCIFParser` defaults to `auth_chains=True` and `auth_residues=True`. This is the correct behavior for matching author-provided chain and residue IDs in publications.

### Chemistry and Visualization

```
# RDKit - CRITICAL: MolFromSequence compatibility
rdkit==2023.3.2

# Matplotlib/Seaborn - Figure generation
matplotlib==3.7.2
seaborn==0.12.2
```

### Scientific Computing

```
numpy==1.24.3
pandas==2.0.3
scipy==1.10.1
```

### Machine Learning (Optional)

```
torch==2.0.1
torch-geometric==2.3.1
scikit-learn==1.3.0
```

## External Tools (Version-Specific)

### PyMOL

**Recommended:** PyMOL 2.5.0+ (Open Source)

**Installation:**
```bash
conda install -c conda-forge pymol-open-source=2.5.0
```

**Critical Settings:**
- DPI: 300 (publication quality)
- Ray tracing: Enabled by default
- Antialias: 2

**Render Presets:**
```python
# In PyMOL scripts
set ray_shadows, 0
set ray_trace_mode, 1
set antialias, 2
set orthoscopic, on
viewport 1200, 900
ray
png output.png, dpi=300
```

### Arpeggio

**Version:** Latest from GitHub (as of 2024)

**Repository:** https://github.com/harryjubb/arpeggio

**Dependencies:**
```
biopython>=1.70
numpy>=1.15
```

**Installation:**
```bash
git clone https://github.com/harryjubb/arpeggio.git
cd arpeggio
# Add to PATH or specify in config
```

### PLIP

**Version:** 2.3.0+

**Installation:**
```bash
pip install plip==2.3.0
```

**Note:** PLIP 2025 refers to the updated version with protein-protein interaction support.

## Chain and Residue ID Policy

### Default: Author IDs (auth)

```python
config = PipelineConfig(
    chain_id_mode="auth",        # Use author chain IDs
    residue_number_mode="auth"   # Use author residue numbers
)
```

**Rationale:**
- Matches published literature
- Consistent with PDB file format
- Default in Biopython MMCIFParser
- Recommended by RCSB PDB

### Alternative: Label IDs (label)

```python
config = PipelineConfig(
    chain_id_mode="label",       # Use internal label chain IDs
    residue_number_mode="label"  # Use internal label residue numbers
)
```

**Use when:**
- Working with computed structures
- Programmatic consistency required
- Cross-referencing with mmCIF label fields

### Critical Consistency Rule

**All components MUST use the same mode:**
- Structure parser
- Interaction extractors (Arpeggio, PLIP)
- Contact matrix generator
- PyMOL scripts
- HTML reports
- TSV exports

**Violation of this rule causes:**
- Mismatched chain IDs in different outputs
- Incorrect residue numbering in contact maps
- PyMOL labels not matching TSV data
- Report inconsistencies

## Render Presets

### PyMOL Publication-Quality Presets

#### Preset 1: Complex Overview
```python
{
    "width": 1200,
    "height": 900,
    "dpi": 300,
    "ray_trace": True,
    "ray_shadows": 0,
    "antialias": 2,
    "orthoscopic": True,
    "protein_style": "surface + cartoon",
    "peptide_style": "sticks",
    "transparency": 0.3
}
```

#### Preset 2: Pocket Zoom
```python
{
    "width": 1200,
    "height": 900,
    "dpi": 300,
    "ray_trace": True,
    "zoom_factor": 8,
    "show_labels": True,
    "label_size": 14,
    "interaction_lines": True,
    "dash_gap": 0.3,
    "dash_radius": 0.15
}
```

#### Preset 3: Interaction Overlay
```python
{
    "width": 1200,
    "height": 900,
    "dpi": 300,
    "ray_trace": True,
    "protein_style": "cartoon",
    "peptide_style": "sticks",
    "show_interacting_residues": True,
    "color_by_interaction_type": True
}
```

### Contact Map Presets

```python
{
    "figsize": (12, 8),
    "dpi": 300,
    "cmap": "YlOrRd",  # For count aggregation
    "cmap_distance": "viridis_r",  # For distance aggregation
    "show_labels": True,
    "annotate_hotspots": True,
    "label_fontsize": 8
}
```

### 3Dmol.js Viewer Presets

```javascript
{
    "backgroundColor": "white",
    "width": "100%",
    "height": "600px",
    "protein_style": "cartoon",
    "protein_color": "spectrum",
    "peptide_style": "stick",
    "peptide_color": "marine",
    "surface_opacity": 0.3,
    "interaction_radius": 0.15,
    "interaction_dashed": true
}
```

## Testing Environment

### Minimal Test Environment (CI/CD)

```bash
pip install numpy pandas biopython rdkit matplotlib seaborn
```

**Capabilities:**
- ✅ Structure parsing
- ✅ Contact map generation
- ✅ 2D peptide rendering
- ✅ HTML report generation
- ❌ Interaction extraction (no Arpeggio/PLIP)
- ❌ PyMOL figures

### Full Test Environment (Local Development)

```bash
# Python packages
pip install -r requirements.txt

# PyMOL
conda install -c conda-forge pymol-open-source=2.5.0

# Arpeggio
git clone https://github.com/harryjubb/arpeggio.git
export PATH=$PATH:$(pwd)/arpeggio

# PLIP
pip install plip==2.3.0
```

## Reproducibility Checklist

- [ ] Python 3.9-3.11
- [ ] Biopython 1.81 (auth_chains=True default)
- [ ] RDKit 2023.3.2
- [ ] Chain ID mode explicitly set (auth/label)
- [ ] Residue number mode explicitly set (auth/label)
- [ ] PyMOL render presets locked
- [ ] Contact map DPI=300
- [ ] 3Dmol.js CDN version pinned (optional)

## Version Pinning Strategy

### Why Lock Versions?

1. **Biopython:** Chain/residue ID handling changes between versions
2. **RDKit:** MolFromSequence API compatibility
3. **PyMOL:** Render command syntax variations
4. **Arpeggio/PLIP:** Interaction type definitions

### When to Update?

- **Security patches:** Always update
- **Bug fixes:** Update if affecting pipeline
- **New features:** Test thoroughly before updating
- **Breaking changes:** Require pipeline code updates

## Platform-Specific Notes

### Windows

```bash
# Use conda for PyMOL
conda install -c conda-forge pymol-open-source

# Arpeggio may require WSL
wsl --install
```

### macOS

```bash
# PyMOL via conda or Homebrew
conda install -c conda-forge pymol-open-source
# or
brew install pymol

# Arpeggio works natively
```

### Linux (Ubuntu/Debian)

```bash
# All tools work natively
sudo apt-get update
sudo apt-get install python3-dev

# PyMOL
conda install -c conda-forge pymol-open-source
```

## Docker Environment (Recommended for Production)

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Arpeggio
RUN git clone https://github.com/harryjubb/arpeggio.git /opt/arpeggio
ENV PATH="/opt/arpeggio:${PATH}"

# Install PLIP
RUN pip install plip==2.3.0

# Install PyMOL (optional)
RUN conda install -c conda-forge pymol-open-source=2.5.0

WORKDIR /app
COPY . .

ENTRYPOINT ["python", "-m", "peptidquantum.pipeline.cli"]
```

## Verification Script

```python
#!/usr/bin/env python
"""Verify environment setup"""

import sys

def check_environment():
    checks = []
    
    # Python version
    py_version = sys.version_info
    if 3.9 <= py_version.minor <= 3.11:
        checks.append(("Python version", True, f"{py_version.major}.{py_version.minor}"))
    else:
        checks.append(("Python version", False, f"{py_version.major}.{py_version.minor} (need 3.9-3.11)"))
    
    # Biopython
    try:
        import Bio
        checks.append(("Biopython", True, Bio.__version__))
    except ImportError:
        checks.append(("Biopython", False, "Not installed"))
    
    # RDKit
    try:
        import rdkit
        checks.append(("RDKit", True, rdkit.__version__))
    except ImportError:
        checks.append(("RDKit", False, "Not installed"))
    
    # PyMOL
    try:
        import pymol
        checks.append(("PyMOL", True, "Available"))
    except ImportError:
        checks.append(("PyMOL", False, "Not installed (optional)"))
    
    # Print results
    print("\nEnvironment Check:")
    print("=" * 60)
    for name, status, version in checks:
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name:20s} {version}")
    print("=" * 60)
    
    return all(status for name, status, _ in checks[:3])  # First 3 are required

if __name__ == "__main__":
    if check_environment():
        print("\n✓ Environment ready!")
        sys.exit(0)
    else:
        print("\n✗ Environment incomplete")
        sys.exit(1)
```

Run with:
```bash
python verify_environment.py
```
