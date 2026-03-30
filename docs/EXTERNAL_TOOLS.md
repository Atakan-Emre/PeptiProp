# External Tools Requirements

> Arşiv/deneysel not. Final makale, aktif README/Pages yüzeyi ve raporlanan sonuçlar external tool extractor kullanımına dayanmaz. Final 2D/3D sanity çıktıları geometric residue-contact fallback ile üretilir.

Bu sayfa yalnız deneysel/yerel araştırma amaçlı external tool notlarını toplar. Aktif final yüzey için zorunlu değildir ve metod/sonuç anlatısında temel dayanak olarak kullanılmamalıdır.

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

**Girdi dosyası:** Arpeggio betiği girişi **PDBParser** ile okur; pipeline bu yüzden mmCIF / çok modelli PDB için önce **BioPython ile yalnızca ilk modeli** (`model 0`) sütun-uyumlu PDB olarak yazar (`structure/parsers/tools_pdb_export.py`). Yayın metninde bu kuralı kısaca belirtin.

**CLI uyarısı:** Upstream `arpeggio.py` içinde `-op` (`--output-postfix`) vardır; **`python -X.Y argparse` `-o` geçişini `-op` ile önek eşleştirmesi yapabildiğinden** wrapper **` -o <dizin>` kullanmaz** (pdb yolu bozuluyordu). Çıktılar, verilen PDB ile **aynı dizine** yazılır (`*.contacts` vb.).

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

**macOS (Apple Silicon, Homebrew, `.venv-mlx`) — tek komut:**

```bash
bash scripts/install_external_tools_macos.sh
```

Bu script `biopython` yükler, `third_party/arpeggio` klonlar ve `.venv-mlx/bin/arpeggio` başlatıcısını yazar.

**Doğrulama (tüm platformlar):**

```bash
source .venv-mlx/bin/activate
export PYTHONPATH=src
python scripts/verify_external_tools.py
```

Statik site üretimi (3D örnek sayfası dahil): `python scripts/build_pages_site.py` — `docs/GITHUB_PAGES_TR.md`.

**Verification:**
```bash
arpeggio --help
```

**Documentation:** https://github.com/harryjubb/arpeggio

**If not installed:** Pipeline will fall back to geometric residue contacts.

---

## Tool Priority and Fallbacks

The pipeline uses a **graceful degradation** strategy:

| Tool | Status | Fallback Behavior |
|------|--------|-------------------|
| **Arpeggio** | Not available | Use geometric fallback |
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
- ❌ Tool-based interaction extraction (requires Arpeggio)
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
INFO - PyMOL: Available ✓
INFO - RDKit: Available ✓
```

## Platform-Specific Notes

### Windows
- PyMOL: Use conda installation
- Arpeggio: May require WSL or manual path configuration

### macOS
- PyMOL: Conda or Homebrew
- Arpeggio: Works natively

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

## Citation

If you use these tools, please cite:

- **PyMOL:** The PyMOL Molecular Graphics System, Version 2.0 Schrödinger, LLC.
- **Arpeggio:** Jubb et al. (2017) J Mol Biol. 429(3):365-371
