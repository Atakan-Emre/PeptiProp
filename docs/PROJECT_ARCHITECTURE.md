# PeptidQuantum - Protein-Peptide Interaction Visualization System

## Architecture Overview

This is a complete **4-layer pipeline** for protein-peptide interaction prediction and visualization:

```
Data Layer → Structure Layer → Interaction Layer → Visualization Layer
```

## Layer 1: Data Acquisition

### Primary Sources
- **RCSB PDB**: Experimental 3D structures + computed models (AlphaFold DB, ModelArchive)
- **PROPEDIA v2.3**: Peptide-protein interaction database with clustered complexes

### Input Formats
- Protein sequence: FASTA
- Peptide sequence: FASTA / plain sequence
- 3D structure: **PDBx/mmCIF** (primary format)
- Chemical peptide: HELM (for modified peptides)
- Metadata: CSV / JSON

### Data Module Structure
```
src/peptidquantum/data/
├── fetchers/
│   ├── rcsb_fetcher.py      # RCSB PDB API integration
│   ├── propedia_fetcher.py  # PROPEDIA database access
│   └── alphafold_fetcher.py # AlphaFold DB integration
├── parsers/
│   ├── mmcif_parser.py      # mmCIF parsing (Biopython/Gemmi)
│   ├── fasta_parser.py      # Sequence parsing
│   └── metadata_parser.py   # Metadata handling
└── models.py                # Data models (Complex, Chain, Residue)
```

## Layer 2: Structure Generation/Completion

### Priority Order
1. Experimental complex (RCSB/PROPEDIA)
2. Experimental monomer + predicted complex
3. Fully predicted complex (AlphaFold 3)

### AlphaFold 3 Integration
- **AlphaFold Server**: For protein-molecule interaction prediction
- **Input format**: Supports proteins, modified residues, ligands (CCD/SMILES), covalent bonds
- Peptide can be treated as second protein chain or modified residue

### Normalization Pipeline
```
src/peptidquantum/structure/
├── prediction/
│   ├── alphafold_client.py  # AlphaFold Server API
│   └── structure_builder.py # Build complexes from sequences
├── normalization/
│   ├── chain_cleaner.py     # Chain ID cleanup, alternate location
│   ├── protonation.py       # Add hydrogens
│   ├── assembly.py          # Biological assembly selection
│   └── pocket_extractor.py  # Extract 6-8Å pocket around peptide
└── quality/
    ├── validator.py         # Structure validation
    └── confidence.py        # pLDDT, interface confidence
```

### Provenance Tracking
Each structure must have:
- `structure_source`: experimental | predicted | hybrid
- `structure_origin`: RCSB | PROPEDIA | AF3 | AF_Server
- `confidence`: pLDDT / model confidence / interface confidence
- `assembly_used`: biological assembly / asymmetric unit

## Layer 3: Interaction Extraction

### Active Final Surface: Geometric Residue Contacts
- Final reported viewer/report outputs use geometric residue-contact summaries
- This keeps visual outputs populated without over-claiming chemistry annotations
- External tool extractors are archival/experimental and not part of final reported results

### Interaction Table Schema
```
protein_chain | protein_residue_id | protein_residue_name |
peptide_chain | peptide_residue_id | peptide_residue_name |
interaction_type | atom_pair | distance | angle |
source_tool | confidence
```

### Module Structure
```
src/peptidquantum/interaction/
├── extractors/
│   ├── arpeggio_wrapper.py  # Archived experimental integration
│   └── merger.py            # Merge helper for experimental tool outputs
├── analysis/
│   ├── contact_matrix.py    # Residue-residue contact matrix
│   ├── fingerprint.py       # Interaction fingerprint
│   └── importance.py        # Residue importance scoring
└── models.py                # Interaction data models
```

## Layer 4: Representation Learning

### Dual Representation System

#### Primary: Residue-Level Interface Graph
```python
Protein Branch:
- nodes: pocket residues
- edges: spatial contact / kNN / same-chain adjacency
- node features: residue type + secondary structure + solvent exposure + ESM embedding
- edge features: distance + orientation + contact type

Peptide Branch:
- nodes: peptide residues
- edges: sequential adjacency + spatial contact
- node features: residue type + position + modification flags
- edge features: peptide bond / spatial distance / interaction context
```

#### Secondary: Peptide Atom-Level Chemical Graph
```python
- RDKit MolFromFASTA / MolFromSequence / MolFromHELM
- 2D chemical depiction
- Modified residue highlighting
- Atom-level motif extraction
```

### Model Architecture
```
src/peptidquantum/models/
├── encoders/
│   ├── protein_encoder.py   # GNN for protein pocket
│   ├── peptide_encoder.py   # GNN for peptide
│   └── chemical_encoder.py  # RDKit-based 2D encoder
├── fusion/
│   ├── cross_attention.py   # Cross-attention mechanism
│   └── co_attention.py      # Co-attention mechanism
├── heads/
│   ├── binding_head.py      # P(bind) prediction
│   ├── contact_head.py      # Residue-residue contact
│   ├── interaction_head.py  # Interaction type scoring
│   └── importance_head.py   # Residue importance
└── full_model.py            # Complete dual-encoder model
```

### Graph Construction
- Use PyTorch Geometric GATConv with `edge_dim` support
- ESM-2 embeddings for sequence context (optional)

## Layer 5: Visualization

### Static Publication-Quality Figures (PyMOL)
```
outputs/figures/
├── complex_overview.png     # Protein surface + cartoon, peptide sticks
├── pocket_zoom.png          # Zoomed binding pocket
├── interactions.png         # H-bonds, salt bridges, hydrophobic contacts
├── surface_electrostatic.png
└── surface_hydrophobic.png
```

### Interactive Analysis
- **Mol\***: RCSB interaction viewer (5Å neighborhood, hydrophobicity coloring)
- **3Dmol.js**: WebGL-based browser viewer (no plugins required)

### Structural Annotation (ChimeraX)
- Residue labeling
- H-bond visualization
- Selection and measurement tools

### 2D Chemical Depiction (RDKit)
```
outputs/chemistry/
├── peptide_2d.png           # 2D chemical structure
├── peptide_2d_highlighted.png  # Active residues highlighted
└── peptide_graph.json       # Atom-level graph
```

### Visualization Module
```
src/peptidquantum/visualization/
├── pymol/
│   ├── renderer.py          # PyMOL script generation
│   ├── coloring.py          # Custom coloring schemes
│   └── sessions.py          # Save PyMOL sessions
├── chimerax/
│   ├── renderer.py          # ChimeraX script generation
│   └── sessions.py          # Save ChimeraX sessions
├── web/
│   ├── molstar_viewer.py    # Mol* integration
│   └── 3dmol_viewer.py      # 3Dmol.js integration
├── chemistry/
│   ├── rdkit_renderer.py    # 2D chemical drawings
│   └── helm_parser.py       # HELM format support
└── reports/
    ├── contact_map.py       # Residue-residue heatmap
    ├── fingerprint.py       # Interaction fingerprint plot
    └── panel_generator.py   # Multi-panel figure assembly
```

## Output System

### Per-Complex Output Structure
```
outputs/{complex_id}/
├── structures/
│   ├── complex.cif          # Full complex (mmCIF)
│   ├── pocket.cif           # Extracted pocket
│   └── peptide.mol2         # Peptide structure
├── chemistry/
│   ├── peptide_2d.png
│   └── peptide_2d.svg
├── figures/
│   ├── complex_overview.png
│   ├── pocket_zoom.png
│   ├── interactions_annotated.png
│   └── contact_map.png
├── data/
│   ├── contacts.tsv         # Interaction table
│   ├── interaction_fingerprint.json
│   ├── residue_matrix.csv   # Contact matrix
│   └── predictions.json     # Binding score, confidence
├── sessions/
│   ├── pymol_session.pse
│   ├── chimerax_session.cxs
│   └── molstar_state.molj
└── report.html              # Comprehensive HTML report
```

## Contact Map System (Protein Contacts Atlas Style)

### Matrix Format
- Rows: peptide residues
- Columns: protein pocket residues
- Cells: contact count / strongest interaction / learned importance
- Color: H-bond (blue) / hydrophobic (yellow) / ionic (red) / dominant type

### Implementation
```
src/peptidquantum/analysis/
├── contact_atlas.py         # Multi-level contact visualization
├── chord_plot.py            # Circular contact representation
└── importance_strip.py      # Residue importance visualization
```

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA ACQUISITION                                         │
│    RCSB PDB / PROPEDIA → sequences + structures            │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 2. STRUCTURE GENERATION/COMPLETION                          │
│    Experimental → use directly                              │
│    Missing → AlphaFold 3 prediction                         │
│    Normalize → mmCIF, clean, protonate, extract pocket      │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 3. INTERACTION EXTRACTION                                   │
│    Geometric residue-contact fallback → interaction table   │
│    Contact matrix, fingerprint, importance scores           │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 4. REPRESENTATION LEARNING                                  │
│    Residue graph (protein ↔ peptide)                        │
│    Chemical graph (peptide 2D)                              │
│    ESM embeddings (optional)                                │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 5. PREDICTION                                               │
│    Dual GNN encoder + co-attention                          │
│    Binding score + contact prediction + importance          │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│ 6. VISUALIZATION                                            │
│    PyMOL → publication figures                              │
│    RDKit → 2D chemistry                                     │
│    Mol*/3Dmol.js → interactive web                          │
│    Contact atlas → residue-level analysis                   │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Core
- Python 3.10+
- PyTorch + PyTorch Geometric
- Biopython + Gemmi (mmCIF parsing)
- RDKit (chemistry)

### Structure
- AlphaFold Server / AlphaFold 3
- Geometric residue-contact fallback (final visualization continuity)
- Experimental tool extractors (archival, not used in reported results)

### Visualization
- PyMOL (static figures)
- ChimeraX (annotation)
- Mol* (interactive RCSB viewer)
- 3Dmol.js (web embedding) — `Viewer3DMol`: PDB/mmCIF format seçimi, güvenli yapı gömme, `viewer_state.json` (`structure_format`, `structure_basename`, zincirler, etkileşimler)

### ML
- PyTorch Geometric GATConv
- ESM-2 (optional embeddings)
- Cross/co-attention mechanisms

## Dependencies

```
# Core
torch>=2.0.0
torch-geometric>=2.3.0
biopython>=1.81
gemmi>=0.6.0
rdkit>=2023.3.1
pandas>=2.0.0
numpy>=1.24.0

# Visualization
pymol-open-source>=2.5.0
py3Dmol>=2.0.0

# Structure prediction
alphafold (via API)

# Interaction analysis
no external interaction extractor required for final active path

# Optional
fair-esm>=2.0.0  # ESM embeddings
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
```

## Key Design Decisions

1. **mmCIF as primary format**: Official wwPDB standard, no atom/residue/chain limitations
2. **Dual representation**: Residue-level for biology + atom-level for chemistry
3. **Final active visualization**: geometric residue-contact fallback
4. **Provenance tracking**: Always know if structure is experimental or predicted
5. **Multi-scale visualization**: From full complex to atom-level interactions
6. **Contact atlas as core output**: Not just prediction, but explanation

## Web / GitHub Pages

- `scripts/build_pages_site.py` → `site/` (index, `data/manifest.json`, `embed/viewer-demo.html` 3Dmol + örnek CIF)
- `.github/workflows/pages.yml` — Actions ile yayın
- Ayrıntı: `docs/GITHUB_PAGES_TR.md`

## Durum özeti (v0.1)

PROPEDIA kanonik hattı, leakage-free split/negatifler, klasik + MLX skorlama, kalibrasyon/sıralama raporları ve 2D/3D çıktılar üretilebilir durumda. Eski “Next Steps” maddeleri çoğunlukla tamamlandı; model ve veri genişletmeleri için `ROADMAP.md` ve `docs/` altındaki TR notlara bakın.
