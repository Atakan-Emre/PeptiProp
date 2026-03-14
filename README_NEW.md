# PeptidQuantum - Protein-Peptide Interaction Visualization System

**Complete 4-layer pipeline for protein-peptide interaction prediction and 3D/2D visualization**

## 🎯 Project Vision

This project implements a comprehensive system that:
1. **Acquires** protein-peptide complex structures from RCSB PDB and PROPEDIA
2. **Generates** missing structures using AlphaFold 3
3. **Extracts** detailed molecular interactions using Arpeggio and PLIP
4. **Predicts** binding affinity and residue-level importance
5. **Visualizes** results in publication-quality 3D and 2D formats

## 🏗️ Architecture

```
Data Layer → Structure Layer → Interaction Layer → ML Layer → Visualization Layer
```

### Layer 1: Data Acquisition
- **RCSB PDB**: Experimental structures + AlphaFold DB
- **PROPEDIA v2.3**: Peptide-protein interaction database
- **Format**: PDBx/mmCIF (official wwPDB standard)

### Layer 2: Structure Processing
- **Parsing**: Biopython + Gemmi for mmCIF
- **Prediction**: AlphaFold 3 / AlphaFold Server for missing structures
- **Normalization**: Chain cleanup, protonation, pocket extraction (6-8Å)

### Layer 3: Interaction Extraction
- **Arpeggio**: Comprehensive interaction analysis (H-bonds, π-π, hydrophobic, etc.)
- **PLIP 2025**: Protein-protein interaction validation
- **Output**: Residue-residue contact matrix, interaction fingerprint

### Layer 4: Machine Learning
- **Dual-encoder**: Protein pocket GNN ↔ Peptide GNN
- **Fusion**: Cross-attention / Co-attention
- **Heads**: Binding prediction + contact prediction + residue importance

### Layer 5: Visualization
- **PyMOL**: Publication-quality 3D figures
- **RDKit**: 2D chemical structures
- **Mol\* / 3Dmol.js**: Interactive web viewers
- **Contact Atlas**: Residue-level interaction heatmaps

## 📁 Project Structure

```
PeptidQuantum/
├── data/
│   ├── raw/GEPPRI/              # Original GEPPRI data
│   ├── processed/               # Processed pair data
│   └── cache/                   # Downloaded structures
│       ├── rcsb/
│       └── propedia/
├── src/peptidquantum/
│   ├── data/
│   │   ├── models.py            # Core data models
│   │   ├── fetchers/
│   │   │   ├── rcsb_fetcher.py
│   │   │   ├── propedia_fetcher.py
│   │   │   └── alphafold_fetcher.py
│   │   └── parsers/
│   │       ├── mmcif_parser.py
│   │       └── fasta_parser.py
│   ├── structure/
│   │   ├── parsers/
│   │   │   └── mmcif_parser.py  # Structure parsing
│   │   ├── prediction/
│   │   │   └── alphafold_client.py
│   │   └── normalization/
│   │       ├── chain_cleaner.py
│   │       └── pocket_extractor.py
│   ├── interaction/
│   │   ├── extractors/
│   │   │   ├── arpeggio_wrapper.py
│   │   │   └── plip_wrapper.py
│   │   └── analysis/
│   │       ├── contact_matrix.py
│   │       └── fingerprint.py
│   ├── models/
│   │   ├── encoders/
│   │   │   ├── protein_encoder.py
│   │   │   └── peptide_encoder.py
│   │   ├── fusion/
│   │   │   └── cross_attention.py
│   │   └── full_model.py
│   ├── visualization/
│   │   ├── pymol/
│   │   │   └── renderer.py
│   │   ├── chemistry/
│   │   │   └── rdkit_renderer.py
│   │   └── reports/
│   │       └── contact_map.py
│   ├── training/
│   │   ├── trainer.py           # Advanced trainer
│   │   └── ablation.py          # Ablation study
│   └── utils/
│       └── data_split.py
├── scripts/
│   ├── prepare_data.py
│   └── download_structures.py
├── models/
│   └── trained/                 # Model checkpoints
├── outputs/
│   └── {complex_id}/
│       ├── structures/
│       ├── figures/
│       ├── data/
│       └── report.html
├── train.py                     # Main training script
├── test_model.py               # Model testing
└── run.bat                     # Windows launcher
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Test Model

```bash
python test_model.py
```

### 3. Prepare Data

```bash
python scripts/prepare_data.py
```

### 4. Train Model

**Single training:**
```bash
python train.py --data data/processed/geppri_all_pairs.jsonl --epochs 100
```

**With ablation study:**
```bash
python train.py --data data/processed/geppri_all_pairs.jsonl --ablation --ablation-mode one_at_a_time
```

**Resume from checkpoint:**
```bash
python train.py --data data/processed/geppri_all_pairs.jsonl --resume models/trained/latest_checkpoint.pt
```

### 5. Windows Quick Start

```bash
run.bat
```

## 📊 Output Structure

For each complex, the system generates:

### Structural Files
- `complex.cif` - Full complex (mmCIF)
- `pocket.cif` - Extracted binding pocket
- `peptide.mol2` - Peptide structure

### Figures
- `complex_overview.png` - 3D complex view
- `pocket_zoom.png` - Binding site detail
- `interactions_annotated.png` - Labeled interactions
- `contact_map.png` - Residue-residue heatmap
- `peptide_2d.png` - 2D chemical structure

### Data
- `contacts.tsv` - Interaction table
- `interaction_fingerprint.json` - Interaction types
- `residue_matrix.csv` - Contact matrix
- `predictions.json` - Binding scores

### Interactive
- `pymol_session.pse` - PyMOL session
- `chimerax_session.cxs` - ChimeraX session
- `report.html` - Comprehensive HTML report

## 🔬 Advanced Features

### Early Stopping
```bash
python train.py --data data.jsonl --patience 15
```

### Learning Rate Scheduling
Automatic ReduceLROnPlateau with factor=0.5, patience=5

### Gradient Clipping
Default max_norm=1.0

### Checkpointing
- Saves every N epochs (default: 5)
- Saves best model based on validation F1
- Saves latest checkpoint for resuming

### Ablation Study
Tests multiple hyperparameter combinations:
- Learning rates: [1e-4, 5e-4, 1e-3]
- Batch sizes: [16, 32, 64]
- Hidden dimensions: [32, 64, 128]
- Message passing steps: [4, 6, 8]
- Attention heads: [4, 8]
- Dense units: [256, 512, 1024]
- Dropout: [0.1, 0.2, 0.3]
- Weight decay: [1e-6, 1e-5, 1e-4]

## 📈 Model Architecture

### Dual-Encoder Design
```
Protein Pocket Encoder (GNN)
        ↓
   Transformer Readout
        ↓
    Co-Attention ← → Peptide Encoder (GNN)
        ↓              ↓
   Interaction    Transformer Readout
      Head
```

### Features
- **Node features**: Residue type + secondary structure + solvent exposure + ESM embeddings
- **Edge features**: Distance + orientation + contact type
- **Attention**: Multi-head cross-attention for protein-peptide fusion
- **Multiple heads**: Binding + contact + importance prediction

## 🎨 Visualization Examples

### 3D Complex View (PyMOL)
- Protein: surface + cartoon
- Peptide: sticks/licorice
- Pocket: semi-transparent surface
- Interactions: dashed lines (H-bonds, salt bridges)

### Contact Map
- Rows: peptide residues
- Columns: protein pocket residues
- Color: interaction type (H-bond=blue, hydrophobic=yellow, ionic=red)

### 2D Chemical Structure (RDKit)
- Standard peptide: FASTA → 2D depiction
- Modified peptide: HELM → 2D depiction
- Highlighted active residues

## 📚 Documentation

- [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) - Complete technical architecture
- [docs/PROJECT_OBJECTIVES_TR.md](docs/PROJECT_OBJECTIVES_TR.md) - Project goals
- [docs/MODEL_IMPROVEMENTS_TR.md](docs/MODEL_IMPROVEMENTS_TR.md) - Model enhancements

## 🔧 Dependencies

### Core
- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- Biopython 1.81+
- RDKit 2023.3+

### Visualization
- PyMOL (optional, for 3D figures)
- py3Dmol (for web viewers)

### Structure Analysis
- Arpeggio (install separately)
- PLIP (install separately)

### Optional
- fair-esm (for ESM embeddings)
- AlphaFold 3 (for structure prediction)

## 🎯 Performance Targets

| Metric | Target |
|--------|--------|
| F1 Score | > 0.75 |
| AUPRC | > 0.80 |
| ROC-AUC | > 0.85 |

## 📝 Citation

If you use this code, please cite:
```
PeptidQuantum: A comprehensive protein-peptide interaction 
visualization and prediction system
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please see CONTRIBUTING.md

## 📧 Contact

For questions and support, please open an issue.
