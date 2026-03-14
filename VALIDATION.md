# Validation and Testing Guide

This document describes the validation strategy for PeptidQuantum v0.1.

## Critical Validation Points

### 1. Chain and Residue ID Consistency

**Problem:** mmCIF files contain both author IDs and label IDs for chains and residues. Mixing these causes silent bugs where different outputs reference different numbering schemes.

**Solution:**
```python
config = PipelineConfig(
    chain_id_mode="auth",        # Default: author chain IDs
    residue_number_mode="auth"   # Default: author residue numbers
)
```

**Validation:**
- All outputs (TSV, contact maps, PyMOL labels, HTML reports) use the same ID scheme
- Config explicitly specifies mode (no implicit defaults)
- Invalid modes are rejected at config validation

**Test:**
```bash
python tests/test_golden_set.py::TestChainResidueIDPolicy
```

### 2. Golden Test Set

Three representative complexes for end-to-end validation:

#### Complex 1: Clean Experimental (1A1M)
- **Type:** Standard protein-peptide complex
- **Expected:** Multiple interactions, clean structure
- **Tests:** Basic pipeline functionality

#### Complex 2: Multi-Chain (1SSH)
- **Type:** Complex with multiple chains
- **Expected:** Correct chain selection and mapping
- **Tests:** Chain ID handling complexity

#### Complex 3: Edge Case (1A2K)
- **Type:** Small peptide, potentially low interactions
- **Expected:** Pipeline completes even with few/no interactions
- **Tests:** Graceful degradation

**Run Golden Tests:**
```bash
python tests/test_golden_set.py::TestGoldenSet -v
```

### 3. Graceful Fallback Validation

**Scenarios:**
1. No external tools (Arpeggio, PLIP, PyMOL) → Minimal report generated
2. Arpeggio only → PLIP skipped
3. PLIP only → Arpeggio skipped
4. No interactions found → Empty contact map, report still generated
5. Missing figures → Report generated with available figures

**Test:**
```bash
python tests/test_golden_set.py::TestGoldenSet::test_graceful_fallback_no_tools
```

### 4. Output Structure Validation

**Required Files (Always):**
```
outputs/{complex_id}/
├── structures/complex.cif
├── data/
│   ├── contacts.tsv
│   └── interaction_fingerprint.json
├── report.html
└── viewer.html
```

**Optional Files (Tool-Dependent):**
```
├── data/residue_residue_matrix.csv
├── figures/
│   ├── complex_overview.png       # PyMOL
│   ├── pocket_zoom.png            # PyMOL
│   ├── interaction_overlay.png   # PyMOL
│   ├── contact_map.png            # matplotlib
│   ├── contact_map_by_type.png   # matplotlib
│   ├── interaction_summary.png   # matplotlib
│   └── peptide_2d.png             # RDKit
```

**Test:**
```bash
python tests/test_golden_set.py::TestOutputStructure
```

## Environment Validation

### Check Environment Setup

```bash
python verify_environment.py
```

**Expected Output:**
```
Environment Check:
============================================================
✓ Python version         3.10
✓ Biopython             1.81
✓ RDKit                 2023.3.2
✓ PyMOL                 Available (optional)
============================================================

✓ Environment ready!
```

### Verify Tool Availability

```bash
python -m peptidquantum.pipeline.cli run --pdb 1A1M --verbose
```

**Check Logs:**
```
INFO - Arpeggio: Available ✓
INFO - PLIP: Available ✓
INFO - PyMOL: Available ✓
INFO - RDKit: Available ✓
```

## Reproducibility Tests

### Test 1: Same Input → Same Output

```bash
# Run 1
python -m peptidquantum.pipeline.cli run --pdb 1A1M --protein A --peptide B

# Run 2 (should produce identical results)
python -m peptidquantum.pipeline.cli run --pdb 1A1M --protein A --peptide B
```

**Validate:**
- Same number of interactions
- Same contact matrix dimensions
- Same residue IDs in outputs

### Test 2: Config-Based Reproducibility

```bash
# Generate config
python -m peptidquantum.pipeline.cli config --output test_config.json

# Edit config, then run
python -m peptidquantum.pipeline.cli run --config test_config.json

# Re-run with same config
python -m peptidquantum.pipeline.cli run --config test_config.json
```

**Validate:**
- Identical outputs across runs
- Config parameters respected

### Test 3: Cross-Platform Consistency

Run on different platforms (Windows, macOS, Linux) and verify:
- Same interaction counts
- Same residue numbering
- Same contact matrix values

## Known Issues and Limitations

### Issue 1: Chain ID Ambiguity

**Problem:** Some PDB structures have non-standard chain IDs or multiple assemblies.

**Mitigation:**
- Explicitly specify `protein_chain` and `peptide_chain`
- Use `chain_id_mode="auth"` for consistency with literature
- Validate chain IDs before running pipeline

### Issue 2: Residue Numbering Gaps

**Problem:** Some structures have non-sequential residue numbering (e.g., 1, 2, 5, 6...).

**Mitigation:**
- Use author residue numbers (default)
- Contact maps handle gaps correctly
- PyMOL labels use actual residue numbers

### Issue 3: External Tool Availability

**Problem:** Arpeggio and PLIP may not be installed in all environments.

**Mitigation:**
- Pipeline continues without them (graceful fallback)
- Minimal functionality still works (parsing, visualization)
- Clear warnings in logs

### Issue 4: PyMOL Render Variations

**Problem:** PyMOL output may vary slightly across versions.

**Mitigation:**
- Lock PyMOL version (2.5.0 recommended)
- Use standardized render presets
- Document render settings in ENVIRONMENT.md

## Validation Checklist

Before releasing a new version:

- [ ] All golden tests pass
- [ ] Chain ID consistency verified
- [ ] Residue numbering consistency verified
- [ ] Graceful fallbacks tested
- [ ] Output structure validated
- [ ] Environment reproducibility confirmed
- [ ] Cross-platform testing completed
- [ ] Documentation updated
- [ ] Example scripts work
- [ ] Known issues documented

## Continuous Integration

### Minimal CI Pipeline (No External Tools)

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install numpy pandas biopython rdkit matplotlib seaborn pytest
      - name: Run tests
        run: |
          pytest tests/test_golden_set.py -v --tb=short
```

### Full CI Pipeline (With External Tools)

```yaml
# Requires Docker or custom runner with tools installed
jobs:
  test-full:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup environment
        run: |
          pip install -r requirements.txt
          # Install Arpeggio, PLIP, PyMOL
      - name: Run full tests
        run: |
          pytest tests/ -v --tb=short
```

## Manual Validation Procedure

### 1. Fresh Environment Test

```bash
# Create fresh conda environment
conda create -n peptidquantum-test python=3.10
conda activate peptidquantum-test

# Install minimal dependencies
pip install numpy pandas biopython rdkit matplotlib seaborn

# Run minimal test
python -m peptidquantum.pipeline.cli run --pdb 1A1M --no-arpeggio --no-plip --no-pymol
```

**Expected:** Report and viewer generated successfully.

### 2. Full Environment Test

```bash
# Install all tools
pip install -r requirements.txt
conda install -c conda-forge pymol-open-source
pip install plip
# Install Arpeggio manually

# Run full test
python -m peptidquantum.pipeline.cli run --pdb 1A1M --protein A --peptide B --verbose
```

**Expected:** All figures generated, interactions extracted.

### 3. Edge Case Test

```bash
# Test with potentially problematic structure
python -m peptidquantum.pipeline.cli run --pdb 1A2K --protein A --peptide B
```

**Expected:** Pipeline completes even if few interactions found.

## Reporting Issues

When reporting validation failures, include:

1. **Environment:**
   - Python version
   - OS and version
   - Package versions (`pip freeze`)
   - External tool versions

2. **Input:**
   - Complex ID or CIF file
   - Config file (if used)
   - Command line arguments

3. **Output:**
   - Error messages
   - Log output (with `--verbose`)
   - Generated files (if any)

4. **Expected vs Actual:**
   - What should have happened
   - What actually happened
   - Differences in outputs

## Version History

### v0.1 (Current)
- Initial release candidate
- Experimental structures only
- Chain/residue ID standardization
- Golden test set established
- Graceful fallback implemented

### Future Versions
- v0.2: AlphaFold fallback
- v0.3: Mol* export
- v1.0: Production release
