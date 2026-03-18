# Sprint 2B Validation Guide

## Overview

Sprint 2B validates the canonical dataset generation pipeline on **100 real complexes** before full-scale production.

**Critical Fix:** PROPEDIA provides PDB format, not mmCIF. We implement PDB→mmCIF normalization via RCSB backfill.

## Validation Strategy

### Sample Distribution

**Total: 100 complexes**
- **50 from PROPEDIA** (PDB format → mmCIF backfill)
- **25 from PepBDB** (mmCIF format)
- **25 from BioLiP2** (mmCIF peptide subset)

**Sample Mix:**
- Short peptides (5-15 aa)
- Long peptides (16-30 aa)
- Extension peptides (31-50 aa)
- Multi-chain structures
- Low interaction structures
- Potentially problematic cases

## 10-Point Validation Checklist

Each complex is validated against 10 criteria:

| # | Check | Pass Criteria |
|---|-------|---------------|
| 1 | **Parse successful** | Structure parsed without errors |
| 2 | **Auth chain found** | Author chain IDs present |
| 3 | **Peptide chain found** | At least one peptide chain identified |
| 4 | **Protein chain found** | At least one protein chain identified |
| 5 | **Peptide length valid** | 5 ≤ length ≤ 50 aa |
| 6 | **Protein length valid** | length ≥ 30 aa |
| 7 | **Pair extractor confidence** | confidence ≥ 0.5 |
| 8 | **No quarantine reason** | No quarantine flags |
| 9 | **Parquet writable** | Can be written to canonical format |
| 10 | **Visualization compatible** | Compatible with existing pipeline |

**Overall Pass:** All 10 checks must pass.

## Critical: PDB→mmCIF Normalization

### Problem

PROPEDIA download page provides:
- Complexes (49,300) - **PDB format (ZIP)**
- Receptors (35,478) - **PDB format (ZIP)**
- Peptides (28,581) - **PDB format (ZIP)**
- Interfaces (49,300) - **PDB format (ZIP)**

Our canonical format: **mmCIF**

### Solution

**PDB→mmCIF backfill from RCSB:**

```python
from peptidquantum.data.processors import PDBToMMCIFConverter

converter = PDBToMMCIFConverter(cache_dir="data/staging/mmcif_cache")

# Convert PDB to mmCIF
mmcif_path, status = converter.convert("propedia_complex.pdb")

# Verify chain consistency
is_consistent, message = converter.verify_chain_consistency(
    pdb_file="propedia_complex.pdb",
    mmcif_file=mmcif_path
)
```

**Strategy:**
1. Extract PDB ID from PROPEDIA file
2. Download authoritative mmCIF from RCSB File Download Services
3. Verify chain ID consistency
4. Cache mmCIF for reuse

**RCSB File Download Services:**
- URL: `https://files.rcsb.org/download/{pdb_id}.cif`
- Provides: PDB, mmCIF, PDBML/XML formats
- Biological assembly mmCIF also available

## Implementation

### 1. PDBToMMCIFConverter (`pdb_to_mmcif.py`)

**Features:**
- Extract PDB ID from filename or HEADER line
- Download mmCIF from RCSB
- Cache downloaded files
- Verify chain consistency
- Batch conversion support
- Rate limiting

**Usage:**
```bash
# Single file
python -m peptidquantum.data.processors.pdb_to_mmcif complex.pdb

# Directory
python -m peptidquantum.data.processors.pdb_to_mmcif propedia/complexes/
```

### 2. ValidationChecklist (`validation_checklist.py`)

**Features:**
- 10-point validation per complex
- PDB→mmCIF conversion integration
- Batch validation
- CSV export (detailed results)
- HTML export (summary report)
- Statistics by source

**Usage:**
```python
from peptidquantum.data.processors import ValidationChecklist

validator = ValidationChecklist(
    mmcif_cache_dir="data/staging/mmcif_cache",
    chain_id_mode="auth",
    residue_number_mode="auth"
)

# Validate batch
results = validator.validate_batch(structure_files, source_db="propedia")

# Export
validator.export_results("validation_results.csv")
validator.export_summary_report("validation_summary.html")
```

### 3. Master Validation Script (`validate_sprint2b.py`)

**Usage:**
```bash
python scripts/validate_sprint2b.py \
  --raw data/raw \
  --staging data/staging \
  --verbose
```

**Options:**
- `--propedia-count 50` - Number of PROPEDIA samples
- `--pepbdb-count 25` - Number of PepBDB samples
- `--biolip2-count 25` - Number of BioLiP2 samples
- `--chain-id-mode auth` - Chain ID mode
- `--residue-number-mode auth` - Residue number mode
- `--verbose` - Detailed logging
- `--log-file` - Log to file

## Running Validation

### Step 1: Download Sample Data

```bash
# Download PROPEDIA (complexes only for testing)
python -m peptidquantum.data.downloaders.propedia \
  --output data/raw/propedia

# Download PepBDB
python -m peptidquantum.data.downloaders.pepbdb \
  --output data/raw/pepbdb \
  --max-length 50

# Download BioLiP2 peptide subset
python -m peptidquantum.data.downloaders.biolip2 \
  --output data/raw/biolip2 \
  --peptides-only \
  --max-length 30
```

### Step 2: Run Validation

```bash
python scripts/validate_sprint2b.py \
  --raw data/raw \
  --staging data/staging \
  --verbose \
  --log-file data/staging/validation.log
```

### Step 3: Review Results

**CSV Results:** `data/staging/validation/validation_results.csv`
- Detailed per-complex results
- All 10 checklist items
- Error messages
- Quarantine reasons

**HTML Summary:** `data/staging/validation/validation_summary.html`
- Overall pass rate
- Checklist statistics
- Quarantine reason breakdown
- Visual dashboard

### Step 4: Interpret Results

**Pass Rate ≥80%:** ✓ Proceed to canonical dataset generation
**Pass Rate 50-80%:** ⚠ Review and fix common issues
**Pass Rate <50%:** ✗ Critical issues, fix before proceeding

## Expected Issues and Solutions

### Issue 1: PDB→mmCIF Conversion Failures

**Symptoms:**
- `pdb_conversion_failed` quarantine reason
- Missing PDB IDs

**Solutions:**
- Check PROPEDIA file naming
- Verify RCSB availability
- Check network connectivity
- Review HEADER lines in PDB files

### Issue 2: Chain ID Inconsistencies

**Symptoms:**
- `auth_label_conflict` quarantine reason
- Chain mapping failures

**Solutions:**
- Use `verify_chain_consistency()` to diagnose
- Check if chains match between PDB and mmCIF
- Review auth vs label ID usage

### Issue 3: Peptide Length Violations

**Symptoms:**
- `peptide_too_short` or `peptide_too_long` quarantine reasons
- High quarantine rate

**Solutions:**
- Review length policy (5-50 aa)
- Check if extension set should be enabled
- Verify chain type classification

### Issue 4: Missing Chains

**Symptoms:**
- `no_peptide_chain` or `no_protein_chain` quarantine reasons
- Low peptide/protein chain found rates

**Solutions:**
- Check chain length thresholds
- Review auto-detection logic
- Consider explicit chain ID specification

## Success Criteria

**Sprint 2B Complete When:**

- [x] PDB→mmCIF converter implemented
- [x] Validation checklist implemented
- [x] Master validation script ready
- [ ] **100 complexes validated**
- [ ] **Pass rate ≥80%**
- [ ] **Quarantine report reviewed**
- [ ] **Chain ID consistency verified**

## Output Files

After validation:

```
data/staging/
├── mmcif_cache/              # Cached mmCIF files from RCSB
│   ├── 1abc.cif
│   └── ...
├── validation/
│   ├── validation_results.csv      # Detailed results
│   └── validation_summary.html     # Summary dashboard
└── validation.log            # Detailed log
```

## Next Steps After Validation

### If Pass Rate ≥80%

1. **Build canonical dataset:**
   ```bash
   python scripts/build_canonical_dataset.py \
     --raw data/raw \
     --staging data/staging \
     --canonical data/canonical \
     --all-sources \
     --limit 1000  # Start with 1000 complexes
   ```

2. **Review quarantine reports:**
   - Check `staging/quarantine/quarantine_log.json`
   - Review quarantine HTML reports
   - Understand common failure modes

3. **Test visualization pipeline:**
   ```bash
   # Test on first canonical complex
   python -m peptidquantum.pipeline.cli run \
     --cif data/canonical/complexes.parquet[0].structure_file \
     --protein A \
     --peptide B
   ```

4. **Proceed to Sprint 3:**
   - Cluster-aware splits
   - QC dashboard
   - Audit gallery

### If Pass Rate <80%

1. **Analyze failures:**
   - Review `validation_results.csv`
   - Group by quarantine reason
   - Identify systematic issues

2. **Fix critical issues:**
   - Update processors if needed
   - Adjust length policies if needed
   - Improve chain detection if needed

3. **Re-run validation:**
   ```bash
   python scripts/validate_sprint2b.py \
     --raw data/raw \
     --staging data/staging \
     --verbose
   ```

## Technical Notes

### Auth vs Label IDs

**Policy:** Use **auth** (author) IDs by default

**Rationale:**
- Matches published literature
- Biopython `MMCIFParser` default
- RCSB recommendation
- Consistent with PDB file format

**Both stored in canonical:**
- `chain_id_auth` and `chain_id_label`
- `residue_number_auth` and `residue_number_label`
- Mode selected at query time

### mmCIF Standardization

**Why mmCIF?**
- Official wwPDB archival format
- No atom/residue/chain count limits
- Better metadata support
- Consistent with RCSB Data API

**PDB limitations:**
- 62 chain limit (A-Z, a-z, 0-9)
- 9999 residue limit per chain
- 99999 atom limit
- Legacy format

### RCSB File Download Services

**Endpoints:**
- mmCIF: `https://files.rcsb.org/download/{pdb_id}.cif`
- PDB: `https://files.rcsb.org/download/{pdb_id}.pdb`
- Biological assembly: `https://files.rcsb.org/download/{pdb_id}-assembly1.cif`

**Rate limiting:** Be respectful, use caching

## Troubleshooting

### Network Issues

```bash
# Test RCSB connectivity
curl https://files.rcsb.org/download/1abc.cif

# Use cache if available
ls data/staging/mmcif_cache/
```

### Parse Failures

```bash
# Test parser directly
python -m peptidquantum.data.processors.mmcif_parser structure.cif

# Check with Gemmi fallback
python -c "import gemmi; gemmi.read_structure('structure.cif')"
```

### Chain Mapping Issues

```bash
# Verify chain consistency
python -m peptidquantum.data.processors.pdb_to_mmcif \
  --verify complex.pdb
```

## Summary

Sprint 2B validates the entire canonical dataset generation pipeline on real data:

1. **PDB→mmCIF normalization** via RCSB backfill
2. **10-point validation** per complex
3. **100 real complexes** (50 PROPEDIA, 25 PepBDB, 25 BioLiP2)
4. **Pass rate ≥80%** required to proceed

**Critical insight:** PROPEDIA is PDB format, not mmCIF. Normalization is essential for consistency.

**Next:** After validation passes, build full canonical dataset and proceed to Sprint 3 (splits + QC).

---

**Last Updated:** 2024-03-15  
**Version:** Sprint 2B - Validation Phase  
**Status:** Ready for execution
