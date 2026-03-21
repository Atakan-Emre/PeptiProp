# PeptidQuantum Data Architecture (Active v0.1)

## Scope

Active data surface is **PROPEDIA-only**.

- Primary source: `data/raw/propedia/`
- Canonical source of truth: `data/canonical/`
- Split strategy: **PDB-level structure-aware split**
- Training data policy: **clean-only**, split-local candidate generation

External datasets are not part of active training flow.

Özet manifest (GitHub Pages / yerel site): `scripts/build_pages_site.py` → `site/data/manifest.json`.

## Directory Layout

```text
data/
  raw/
    propedia/                      # Active raw source
  external_frozen/                 # Frozen, non-active sources
    GEPPRI/
    pepbdb/
    biolip2/
  staging/                         # Intermediate processing artifacts
  canonical/
    complexes.parquet
    chains.parquet
    residues.parquet
    provenance.parquet
    splits/
      train_ids.txt
      val_ids.txt
      test_ids.txt
    pairs/
      train_pairs.parquet
      val_pairs.parquet
      test_pairs.parquet
      pair_data_report.json
      candidate_set_report.json
  reports/
    qc*/
    audit_gallery_propedia/
```

## Canonical Policies

- chain id mode: `auth`
- residue numbering mode: `auth`
- peptide length: `5-50 aa`
- protein minimum length: `>= 30 aa`
- max pairs per structure: `50`
- duplicate pairs: `0` (required)
- split overlap leakage: `0` (required)

## Candidate Set Policy

Per protein:

- `1` positive peptide
- `5` negative peptides
- candidate size = `6`

Target negative mix:

- train: `50% easy / 30% hard / 20% structure_hard`
- val/test: `70% easy / 30% hard`

If target ratio cannot be reached, shortfall is reported in:

- `data/canonical/pairs/candidate_set_report.json`

## Active Training Output Surface

Primary final run folder:

- `outputs/training/peptidquantum_v0_1_final_best_classical_100ep_r2/`

Required outputs include:

- `metrics.json`
- `ranking_metrics.json`
- `calibration_metrics.json`
- `pair_data_report.json`
- `candidate_set_report.json`
- ROC/PR/confusion/threshold/calibration plots
- top-k ranking tables

## 3D/2D Output Surface

Per-complex outputs:

- `report.html`
- `viewer.html`
- `data/viewer_state.json`
- `figures/peptide_2d.png`

Sanity summaries:

- `outputs/analysis_propedia_batch_100ep_r2/visualization_sanity_summary.json`
- `outputs/analysis_propedia_top_ranked_batch_100ep_r2/visualization_sanity_summary.json`
