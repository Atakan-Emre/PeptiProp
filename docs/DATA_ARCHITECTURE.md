# PeptiProp — Veri Mimarisi

*(Kod paketi: `peptidquantum`. Aktif model: GNN+ESM-2 v0.2.)*

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

- train: `~70% easy / ~30% hard`
- val/test: `~80% easy / ~20% hard`

If target ratio cannot be reached, shortfall is reported in:

- `data/canonical/pairs/candidate_set_report.json`

## ESM-2 Embedding Surface

ESM-2 per-rezidü embedding'ler `data/embeddings/` altında saklanır:

- `data/embeddings/esm2_residue/*.npz` — her benzersiz sekans için 1 dosya (320-d float16)
- `data/embeddings/esm2_chain_lookup.json` — (complex_id::chain_id) → NPZ dosya eşlemesi

Script: `scripts/extract_esm2_embeddings.py`

## Graph Surface

Rezidü-seviye PyG grafları `data/graphs/` altında saklanır:

- `data/graphs/{complex_id}__{chain_id}.pt` — her zincir için 1 PyG Data objesi
- Node features: ESM-2 (320-d) + yapısal (6-d) = 326-d
- Edge features: mesafe (1-d) + yön vektörü (3-d) = 4-d

Script: `scripts/build_residue_graphs.py`

## Active Training Output Surface

GNN+ESM-2 (v0.2): `outputs/training/peptiprop_v0_2_gnn_esm2/`

- `best_model.pt` — en iyi model ağırlıkları
- `metrics.json` — test metrikleri
- `ranking_metrics.json` — sıralama metrikleri
- `top_ranked_examples.json` — en iyi tahmin örnekleri
- `figures/` — ROC, PR, confusion matrix, histogram PNG'leri

MLP baseline (v0.1): `publish/github_pages_training_bundle/` (senkronize snapshot)
