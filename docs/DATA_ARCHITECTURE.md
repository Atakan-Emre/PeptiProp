# PeptiProp - Veri Mimarisi

*(Kod paketi: `peptidquantum`. Aktif final model: GATv2 + ESM-2 v0.2. MLX v0.1 baseline olarak korunur.)*

## Scope

Aktif veri yüzeyi yalnızca **PROPEDIA**'dır.

- Ham veri: `data/raw/propedia/`
- Kanonik doğruluk kaynağı: `data/canonical/`
- Split stratejisi: **sequence-cluster leakage-free split**
- Eğitim politikası: **clean-only**, split-local candidate generation

Harici veri setleri (`GEPPRI`, `PepBDB`, `BioLiP2`) repoda dursa da aktif train/val/test akışına girmez.

Güncel repo durum özeti:

- Makine tarafından üretilen manifest: `data/reports/project_state_manifest.json`
- Site özeti: `site/data/manifest.json`

## Directory Layout

```text
data/
  raw/
    propedia/
      complexes/
      interfaces/
      peptides/
      receptors/
      sequence_meta/
  canonical/
    complexes.parquet
    chains.parquet
    residues.parquet
    provenance.parquet
    splits/
      train_ids.txt
      val_ids.txt
      test_ids.txt
      split_summary.txt
    pairs/
      train_pairs.parquet
      val_pairs.parquet
      test_pairs.parquet
      pair_data_report.json
      candidate_set_report.json
  embeddings/
    esm2_residue/
    esm2_chain_lookup.json
  graphs/
    *.pt
  mlx/
    features_v0_1_m4/
  reports/
    project_state_manifest.json
    audit_gallery_propedia/
      sample_list_final_best_mlx_model.txt
      sample_list_top_ranked_gnn_v0_2.txt
```

## Canonical Policies

- chain id mode: `auth`
- residue numbering mode: `auth`
- peptide length: `5-50 aa`
- protein minimum length: `>= 30 aa`
- duplicate pair count: `0` zorunlu
- pair quality: aktif hatta yalnız `clean`
- split-local candidate generation: zorunlu

## Split Policy

Split script: `scripts/build_pdb_level_splits.py`

Gerçekte kullanılan yöntem:

1. Protein zincir sekansları MMseqs2 ile `%30` kimlik eşiğinde kümelenir.
2. Aynı kümedeki tüm kompleksler tek split'e atanır.
3. MMseqs2 yoksa exact-sequence fallback kullanılır.

Bu nedenle dosya adı `build_pdb_level_splits.py` olsa da aktif bilimsel protokol **PDB-level değil, sequence-cluster-aware leakage control**'dür.

## Candidate Set Policy

Her protein için:

- `1` pozitif
- `5` negatif
- toplam candidate size = `6`

Negatif karışımı:

- train: `~70% easy / ~30% hard`
- val/test: `~80% easy / ~20% hard`

Hard negatifler SCOP/CATH etiketinden değil, **aynı split içinde benzer protein sequence/bucket havuzundan** üretilir. Ratio sapmaları `data/canonical/pairs/candidate_set_report.json` içine yazılır.

## Feature / Graph Surfaces

MLX dense features:

- klasör: `data/mlx/features_v0_1_m4/`
- script: `scripts/export_mlx_features.py`
- kullanım: MLX MLP baseline

ESM-2 residue embeddings:

- klasör: `data/embeddings/esm2_residue/`
- lookup: `data/embeddings/esm2_chain_lookup.json`
- script: `scripts/extract_esm2_embeddings.py`

PyG residue graphs:

- klasör: `data/graphs/`
- dosya: `{complex_id}__{chain_id}.pt`
- node features: `320-d ESM-2 + 6-d structural = 326-d`
- edge features: `distance + direction = 4-d`
- script: `scripts/build_residue_graphs.py`

## Training Output Surfaces

Aktif final GNN run:

- `outputs/training/peptiprop_v0_2_gnn_esm2/`

Beklenen ana artifact'lar:

- `metrics.json`
- `ranking_metrics.json`
- `best_thresholds.json`
- `pair_data_report.json`
- `candidate_set_report.json`
- `calibration_metrics.json`
- `test_summary.txt`
- `test_topk_candidates.csv`
- `test_topk_positive_hits.csv`
- `top_ranked_examples.json`
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `score_histogram_pos_neg.png`
- `validation_score_histogram_pos_neg.png`
- `validation_threshold_sweep.png`
- `calibration_curve.png`

MLX baseline run:

- `outputs/training/peptidquantum_v0_1_final_mlx_m4/`

Visualization sanity outputs:

- MLX: `outputs/analysis_propedia_batch_mlx/`, `outputs/analysis_propedia_top_ranked_batch_mlx/`
- GNN: `outputs/analysis_propedia_batch_gnn/`, `outputs/analysis_propedia_top_ranked_batch_gnn/`

## Notes

- Site build `scripts/build_pages_site.py` aktif olarak GNN final klasörünü, fallback olarak MLX klasörünü kullanır.
- GitHub Pages için commitlenecek hafif eğitim bundle'ı `scripts/sync_pages_training_bundle.py` doldurur.
