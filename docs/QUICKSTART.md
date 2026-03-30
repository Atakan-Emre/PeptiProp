# PeptiProp Quick Start

Aktif workflow: **PROPEDIA-only -> canonical -> sequence-cluster split -> candidate pairs -> GNN final scoring + MLX baseline -> visualization sanity -> Pages**.

Repo bu makinede `python`'ı global vermiyor; komutları venv ile çalıştırın:

```bash
source .venv-mlx/bin/activate
```

## 1. Prepare Data

```bash
source .venv-mlx/bin/activate

python scripts/build_pdb_level_splits.py \
  --canonical data/canonical \
  --propedia-meta data/raw/propedia \
  --out data/canonical/splits \
  --seed 42

python scripts/generate_negative_pairs.py \
  --canonical data/canonical \
  --splits data/canonical/splits \
  --output data/canonical/pairs \
  --seed 42
```

Kontrol dosyaları:

- `data/canonical/splits/split_summary.txt`
- `data/canonical/pairs/pair_data_report.json`
- `data/canonical/pairs/candidate_set_report.json`

## 2. Train the Active GNN Final Model

```bash
source .venv-mlx/bin/activate

python scripts/train_gnn_esm2.py \
  --config configs/train_v0_2_gnn_esm2.yaml

python scripts/generate_gnn_predictions.py
```

Final GNN output:

- `outputs/training/peptiprop_v0_2_gnn_esm2/`

Ana dosyalar:

- `metrics.json`
- `ranking_metrics.json`
- `best_thresholds.json`
- `calibration_metrics.json`
- `test_summary.txt`
- `test_topk_candidates.csv`
- `test_topk_positive_hits.csv`
- `top_ranked_examples.json`

## 3. Run the MLX Baseline

```bash
source .venv-mlx/bin/activate

python scripts/export_mlx_features.py \
  --config configs/train_v0_1_scoring_mlx_m4.yaml

python scripts/train_scoring_mlx.py \
  --config configs/train_v0_1_scoring_mlx_m4.yaml
```

MLX output:

- `outputs/training/peptidquantum_v0_1_final_mlx_m4/`

MLX ablation gerekiyorsa:

```bash
source .venv-mlx/bin/activate

python scripts/run_final_ablation_mlx.py \
  --smoke-epochs 8 \
  --smoke-patience 4 \
  --full-epochs 200 \
  --full-patience 20 \
  --finalists-per-family 1
```

## 4. Run Visualization Sanity

MLX sample list hazır:

- `data/reports/audit_gallery_propedia/sample_list_final_best_mlx_model.txt`

GNN sample list `scripts/generate_gnn_predictions.py` tarafından üretilir:

- `data/reports/audit_gallery_propedia/sample_list_top_ranked_gnn_v0_2.txt`

Komutlar:

```bash
source .venv-mlx/bin/activate

python scripts/run_visualization_sanity.py \
  --canonical data/canonical \
  --sample-list data/reports/audit_gallery_propedia/sample_list_top_ranked_gnn_v0_2.txt \
  --output outputs/analysis_propedia_top_ranked_batch_gnn \
  --limit 10

python scripts/run_visualization_sanity.py \
  --canonical data/canonical \
  --sample-list data/reports/audit_gallery_propedia/sample_list_final_best_mlx_model.txt \
  --output outputs/analysis_propedia_top_ranked_batch_mlx \
  --limit 10
```

Beklenen sample-level artifact'lar:

- `report.html`
- `viewer.html`
- `data/viewer_state.json`
- `data/interaction_provenance.json`
- `figures/peptide_2d.png`

## 5. Validate

```bash
source .venv-mlx/bin/activate

python tests/test_mlx_leakage_guards.py
python -m unittest tests.test_propedia_active_pipeline -v
python -m unittest tests.test_propedia_active_pipeline_mlx -v
```

## 6. Refresh Repo State Manifest

```bash
source .venv-mlx/bin/activate

python scripts/build_project_state_manifest.py
```

Yazılan özet:

- `data/reports/project_state_manifest.json`

## 7. Rebuild GitHub Pages

```bash
source .venv-mlx/bin/activate

python scripts/build_pages_site.py
python scripts/sync_pages_training_bundle.py
```

Ayrıntı:

- `docs/GITHUB_PAGES_TR.md`
- `site/data/manifest.json`
