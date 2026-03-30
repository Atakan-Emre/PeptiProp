# Validation Guide

Bu checklist aktif PROPEDIA-only scoring pipeline ile hizalıdır.

Ön koşul:

```bash
source .venv-mlx/bin/activate
```

## 1) Leakage Guards

```bash
python tests/test_mlx_leakage_guards.py
```

Beklenen:

- testler `PASS`
- MLX feature export native peptide leakage barındırmaz
- split-local pair surface korunur

## 2) Canonical Data Integrity

Kontrol dosyaları:

- `data/canonical/splits/split_summary.txt`
- `data/canonical/pairs/pair_data_report.json`
- `data/canonical/pairs/candidate_set_report.json`

Doğrulanacaklar:

- sequence-cluster leakage = `0`
- split column consistency = `true`
- duplicate pair count = `0`
- quality flag = `clean`
- candidate size = `6`
- hard-negative shortfall policy pass

## 3) Active Pipeline Tests

```bash
python -m unittest tests.test_propedia_active_pipeline -v
python -m unittest tests.test_propedia_active_pipeline_mlx -v
```

GNN testi aktif final run klasörünü doğrular:

- `outputs/training/peptiprop_v0_2_gnn_esm2/`

MLX testi baseline veya ablation run'ı doğrular:

- `outputs/training/peptidquantum_v0_1_final_mlx_m4/`
- veya `outputs/training/peptidquantum_v0_1_final_best_mlx_ablation/`

## 4) GNN Output Checks

Zorunlu GNN artifact'lar:

- `metrics.json`
- `ranking_metrics.json`
- `best_thresholds.json`
- `pair_data_report.json`
- `candidate_set_report.json`
- `calibration_metrics.json`
- `threshold_vs_f1_table.csv`
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

Not:

- `train_log.csv` yalnız yeni eğitim rerun'larında doğrudan training script tarafından üretilir.

## 5) MLX Output Checks

MLX tarafında aşağıdakiler beklenir:

- `metrics.json`
- `ranking_metrics.json`
- `best_thresholds.json`
- `pair_data_report.json`
- `candidate_set_report.json`
- `calibration_metrics.json`
- `train_log.csv`
- `test_summary.txt`
- `threshold_vs_f1_table.csv`
- ROC/PR/confusion/histogram/calibration plot'ları

## 6) Visualization Sanity

GNN:

```bash
python scripts/run_visualization_sanity.py \
  --canonical data/canonical \
  --sample-list data/reports/audit_gallery_propedia/sample_list_top_ranked_gnn_v0_2.txt \
  --output outputs/analysis_propedia_top_ranked_batch_gnn \
  --limit 10
```

MLX:

```bash
python scripts/run_visualization_sanity.py \
  --canonical data/canonical \
  --sample-list data/reports/audit_gallery_propedia/sample_list_final_best_mlx_model.txt \
  --output outputs/analysis_propedia_top_ranked_batch_mlx \
  --limit 10
```

Beklenen:

- `visualization_sanity_summary.json` içinde `10/10 success`
- her sample altında:
  - `report.html`
  - `viewer.html`
  - `data/viewer_state.json`
  - `figures/peptide_2d.png`

## 7) Repo State Manifest

Repo yüzeyini tek JSON dosyada yenilemek için:

```bash
python scripts/build_project_state_manifest.py
```

Çıktı:

- `data/reports/project_state_manifest.json`
