# PeptidQuantum

PROPEDIA-only, leakage-free protein-peptide interaction scoring and visualization platform.

## Active Scope

- Primary dataset: PROPEDIA only
- Split: PDB-level structure-aware split
- Task: interaction scoring + candidate reranking
- Model family: classical graph models (active final run is GATv2-based scoring)

## Active Pipeline

```text
PROPEDIA raw
-> canonical dataset
-> PDB-level structure-aware split
-> split-local clean-only candidate/negative generation
-> leakage-free scoring model training
-> threshold + calibration + ranking evaluation
-> 2D/3D viewer/report sanity checks
```

## Final v0.1 Run (Current)

Active final training folder:

- `outputs/training/peptidquantum_v0_1_final_best_classical_100ep_r2/`

Final snapshot (test):

- AUROC: `0.8600`
- AUPRC: `0.5115`
- F1: `0.5760`
- MCC: `0.4879`
- MRR: `0.7419`
- Hit@1: `0.5726`
- Hit@3: `0.9187`
- Hit@5: `0.9877`
- Brier: `0.1606`

Validation snapshot:

- AUROC: `0.8566`
- AUPRC: `0.5087`
- F1: `0.5672`
- MCC: `0.4773`
- MRR: `0.7270`

## Data Snapshot

- canonical complexes: `41,572` (`27,387 clean`)
- pair sets: train `114,366`, val `25,542`, test `24,414`
- candidate set: `1 positive + 5 negatives` per protein
- duplicate pairs: `0`
- split overlap leakage: `0`
- leakage guard tests: pass (`2/2`)

## Where To Analyze Outputs

Primary folder:

- `outputs/training/peptidquantum_v0_1_final_best_classical_100ep_r2/`

Key files:

- `metrics.json`
- `ranking_metrics.json`
- `train_log.csv`
- `pair_data_report.json`
- `candidate_set_report.json`
- `calibration_metrics.json`
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `calibration_curve.png`
- `validation_threshold_sweep.png`
- `validation_score_histogram_pos_neg.png`
- `score_histogram_pos_neg.png`
- `test_topk_candidates.csv`
- `test_topk_positive_hits.csv`
- `top_ranked_examples.json`

## 2D/3D Outputs

Run-specific sanity outputs:

- `outputs/analysis_propedia_batch_100ep_r2/visualization_sanity_summary.json`
- `outputs/analysis_propedia_top_ranked_batch_100ep_r2/visualization_sanity_summary.json`

Per-complex outputs include:

- `report.html` (includes complex/protein/peptide chain metadata)
- `viewer.html` (includes complex/protein/peptide chain metadata)
- `data/viewer_state.json`
- `figures/peptide_2d.png` (title now includes complex + protein + peptide info)

## Command Order (Final)

```bash
python scripts/build_pdb_level_splits.py --canonical data/canonical --propedia-meta data/raw/propedia --out data/canonical/splits --seed 42
python scripts/generate_negative_pairs.py --canonical data/canonical --splits data/canonical/splits --output data/canonical/pairs --seed 42
python -m unittest tests/test_baseline_leakage_guards.py -v
python scripts/train_scoring_model.py --config configs/train_v0_1_final_best_classical_100ep.yaml
python scripts/run_visualization_sanity.py --canonical data/canonical --sample-list data/reports/audit_gallery_propedia/sample_list_top_ranked_100ep_r2.txt --output outputs/analysis_propedia_top_ranked_batch_100ep_r2 --limit 10
```

## MLX (M4 MacBook) Optional Backend

Apple Silicon için ayrı MLX çalışma yapısı eklendi:

- setup ve kullanım: `mlx/README.md`
- feature export: `scripts/export_mlx_features.py`
- MLX training: `scripts/train_scoring_mlx.py`
- M4 config: `configs/train_v0_1_scoring_mlx_m4.yaml`
