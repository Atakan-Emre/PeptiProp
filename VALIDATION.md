# Validation Guide (Active v0.1)

This validation checklist is aligned with the active PROPEDIA-only scoring pipeline.

## 1) Leakage Guards (Mandatory)

Run:

```bash
python -m unittest tests/test_baseline_leakage_guards.py -v
```

Expected:

- `2/2` tests pass
- negative pair features do not depend on native peptide
- negative protein graph does not depend on native peptide

## 2) Data Integrity Checks

Use reports:

- `data/canonical/pairs/pair_data_report.json`
- `data/canonical/pairs/candidate_set_report.json`

Confirm:

- split overlap leakage = `0`
- split column consistency = `true`
- duplicate pair count = `0`
- quality flag = `clean` only
- candidate size distribution = `6` (1 positive + 5 negatives)

## 3) Training Output Checks

Final run folder:

- `outputs/training/peptidquantum_v0_1_final_best_classical_100ep_r2/`

Required files:

- `metrics.json`
- `ranking_metrics.json`
- `best_thresholds.json`
- `calibration_metrics.json`
- `train_log.csv`
- ROC/PR/confusion/threshold/calibration plots

## 4) Overfitting / Generalization Check

Inspect `train_log.csv` and compare train vs val:

- `train_mrr` vs `val_mrr`
- `train_hit@3` vs `val_hit@3`
- `train_loss` vs `val_loss`

Guideline:

- mild train-val gap is acceptable
- strong gap with degrading val/test metrics indicates overfitting

## 5) 3D/2D Sanity Check

Run:

```bash
python scripts/run_visualization_sanity.py --canonical data/canonical --sample-list data/reports/audit_gallery_propedia/sample_list_top_ranked_100ep_r2.txt --output outputs/analysis_propedia_top_ranked_batch_100ep_r2 --limit 10
```

Expected:

- `10/10` pass in `visualization_sanity_summary.json`
- each sample contains:
  - `report.html`
  - `viewer.html`
  - `data/viewer_state.json`
  - `figures/peptide_2d.png`

## 6) Metadata Visibility in Visual Outputs

Verify:

- `viewer.html` shows complex id, protein chain ids, peptide chain ids
- `report.html` includes protein/peptide chain id fields
- `peptide_2d.png` title includes complex + protein + peptide metadata
