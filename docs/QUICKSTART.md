# PeptidQuantum Quick Start

Active workflow is PROPEDIA-only and scoring-focused.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Prepare Data

```bash
python scripts/build_pdb_level_splits.py --canonical data/canonical --propedia-meta data/raw/propedia --out data/canonical/splits --seed 42
python scripts/generate_negative_pairs.py --canonical data/canonical --splits data/canonical/splits --output data/canonical/pairs --seed 42
```

Check reports:

- `data/canonical/pairs/pair_data_report.json`
- `data/canonical/pairs/candidate_set_report.json`

## 3. Train Scoring Runs

**Klasik:**

```bash
python scripts/train_scoring_model.py --config configs/train_v0_1_final_best_classical_100ep.yaml
```

Output: `outputs/training/peptidquantum_v0_1_final_best_classical_100ep_r2/`

**MLX (M4, ayrı venv):** bkz. `mlx/README.md` — eğitim çıktısı `outputs/training/peptidquantum_v0_1_final_mlx_m4/`.

## 4. Analyze Last Training Graphs

Active final run:

- `outputs/training/peptidquantum_v0_1_final_best_classical_100ep_r2/`

Open these for analysis:

- `metrics.json`
- `ranking_metrics.json`
- `train_log.csv`
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `validation_threshold_sweep.png`
- `validation_score_histogram_pos_neg.png`
- `score_histogram_pos_neg.png`
- `calibration_curve.png`
- `calibration_metrics.json`
- `test_topk_candidates.csv`
- `test_topk_positive_hits.csv`
- `top_ranked_examples.json`

## 5. 3D/2D Sanity

Arpeggio / PLIP / Open Babel: `EXTERNAL_TOOLS.md` ve macOS için `scripts/install_external_tools_macos.sh`.

```bash
python scripts/run_visualization_sanity.py --canonical data/canonical --sample-list data/reports/audit_gallery_propedia/sample_list_top_ranked_100ep_r2.txt --output outputs/analysis_propedia_batch_100ep_r2 --limit 10
python scripts/run_visualization_sanity.py --canonical data/canonical --sample-list data/reports/audit_gallery_propedia/sample_list_top_ranked_100ep_r2.txt --output outputs/analysis_propedia_top_ranked_batch_100ep_r2 --limit 10
```

Beklenen çıktılar:

- `report.html`, `viewer.html`
- `data/viewer_state.json` (`structure_format`, `structure_basename`, `chains`, `interactions`, …)
- `data/interaction_provenance.json`
- `figures/peptide_2d.png`

## 6. GitHub Pages (statik site)

```bash
python scripts/build_pages_site.py
```

Ayrıntı: `docs/GITHUB_PAGES_TR.md`. CI: `.github/workflows/pages.yml`.

## 7. Leakage Guard Tests

```bash
python -m unittest tests/test_baseline_leakage_guards.py -v
```
