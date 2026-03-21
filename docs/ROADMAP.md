# PeptidQuantum Roadmap

## Current Status (Final v0.1 Surface)

Repository is stabilized on a PROPEDIA-only, leakage-free scoring stack (classical + optional MLX on Apple Silicon).

- dataset: PROPEDIA
- split: sekans-küme bazlı (MMseqs2 %30 kimlik)
- task: interaction scoring + candidate reranking
- MLX eğitim: `outputs/training/peptidquantum_v0_1_final_mlx_m4`
- web: GitHub Pages via `scripts/build_pages_site.py` + `.github/workflows/pages.yml`
- 3D viewer: PDB/mmCIF-aware `addModel`, safe JSON embedding, fixed embedded-report `div_id` handling (`viewer_3dmol.py`)

## Phase A - Data Integrity (Completed)

- canonical dataset locked under `data/canonical/`
- split metadata synchronized and fail-fast checks active
- clean-only pair generation active
- duplicate pair count is `0`
- pair/candidate reports produced
- leakage guard tests passing (`2/2`)

## Phase B - Model Selection (Completed)

Model family ablation completed (MPNN / GATv2 / GIN).

Selected final run:

- `outputs/training/peptidquantum_v0_1_final_best_classical_100ep_r2/`

Final test summary:

- AUROC `0.8600`
- AUPRC `0.5115`
- F1 `0.5760`
- MCC `0.4879`
- MRR `0.7419`
- Hit@1 `0.5726`
- Hit@3 `0.9187`
- Hit@5 `0.9877`
- Brier `0.1606`

Interpretation:

- ranking behavior is strong and stable
- calibration is materially improved over prior collapsed-score runs
- no severe generalization collapse observed

## Phase C - Visualization/QC (Completed)

Run-specific 3D/2D sanity outputs:

- `outputs/analysis_propedia_batch_100ep_r2/visualization_sanity_summary.json` -> `10/10`
- `outputs/analysis_propedia_top_ranked_batch_100ep_r2/visualization_sanity_summary.json` -> `10/10`

Metadata visibility upgrades completed:

- viewer shows complex/protein/peptide chain info
- report shows protein/peptide chain IDs
- peptide 2D title includes complex + protein + peptide info

## v0.2 Delivery Schedule (Locked)

Start date: **March 23, 2026**  
Target finish: **May 8, 2026**

| Window | Sprint Goal | Main Tasks | Exit Criteria |
|---|---|---|---|
| Mar 23 - Apr 3, 2026 | S1 - Candidate Quality | Hard negative coverage recovery, ratio shortfall reduction, ratio warnings + QC hardening | `negative_type_ratio_shortfall` for hard <= 0.08 (train/val/test) |
| Apr 6 - Apr 17, 2026 | S2 - Calibration | Reliability improvement, threshold stability pass, calibration-aware checkpoint selection | test Brier <= 0.14 and no threshold collapse |
| Apr 20 - May 1, 2026 | S3 - Ranking Lift | Loss/feature tuning inside current classical family, no leakage regressions | test MRR +0.03 over v0.1 and Hit@1 +0.03 |
| May 4 - May 8, 2026 | S4 - Release Hardening | Reproducibility run, artifact freeze, doc freeze, final sanity pack | same-seed rerun deviation <= 2% on key metrics and 3D sanity 10/10 |

## Out of Scope (Frozen)

- new datasets in active train flow
- legacy mixed-data train paths
- large architecture refactors
- quantum or GAN integration in active line

