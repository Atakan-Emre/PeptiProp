# Dataset Download Guide (PROPEDIA-only)

This guide describes only the active v0.1 data flow. Güncel komut özeti: [README.md](README.md). Yayınlanan özet JSON: `site/data/manifest.json` (site script’i ile üretilir).

## Active Dataset

- **PROPEDIA** is the only active source for training/evaluation data generation.
- External datasets (`GEPPRI`, `PepBDB`, `BioLiP2`) are frozen and not part of active training.

## 1) Prepare Raw Folder

```bash
mkdir -p data/raw/propedia
```

Place downloaded PROPEDIA archives/files under:

- `data/raw/propedia/`

## 2) Build or Refresh Canonical Splits

```bash
python scripts/build_pdb_level_splits.py --canonical data/canonical --propedia-meta data/raw/propedia --out data/canonical/splits --seed 42
```

This step enforces PDB-level split consistency.

## 3) Generate Candidate/Negative Pairs

```bash
python scripts/generate_negative_pairs.py --canonical data/canonical --splits data/canonical/splits --output data/canonical/pairs --seed 42
```

Reports written to:

- `data/canonical/pairs/pair_data_report.json`
- `data/canonical/pairs/candidate_set_report.json`

## 4) Required Data Checks

Before training, verify:

- split leakage = `0`
- duplicate pairs = `0`
- quality flag includes only `clean`
- candidate set size is stable (`6`: 1 positive + 5 negatives)

## 5) Train Final Classical Model

```bash
python scripts/train_scoring_model.py --config configs/train_v0_1_final_best_classical_100ep.yaml
```

Final outputs:

- `outputs/training/peptidquantum_v0_1_final_best_classical_100ep_r2/`

## Notes

- Do not mix frozen external datasets into active train/val/test.
- Do not bypass split-local candidate generation.
- Keep leakage guard tests green before each final training run.
