# PeptiProp Final Repo State

Bu doküman, repo içindeki aktif final yapının kısa referansıdır. Makine-okunur eşdeğeri:

- `data/reports/project_state_manifest.json`

## 1. Amaç

Aktif proje omurgası:

- PROPEDIA-only veri
- leakage-free sequence-cluster split
- split-local candidate generation
- interaction scoring + candidate reranking
- görsel sanity report/viewer/2D surface

## 2. Aktif Model ve Karşılaştırma

Aktif final model:

- `outputs/training/peptiprop_v0_2_gnn_esm2/`
- GATv2 + ESM-2 v0.2

Karşılaştırma baseline:

- `outputs/training/peptidquantum_v0_1_final_mlx_m4/`
- MLX MLP v0.1

## 3. Veri Yüzeyi

Ham veri:

- `data/raw/propedia/`

Kanonik tablolar:

- `data/canonical/complexes.parquet`
- `data/canonical/chains.parquet`
- `data/canonical/residues.parquet`

Split ve çift raporları:

- `data/canonical/splits/`
- `data/canonical/pairs/pair_data_report.json`
- `data/canonical/pairs/candidate_set_report.json`

## 4. İşleme Katmanları

ESM-2 embedding:

- `data/embeddings/esm2_residue/`

PyG residue graph:

- `data/graphs/`

MLX dense feature:

- `data/mlx/features_v0_1_m4/`

## 5. Sonuç Yüzeyi

GNN final artifact'ları:

- metrics / ranking / thresholds / calibration / top-k / plots

MLX artifact'ları:

- metrics / ranking / thresholds / calibration / top-k / train log / plots

Visualization sanity klasörleri:

- `outputs/analysis_propedia_batch_gnn/`
- `outputs/analysis_propedia_top_ranked_batch_gnn/`
- `outputs/analysis_propedia_batch_mlx/`
- `outputs/analysis_propedia_top_ranked_batch_mlx/`

## 6. Güncelleme Komutları

```bash
source .venv-mlx/bin/activate
python scripts/generate_gnn_predictions.py
python scripts/build_project_state_manifest.py
python scripts/build_pages_site.py
python scripts/sync_pages_training_bundle.py
```
