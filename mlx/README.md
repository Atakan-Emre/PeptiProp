# MLX (M4 MacBook) Pipeline

Bu klasör, mevcut CUDA/Torch graph hattına dokunmadan Apple Silicon (M4) için ayrı bir MLX çalışma yoludur.

## Amaç

- PROPEDIA-only pair verisinden leakage-free dense feature üretmek
- MLX ile 0-1 interaction scoring modeli eğitmek
- Ranking + calibration artefact üretmek

## Kurulum (macOS, Apple Silicon)

```bash
cd /path/to/PeptidQuantum
python3 -m venv .venv-mlx
source .venv-mlx/bin/activate
pip install -U pip
pip install -r mlx/requirements-m4.txt
```

## Çalıştırma

```bash
python scripts/export_mlx_features.py --config configs/train_v0_1_scoring_mlx_m4.yaml
python scripts/train_scoring_mlx.py --config configs/train_v0_1_scoring_mlx_m4.yaml
```

## Tam ablation + final senkron

```bash
python scripts/run_final_ablation_mlx.py \
  --smoke-epochs 8 --smoke-patience 4 \
  --full-epochs 200 --full-patience 20 \
  --finalists-per-family 2
```

Seçilen en iyi çalışma: `outputs/training/peptidquantum_v0_1_final_best_mlx_ablation/` (`ablation_heatmap.png`, `metrics.json`, …). Statik site bu klasörden görselleri kopyalayabilir: `python scripts/build_pages_site.py` — `docs/GITHUB_PAGES_TR.md`.

## Çıktılar (tek MLX koşusu örneği)

- `outputs/training/peptidquantum_v0_1_final_mlx_m4/metrics.json`
- `outputs/training/peptidquantum_v0_1_final_mlx_m4/ranking_metrics.json`
- `outputs/training/peptidquantum_v0_1_final_mlx_m4/calibration_metrics.json`
- `outputs/training/peptidquantum_v0_1_final_mlx_m4/roc_curve.png`
- `outputs/training/peptidquantum_v0_1_final_mlx_m4/pr_curve.png`
- `outputs/training/peptidquantum_v0_1_final_mlx_m4/confusion_matrix.png`
- `outputs/training/peptidquantum_v0_1_final_mlx_m4/calibration_curve.png`

## Not

Bu yol, klasik graph encoder’ın birebir MLX re-implementation’ı değildir; Mac tarafında hızlı deney ve taşınabilir scoring backend sağlar.

