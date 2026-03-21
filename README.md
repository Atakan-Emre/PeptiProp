# PeptidQuantum

PROPEDIA odaklı, **sızıntısız (leakage-free)** protein–peptid etkileşim **skorlama**, aday **yeniden sıralama** ve **2D/3D görselleştirme** platformu.

## Canlı demo (GitHub Pages)

Depoda **GitHub Actions → Pages** açıldıktan sonra kök URL’de statik site yayınlanır.

- Ana sayfa: proje özeti, veri, ablation görselleri (yerel çıktılardan kopyalanır), metrik özeti, `manifest.json`
- **3D demo:** `embed/viewer-demo.html` — tarayıcıda **3Dmol.js** + yerel `1crn.cif` (build sırasında indirilir)

Yerelde site üretmek:

```bash
python scripts/build_pages_site.py
# Çıktı: site/ — index.html, embed/viewer-demo.html, data/manifest.json
```

Ayrıntı: [docs/GITHUB_PAGES_TR.md](docs/GITHUB_PAGES_TR.md)

## Aktif kapsam

| Öğe | Açıklama |
|-----|----------|
| Veri | PROPEDIA → kanonik tablolar, PDB düzeyinde split |
| Görev | Skorlama + reranking (1 pozitif + K negatif / aday seti) |
| Modeller | **MLX** (Apple Silicon) ve **klasik** PyTorch yolları |
| Görsel | RDKit **2D** peptide, **3Dmol** viewer, HTML rapor; `viewer_state.json` şeması güncel |

## Pipeline (özet)

```text
PROPEDIA raw → canonical → PDB-level split → split-local negatifler
→ özellik ihracı → eğitim (MLX veya klasik)
→ kalibrasyon + sıralama metrikleri → 2D/3D rapor / sanity
```

## Son çıktı klasörleri (referans)

| Çalıştırma | Klasör |
|------------|--------|
| MLX final (ablation sonrası senkron) | `outputs/training/peptidquantum_v0_1_final_best_mlx_ablation/` |
| Klasik final (örnek) | `outputs/training/peptidquantum_v0_1_final_best_classical_100ep_r2/` |

Önemli dosyalar: `metrics.json`, `ranking_metrics.json`, `ablation_heatmap.png`, `train_log.csv`, `pair_data_report.json`, `candidate_set_report.json`, `calibration_metrics.json`, eğri PNG’leri, `top_ranked_examples.json`.

## 2D / 3D çıktıları ve JSON

Her kompleks çalıştırmasında tipik yapı:

- `report.html` — özet + gömülü/viewer bölümü, etkileşim kaynağı özeti
- `viewer.html` — bağımsız 3Dmol görüntüleyici
- `data/viewer_state.json` — `complex_id`, `structure_format` (`pdb` \| `cif`), `structure_basename`, `chains`, `interactions`, `view_config`
- `data/interaction_provenance.json` — PLIP/Arpeggio oranları
- `figures/peptide_2d.png` — peptide 2D

**Düzeltmeler (v0.1+):** 3D viewer artık **PDB dosyalarını `pdb`, mmCIF’i `cif`** ile yükler; yapı metni **JSON kaçışlı** gömülür (backtick kırılması giderildi); boş atom seçimlerinde silindir çizimi atlanır; gömülü rapor için **yanlış `viewer` string replace** kaldırıldı.

## Komutlar

**Negatif çiftler (train / val-test oranları ayrı; hard bucket genişletilmiş):**

```bash
python scripts/generate_negative_pairs.py \
  --canonical data/canonical --splits data/canonical/splits \
  --output data/canonical/pairs
```

**MLX ablation (örnek):**

```bash
source .venv-mlx/bin/activate
export PYTHONPATH=src
python scripts/run_final_ablation_mlx.py \
  --smoke-epochs 8 --smoke-patience 4 \
  --full-epochs 200 --full-patience 20 \
  --finalists-per-family 2
```

**Görsel sanity (PLIP/Arpeggio için kurulum: `scripts/install_external_tools_macos.sh`):**

```bash
python scripts/run_visualization_sanity.py \
  --canonical data/canonical \
  --sample-list data/reports/audit_gallery_propedia/sample_list_final_best_mlx_model.txt \
  --output outputs/analysis_propedia_batch_mlx \
  --limit 10
```

## Dokümantasyon

| Dosya | İçerik |
|-------|--------|
| [QUICKSTART.md](QUICKSTART.md) | Hızlı başlangıç |
| [EXTERNAL_TOOLS.md](EXTERNAL_TOOLS.md) | Arpeggio, PLIP, Open Babel, PyMOL |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Ortam kurulumu |
| [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md) | Veri şeması |
| [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) | Mimari |
| [VALIDATION.md](VALIDATION.md) | Doğrulama |
| [mlx/README.md](mlx/README.md) | MLX / M4 |
| [docs/GITHUB_PAGES_TR.md](docs/GITHUB_PAGES_TR.md) | Pages yayını |
| [docs/PROJECT_OBJECTIVES_TR.md](docs/PROJECT_OBJECTIVES_TR.md) | Bilimsel hedefler (TR) |

## Lisans

Proje lisansı depo kökündeki LICENSE dosyasına tabidir (varsa).
