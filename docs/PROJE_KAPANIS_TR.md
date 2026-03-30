# PeptiProp - proje kapanış özeti

**Tarih:** Mart 2026  
**Aktif final hat:** PROPEDIA + GATv2 + ESM-2  
**Karşılaştırma hattı:** MLX MLP baseline / MLX ablation

## Tamamlanan hat

- PROPEDIA-only canonical veri yüzeyi
- sequence-cluster leakage-free split
- split-local candidate / negative pair generation
- GNN+ESM-2 interaction scoring + reranking
- MLX baseline scoring
- ROC / PR / threshold / calibration / top-k rapor yüzeyi
- HTML report + viewer + peptide 2D sanity pipeline
- GitHub Pages site üretimi

## Doğrulama

- Leakage guard: `python tests/test_mlx_leakage_guards.py`
- GNN active pipeline: `python -m unittest tests.test_propedia_active_pipeline -v`
- MLX active pipeline: `python -m unittest tests.test_propedia_active_pipeline_mlx -v`
- Golden E2E: `pytest tests/golden_set_e2e_pytest.py -v`

## Repo durum dosyaları

- İnsan-okunur giriş: `README.md`
- Veri mimarisi: `docs/DATA_ARCHITECTURE.md`
- Bilimsel metod: `docs/SCIENTIFIC_METHOD_TR.md`
- Hızlı başlangıç: `docs/QUICKSTART.md`
- Makine-manifest: `data/reports/project_state_manifest.json`

## Bilinen sınırlar

- Final aktif görsel hat geometric residue-contact fallback kullanır; external tool extractor çıktıları raporlanan sonuçların parçası değildir
- GNN mevcut publish run'ı postprocess ile tamamlanmıştır; yeni rerun yapılırsa `train_log.csv` da aynı klasöre yazılır
- Cross-attention veya daha büyük ESM varyantları bu sprint kapsamına alınmadı

Bu sürüm, tekrar üretilebilir komutlar, testler ve final artifact yüzeyi ile kapatılmış kabul edilir.
