# PeptidQuantum v0.1 — proje kapanış özeti

**Tarih:** Mart 2026 (aktif hat PROPEDIA + klasik/MLX eğitim seçenekleri).

## Tamamlanan hat

- Kanonik veri ve çift üretimi; sızıntı korumaları (`test_baseline_leakage_guards`).
- Skorlama pipeline’ı, görselleştirme ve GitHub Pages site üretimi (`scripts/build_pages_site.py`).
- Arpeggio / Open Babel 3 uyumluluğu (`third_party/arpeggio/arpeggio.py`); wrapper çıktı dizini düzeltmesi.

## Doğrulama

- `VALIDATION.md` içindeki komutlar ve tam keşif: `python -m unittest discover -s tests -p 'test_*.py'`.
- Golden E2E: `pytest tests/golden_set_e2e_pytest.py` (pytest gerekir).

## Bilinen sınırlar

- Bazı PDB’lerde Arpeggio seçici uyarıları (`SelectionError` vb.) görülebilir; pipeline genelde `success` ile tamamlanır, etkileşim sayısı örneğe göre değişir.
- Klasik eğitim + batch analiz özeti olmadan `TestPropediaActivePipelineClassicalArtifacts` atlanır (isteğe bağlı).

Bu sürüm, dokümantasyondaki komutlarla yeniden üretilebilir çıktılar ve testlerle kapatılmış kabul edilir.
