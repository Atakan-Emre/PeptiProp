# Proje Amaçları

> **Not:** Aktif ürün hattı PROPEDIA + skorlama + MLX/klasik eğitim + 2D/3D görselleştirmedir. Aşağıdaki maddeler tarihsel/planlama notları içerir; güncel komutlar ve klasörler için kök [README.md](../README.md) ve [docs/GITHUB_PAGES_TR.md](GITHUB_PAGES_TR.md) esas alınmalıdır.

## Ana Amaç

Protein-peptit etkileşimini yüksek doğrulukta tahmin eden, GAINET benzeri görsel çıktı üreten, MPNN-inspired geliştirilmiş mimariye sahip, sade ve tekrar çalıştırılabilir bir proje hattı kurmak.

## Alt Amaçlar

1. GEPPRI ham verisini doğrudan kullanmak.
2. Residue-level etiketlerden peptide-protein pair veri üretmek.
3. **Geliştirilmiş çift-kollu graph model (Improved PeptGAINET) ile yüksek performanslı eğitim yapmak.**
4. **Otomatik threshold optimizasyonu ile optimal sınıflandırma yapmak.**
5. Tahminleri makaledeki stile yakın panel görsel ile sunmak.
6. **COVID-19 drug interaction projesindeki başarılı MPNN yaklaşımını entegre etmek.**
7. Projeyi gereksiz dosyalardan arındırmak ve yönetilebilir hale getirmek.

## Başarı Kriterleri

### Temel Akış
- `data/raw/GEPPRI/` altında ham veri mevcut.
- `scripts/build_geppri_pair_dataset.py` ile pair jsonl üretilebiliyor.
- `scripts/train_peptgainet_improved.py` ile geliştirilmiş model eğitilebiliyor.
- `scripts/predict_peptgainet_improved.py` ile optimal threshold ile tahmin yapılabiliyor.
- `scripts/plot_gainet_style_panel.py` ile panel görsel üretiliyor.

### Performans Hedefleri
- **F1 Score**: ≥ 0.70 (orijinal 0.55 yerine)
- **AUPRC**: ≥ 0.75 (orijinal 0.57 yerine)
- **ROC-AUC**: ≥ 0.80 (orijinal 0.60 yerine)
- **Optimal Threshold**: Otomatik bulunacak (~0.45)

### Teknik Gereksinimler
- MPNN-inspired message passing (6 adım)
- Transformer encoder readout
- Early stopping ve learning rate scheduling
- Gradient clipping ve stabil eğitim
