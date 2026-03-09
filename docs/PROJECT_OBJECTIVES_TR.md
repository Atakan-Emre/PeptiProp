# Proje Amaçları

## Ana Amaç

Protein-peptit etkileşimini tahmin eden, GAINET benzeri görsel çıktı üreten, sade ve tekrar çalıştırılabilir bir proje hattı kurmak.

## Alt Amaçlar

1. GEPPRI ham verisini doğrudan kullanmak.
2. Residue-level etiketlerden peptide-protein pair veri üretmek.
3. Çift-kollu graph model (PeptGAINET) ile gerçek eğitim yapmak.
4. Tahminleri makaledeki stile yakın panel görsel ile sunmak.
5. Projeyi gereksiz dosyalardan arındırmak ve tek bir README ile yönetilebilir hale getirmek.

## Başarı Kriterleri

- `data/raw/GEPPRI/` altında ham veri mevcut.
- `scripts/build_geppri_pair_dataset.py` ile pair jsonl üretilebiliyor.
- `scripts/train_peptgainet.py` ile model eğitilebiliyor.
- `scripts/predict_peptgainet_dataset.py` ile batch tahmin CSV üretiliyor.
- `scripts/plot_gainet_style_panel.py` ile panel görsel üretiliyor.
