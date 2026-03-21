# Görselleştirme Standardı (Makale Stili)

Üretim çıktıları: `report.html`, `viewer.html`, `data/viewer_state.json` (`structure_format`, zincirler, etkileşim listesi). GitHub Pages 3D örneği: `docs/GITHUB_PAGES_TR.md`.

Bu proje için hedef görsel, GAINET makalesindeki panel düzenine benzer olmalıdır.

## Panel Yapısı (Satır Başına)

1. Sol: Molekül A (ham çizim)
2. Orta: Molekül B (ham çizim)
3. Sağ: Molekül A üzerinde kırmızı highlight edilmiş alt-yapı
4. Orta-sağ: ok (`->`) ile dönüşüm/odak yönü
5. Üst metin:
- `Interaction Prediction 1`
- `Interaction Prediction 2`
- `Average Interaction Prediction`
- `Actual`

## Stil Kuralları

- Açık gri arka plan (`#e9e9e9`)
- Satır etiketi: `(a)`, `(b)`, `(c)` ...
- Highlight rengi: kırmızı tonları (atom+bond)
- Yüksek çözünürlük: en az 300 DPI

## Veri Kaynağı Kuralları

Öncelik sırası:
1. Doğrudan `smiles1/smiles2` kolonları
2. Yoksa `peptide_seq` ve `protein_seq`'den RDKit ile türetme

Protein sekansı çok uzunsa çizim için ilk `--max-seq-len-draw` kadar fragman kullanılır.

## Kullanım

```bash
PYTHONPATH=src python3 scripts/plot_gainet_style_panel.py \
  --pairs-jsonl data/processed/geppri_test1_pairs.jsonl \
  --pred-csv outputs/geppri_test1_predictions.csv \
  --output-png outputs/geppri_test1_gainet_style_panel.png \
  --mode all \
  --max-rows 5 \
  --threshold 0.5
```

## Mode Seçimi

- `all`: en güvenli örnekler
- `tp`: doğru pozitif örnekler
- `tn`: doğru negatif örnekler
- `fp`: yanlış pozitif örnekler
- `fn`: yanlış negatif örnekler
- `pos`: skor eşik üstü
- `neg`: skor eşik altı
