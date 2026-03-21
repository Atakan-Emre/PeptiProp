# Pozitif peptit (etiket 1) ve model skoru — ne fark eder?

## Etiket 1 = “skor 1” değil

Eğitimde **`label == 1`** (veya `label_eval == 1`), modelin ürettiği bir skor değildir. **Yapısal pozitif** (native / co-crystal) çifti tanımlar:

- Aynı **PROPEDIA** kompleksinde, deney yapısında **birlikte** görünen **protein zinciri + peptit zinciri**.
- `scripts/generate_negative_pairs.py` içinde `get_positive_pairs`: her kompleks için `pair_id = …_pos`, `negative_type = positive`, `label = 1`.
- **Negatifler (`label = 0`)**: Aynı protein (veya aynı PDB) için seçilen **başka komplekslerden** peptitler (easy / hard vb.); yapıda bu ikili birlikte ölçülmemiş olabilir.

Özet: **“Kabul ettiğimiz peptit” = o PDB kaydının peptit zinciri; “kabul” gerekçesi = kristalografik tanım, model değil.**

## Model skoru

**Skor**, eğitimli skorlama modelinin çıktısıdır (ör. sigmoid sonrası olasılık). Anlamı:

- Aynı **protein grubu** (aynı protein kompleksi + protein zinciri) altındaki **aday peptitler** arasında kıyaslama yapılır.
- `top_ranked_examples.json` içindeki **`score`** bu model çıktısıdır.
- `best_true_positive` satırında skorun **1.0** görünmesi, o örnekte modelin çıktısının üst sınıra oturduğu anlamına gelir; “etiket otomatik 1” anlamına gelmez (etiket zaten çift tablosundan gelir).

## Rapor ve site çıktıları

- Pipeline **`report.html`** üretirken yapıyı ve etkileşimleri gösterir; isteğe bağlı **`report_metadata`** ile `pair_id`, model skoru, eğitim etiketi eklenebilir (`pipeline.run(..., report_metadata={...})`).
- Statik sitede **2D peptit paneli**, `top_ranked_examples.json` önizlemesi ve `chains.parquet` sekansı ile üretilir; her görselde **PDB, zincir, sekans özeti, skor, etiket, uzunluk** yazılır.

## Dosyalar

| Dosya | İçerik |
|--------|--------|
| `data/canonical/pairs/*_pairs.parquet` | `label`, `negative_type`, çift kimlikleri |
| `outputs/training/.../top_ranked_examples.json` | Test sıralama örnekleri, skorlar |
| `scripts/generate_negative_pairs.py` | Pozitif/negatif üretim kuralları |
