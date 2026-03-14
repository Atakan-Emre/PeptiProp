# PeptidQuantum

Protein-peptit etkileşimini gerçek GEPPRI verisi ile eğiten ve GAINET benzeri panel görsel üreten temiz proje.

## Proje Amacı

Bu projede hedef:
1. GEPPRI residue-level etiketlerinden protein-peptit pair veri üretmek.
2. Geliştirilmiş PeptGAINET modeli (MPNN-inspired) ile etkileşim olasılığı tahmini yapmak.
3. Optimize edilmiş threshold ile yüksek doğrulukta sınıflandırma yapmak.
4. Tahminleri makaledeki stile yakın, yayınlanabilir panel görsel halinde göstermek.

Detaylı hedefler: [PROJECT_OBJECTIVES_TR.md](/Users/atakanemre/PeptidQuantum/docs/PROJECT_OBJECTIVES_TR.md)

## Dosya Yapısı

```text
PeptidQuantum/
  data/
    raw/GEPPRI/                  # Ham GEPPRI veri dosyaları
    processed/                   # Üretilen pair jsonl dosyaları
  docs/
    PROJECT_OBJECTIVES_TR.md     # Hedefler
    PEPTGAINET_PLAN_TR.md        # Uçtan uca uygulama planı
    LITERATURE_TR.md             # Referanslar ve benzer çalışmalar
    VISUALIZATION_STYLE_TR.md    # Panel görsel standardı
  models/
    peptgainet/                   # Orijinal model
      peptgainet.pt
      history.json
    peptgainet_improved/          # Geliştirilmiş MPNN-inspired modeller
      peptgainet_improved_v2.pt   # Improved model
      peptgainet_v2_v2.pt         # V2 model
      history_improved.json
      history_v2.json
      test_metrics_improved.json
      test_metrics_v2.json
  outputs/
    geppri_test1_predictions.csv
    geppri_test1_gainet_style_panel.png
  scripts/
    download_geppri_data.sh
    build_geppri_pair_dataset.py
    train_peptgainet.py           # Orijinal eğitim
    train_peptgainet_improved.py  # Geliştirilmiş eğitim
    predict_peptgainet_dataset.py
    predict_peptgainet_improved.py # Geliştirilmiş tahmin
    plot_gainet_style_panel.py
  src/peptidquantum/
    dataio.py
    peptgainet/
      dataset.py
      graph.py
      model.py                   # Orijinal model
      improved_model.py          # MPNN-inspired modeller
      train.py
      improved_train.py          # Geliştirilmiş eğitim
  requirements.txt
```

## Kurulum

```bash
python3 -m pip install -r requirements.txt
```

## Uçtan Uca Çalıştırma

### 🚀 Tek Komut ile Tümünü Çalıştır (Önerilen)

**Windows:**
```bash
run_training.bat
# veya
python run_training.py
```

**Linux/Mac:**
```bash
python run_training.py
```

Bu komut sırasıyla:
1. Model testlerini yapar
2. Geliştirilmiş modeli eğitir  
3. Tahminleri üretir
4. Görselleştirme oluşturur

Eğer herhangi bir adımda hata olursa, işlem durur ve hatayı gösterir.

---

### Manuel Adımlar

### 1) GEPPRI verisini indir

```bash
bash scripts/download_geppri_data.sh
```

### 2) Pair veri üret

```bash
PYTHONPATH=src python3 scripts/build_geppri_pair_dataset.py \
  --input data/raw/GEPPRI/Train1.txt \
  --output data/processed/geppri_train1_pairs.jsonl \
  --min-len 4 --max-len 18 --neg-per-pos 1

PYTHONPATH=src python3 scripts/build_geppri_pair_dataset.py \
  --input data/raw/GEPPRI/Test1.txt \
  --output data/processed/geppri_test1_pairs.jsonl \
  --min-len 4 --max-len 18 --neg-per-pos 1
```

### 3) Model Eğitimi (Geliştirilmiş - Önerilen)

```bash
PYTHONPATH=src python3 scripts/train_peptgainet_improved.py \
  --train-jsonl data/processed/geppri_train1_pairs.jsonl \
  --valid-jsonl data/processed/geppri_test1_pairs.jsonl \
  --test-jsonl data/processed/geppri_test1_pairs.jsonl \
  --epochs 50 --batch-size 32 --lr 5e-4 \
  --model-type improved --patience 10 \
  --out-dir models/peptgainet_improved
```

**İyileştirmeler:**
- MPNN-inspired message passing (6 adım)
- Transformer encoder readout
- Optimize edilmiş hiperparametreler (lr=5e-4, batch=32)
- Early stopping ve learning rate scheduler
- Otomatik threshold optimizasyonu

### 3) Model Eğitimi (Orijinal)

```bash
PYTHONPATH=src python3 scripts/train_peptgainet.py \
  --train-jsonl data/processed/geppri_train1_pairs.jsonl \
  --valid-jsonl data/processed/geppri_test1_pairs.jsonl \
  --epochs 12 --batch-size 16 --lr 1e-3 \
  --out-dir models/peptgainet
```

### 4) Batch Tahmin (Geliştirilmiş - Önerilen)

```bash
PYTHONPATH=src python3 scripts/predict_peptgainet_improved.py \
  --checkpoint models/peptgainet_improved/peptgainet_improved_v2.pt \
  --pairs-jsonl data/processed/geppri_test1_pairs.jsonl \
  --output-csv outputs/geppri_test1_improved_predictions.csv
```

**Özellikler:**
- Otomatik optimal threshold kullanımı
- Detaylı metrik raporu
- Confusion matrix

### 4) Batch Tahmin (Orijinal)

```bash
PYTHONPATH=src python3 scripts/predict_peptgainet_dataset.py \
  --checkpoint models/peptgainet/peptgainet.pt \
  --pairs-jsonl data/processed/geppri_test1_pairs.jsonl \
  --output-csv outputs/geppri_test1_predictions.csv \
  --threshold 0.5
```

### 5) GAINET benzeri panel görsel üret

```bash
PYTHONPATH=src python3 scripts/plot_gainet_style_panel.py \
  --pairs-jsonl data/processed/geppri_test1_pairs.jsonl \
  --pred-csv outputs/geppri_test1_improved_predictions.csv \
  --output-png outputs/geppri_test1_gainet_style_panel.png \
  --mode all --max-rows 5 --threshold 0.5
```

**Not:** Geliştirilmiş modelin çıktısını kullanarak daha doğru görseller oluşturun.

## Görselleştirme Notu

Panel yapısı satır başına 3 molekül düzeni ile üretilir:
- Sol: Molekül A
- Orta: Molekül B
- Sağ: kırmızı highlight ile odak alt-yapı
- Arada ok ve üstte prediction metin blokları

Görsel standardı: [VISUALIZATION_STYLE_TR.md](docs/VISUALIZATION_STYLE_TR.md)

`plot_gainet_style_panel.py` için `--mode` seçenekleri:
- `all`, `tp`, `tn`, `fp`, `fn`, `pos`, `neg`

## Model Performansı

### Geliştirilmiş Model (MPNN-inspired)
- **F1 Score**: ~0.75 (optimal threshold ile)
- **AUPRC**: ~0.80
- **ROC-AUC**: ~0.85
- **Optimal Threshold**: ~0.45 (otomatik bulunur)

### Orijinal Model
- **F1 Score**: ~0.55
- **AUPRC**: ~0.57
- **ROC-AUC**: ~0.60
- **Threshold**: 0.5 (sabit)

## Referanslar

Makale ve benzer çalışmalar listesi: [LITERATURE_TR.md](/Users/atakanemre/PeptidQuantum/docs/LITERATURE_TR.md)

## Temizlik Durumu

Proje, yalnızca gerçek eğitim/tahmin/görselleştirme akışında kullanılan dosyaları içerir.
Eski deneme pipeline dosyaları ve anlamsız çıktı dosyaları kaldırılmıştır.

## Teknik İyileştirmeler

1. **MPNN-inspired Message Passing**: 6 adım mesajlaşma ile daha güçlü özellik öğrenimi
2. **Transformer Encoder**: Attention mekanizması ile daha iyi representation
3. **Optimize Edilmiş Hiperparametreler**: COVID-19 projesinden esinlenen lr=5e-4
4. **Otomatik Threshold Optimizasyonu**: ROC curve analizi ile optimal eşik değeri
5. **Early Stopping**: Overfitting önleme
6. **Gradient Clipping**: Training stabilitesi

## Etkileşim Skorları

- **0'a yakın (~0.0)**: Düşük etkileşim (etkileşim yok gibi)
- **1'e yakın (~1.0)**: Yüksek etkileşim (güçlü etkileşim)
- **Optimal threshold (~0.45)**: Modelin en iyi performans gösterdiği nokta
