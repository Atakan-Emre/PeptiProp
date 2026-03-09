# PeptidQuantum

Protein-peptit etkileşimini gerçek GEPPRI verisi ile eğiten ve GAINET benzeri panel görsel üreten temiz proje.

## Proje Amacı

Bu projede hedef:
1. GEPPRI residue-level etiketlerinden protein-peptit pair veri üretmek.
2. PeptGAINET modeli ile etkileşim olasılığı tahmini yapmak.
3. Tahminleri makaledeki stile yakın, yayınlanabilir panel görsel halinde göstermek.

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
  models/peptgainet/
    peptgainet.pt                # Eğitilmiş model
    history.json                 # Eğitim geçmişi
  outputs/
    geppri_test1_predictions.csv
    geppri_test1_gainet_style_panel.png
  scripts/
    download_geppri_data.sh
    build_geppri_pair_dataset.py
    train_peptgainet.py
    predict_peptgainet_dataset.py
    plot_gainet_style_panel.py
  src/peptidquantum/
    dataio.py
    peptgainet/
      dataset.py
      graph.py
      model.py
      train.py
  requirements.txt
```

## Kurulum

```bash
python3 -m pip install -r requirements.txt
```

## Uçtan Uca Çalıştırma

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

### 3) Model eğit

```bash
PYTHONPATH=src python3 scripts/train_peptgainet.py \
  --train-jsonl data/processed/geppri_train1_pairs.jsonl \
  --valid-jsonl data/processed/geppri_test1_pairs.jsonl \
  --epochs 12 --batch-size 16 --lr 1e-3 \
  --out-dir models/peptgainet
```

### 4) Batch tahmin al

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
  --pred-csv outputs/geppri_test1_predictions.csv \
  --output-png outputs/geppri_test1_gainet_style_panel.png \
  --mode all --max-rows 5 --threshold 0.5
```

## Görselleştirme Notu

Panel yapısı satır başına 3 molekül düzeni ile üretilir:
- Sol: Molekül A
- Orta: Molekül B
- Sağ: kırmızı highlight ile odak alt-yapı
- Arada ok ve üstte prediction metin blokları

Görsel standardı: [VISUALIZATION_STYLE_TR.md](/Users/atakanemre/PeptidQuantum/docs/VISUALIZATION_STYLE_TR.md)

`plot_gainet_style_panel.py` için `--mode` seçenekleri:
- `all`, `tp`, `tn`, `fp`, `fn`, `pos`, `neg`

## Referanslar

Makale ve benzer çalışmalar listesi: [LITERATURE_TR.md](/Users/atakanemre/PeptidQuantum/docs/LITERATURE_TR.md)

## Temizlik Durumu

Proje, yalnızca gerçek eğitim/tahmin/görselleştirme akışında kullanılan dosyaları içerir.
Eski deneme pipeline dosyaları ve anlamsız çıktı dosyaları kaldırılmıştır.
