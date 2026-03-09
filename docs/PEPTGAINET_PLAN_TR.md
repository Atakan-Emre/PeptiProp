# PeptGAINET Uygulama Planı (GEPPRI Tabanlı)

## Kapsam

Bu plan, mevcut proje için gerçek ve çalışır üretim hattını tanımlar:
- ham GEPPRI verisini almak
- pair veri üretmek
- modeli eğitmek
- batch tahmin almak
- makale stili panel görsel üretmek

## Adım 1: Ham Veri

Komut:

```bash
bash scripts/download_geppri_data.sh
```

Çıktı:
- `data/raw/GEPPRI/Train1.txt`
- `data/raw/GEPPRI/Test1.txt`
- `data/raw/GEPPRI/Train2.txt`
- `data/raw/GEPPRI/Test2.txt`

## Adım 2: Pair Veri Üretimi

Komut:

```bash
PYTHONPATH=src python3 scripts/build_geppri_pair_dataset.py \
  --input data/raw/GEPPRI/Train1.txt \
  --output data/processed/geppri_train1_pairs.jsonl

PYTHONPATH=src python3 scripts/build_geppri_pair_dataset.py \
  --input data/raw/GEPPRI/Test1.txt \
  --output data/processed/geppri_test1_pairs.jsonl
```

Mantık:
- Etiket `1` olan contiguous residue segmentleri: pozitif peptide
- Aynı uzunlukta `0` etiketli segmentler: negatif peptide

## Adım 3: Model Eğitimi

Komut:

```bash
PYTHONPATH=src python3 scripts/train_peptgainet.py \
  --train-jsonl data/processed/geppri_train1_pairs.jsonl \
  --valid-jsonl data/processed/geppri_test1_pairs.jsonl \
  --epochs 12 \
  --batch-size 16 \
  --out-dir models/peptgainet
```

## Adım 4: Batch Tahmin

Komut:

```bash
PYTHONPATH=src python3 scripts/predict_peptgainet_dataset.py \
  --checkpoint models/peptgainet/peptgainet.pt \
  --pairs-jsonl data/processed/geppri_test1_pairs.jsonl \
  --output-csv outputs/geppri_test1_predictions.csv
```

## Adım 5: Makale Stili Görsel

Komut:

```bash
PYTHONPATH=src python3 scripts/plot_gainet_style_panel.py \
  --pairs-jsonl data/processed/geppri_test1_pairs.jsonl \
  --pred-csv outputs/geppri_test1_predictions.csv \
  --output-png outputs/geppri_test1_gainet_style_panel.png \
  --mode all \
  --max-rows 5
```

## Teknik Notlar

- Model: çift graph encoder + co-attention + bilinear scorer
- Görselleştirme: RDKit ile molekül çizimi + MCS tabanlı highlight
- Protein çiziminde uzun sekanslar fragmana kısaltılır (`--max-seq-len-draw`)
