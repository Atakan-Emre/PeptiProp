# PeptiProp

**Yapısal protein–peptid etkileşim tahmini: PROPEDIA verisinden skor üretimi, aday sıralama ve görsel raporlama.**

[![GitHub Pages](https://img.shields.io/badge/demo-GitHub%20Pages-blue)](https://atakan-emre.github.io/PeptiProp/)

---

## Amaç

Protein–peptid etkileşimleri ilaç tasarımı, sinyal yolağı analizi ve biyomalzeme mühendisliğinde merkezi bir rol oynar. Peptid-bazlı terapötikler küçük moleküllere kıyasla daha yüksek hedef seçiciliği ve daha düşük toksisite sunsa da, hangi peptidin bir protein yüzeyine gerçekten bağlanacağını tahmin etmek hesaplamalı olarak zorlu bir problemdir: bağlanma yüzeyi büyük, esneklik yüksek, ve konformasyon uzayı geniştir.

PeptiProp, deneysel olarak çözülmüş kristal yapılarından (co-crystal) yola çıkarak bir protein yüzeyine **hangi peptidin gerçekten bağlandığını** ayırt edebilen bir **skorlama ve sıralama (reranking)** modeli sunar. Temel hedef, verilen bir protein için aday peptitler arasında native (doğal) bağlayıcıyı üst sıralara taşımaktır.

### Neden Geleneksel Yöntemler Yetmiyor?

Geleneksel makine öğrenimi yaklaşımları (MLP, Random Forest, XGBoost), protein–peptid çiftlerini **özet istatistiklerle** (ortalama uzunluk, aminoasit oranları, genel arayüz skoru) temsil eder. Bu temsil, **rezidü seviyesindeki yapısal bilgiyi** — hangi aminoasidin hangi pozisyonda hangi komşusuyla etkileştiğini — kaybeder. PeptiProp, her rezidüyü bir **graf düğümü** olarak modelleyerek bu bilgiyi korur ve ESM-2 protein dil modelinin evrimsel bağlam bilgisiyle zenginleştirir.

## Canlı Demo

**GitHub Pages:** [atakan-emre.github.io/PeptiProp](https://atakan-emre.github.io/PeptiProp/)

Statik site 8 metrik kartı, 6 fazlı detaylı pipeline diyagramı, MLP vs GNN ablation karşılaştırma grafiği, ROC/PR eğrileri, confusion matrix, skor dağılımı histogramı, top-ranked tahmin tablosu, 5 farklı skor ve uzunlukta 2D peptit görseli (lightbox ile büyütme), gömülü 3D yapı önizlemesi (3Dmol.js) ve responsive tasarım içerir.

---

## Yöntem

### 1. Veri Kaynağı: PROPEDIA

[PROPEDIA](https://doi.org/10.1093/bioinformatics/btab524) (Protein–Peptide Binding Database), PDB'den derlenen ve deneysel olarak çözülmüş protein–peptid ko-kristal yapılarını barındıran kapsamlı bir veri tabanıdır. Her kayıt, bir protein zincirine fiziksel olarak bağlı peptid zincirinin atomik koordinatlarını içerir.

| Öğe | Değer |
|-----|-------|
| Kompleks sayısı | 42,375 |
| Zincir sayısı | 84,750 |
| Toplam rezidü | ~12.3 milyon |
| Arayüz rezidüsü | ~134,000 (5 Å eşik) |
| Pocket rezidüsü | ~605,000 (8 Å eşik) |
| Kaynak format | mmCIF (Macromolecular Crystallographic Information File) |

### 2. Kanonik Veri Oluşturma

Ham PROPEDIA yapıları, üç standart Parquet tablosuna dönüştürülür:

- **complexes** — her kompleks için PDB ID, kaynak, kalite bayrağı.
- **chains** — her zincir için tip (protein/peptid), uzunluk, ikincil yapı oranı, aminoasit bileşimi.
- **residues** — her rezidü için koordinatlar, arayüz/pocket bayrağı, yerel yoğunluk.

Arayüz ve pocket anotasyonu mesafe tabanlıdır: protein–peptid atomları arası mesafe 5 Å altındaysa *arayüz*, 8 Å altındaysa *pocket* olarak etiketlenir.

### 3. Veri Sızıntısı Koruması: Sekans-Küme Bazlı Split

Benzer sekans yapısına sahip proteinlerin hem eğitim hem test setinde bulunması, gerçek dünya performansını şişirir. Bunu önlemek için:

1. Tüm protein sekansları **MMseqs2** ile %30 kimlik eşiğinde kümelenir.
2. Her küme bölünmez bir birim olarak tek bir split'e (train / val / test) atanır.
3. Böylece test setindeki hiçbir protein, eğitim setindekilerle yüksek sekans benzerliği taşımaz.

### 4. Negatif Çift Üretimi

Modelin "bağlanmayan" örnekleri öğrenmesi için iki tür negatif çift üretilir:

| Tür | Açıklama | Zorluk |
|-----|----------|--------|
| **Easy** | Protein ile tamamen rastgele bir peptid eşleştirilir | Düşük |
| **Hard** | Protein ile aynı SCOP/CATH ailesinden farklı bir peptid eşleştirilir | Yüksek |

Her protein grubu için 1 pozitif (native) + 4 easy + 1 hard = 6 aday oluşturulur.

### 5. Özellik Mühendisliği

#### v0.1 MLP Baseline (Özet Vektör)

Her protein–peptid çifti için aşağıdaki bilgiler 131 boyutlu dense bir özellik vektörüne dönüştürülür:

| Kategori | Özellikler | Boyut |
|----------|------------|-------|
| Yapısal | Zincir uzunluğu, ikincil yapı oranları (helix/sheet/coil) | 4+4 |
| Sekans | Aminoasit bileşimi (20-boyutlu frekans vektörü) | 20+20 |
| Arayüz | Arayüz rezidü sayısı, arayüz oranı | 2+2 |
| Pocket | Pocket rezidü sayısı, pocket oranı | 2+2 |
| Yerel yoğunluk | 8 Å yarıçapında komşu atom yoğunluğu (ort/std) | 2+2 |
| Çift-bazlı | Kosinus benzerliği, L2 mesafesi, dot product | 3 |

Bu temsil, protein–peptid etkileşiminin **rezidü düzeyindeki detaylarını** kaybeder: hangi aminoasidin etkileşim yüzeyinde olduğu, komşuluk ilişkileri ve yerel yapısal motifler bilgisi yok olur.

#### v0.2 GNN+ESM-2 (Rezidü-Seviye Graf)

Her rezidü bir graf düğümü olarak 326 boyutlu özellik vektörüyle temsil edilir:

| Kaynak | Özellik | Boyut | Açıklama |
|--------|---------|-------|----------|
| ESM-2 | Per-rezidü embedding | 320 | Evrimsel bağlam: benzer sekans ailelerinden öğrenilmiş anlamsal temsil |
| Yapısal | is_interface | 1 | 5 Å arayüzde mi? |
| Yapısal | is_pocket | 1 | 8 Å pocket bölgesinde mi? |
| Yapısal | ss_onehot | 3 | İkincil yapı tipi (helix / sheet / coil) |
| Yapısal | local_density | 1 | 8 Å yarıçapında komşu rezidü yoğunluğu |

Kenar bilgisi: Cα atomları arası mesafe < 8 Å olan rezidü çiftleri birbirine bağlanır. Her kenar 4 boyutlu özellik taşır: `[mesafe, birim_yön_x, birim_yön_y, birim_yön_z]`.

### 6. Model Mimarisi

#### ESM-2 Nedir?

**ESM-2** (Evolutionary Scale Modeling 2), Meta AI tarafından geliştirilen, milyarlarca protein sekansı üzerinde **self-supervised** olarak eğitilmiş bir protein dil modelidir. Tıpkı GPT'nin doğal dil için yaptığı gibi, ESM-2 aminoasit dizilerinin istatistiksel örüntülerini öğrenir ve her rezidü için **evrimsel bağlam bilgisi** içeren yoğun vektör temsilleri üretir.

PeptiProp'ta `esm2_t6_8M_UR50D` varyantı kullanılır: 6 transformer katmanı, 8 milyon parametre, UniRef50 üzerinde eğitilmiş. Her rezidü için 320 boyutlu embedding üretir. 1022 token'dan uzun sekanslar için kayar pencere (stride 512) ve örtüşen bölgelerin ortalaması uygulanır.

#### GATv2 Nedir?

**GATv2** (Graph Attention Network v2), graf düğümleri arasındaki mesaj iletiminde **dinamik attention** mekanizması kullanan bir graf sinir ağıdır. Standart GAT'tan farkı, attention skorlarının hem kaynak hem hedef düğüme bağlı olarak hesaplanmasıdır — bu sayede model, etkileşim yüzeyindeki kritik rezidülere daha fazla ağırlık verebilir.

#### Mimari Detayı

Proje iki model mimarisini destekler:

**v0.1 — MLP Baseline:** 3 katmanlı tam bağlantılı sinir ağı. Her çift için özet istatistiklerden (131-d) oluşan dense vektörü girdi alır. Apple MLX framework'ü ile eğitilir.

**v0.2 — GATv2 + ESM-2 (Aktif):** Çift kanallı (dual-encoder) graf sinir ağı ile ön-eğitimli protein dil modeli birleşimi. Her rezidü bir graf düğümü olarak temsil edilir; düğüm özellikleri ESM-2'nin per-rezidü embedding'leri (320-d) ve yapısal anotasyonları (6-d) içerir.

```
Protein Rezidü Grafı ──→ GATv2 Encoder (4 katman, 4 head) ──→ Attention Pooling ──→ protein_vec (128-d)
                                                                                           ↘
                                                                                   [concat; hadamard; |diff|]
                                                                                           ↗       ↓
Peptid Rezidü Grafı  ──→ GATv2 Encoder (4 katman, 4 head) ──→ Attention Pooling ──→ peptide_vec (128-d)
                                                                                               MLP Head (512→256→128→1) → skor
```

**Etkileşim vektörü:** `[prot; pep; prot⊙pep; |prot−pep|]` = 512 boyutlu birleşik temsil. Hadamard çarpımı eleman-bazlı benzerliği, mutlak fark ise ayrışma bilgisini kodlar.

**Kayıp fonksiyonu:** BCE (α=0.5) + Pairwise Ranking Loss (margin=0.2). Ranking loss, aynı protein grubundaki pozitif-negatif çiftleri arasında en az `margin` kadar skor farkı olmasını zorlar.

| Parametre | MLP (v0.1) | GNN+ESM-2 (v0.2) |
|-----------|-----------|-------------------|
| Girdi boyutu | 131-d özet vektör | Rezidü grafı (326 d/node) |
| Mimari | MLP 3-katman | GATv2 4-katman × 4 head + Attention Pool |
| Toplam parametre | ~50K | ~554K |
| Protein temsili | Ortalama/oran istatistikleri | ESM-2 per-rezidü embedding |
| Yapısal bilgi | Kayıp (ortalamaya ezilmiş) | Korunmuş (graf topolojisi) |
| Loss | BCE + Ranking | BCE + Ranking |
| Erken durdurma | Val MRR, patience=12 | Val MRR, patience=15 |
| Eğitim süresi | ~5 dk (MLX, Apple M4) | ~4 saat (CPU, 80 epoch) |

### 7. Kullanılan Araç ve Teknolojiler

| Kategori | Araç | Versiyon | Kullanım Amacı |
|----------|------|----------|----------------|
| **Protein dil modeli** | ESM-2 (esm2_t6_8M_UR50D) | fair-esm ≥ 2.0 | Per-rezidü 320-d embedding çıkarımı; evrimsel bağlam kodlama |
| **Graf sinir ağı** | GATv2 (PyTorch Geometric) | pyg ≥ 2.5 | Rezidü-seviye yapısal öğrenme; attention-tabanlı mesaj iletimi |
| **Derin öğrenme** | PyTorch | ≥ 2.2 | GNN eğitimi, model tanımı, gradient hesabı |
| **Apple Silicon ML** | MLX | — | MLP baseline eğitimi (M-serisi çiplerde hızlandırılmış) |
| **Sekans kümeleme** | MMseqs2 | — | %30 sekans kimliği eşiğinde kümeleme; veri sızıntısı koruması |
| **Yapı ayrıştırma** | BioPython, Gemmi | — | mmCIF dosya parse; atomik koordinat çıkarma |
| **2D görselleştirme** | RDKit | — | Peptit 2D bağ yapısı çizimi (aminoasit dizisinden) |
| **3D görselleştirme** | 3Dmol.js | — | Tarayıcı-içi interaktif 3D protein/peptid viewer |
| **Veri formatı** | Apache Parquet | — | Sıkıştırılmış, sütun-bazlı veri depolama (~%70 boyut kazancı) |
| **CI/CD** | GitHub Actions + Pages | — | Otomatik site build ve yayın; her push'ta güncelleme |
| **Grafik çıktılar** | Matplotlib / Seaborn | — | ROC, PR eğrileri, confusion matrix, skor histogramları |

### 8. Çıktılar ve Görselleştirme

Model çıktıları dört katmanda raporlanır:

1. **Metrik tablosu** — 8 temel metrik: AUROC, AUPRC, F1, MCC (sınıflandırma) + MRR, Hit@1, Hit@3, Hit@5 (sıralama). Her metrik belirli bir yönü ölçer:

   | Metrik | Tür | Ne Ölçer? |
   |--------|-----|-----------|
   | **AUROC** | Sınıflandırma | Eşik-bağımsız pozitif/negatif ayırma gücü |
   | **AUPRC** | Sınıflandırma | Dengesiz sınıflarda precision-recall dengesi |
   | **F1** | Sınıflandırma | Precision ile recall'un harmonik ortalaması |
   | **MCC** | Sınıflandırma | Dengeli ikili sınıflandırma korelasyonu (-1 ile +1 arası) |
   | **MRR** | Sıralama | Doğru adayın ortalama ters sırası; 1.0 = daima 1. sırada |
   | **Hit@k** | Sıralama | İlk k aday içinde native peptit bulunma oranı |

2. **2D peptid görselleri** — RDKit ile çizilen 5 farklı peptit: en yüksek/düşük skorlu, orta skorlu, uzun ve kısa zincirlerden. Her görselde PDB ID, skor ve uzunluk bilgisi yer alır; tıklanınca lightbox ile büyütülür.
3. **3D yapı önizleme** — 3Dmol.js ile gömülü interaktif 3D viewer. Demo sayfasında 1CRN (Crambin, 46 aa) kristal yapısı gösterilir; tam pipeline çıktıları local ortamda üretilir. Cartoon, stick, sphere ve yüzey görünüm modları desteklenir.
4. **ROC / PR eğrileri, skor histogramları, confusion matrix** — Eğitim sonuçlarının detaylı grafikleri. GNN+ESM-2 eğitimi sonrası `generate_gnn_predictions.py` ile üretilir.

---

## Sonuçlar

### Ablation: MLP Baseline vs GNN+ESM-2

| Metrik | MLP v0.1 (68 ep) | GNN+ESM-2 v0.2 (80 ep) | Fark |
|--------|:-----------------:|:-----------------------:|:----:|
| **AUROC** | 0.8388 | **0.8813** | +0.0425 |
| **AUPRC** | 0.4348 | **0.5566** | +0.1218 |
| **F1** | 0.5074 | **0.5884** | +0.0810 |
| **MCC** | 0.4134 | **0.5037** | +0.0903 |
| **MRR** | 0.7120 | **0.7776** | +0.0656 |
| **Hit@1** | 0.5121 | **0.6210** | +0.1089 |
| **Hit@3** | 0.9275 | **0.9469** | +0.0194 |
| **Hit@5** | 0.9952 | **0.9965** | +0.0013 |

> GNN+ESM-2, tüm metriklerde MLP baseline'ı geçiyor. En büyük kazanım AUPRC (+12.2 pp) ve Hit@1 (+10.9 pp) metriklerinde; model pozitif çiftleri ilk sıraya taşımada belirgin şekilde daha başarılı.

### Veri Split İstatistikleri

| Split | PDB Grup | Pozitif | Negatif (easy + hard) | Toplam |
|-------|----------|---------|----------------------|--------|
| Train | 19,542 | 19,542 | 97,710 (68,397 + 29,313) | 117,252 |
| Val | 3,985 | 3,985 | 19,925 (15,940 + 3,985) | 23,910 |
| Test | 4,538 | 4,538 | 22,690 (18,152 + 4,538) | 27,228 |

> Toplam 168,390 çift; pozitif/negatif oranı yaklaşık 1:5.

---

## Pipeline

```
Faz 1: PROPEDIA mmCIF (42,375 kompleks)
  │  Ham deneysel ko-kristal yapılarının indirilmesi ve standardizasyonu
  ▼
Faz 2: Kanonik tablolar (complexes · chains · residues → Parquet)
  │  18.7K kompleks, 48K zincir, 3.5M rezidü; arayüz (5Å) + pocket (8Å) anotasyonu
  ▼
Faz 3: Sekans-küme split (MMseqs2 %30 kimlik → train 70% / val 15% / test 15%)
  │  Veri sızıntısı koruması: benzer proteinler aynı split'te
  ▼
Faz 4: Negatif çift üretimi (easy: rastgele peptit + hard: aynı aile)
  │  Her pozitife 5 negatif → toplam ~168K çift
  ▼
Faz 5: ESM-2 embedding çıkarımı (esm2_t6_8M_UR50D)
  │  Her benzersiz zincir sekansı için per-rezidü 320-d vektör → NPZ arsivleri
  ▼
Faz 6: Rezidü-seviye graf inşası (Cα < 8Å komşuluk)
  │  326-d düğüm (ESM-2 + yapısal) + 4-d kenar (mesafe + yön) → PyG .pt dosyaları
  ▼
Faz 7: GATv2 Dual-Encoder eğitimi
  │  BCE + pairwise ranking loss; 80 epoch; AdamW lr=5e-4; val MRR erken durdurma
  ▼
Faz 8: Değerlendirme + tahmin üretimi
  │  Test: AUROC=0.881, MRR=0.778, Hit@1=0.621 → top_ranked_examples.json
  ▼
Faz 9: Raporlama
       ROC/PR eğrileri · confusion matrix · 2D peptit (RDKit) · 3D viewer (3Dmol.js) · GitHub Pages site
```

---

## Hızlı Başlangıç

```bash
# 1. Ortam kurulumu
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Kanonik veri oluşturma (PROPEDIA mmCIF'lerden)
python scripts/build_canonical_dataset.py --source data/propedia --output data/canonical

# 3. Arayüz ve pocket anotasyonu
python scripts/annotate_interface_pocket.py --canonical data/canonical

# 4. Sekans-küme bazlı split
python scripts/build_pdb_level_splits.py --canonical data/canonical

# 5. Negatif çift üretimi
python scripts/generate_negative_pairs.py \
  --canonical data/canonical --splits data/canonical/splits \
  --output data/canonical/pairs

# 6a. MLP baseline (v0.1)
python scripts/export_mlx_features.py --config configs/train_v0_1_scoring_mlx_m4.yaml
python scripts/train_scoring_mlx.py --config configs/train_v0_1_scoring_mlx_m4.yaml

# 6b. GNN + ESM-2 (v0.2)
python scripts/extract_esm2_embeddings.py --model esm2_t6_8M  # ~35K NPZ dosyası üretir
python scripts/build_residue_graphs.py --config configs/train_v0_2_gnn_esm2.yaml  # PyG .pt grafları
python scripts/train_gnn_esm2.py --config configs/train_v0_2_gnn_esm2.yaml  # 80 epoch GATv2 eğitimi
python scripts/generate_gnn_predictions.py --config configs/train_v0_2_gnn_esm2.yaml  # Tahmin + grafikler

# 7. Statik site üretimi
pip install -r scripts/requirements-pages.txt
python scripts/build_pages_site.py
```

---

## Proje Yapısı

```
PeptiProp/
├── README.md
├── requirements.txt
├── .github/workflows/pages.yml    # GitHub Pages CI/CD
│
├── src/peptidquantum/             # Ana Python paketi
│   ├── models/                    # Model tanımları
│   │   ├── gnn_esm2.py            # PeptiPropGNN (GATv2 dual-encoder)
│   │   └── graph_builder.py       # Rezidü grafı inşa modülü
│   ├── data/
│   │   ├── processors/            # canonical_builder, mmcif_parser, pair_extractor
│   │   ├── downloaders/           # PROPEDIA, PepBDB, BioLip2, GEPPRI indirici
│   │   └── fetchers/              # RCSB mmCIF fetcher
│   ├── interaction/               # Arpeggio/PLIP wrapper, contact matrix, fingerprint
│   ├── training/                  # Trainer, ablation runner
│   ├── visualization/
│   │   ├── chemistry/             # RDKit 2D peptit renderer
│   │   ├── plots/                 # site_extras, contact_map
│   │   ├── web/                   # 3Dmol.js viewer, HTML rapor
│   │   └── pymol/                 # PyMOL renderer
│   ├── pipeline/                  # CLI, config, pipeline orchestrator
│   └── utils/                     # data_split, yardımcılar
│
├── scripts/                       # Pipeline betikleri
│   ├── build_canonical_dataset.py
│   ├── annotate_interface_pocket.py
│   ├── build_pdb_level_splits.py
│   ├── generate_negative_pairs.py
│   ├── export_mlx_features.py
│   ├── train_scoring_mlx.py       # MLP eğitim betiği (v0.1)
│   ├── extract_esm2_embeddings.py # ESM-2 per-rezidü embedding çıkarımı
│   ├── build_residue_graphs.py    # PyG graf pre-build
│   ├── train_gnn_esm2.py          # GNN+ESM-2 eğitim (v0.2)
│   ├── generate_gnn_predictions.py # GNN tahmin + grafik üretici
│   ├── build_pages_site.py        # Statik site üretici
│   └── sync_pages_training_bundle.py
│
├── configs/
│   ├── train_v0_2_gnn_esm2.yaml             # GNN+ESM-2 konfigürasyonu (aktif)
│   ├── train_v0_1_scoring_mlx_m4.yaml       # MLP baseline konfigürasyonu
│   └── ablation_generated_mlx/              # MLP ablation deneyleri
│
├── tests/                         # Birim ve entegrasyon testleri
├── examples/                      # Örnek kullanım betiği
├── mlx/                           # Apple Silicon eğitim ortamı gereksinimleri
├── docs/                          # Detaylı dokümantasyon (Türkçe)
├── publish/github_pages_training_bundle/    # Pages metrik/görsel paketi
└── outputs/                       # Eğitim çıktıları (gitignore)
```

---

## Konfigürasyon

Aktif eğitim konfigürasyonu `configs/train_v0_2_gnn_esm2.yaml` dosyasındadır:

```yaml
model:
  node_feat_dim: 326     # ESM-2 (320) + yapısal (6)
  hidden_dim: 128
  num_gnn_layers: 4
  heads: 4
  mlp_hidden: 256
  dropout: 0.1
training:
  seed: 42
  batch_size: 32
  epochs: 80
  early_stopping_patience: 15
  lr: 0.0005
loss:
  use_ranking: true
  margin: 0.2
  bce_alpha: 0.5
evaluation:
  monitor_metric: val_mrr
  threshold_selection_metric: mcc
```

MLP baseline konfigürasyonu: `configs/train_v0_1_scoring_mlx_m4.yaml`. Ablation deneyleri: `configs/ablation_generated_mlx/`.

---

## GitHub Pages Yayını

Eğitim sonrası metrikleri ve görselleri Pages'a aktarmak için:

```bash
python scripts/sync_pages_training_bundle.py outputs/training/<egitim_klasoru>
git add publish/github_pages_training_bundle && git commit -m "Pages: eğitim bundle güncelle"
git push
```

CI pipeline otomatik olarak `build_pages_site.py` çalıştırır ve siteyi yayınlar.

---

## Testler

```bash
python -m pytest tests/ -v
python -m unittest discover -s tests -v
```

---

## Dokümantasyon

| Dosya | İçerik |
|-------|--------|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Hızlı başlangıç rehberi |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Ortam kurulumu ve bağımlılıklar |
| [docs/DATA_ARCHITECTURE.md](docs/DATA_ARCHITECTURE.md) | Kanonik veri şeması |
| [docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md) | Proje mimarisi detayı |
| [docs/EXTERNAL_TOOLS.md](docs/EXTERNAL_TOOLS.md) | Arpeggio, PLIP, Open Babel kurulumu |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Doğrulama rehberi |
| [docs/DATASET_DOWNLOAD_GUIDE.md](docs/DATASET_DOWNLOAD_GUIDE.md) | Veri indirme adımları |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Proje yol haritası |
| [docs/SCIENTIFIC_METHOD_TR.md](docs/SCIENTIFIC_METHOD_TR.md) | Bilimsel yöntem açıklaması |
| [docs/POZITIF_PEPTIT_VE_SKOR_TR.md](docs/POZITIF_PEPTIT_VE_SKOR_TR.md) | Pozitif etiket ve skor mantığı |
| [docs/VERI_VE_GORSEL_GERCEK_TR.md](docs/VERI_VE_GORSEL_GERCEK_TR.md) | Veri ve görsel doğruluğu |
| [docs/GITHUB_PAGES_TR.md](docs/GITHUB_PAGES_TR.md) | Pages yayın detayları |
| [docs/LITERATURE_TR.md](docs/LITERATURE_TR.md) | İlgili literatür |
| [docs/MODEL_IMPROVEMENTS_TR.md](docs/MODEL_IMPROVEMENTS_TR.md) | Model iyileştirme notları |

---

## Kaynaklar

1. **PROPEDIA** — Martins, P.M. et al. (2021). *PROPEDIA: a database for protein–peptide identification.* Bioinformatics, 37(8), 1180–1182. [doi:10.1093/bioinformatics/btab524](https://doi.org/10.1093/bioinformatics/btab524)
2. **MMseqs2** — Steinegger, M. & Söding, J. (2017). *MMseqs2 enables sensitive protein sequence searching for the analysis of massive data sets.* Nature Biotechnology, 35, 1026–1028. [doi:10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988)
3. **RDKit** — Open-source cheminformatics toolkit. [rdkit.org](https://www.rdkit.org/)
4. **3Dmol.js** — Rego, N. & Koes, D. (2015). *3Dmol.js: molecular visualization with WebGL.* Bioinformatics, 31(8), 1322–1324. [doi:10.1093/bioinformatics/btu829](https://doi.org/10.1093/bioinformatics/btu829)
5. **ESM-2** — Lin, Z. et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science, 379(6637), 1123–1130. [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)
6. **PyTorch Geometric** — Fey, M. & Lenssen, J.E. (2019). *Fast graph representation learning with PyTorch Geometric.* ICLR Workshop. [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)
7. **MLX** — Apple Machine Learning Research. [ml-explore/mlx](https://github.com/ml-explore/mlx)

---

## Lisans

Proje lisansı depo kökündeki LICENSE dosyasına tabidir.
