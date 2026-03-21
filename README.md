# PeptiProp

**Yapısal protein–peptid etkileşim tahmini: PROPEDIA verisinden skor üretimi, aday sıralama ve görsel raporlama.**

[![GitHub Pages](https://img.shields.io/badge/demo-GitHub%20Pages-blue)](https://atakan-emre.github.io/PeptiProp/)

---

## Amaç

Protein–peptid etkileşimleri ilaç tasarımı, sinyal yolağı analizi ve biyomalzeme mühendisliğinde merkezi bir rol oynar. PeptiProp, deneysel olarak çözülmüş kristal yapılarından (co-crystal) yola çıkarak bir protein yüzeyine hangi peptidin gerçekten bağlandığını ayırt edebilen bir skorlama ve sıralama modeli sunar. Temel hedef, verilen bir protein için aday peptitler arasında native (doğal) bağlayıcıyı üst sıralara taşımaktır.

## Canlı Demo

**GitHub Pages:** [atakan-emre.github.io/PeptiProp](https://atakan-emre.github.io/PeptiProp/)

Statik site 8 metrik kartı, pipeline diyagramı, top-ranked tahmin tablosu, 5 farklı skor ve uzunlukta 2D peptit görseli (lightbox ile büyütme), gömülü 3D yapı önizlemesi (3Dmol.js) ve responsive tasarım içerir.

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

Her protein–peptid çifti için aşağıdaki bilgiler dense bir özellik vektörüne dönüştürülür:

| Kategori | Özellikler |
|----------|------------|
| Yapısal | Zincir uzunluğu, ikincil yapı oranları (helix/sheet/coil) |
| Sekans | Aminoasit bileşimi (20-boyutlu frekans vektörü) |
| Arayüz | Arayüz rezidü sayısı, arayüz oranı |
| Pocket | Pocket rezidü sayısı, pocket oranı |
| Yerel yoğunluk | 8 Å yarıçapında komşu atom yoğunluğu |
| Çift-bazlı | Kosinus benzerliği, L2 mesafesi, dot product (protein vs peptid özet vektörü) |

### 6. Model Mimarisi

Proje iki model mimarisini destekler:

**v0.1 — MLP Baseline:** 3 katmanlı tam bağlantılı sinir ağı. Her çift için özet istatistiklerden (131-d) oluşan dense vektörü girdi alır. Apple MLX framework'ü ile eğitilir.

**v0.2 — GATv2 + ESM-2 (Aktif):** Çift kanallı graf sinir ağı (Graph Attention Network v2) ile ön-eğitimli protein dil modeli (ESM-2) birleşimi. Her rezidü bir graf düğümü olarak temsil edilir; düğüm özellikleri ESM-2'nin per-rezidü embedding'leri (320-d) ve yapısal anotasyonları (arayüz, pocket, ikincil yapı) içerir. Rezidüler arası 8 Å mesafe eşiğiyle komşuluk grafı oluşturulur.

```
Protein Rezidü Grafı ──→ GATv2 (4 katman, 4 head) ──→ Attention Pooling ──→ protein_vec
                                                                                    ↘
                                                                            [concat; hadamard; |diff|]
                                                                                    ↗       ↓
Peptid Rezidü Grafı  ──→ GATv2 (4 katman, 4 head) ──→ Attention Pooling ──→ peptide_vec    MLP Head → skor
```

| Parametre | MLP (v0.1) | GNN+ESM-2 (v0.2) |
|-----------|-----------|-------------------|
| Girdi | 131-d özet vektör | Rezidü grafı (320+6 d/node) |
| Mimari | MLP 3-katman | GATv2 4-katman + Attention Pool |
| Parametreler | ~50K | ~554K |
| Protein temsili | Ortalama/oran | ESM-2 per-rezidü embedding |
| Loss | BCE + Ranking | BCE + Ranking |
| Erken durdurma | Val MRR, patience=12 | Val MRR, patience=15 |

### 7. Çıktılar ve Görselleştirme

Model çıktıları üç katmanda raporlanır:

- **Metrik tablosu** — AUROC, AUPRC, F1, MCC, MRR, Hit@k gibi 8 temel metrik.
- **2D peptid görselleri** — RDKit ile çizilen 5 farklı peptit: en yüksek/düşük skorlu, orta skorlu, uzun ve kısa zincirlerden. Her görselde PDB ID, skor ve uzunluk bilgisi yer alır; tıklanınca lightbox ile büyütülür.
- **3D yapı önizleme** — 3Dmol.js ile gömülü interaktif 3D viewer. Demo sayfasında 1CRN (Crambin) kristal yapısı gösterilir; tam pipeline çıktıları local ortamda üretilir.
- **ROC / PR eğrileri, skor histogramları, confusion matrix** — eğitim sonuçlarının detaylı grafikleri.

---

## Sonuçlar

### Test Metrikleri (v0.1 — MLP, 68 epoch)

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **AUROC** | 0.8388 | İkili sınıflandırma ayrım gücü |
| **AUPRC** | 0.4348 | Dengesiz sınıf altında hassasiyet-duyarlılık dengesi |
| **F1** | 0.5074 | Eşik-bazlı F1 skoru (threshold = 0.59) |
| **MCC** | 0.4134 | Matthews korelasyonu — sınıf dengesizliğine dayanıklı |
| **MRR** | 0.7120 | Ortalama ters sıralama (Mean Reciprocal Rank) |
| **Hit@1** | 0.5121 | Grupların %51'inde native peptit 1. sırada |
| **Hit@3** | 0.9275 | Grupların %93'ünde native peptit ilk 3'te |
| **Hit@5** | 0.9952 | Grupların %99.5'inde native peptit ilk 5'te |

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
PROPEDIA mmCIF (42,375 kompleks)
    │
    ▼
Kanonik tablolar (complexes · chains · residues → Parquet)
    │
    ▼
Arayüz + Pocket anotasyonu (5 Å / 8 Å mesafe tabanlı)
    │
    ▼
Sekans-küme split (MMseqs2 %30 kimlik → train / val / test)
    │
    ▼
Negatif çift üretimi (easy + hard; her grup için 6 aday)
    │
    ▼
Özellik ihracı (yapı + sekans + arayüz + yerel yoğunluk → dense vektör)
    │
    ▼
ESM-2 embedding çıkarımı (per-rezidü 320-d)
    │
    ▼
Rezidü-seviye graf inşası (8 Å komşuluk)
    │
    ▼
GATv2 eğitimi (BCE + pairwise ranking loss; val MRR ile erken durdurma)
    │
    ▼
Test metrikleri (AUROC, AUPRC, MRR, Hit@k, F1, MCC)
    │
    ▼
Rapor: ROC/PR eğrileri · 2D peptid görselleri · 3D viewer · statik site
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
python scripts/extract_esm2_embeddings.py --model esm2_t6_8M
python scripts/train_gnn_esm2.py --config configs/train_v0_2_gnn_esm2.yaml

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
│   ├── train_scoring_mlx.py       # MLP eğitim betiği
│   ├── build_pages_site.py        # Statik site üretici
│   └── sync_pages_training_bundle.py
│
├── configs/
│   ├── train_v0_1_scoring_mlx_m4.yaml       # Aktif eğitim konfigürasyonu
│   └── ablation_generated_mlx/              # Ablation deneyleri
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

Aktif eğitim konfigürasyonu `configs/train_v0_1_scoring_mlx_m4.yaml` dosyasındadır. Temel parametreler:

```yaml
model:
  hidden_dim: 192
  num_layers: 3
training:
  seed: 42
  batch_size: 512
  epochs: 100
  early_stopping_patience: 12
  lr: 0.001
loss:
  use_ranking: true
  margin: 0.2
  bce_alpha: 0.5
evaluation:
  monitor_metric: val_mrr
  threshold_selection_metric: mcc
```

Ablation deneyleri `configs/ablation_generated_mlx/` dizinindeki YAML dosyaları ile yürütülür (model boyutu: S/M/L, özellik kombinasyonları, dropout oranları).

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
