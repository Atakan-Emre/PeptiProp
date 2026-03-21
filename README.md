# PeptiProp

**Yapısal protein–peptid komplekslerinde skor üretimi, aday peptitler arasında sıralama (reranking) ve sonuçların 2D kimya + 3D yapı ile raporlanması.**

Kod kökü: Python paketi `peptidquantum` (`src/peptidquantum/`).

## Canlı Demo

**GitHub Pages:** [atakan-emre.github.io/PeptiProp](https://atakan-emre.github.io/PeptiProp/)

Statik site 8 metrik kartı, 5 adet farklı skor/uzunlukta 2D peptit görseli, gömülü 3D yapı önizlemesi, detaylı pipeline diyagramı ve top-ranked tahmin tablosu içerir. `python scripts/build_pages_site.py` ile yerelde üretilir; GitHub Actions ile otomatik yayınlanır.

## Proje Özeti

| Öğe | Açıklama |
|-----|----------|
| Veri kaynağı | PROPEDIA — yapısal protein–peptid kompleksleri (mmCIF) |
| Kanonik veri | 42,375 kompleks · 84,750 zincir · 12.3M rezidü |
| Anotasyon | Arayüz (5Å) ve pocket (8Å) mesafe tabanlı anotasyon |
| Split stratejisi | MMseqs2 ile %30 kimlik eşiğinde sekans kümeleme; küme-bazlı train/val/test |
| Negatif üretimi | Easy (rastgele peptit) + Hard (aynı protein ailesi) |
| Özellik vektörü | Yapı + sekans + arayüz + yerel yoğunluk → dense vektör |
| Model | MLX MLP; BCE + pairwise ranking loss; validasyon MRR ile erken durdurma |
| Görselleştirme | RDKit 2D peptit · 3Dmol.js 3D viewer · HTML rapor |

## Son Eğitim Sonuçları

| Metrik | Test | Açıklama |
|--------|------|----------|
| AUROC | 0.8388 | İkili sınıflandırma gücü |
| AUPRC | 0.4348 | Dengesiz sınıf performansı |
| MRR | 0.7120 | Sıralama başarısı (ortalama ters sıra) |
| Hit@1 | 0.5121 | İlk sırada doğru pozitif |
| Hit@3 | 0.9275 | İlk 3'te doğru pozitif |
| Hit@5 | 0.9952 | İlk 5'te doğru pozitif |
| F1 | 0.5074 | Eşik-bazlı F1 skoru |
| MCC | 0.4134 | Matthews korelasyonu |

**Split istatistikleri:**

| Split | PDB | Pozitif | Negatif (easy + hard) | Toplam |
|-------|-----|---------|----------------------|--------|
| Train | 29,629 | 19,542 | 97,710 (68,397 + 29,313) | 117,252 |
| Val | 5,969 | 3,985 | 19,925 (15,940 + 3,985) | 23,910 |
| Test | 6,777 | 4,538 | 22,690 (18,152 + 4,538) | 27,228 |

## Pipeline

```
PROPEDIA mmCIF
    ↓
Kanonik tablolar (complexes, chains, residues Parquet)
    ↓
Arayüz + Pocket anotasyonu (5Å / 8Å mesafe)
    ↓
Sekans-küme split (MMseqs2 %30 kimlik → train / val / test)
    ↓
Negatif çift üretimi (easy + hard)
    ↓
Özellik ihracı (yapı + sekans + arayüz + yerel yoğunluk → dense vektör)
    ↓
MLX MLP eğitimi (BCE + pairwise ranking loss; val MRR ile erken durdurma)
    ↓
Test metrikleri (AUROC, MRR, Hit@k, F1, MCC)
    ↓
Rapor: ROC/PR eğrileri, 2D peptit, 3D viewer, statik site
```

## Hızlı Başlangıç

```bash
# 1. Ortam
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Kanonik veri oluşturma
python scripts/build_canonical_dataset.py --source data/propedia --output data/canonical

# 3. Arayüz/pocket anotasyonu
python scripts/annotate_interface_pocket.py --canonical data/canonical

# 4. Sekans-küme split
python scripts/build_pdb_level_splits.py --canonical data/canonical

# 5. Negatif çift üretimi
python scripts/generate_negative_pairs.py \
  --canonical data/canonical --splits data/canonical/splits \
  --output data/canonical/pairs

# 6. Özellik ihracı ve eğitim
python scripts/export_mlx_features.py --config configs/train_v0_1_scoring_mlx_m4.yaml
python scripts/train_scoring_mlx.py --config configs/train_v0_1_scoring_mlx_m4.yaml

# 7. Site üretimi
pip install -r scripts/requirements-pages.txt
python scripts/build_pages_site.py
```

## Proje Yapısı

```
PeptiProp/
├── src/peptidquantum/          # Ana Python paketi
│   ├── data/processors/        # Kanonik veri oluşturucu
│   ├── visualization/          # 2D/3D görselleştirme
│   │   ├── chemistry/          # RDKit 2D peptit renderer
│   │   └── plots/              # Site görselleri, site_extras
│   └── analysis/               # Pipeline ve analiz
├── scripts/                    # Pipeline betikleri
│   ├── build_canonical_dataset.py
│   ├── annotate_interface_pocket.py
│   ├── build_pdb_level_splits.py
│   ├── generate_negative_pairs.py
│   ├── export_mlx_features.py
│   ├── train_scoring_mlx.py
│   └── build_pages_site.py
├── configs/                    # Eğitim konfigürasyonları
├── data/canonical/             # Kanonik tablolar + splits + pairs
├── outputs/training/           # Eğitim çıktıları (repo dışı)
├── publish/github_pages_training_bundle/  # Pages için metrik/görsel paketi
├── site/                       # Üretilen statik site
├── tests/                      # Unit testler
└── docs/                       # Ek dokümantasyon (TR)
```

## GitHub Pages Yayını

Eğitim sonrası metrikleri ve görselleri Pages'a aktarmak için:

```bash
python scripts/sync_pages_training_bundle.py outputs/training/<egitim_klasoru>
git add publish/github_pages_training_bundle && git commit -m "Pages: eğitim bundle güncelle"
git push
```

CI pipeline otomatik olarak `build_pages_site.py` çalıştırır ve siteyi yayınlar. Bundle yoksa yer tutucu PNG üretilir.

## Testler

```bash
python -m pytest tests/ -v
python -m unittest discover -s tests -v
```

## Dokümantasyon

| Dosya | İçerik |
|-------|--------|
| [QUICKSTART.md](QUICKSTART.md) | Hızlı başlangıç |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Ortam kurulumu |
| [DATA_ARCHITECTURE.md](DATA_ARCHITECTURE.md) | Veri şeması |
| [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) | Proje mimarisi |
| [EXTERNAL_TOOLS.md](EXTERNAL_TOOLS.md) | Arpeggio, PLIP, Open Babel |
| [VALIDATION.md](VALIDATION.md) | Doğrulama rehberi |
| [docs/GITHUB_PAGES_TR.md](docs/GITHUB_PAGES_TR.md) | Pages yayın detayları |
| [docs/POZITIF_PEPTIT_VE_SKOR_TR.md](docs/POZITIF_PEPTIT_VE_SKOR_TR.md) | Etiket 1 (native) vs model skoru |
| [docs/SCIENTIFIC_METHOD_TR.md](docs/SCIENTIFIC_METHOD_TR.md) | Bilimsel yöntem açıklaması |
| [docs/PROJE_ADI_TR.md](docs/PROJE_ADI_TR.md) | Ürün adı gerekçesi |
| [docs/VERI_VE_GORSEL_GERCEK_TR.md](docs/VERI_VE_GORSEL_GERCEK_TR.md) | Veri doğruluğu, 2D/3D anlamı |

## Lisans

Proje lisansı depo kökündeki LICENSE dosyasına tabidir.
