# PeptiProp Bilimsel Metod

Bu doküman, projenin bilimsel metodunu ve kanıt yüzeyini tek yerde toplar.
Genel özet: [README.md](../README.md), Pages: [GITHUB_PAGES_TR.md](GITHUB_PAGES_TR.md).

## 1) Problem Tanımı

Ana görev klasik binary sınıflandırma değil, **interaction scoring + candidate reranking** olarak tanımlanmıştır:

- Her protein–peptid çifti için `score ∈ [0, 1]`
- Aynı protein için aday peptidler arasında sıralama
- Ana değerlendirme: `MRR`, `Hit@1/3/5`
- Yardımcı değerlendirme: `AUROC`, `AUPRC`, `F1`, `MCC`

## 2) Veri Kaynağı ve Split Protokolü

Veri kaynağı yalnızca **PROPEDIA** tabanlı canonical corpus (42,375 kompleks, ~12.3M rezidü).

Split protokolü:

- Tüm protein sekansları **MMseqs2** ile %30 kimlik eşiğinde kümelenir
- Her küme bölünmez birim olarak tek split'e atanır (leakage-free)
- Dosya: `scripts/build_pdb_level_splits.py`
- Leakage check: **PASSED** (splitler arası PDB örtüşmesi = 0)

Split dağılımı:

| Split | Pozitif | Negatif (easy + hard) | Toplam |
|-------|---------|----------------------|--------|
| Train | 19,542 | 97,710 (68,397 + 29,313) | 117,252 |
| Val | 3,985 | 19,925 (15,940 + 3,985) | 23,910 |
| Test | 4,538 | 22,690 (18,152 + 4,538) | 27,228 |

## 3) Kalite Filtresi

Eğitim/veri üretimi sırasında yalnızca `quality_flag = clean` kompleksler kullanılır (`42,375` toplam kompleks içinden `28,065` clean kompleks). Düşük güvenilirlikli yapıların etiket gürültüsünü azaltır.

## 4) Negatif Çift Tasarımı

Her protein grubu için 1 pozitif (native co-crystal) + 4 easy + 1 hard = 6 aday:

| Tür | Açıklama | Zorluk |
|-----|----------|--------|
| Easy | Tamamen rastgele peptid eşleşmesi | Düşük |
| Hard | Aynı split içinde, benzer protein sequence/bucket havuzundan farklı peptid | Yüksek |

## 5) Model Mimarisi: GNN + ESM-2 (v0.2)

### 5.1 Girdi Temsili

Her rezidü bir graf düğümü olarak temsil edilir:

- **ESM-2 embedding** (320-d): `esm2_t6_8M_UR50D` modeli ile per-rezidü çıkarım
- **Yapısal özellikler** (6-d): is_interface, is_pocket, ikincil yapı one-hot (helix/sheet/coil/unknown)
- **Kenarlar**: Cα atomları arası 8 Å mesafe eşiği ile komşuluk grafı
- **Kenar özellikleri** (4-d): Öklidyen mesafe + normalize yön vektörü

### 5.2 Mimari

Çift kanallı (dual-encoder) GATv2 mimarisi:

1. **Protein encoder**: 4-katman GATv2Conv (4 head, 128 hidden) + residual + LayerNorm → Attention Pooling → protein_vec
2. **Peptid encoder**: Aynı mimari (ayrı ağırlıklar) → peptide_vec
3. **Etkileşim hesabı**: [concat; hadamard product; absolute difference] → 512-d
4. **MLP head**: 3-katman (256 → 128 → 1) + SiLU → sigmoid → skor

Toplam parametre: ~554K

### 5.3 Eğitim

- **Loss**: BCE (α=0.5) + Pairwise Ranking Loss (margin=0.2)
- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-4)
- **Erken durdurma**: Val MRR, patience=15
- **Eşik seçimi**: MCC-optimal threshold (0.21)
- **Batch size**: 32
- **Epoch**: 80 (erken durdurma tetiklenmedi)

## 6) MLP Baseline (v0.1)

Karşılaştırma amacıyla 3-katmanlı MLP baseline (Apple MLX ile eğitildi):

- Girdi: 131-d dense vektör (özet istatistikler)
- Parametre: ~50K
- Epoch: 68

## 7) Ablation Sonuçları

| Metrik | MLP v0.1 | GNN+ESM-2 v0.2 | Fark |
|--------|:--------:|:---------------:|:----:|
| AUROC | 0.8388 | **0.8813** | +0.0425 |
| AUPRC | 0.4348 | **0.5566** | +0.1218 |
| F1 | 0.5074 | **0.5884** | +0.0810 |
| MCC | 0.4134 | **0.5037** | +0.0903 |
| MRR | 0.7120 | **0.7776** | +0.0656 |
| Hit@1 | 0.5121 | **0.6210** | +0.1089 |
| Hit@3 | 0.9275 | **0.9469** | +0.0194 |
| Hit@5 | 0.9952 | **0.9965** | +0.0013 |

GNN+ESM-2, rezidü-seviye yapısal bilgiyi kullanarak tüm metriklerde MLP baseline'ı geçti. En büyük kazanım AUPRC ve Hit@1 metriklerinde.

## 8) Bilimsel Savunulabilirlik

- PROPEDIA-only kapsam net ve tekrar üretilebilir
- Sekans kümeleme ile leakage koruması kanıtlanmış
- Negatif tasarım (easy + hard) problem zorluğunu kontrol altında tutar
- ESM-2 ön-eğitimli temsil kullanımı mevcut literatürle uyumlu
- GATv2 attention mekanizması hangi rezidülerin önemli olduğunu öğrenebilir
- Ablation karşılaştırması (MLP vs GNN) mimari katkıyı izole eder
- MRR/Hit@K metrikleri sıralama problemine uygun
- Final raporlanan 2D/3D sanity çıktıları, geometric residue-contact fallback ile üretilir; external tool extractor çıktıları final metrik/makale yüzeyine dahil edilmez

## 9) Sınırlar

- Visualization tarafında 3D temas çizgileri tool-annotated chemistry değil, geometric residue-contact özetidir
- ESM-2 t6-8M (en küçük model) kullanıldı; daha büyük modeller potansiyel iyileşme sağlayabilir
- Cross-attention (protein–peptid arası dikkat) henüz uygulanmadı
- Eğitim CPU üzerinde gerçekleştirildi; GPU/MPS ile daha fazla epoch/hiperparametre araması mümkün
