# Model İyileştirmeleri: MLP → GNN+ESM-2

> Aktif model: GATv2 + ESM-2 dual-encoder (v0.2) — [README.md](../README.md).

## Motivasyon

MLP baseline (v0.1) protein–peptid çiftlerini özet istatistiklerle (131-d dense vektör) temsil eder. Bu yaklaşım rezidü-seviye yapısal bilgiyi kaybeder. GNN+ESM-2 mimarisi, her rezidüyü ayrı bir graf düğümü olarak temsil ederek atomik detay seviyesinde öğrenme sağlar.

## Mimari Karşılaştırma

### v0.1 — MLP Baseline

- **Girdi**: Protein ve peptid için özet istatistikler (uzunluk, AA bileşimi, arayüz/pocket oranları, kosinus benzerliği vb.) → 131-d vektör
- **Model**: 3-katman feedforward MLP (SiLU aktivasyon, 192 hidden)
- **Parametre**: ~50K
- **Framework**: Apple MLX

### v0.2 — GATv2 + ESM-2 (Aktif)

- **Girdi**: Per-rezidü ESM-2 embedding (320-d) + yapısal özellikler (6-d) → rezidü grafı
- **Model**: Çift kanallı GATv2 encoder (4 katman, 4 attention head, 128 hidden) + Attention Pooling + MLP head
- **Parametre**: ~554K
- **Framework**: PyTorch + PyTorch Geometric

### Mimari Farklar

| Özellik | MLP (v0.1) | GNN+ESM-2 (v0.2) |
|---------|-----------|-------------------|
| Protein temsili | Global özet | Per-rezidü embedding |
| Yapısal bilgi | Oran/ortlama | 3D uzaysal graf |
| Attention | Yok | Multi-head (GATv2) + pooling |
| Ön-eğitim | Yok | ESM-2 protein dil modeli |
| Mesafe bilgisi | L2/kosinus | Kenar mesafe + yön vektörü |

## Ablation Sonuçları

| Metrik | MLP v0.1 | GNN+ESM-2 v0.2 | Fark |
|--------|:--------:|:---------------:|:----:|
| AUROC | 0.8388 | **0.8813** | +4.3% |
| AUPRC | 0.4348 | **0.5566** | +12.2% |
| MRR | 0.7120 | **0.7776** | +6.6% |
| Hit@1 | 0.5121 | **0.6210** | +10.9% |
| Hit@3 | 0.9275 | **0.9469** | +1.9% |
| MCC | 0.4134 | **0.5037** | +9.0% |

## Neden GATv2?

1. **Attention mekanizması**: Hangi komşu rezidülerin etkileşim tahmini için önemli olduğunu öğrenir
2. **Residual connection**: Derin ağlarda gradient akışını korur
3. **Edge features**: Rezidüler arası mesafe ve yön bilgisini doğrudan kullanır
4. **Dual encoder**: Protein ve peptid ayrı kodlanır, etkileşim birleşim katmanında hesaplanır

## Neden ESM-2?

1. **Transfer learning**: Milyonlarca protein sekansı üzerinde ön-eğitimli
2. **Per-rezidü temsil**: Her aminoasit için bağlam-duyarlı 320-d vektör
3. **Evrimsel bilgi**: Sekans homolojisi ve yapısal motif bilgisini içerir
4. **Verimlilik**: t6-8M modeli hızlı çıkarım sağlar (27K sekans ~10 dk)

## Gelecek Adımlar

- Daha büyük ESM-2 modeli (t12-35M veya t33-650M) ile embedding kalitesi artışı
- Protein–peptid arası cross-attention mekanizması
- GPU ile daha uzun eğitim ve hiperparametre araması
- MLP + GNN ensemble skorlama
