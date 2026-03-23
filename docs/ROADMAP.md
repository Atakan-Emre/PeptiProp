# PeptiProp Roadmap

## Current Status (v0.2 — GNN+ESM-2)

Proje, PROPEDIA tabanlı leakage-free veri üzerinde GATv2 + ESM-2 dual-encoder mimarisi ile eğitilmiştir.

- **Veri**: PROPEDIA (42,375 kompleks, ~12.3M rezidü)
- **Split**: Sekans-küme bazlı (MMseqs2 %30 kimlik)
- **Model**: GATv2 (4 katman, 4 head) + ESM-2 per-rezidü embedding
- **Eğitim**: PyTorch + PyG, CPU üzerinde 80 epoch
- **Test AUROC**: 0.8813, **MRR**: 0.7776, **Hit@3**: 0.9469
- **Web**: GitHub Pages + 3Dmol.js + RDKit 2D görseller

## Completed Phases

### Phase A — Data Integrity
- Canonical dataset (`data/canonical/`)
- Leakage-free split (MMseqs2 %30 kimlik)
- Clean-only kalite filtresi
- Easy + hard negatif çift üretimi

### Phase B — MLP Baseline (v0.1)
- 3-katman MLP, Apple MLX ile eğitim
- Test AUROC: 0.8388, MRR: 0.7120

### Phase C — GNN+ESM-2 (v0.2)
- ESM-2 t6-8M per-rezidü embedding çıkarımı (27,303 sekans)
- Rezidü-seviye graf inşası (8 Å cutoff, 55,958 graf)
- GATv2 dual-encoder eğitimi (554K parametre, 80 epoch)
- Tüm metriklerde MLP baseline'ı geçti

### Phase D — Visualization & Site
- GitHub Pages statik site (ablation tablosu, metrikler, pipeline)
- 2D peptit görselleri (RDKit, lightbox)
- 3D yapı viewer (3Dmol.js)
- ROC/PR/confusion matrix grafikleri

## Potential Future Work

| Alan | Açıklama |
|------|----------|
| ESM-2 büyük model | t12-35M veya t33-650M ile embedding kalitesi artışı |
| Cross-attention | Protein–peptid arası dikkat mekanizması |
| GPU eğitim | CUDA ile daha fazla epoch ve hiperparametre araması |
| Ensemble | MLP + GNN ensemble skorlama |
| Dış veri | PepBDB, BioLip2 ile eğitim setini genişletme |
| Binding affinity | Binary sınıflandırmadan sürekli skor tahmine geçiş |
