# PeptidQuantum v0.1 Bilimsel Metod (PROPEDIA-only, Leakage-free)

Bu doküman, mevcut aktif hattın bilimsel metodunu ve kanıt yüzeyini tek yerde toplar.  
Kapsam yalnızca **PROPEDIA-only** hattıdır; legacy/dış dataset eğitim hattına dahil değildir.  
Genel özet ve Pages: [README.md](../README.md), [GITHUB_PAGES_TR.md](GITHUB_PAGES_TR.md).

## 1) Problem Tanımı

Ana görev klasik binary sınıflandırma değil, **interaction scoring + candidate reranking** olarak tanımlanmıştır:

- Her protein-peptid çifti için `score in [0, 1]`
- Aynı protein için aday peptidler arasında sıralama
- Ana değerlendirme: `MRR`, `Hit@1/3/5`
- Yardımcı değerlendirme: `AUROC`, `AUPRC`, `F1`, `MCC`, calibration metrikleri

## 2) Veri Kaynağı ve Split Protokolü

Veri kaynağı yalnızca PROPEDIA tabanlı canonical corpus.

Split adımı:

- PDB-level structure-aware split
- Dosya: `scripts/build_pdb_level_splits.py`
- Kanıt logu: `outputs/logs/final_ablation_20260319_112457/01_split.log`

Gözlenen split sayıları:

- Total complexes: `41572`
- Unique PDB IDs: `13386`
- Train complexes: `28986`
- Val complexes: `6527`
- Test complexes: `6059`
- Leakage check: `PASSED (no PDB overlap between splits)`

## 3) Kalite Filtresi (clean-only)

Eğitim/veri üretimi sırasında yalnızca `quality_flag = clean` kompleksler kullanılır.

- Kanıt logu: `outputs/logs/final_ablation_20260319_112457/02_pairs.log`
- Uygulama sonucu: `41572 -> 27387` kompleks

Bilimsel gerekçe:

- Düşük güvenilirlikli yapıların etiket/girdi gürültüsünü azaltır
- İç geçerliliği ve tekrar üretilebilirliği artırır

## 4) Candidate-set Üretimi ve Negatif Tasarımı

Aday set tasarımı:

- Her protein için `1 pozitif + 5 negatif` (candidate size = 6)
- Train/val/test için split sonrası üretim
- Duplicate pair = 0

Dosyalar:

- `data/canonical/pairs/pair_data_report.json`
- `data/canonical/pairs/candidate_set_report.json`

Doğrulanan dağılımlar:

- Train: `117252` (pos `19542`, neg `97710` — 68,397 easy + 29,313 hard)
- Val: `23910` (pos `3985`, neg `19925` — 15,940 easy + 3,985 hard)
- Test: `27228` (pos `4538`, neg `22690` — 18,152 easy + 4,538 hard)
- Tüm splitlerde `split_column_consistent = true`, `duplicate_pair_count = 0`

Negatif tip oranı (mevcut koşu):

- Hedef: train/val/test için `easy + hard` ağırlıklı negatif dağılımı
- Gerçekleşen:
  - Train hard shortfall ~ `0.0` (hedefe çok yakın)
  - Val hard shortfall ~ `0.0886`
  - Test hard shortfall ~ `0.1054`

Not: Val/test’te hard örnek havuzu sınırlı olduğunda shortfall raporlanır; sessizce gizlenmez.

## 5) Leakage Koruması

Aktif olarak korunan iki kritik leakage yasağı:

1. Pair feature’larda native bound peptide’den türetilmiş bilgi kullanılmaması  
2. Negatif örneklerde native pocket’a göre crop yapılmaması

Otomatik test:

- Dosya: `tests/test_baseline_leakage_guards.py`
- Kanıt logu: `outputs/logs/final_ablation_20260319_112457/03_leakage_tests.log`
- Sonuç: `Ran 2 tests ... OK`

## 6) Model ve Eğitim Protokolü

Model ailesi (ablation):

- `MPNN`
- `GATv2`
- `GIN`

Eğitim yaklaşımı:

- 200 epoch üst sınırı
- early stopping aktif
- balanced sampler aktif
- `pos_weight` otomatik (dengesizlik için)
- threshold seçimi: validation MCC öncelikli

İlgili script/config:

- `scripts/run_classical_ablation.py`
- `scripts/train_scoring_model.py`
- `configs/ablation_generated/*.yaml`

## 7) Tamamlanan Full Run Kanıtları (Snapshot)

### 7.1 GIN full (en güçlü tamamlanan run)

Klasör: `outputs/training/ablation_gin_f1_l1_c0_full`

Kanıt dosyaları:

- `metrics.json`
- `ranking_metrics.json`
- `calibration_metrics.json`
- `test_summary.txt`

Ana sonuçlar:

- Val AUROC: `0.7236`
- Val MRR: `0.6095`
- Val Hit@3: `0.7546`
- Test AUROC: `0.7365`
- Test AUPRC: `0.3506`
- Test F1: `0.4065`
- Test MCC: `0.2713`
- Test MRR: `0.6109`
- Test Hit@1/3/5: `0.3975 / 0.7723 / 0.9487`
- Test Brier: `0.2236`
- Test score range: `0.8078` (dar banda çökme yok)

### 7.2 GATv2 full (karşılaştırma)

Klasör: `outputs/training/ablation_gatv2_f2_l2_c0_full`

Ana sonuçlar:

- Test AUROC: `0.5320`
- Test MCC: `-0.0045`
- Test MRR: `0.4335`
- Test Brier: `0.2960`
- Test score range: `0.0300` (skor sıkışması belirgin)

Yorum: Model ailesi farkı güçlü; GIN bu kurulumda belirgin daha iyi.

## 8) Devam Eden Run Durumu (MPNN)

Aktif klasör: `outputs/training/ablation_mpnn_f2_l2_c0_full`

Şu anki kayıtlı ilerleme (train_log snapshot):

- Son kayıtlı epoch: `10`
- Val AUROC: `0.6332`
- Val MRR: `0.5179`

Not: Bu run tamamlanınca nihai model seçimi tekrar yapılmalıdır.

## 9) Bilimsel Savunulabilirlik Özeti

Bu setup aşağıdaki nedenle hakem karşısında savunulabilir:

- PROPEDIA-only kapsam net
- Split leakage kontrolü açık ve geçmiş
- Leakage guard testleri otomatik ve geçmiş
- Clean-only protokolü şeffaf
- Candidate-set/reranking problemi açık tanımlı
- Ana metrikler problem tipi ile uyumlu (MRR/Hit@K)
- Calibration raporları üretiliyor
- Zayıf noktalar (hard shortfall, model farkı) açık raporlanıyor

## 10) Açık Sınırlar

- Val/test hard-negative shortfall tam kapanmış değil
- Bazı koşul/ailerde skor sıkışması görülebiliyor
- MPNN final run henüz tamamlanmadan tek nihai seçim kilitlenmemeli

---

## Kanıt Dosyaları Hızlı Liste

- `outputs/logs/final_ablation_20260319_112457/01_split.log`
- `outputs/logs/final_ablation_20260319_112457/02_pairs.log`
- `outputs/logs/final_ablation_20260319_112457/03_leakage_tests.log`
- `data/canonical/pairs/pair_data_report.json`
- `data/canonical/pairs/candidate_set_report.json`
- `outputs/training/ablation_gin_f1_l1_c0_full/metrics.json`
- `outputs/training/ablation_gin_f1_l1_c0_full/calibration_metrics.json`
- `outputs/training/ablation_gatv2_f2_l2_c0_full/metrics.json`
- `outputs/training/ablation_mpnn_f2_l2_c0_full/train_log.csv`
