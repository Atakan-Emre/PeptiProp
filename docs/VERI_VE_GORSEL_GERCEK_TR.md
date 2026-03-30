# Veri ve görseller: gerçeklik, halüsinasyon, “bağlanma oranı”

Bu not, **amacınızla uyumluluk** ve **2D/3D çıktıların ne ölçtüğü** sorusunu netleştirmek içindir; sayılar yerelde üretilmiş dosyalara dayanır.

## 1. Eğitim verisi gerçek mi, halüsinasyon var mı?

- **Kaynak:** Aktif hat **PROPEDIA** → `data/canonical/` (tek doğruluk kaynağı). `DATA_ARCHITECTURE.md` ile hizalı.
- **Çiftler:** `data/canonical/pairs/pair_data_report.json` gerçek sayımları içerir (örnek, bu repoda görülen özet):
  - **Train:** 117,252 çift (19,542 pozitif, 97,710 negatif), duplicate 0, hepsi `clean`.
  - **Val:** 23,910 çift · **Test:** 27,228 çift; sızıntı koruması: MMseqs2 sekans-küme bazlı split.
- **Sonuç:** Pozitif/negatif etiketler **kristal yapıdan türetilmiş kanonik şema** ile tanımlı; model çıktısı “uydurma PDB” değildir. Halüsinasyon riski başlıca **tahmin aşamasında** (yanlış sıralama/eşik), veri tablosunun kendisinde değil.

## 2. “Bağlanma oranı” derken ne kastedilmeli?

İki farklı kavram karıştırılmamalı:

| Ne | Anlamı | Örnek (bu repoda) |
|----|--------|-------------------|
| **Model performansı** | Pozitif/negatif veya sıralama başarısı | Test **AUROC ≈ 0.84**, **MRR ≈ 0.71**, **Hit@3 ≈ 0.93** (`metrics.json` özetleri) |
| **Etkileşim tespiti (3D çizim)** | Geometric residue-contact fallback ile çizilen temas/çizgi sayısı | Kompleks başına değişir; **“% kaç bağ bulundu”** tek cümlede anlamlı değildir (eşik ve geometri tanımı değişir) |

Yani **2D şekil** veya **3D silindir sayısı**, “model %X bağlanma buldu” anlamına gelmez; **görselleştirme veya geometri özetidir**.

## 3. 2D peptit görseli ne yapar, ne yapmaz?

- **Yapar:** RDKit ile sekansın **2D bağ çizimi** (sunum/rapor); başlıkta kompleks + zincir bilgisi.
- **Yapmaz:** Bağlanma olasılığı veya arayüz enerjisi **tahmin etmez**. Halüsinasyon “2D’de görünen bağ = gerçekte var” varsayımıdır — bu doğru değil.

## 4. 3D görsel ve etkileşim kaynağı

- Pipeline `interaction_provenance.json` yazar: final aktif hatta beklenen mod **geometric_fallback**’tır.
- External tool extractor denemeleri repoda arşiv/deneysel olarak bulunabilir; fakat **makale ve final aktif rapor yüzeyinde kullanılmazlar**.
- Bu repoda güncel GNN sanity toplu çıktıları (`outputs/analysis_propedia_*_gnn/**/data/interaction_provenance.json`) üzerinde özet:
  - **`tool_based_interaction_fraction` ortalaması: 0.0**
  - **`tools_succeeded`:** boş; kayıtlar **geometric_fallback** ile üretilmiş.

Bu yüzden final bilimsel yorumda, 3D çizgiler tool-annotated chemistry olarak değil, geometrik temas özeti olarak okunmalıdır.

## 5. Amaçla uyum

- **Amaç:** Leakage-free çiftler üzerinde skor + reranking + raporlama → veri mimarisi ve metrikler bu hedefle uyumlu.
- **PROJECT_OBJECTIVES_TR.md** içindeki GEPPRI / PeptGAINET maddeleri **tarihsel/plan** notu; aktif ürün PROPEDIA hattıdır (`README` ile uyumlu).

## 6. Önerilen ek 2–3 görsel (bağlantı + peptit dizilimi)

Zaten üretilen türler: `interaction_summary.png`, `contact_map.png`, `contact_map_by_type.png`, `peptide_2d.png` (`outputs/analysis_propedia*/**/figures/`).

Anlamlı ekler:

1. **Peptit uzunluk histogramı** — `complexes.parquet` içindeki peptit uzunluğu; veri örtümesi kontrolü.
2. **Etkileşim tipi çubuk grafiği** — birden fazla kompleks için `interaction_summary` özetinin birleşik paneli (jupyter veya mevcut `ContactMapPlotter.plot_interaction_summary` ile toplu klasör döngüsü).
3. **Tek sayfa “kart”** — PDB ID, peptit **tek harf sekansı**, pozitif çift skoru, temas sayısı (geometrik); makale ekine uygun.

**Otomasyon:** `python scripts/build_pages_site.py` çalıştırıldığında `peptidquantum.visualization.plots.site_extras` modülü (varsa veri ve `outputs/`) şunları üretir: `site/assets/img/peptide_length_histogram.png`, `site/assets/img/interaction_summary_panel.png`, `site/embed/complex-cards.html`; ana `index.html` içinde **Ek veri görselleri** bölümüne bağlanır. Veri veya çıktı klasörü yoksa bu adımlar sessizce atlanır.

## 7. Tek komutla provenance özeti (yerel)

```bash
python3 -c "
import json
from pathlib import Path
ps = list(Path('outputs').rglob('**/interaction_provenance.json'))
xs = [json.loads(p.read_text())['tool_based_interaction_fraction'] for p in ps]
print('N=', len(ps), 'mean tool_frac=', sum(xs)/len(xs) if xs else None)
"
```

---

*Özet: Veri tabanı tarafı kanıtlanabilir; 2D/3D görseller açıklayıcıdır, “bağlanma %” yerine model metrikleri ve provenance oranları kullanın.*
