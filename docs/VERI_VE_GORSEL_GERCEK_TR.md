# Veri ve görseller: gerçeklik, halüsinasyon, “bağlanma oranı”

Bu not, **amacınızla uyumluluk** ve **2D/3D çıktıların ne ölçtüğü** sorusunu netleştirmek içindir; sayılar yerelde üretilmiş dosyalara dayanır.

## 1. Eğitim verisi gerçek mi, halüsinasyon var mı?

- **Kaynak:** Aktif hat **PROPEDIA** → `data/canonical/` (tek doğruluk kaynağı). `DATA_ARCHITECTURE.md` ile hizalı.
- **Çiftler:** `data/canonical/pairs/pair_data_report.json` gerçek sayımları içerir (örnek, bu repoda görülen özet):
  - **Train:** 117 738 çift (19 623 pozitif, 98 115 negatif), duplicate 0, hepsi `clean`.
  - **Val / test:** benzer şekilde raporlanmış; sızıntı politikası dokümante.
- **Sonuç:** Pozitif/negatif etiketler **kristal yapıdan türetilmiş kanonik şema** ile tanımlı; model çıktısı “uydurma PDB” değildir. Halüsinasyon riski başlıca **tahmin aşamasında** (yanlış sıralama/eşik), veri tablosunun kendisinde değil.

## 2. “Bağlanma oranı” derken ne kastedilmeli?

İki farklı kavram karıştırılmamalı:

| Ne | Anlamı | Örnek (bu repoda) |
|----|--------|-------------------|
| **Model performansı** | Pozitif/negatif veya sıralama başarısı | Test **AUROC ≈ 0.79**, **MRR ≈ 0.68** (`metrics.json` özetleri) |
| **Etkileşim tespiti (3D çizim)** | PLIP/Arpeggio veya geometrik fallback ile çizilen temas/çizgi sayısı | Kompleks başına değişir; **“% kaç bağ bulundu”** tek cümlede anlamlı değildir (eşik, mesafe, kaynak aracı değişir) |

Yani **2D şekil** veya **3D silindir sayısı**, “model %X bağlanma buldu” anlamına gelmez; **görselleştirme veya geometri özetidir**.

## 3. 2D peptit görseli ne yapar, ne yapmaz?

- **Yapar:** RDKit ile sekansın **2D bağ çizimi** (sunum/rapor); başlıkta kompleks + zincir bilgisi.
- **Yapmaz:** Bağlanma olasılığı veya arayüz enerjisi **tahmin etmez**. Halüsinasyon “2D’de görünen bağ = gerçekte var” varsayımıdır — bu doğru değil.

## 4. 3D görsel ve etkileşim kaynağı

- Pipeline `interaction_provenance.json` yazar: **PLIP / Arpeggio / geometric_fallback** oranları.
- **PLIP / Arpeggio neden “kullanılmamış” görünür?** `pipeline.py` içinde `_extract_interactions` yalnızca her aracın `is_available()` True olduğunda (komut `PATH`’te ve genelde `--help` başarılı) çalıştırır; aksi halde bu adımlar **atlanır** ve `interaction_sets` boş kalırsa **geometrik fallback** devreye girer. Kurulum doğrulaması: `scripts/verify_external_tools.py`, ayrıntı: `EXTERNAL_TOOLS.md`.
- Bu repoda **örnek sanity toplu çıktıları** (`outputs/analysis_propedia_*_mlx/**/data/interaction_provenance.json`, N≈20) üzerinde özet:
  - **`tool_based_interaction_fraction` ortalaması: 0.0**
  - **`tools_succeeded`:** boş; kayıtlar **geometric_fallback** ile üretilmiş.

Bu, “araçlar kurulu değil / çalışmadı / CIF-PDB yolu uyumsuz” gibi nedenlerle olabilir; **3D çizim yine de yapı + geometrik mesafe kurallarıyla tutarlı** olabilir, fakat “PLIP onaylı bağ” iddiası bu örneklerde geçerli değildir. Üretimde PLIP’in dolması için `EXTERNAL_TOOLS.md` ve `run_visualization_sanity.py` ortamını doğrulamak gerekir.

## 5. Amaçla uyum

- **Amaç:** Leakage-free çiftler üzerinde skor + reranking + raporlama → veri mimarisi ve metrikler bu hedefle uyumlu.
- **PROJECT_OBJECTIVES_TR.md** içindeki GEPPRI / PeptGAINET maddeleri **tarihsel/plan** notu; aktif ürün PROPEDIA hattıdır (`README` ile uyumlu).

## 6. Önerilen ek 2–3 görsel (bağlantı + peptit dizilimi)

Zaten üretilen türler: `interaction_summary.png`, `contact_map.png`, `contact_map_by_type.png`, `peptide_2d.png` (`outputs/analysis_propedia*/**/figures/`).

Anlamlı ekler:

1. **Peptit uzunluk histogramı** — `complexes.parquet` içindeki peptit uzunluğu; veri örtümesi kontrolü.
2. **Etkileşim tipi çubuk grafiği** — birden fazla kompleks için `interaction_summary` özetinin birleşik paneli (jupyter veya mevcut `ContactMapPlotter.plot_interaction_summary` ile toplu klasör döngüsü).
3. **Tek sayfa “kart”** — PDB ID, peptit **tek harf sekansı**, pozitif çift skoru, temas sayısı (geometrik veya PLIP); makale ekine uygun.

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
