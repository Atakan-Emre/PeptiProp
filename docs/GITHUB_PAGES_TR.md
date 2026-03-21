# GitHub Pages yayını

## Ne yayınlanır?

Workflow `.github/workflows/pages.yml` her `main` (veya `master`) push’ında:

1. `python scripts/build_pages_site.py` çalıştırır
2. `site/` klasörünü GitHub Pages’e yükler

İçerik:

- `index.html` — yan menülü düzen; dört metrik kartı `manifest.json` → `metrics` (kaynak: `training_dir`/`metrics.json`); **eğitim PNG** bölümü çıktı klasöründe varsa ROC/PR, kalibrasyon vb. kopyalar
- `embed/viewer-demo.html` — **3Dmol.js** ile tarayıcıda **1CRN** (mmCIF, `assets/demo/1crn.cif`)
- `data/manifest.json` — makine-okur özet
- `assets/css/site.css` — gündüz/gece CSS değişkenleri, responsive düzen
- `assets/js/site-theme.js` — tema anahtarı (`pq-site-theme`); ana sayfa ve 3D demo ortak kullanır

> `site/` `.gitignore` içindedir; **yayınlanan dosyalar CI ürünüdür**, repoya commit edilmesi gerekmez.

## Repo ayarları

1. GitHub → **Settings → Pages**
2. **Build and deployment**: kaynak **GitHub Actions**
3. İlk çalıştırmadan sonra environment **github-pages** için onay istenebilir; onaylayın.

## Project Pages (alt yol)

Depo `username.github.io/PeptidQuantum/` altında ise demo yolları göreli kalır (`../assets/...`). Başka bir base path kullanıyorsanız `scripts/build_pages_site.py` içindeki `fetch('../assets/demo/1crn.cif')` yolunu güncelleyin.

## Yerel önizleme

```bash
python scripts/build_pages_site.py
cd site && python -m http.server 8080
# http://localhost:8080/embed/viewer-demo.html
```

## Görseller eksikse

`ablation_heatmap.png` ve benzeri dosyalar yalnızca yerelde ilgili eğitim çıktı klasörü varsa kopyalanır. Önce eğitimi çalıştırıp ardından site script’ini tekrar çalıştırın.
