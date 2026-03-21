# GitHub Pages yayını

## Ne yayınlanır?

Workflow `.github/workflows/pages.yml` her `main` (veya `master`) push’ında:

1. `pip install -r scripts/requirements-pages.txt` (site için **matplotlib** — CI’da eksik eğitim PNG’lerine yer tutucu üretir)
2. `python scripts/build_pages_site.py` çalıştırır (`PEPTIPROP_SITE_URL` ile `canonical` / `og:url` meta; örn. `https://atakan-emre.github.io/PeptiProp`)
3. `site/` klasörünü GitHub Pages’e yükler

İçerik:

- `index.html` — tek sütun genişliğinde hizalı **site-shell**, üst çubukta GitHub + tema; **proje akışı** diyagramı (`#akim`); dört metrik kartı; eğitim PNG’leri (veya yer tutucu)
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

Canlı örnek: [PeptiProp — GitHub Pages](https://atakan-emre.github.io/PeptiProp/index.html). Depo `username.github.io/RepoAdi/` altında göreli yollar (`assets/...`, `../assets/...`) tarayıcıda doğru çözülür; 3D demo CIF yüklemesi `fetch(new URL('../assets/demo/1crn.cif', window.location.href))` ile sayfa adresine göre çözülür.

## Yerel önizleme

```bash
python scripts/build_pages_site.py
cd site && python -m http.server 8080
# http://localhost:8080/embed/viewer-demo.html
```

## Görseller eksikse

CI’da `outputs/training/` yoksa derleme **matplotlib** ile **yer tutucu PNG** üretir (kırık `<img>` olmaz). Gerçek ROC/ablation görselleri için yerelde eğitim/ablation sonrası `build_pages_site.py` çalıştırın.
