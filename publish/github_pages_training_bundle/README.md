# GitHub Pages — eğitim özeti (bundle)

Bu klasör **bilerek repoda tutulur**: [GitHub Actions](https://github.com/Atakan-Emre/PeptiProp/actions) `outputs/training/` göremediği için `scripts/build_pages_site.py` buradaki `metrics.json` ve PNG’leri kullanır.

## Güncelleme

Yerelde tamamlanmış bir eğitim klasörünüz varsa:

```bash
python scripts/sync_pages_training_bundle.py
# veya özel klasör:
python scripts/sync_pages_training_bundle.py outputs/training/baska_kosum
git add publish/github_pages_training_bundle
git commit -m "Pages: eğitim bundle güncelle"
git push
```

Sonraki Pages deploy’da [site](https://atakan-emre.github.io/PeptiProp/index.html) metrik kartları ve eğitim görselleri gerçek dosyalarla dolar.

## Öncelik

Yerelde `outputs/training/...` varsa derleme önce onu kullanır; yalnızca CI’da (veya outputs silinmiş makinede) bu bundle devreye girer.
