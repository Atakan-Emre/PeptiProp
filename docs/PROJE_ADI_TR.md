# Proje adı: PeptiProp

## Neden?

- **PROPEDIA** + **peptit** bağlamını kısa ve okunur biçimde taşır.
- **PeptidQuantum** adı, kuantum hesaplama veya PennyLane başlığı olmadan okunduğunda yanıltıcı olabiliyor; aktif ürün hattı öncelikle **yapısal skorlama + reranking + 2D/3D rapor**.
- GitHub depo adı `PeptidQuantum` olarak kalabilir (URL değişmez); **görünen ürün adı** `PeptiProp` kullanılır.

## Teknik not

- Python paketi ve `src/peptidquantum/` dizin adı **değiştirilmedi** (import kırılmaz).
- Statik site ve `manifest.json` içindeki `project` alanı **PeptiProp** üretilir (`scripts/build_pages_site.py` → `PROJECT_DISPLAY_NAME`).

İsterseniz yalnızca görünen metni değiştirmek için `PROJECT_DISPLAY_NAME` sabitini düzenlemeniz yeterli.
