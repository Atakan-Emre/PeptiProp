#!/usr/bin/env python3
"""
GitHub Pages statik site üretir: manifest, görseller (varsa), 3D demo, index.

Çalıştır: python scripts/build_pages_site.py
Çıktı: site/ (workflow bu klasörü yayınlar)
"""
from __future__ import annotations

import html as html_module
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
SITE = ROOT / "site"
# Görünen ürün adı (Python paketi `peptidquantum` olarak kalır)
PROJECT_DISPLAY_NAME = "PeptiProp"
REPO_URL = "https://github.com/Atakan-Emre/PeptiProp"
MLX_BEST = ROOT / "outputs" / "training" / "peptidquantum_v0_1_final_mlx_m4"
# GitHub Actions’ta outputs/ yok; metrik + PNG’ler buradan kopyalanır (sync betiği ile güncellenir).
PAGES_TRAINING_BUNDLE = ROOT / "publish" / "github_pages_training_bundle"
DEMO_CIF_URL = "https://files.rcsb.org/download/1CRN.cif"


def _copy_if(src: Path, dest: Path) -> bool:
    if not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _first_glob(patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        for p in ROOT.glob(pat):
            if p.is_file():
                return p
    return None


def load_metrics(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_training_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """metrics.json hem düz (test_auroc) hem iç içe (test_metrics.auroc) anahtarları destekler."""
    if not raw:
        return {}
    tm = raw.get("test_metrics") if isinstance(raw.get("test_metrics"), dict) else {}
    tr = raw.get("test_ranking_metrics") if isinstance(raw.get("test_ranking_metrics"), dict) else {}

    def first(*vals: Any) -> Any:
        for v in vals:
            if v is not None:
                return v
        return None

    hit3 = first(raw.get("test_hit3"), tr.get("hit@3"), tr.get("hit_3"))

    cal = raw.get("calibration") if isinstance(raw.get("calibration"), dict) else {}
    return {
        "test_auroc": first(raw.get("test_auroc"), tm.get("auroc")),
        "test_auprc": first(raw.get("test_auprc"), tm.get("auprc")),
        "test_f1": first(raw.get("test_f1"), tm.get("f1")),
        "test_mcc": first(raw.get("test_mcc"), tm.get("mcc")),
        "test_brier": first(raw.get("test_brier"), cal.get("brier_score")),
        "test_mrr": first(raw.get("test_mrr"), tr.get("mrr")),
        "test_hit1": first(raw.get("test_hit1"), tr.get("hit@1"), tr.get("hit_1")),
        "test_hit3": hit3,
        "test_hit5": first(raw.get("test_hit5"), tr.get("hit@5"), tr.get("hit_5")),
        "epochs": raw.get("epochs_completed"),
        "threshold": raw.get("selected_threshold"),
        "train_groups": (raw.get("candidate_group_integrity", {}).get("train", {}) or {}).get("total_groups"),
        "val_groups": (raw.get("candidate_group_integrity", {}).get("val", {}) or {}).get("total_groups"),
        "test_groups": (raw.get("candidate_group_integrity", {}).get("test", {}) or {}).get("total_groups"),
    }


def _pick_training_dir(*name_tokens: str) -> Optional[Path]:
    """outputs/training altında metrics.json içeren klasör; tüm token'lar isimde (küçük harf) geçmeli."""
    td = ROOT / "outputs" / "training"
    if not td.is_dir():
        return None
    cands = [d for d in td.iterdir() if d.is_dir() and (d / "metrics.json").is_file()]
    if not cands:
        return None
    toks = tuple(t.lower() for t in name_tokens)
    matched = [d for d in cands if all(t in d.name.lower() for t in toks)]
    pool = matched if matched else []
    if not pool:
        return None
    return max(pool, key=lambda p: (p / "metrics.json").stat().st_mtime)


def _resolve_primary_training_dir() -> Optional[Path]:
    """Site görselleri / metrik özeti için tercih edilen eğitim çıktısı (klasör adında sıkça 'mlx' geçer; bu yalnızca keşif ipucu)."""
    if MLX_BEST.is_dir() and (MLX_BEST / "metrics.json").is_file():
        return MLX_BEST
    d = _pick_training_dir("mlx")
    if d:
        return d
    td = ROOT / "outputs" / "training"
    if td.is_dir():
        any_m = [x for x in td.iterdir() if x.is_dir() and (x / "metrics.json").is_file()]
        if any_m:
            return max(any_m, key=lambda p: (p / "metrics.json").stat().st_mtime)
    if PAGES_TRAINING_BUNDLE.is_dir() and (PAGES_TRAINING_BUNDLE / "metrics.json").is_file():
        return PAGES_TRAINING_BUNDLE
    return None


def download_demo_cif(dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(DEMO_CIF_URL, timeout=60) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as exc:
        print(f"[WARN] Demo CIF indirilemedi: {exc}")
        return False


def write_placeholder_png(path: Path, title: str, subtitle: str = "") -> bool:
    """Eğitim/ablation PNG yoksa (ör. GitHub Actions) kırık img önlemek için yer tutucu üretir."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=110)
        ax.set_facecolor("#161b22")
        fig.patch.set_facecolor("#0d1117")
        ax.text(0.5, 0.62, title, ha="center", va="center", color="#c9d1d9", fontsize=12)
        if subtitle:
            ax.text(0.5, 0.42, subtitle, ha="center", va="center", color="#8b949e", fontsize=9)
        ax.text(
            0.5,
            0.18,
            "CI’da outputs/training yoksa otomatik yer tutucu — yerelde eğitim sonrası build ile değişir.",
            ha="center",
            va="center",
            color="#58a6ff",
            fontsize=8,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.4, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path.is_file()
    except Exception as exc:
        print(f"[WARN] Yer tutucu PNG yazılamadı ({path}): {exc}")
        return False


def _load_metrics_bundle(dir_path: Path) -> Dict[str, Any]:
    """metrics.json + gerekirse ranking_metrics.json içindeki test sıralama blokları."""
    m = load_metrics(dir_path / "metrics.json")
    if not m:
        return {}
    if not m.get("test_ranking_metrics"):
        rnk = load_metrics(dir_path / "ranking_metrics.json")
        test_blk = rnk.get("test") if isinstance(rnk.get("test"), dict) else None
        if test_blk:
            m = {**m, "test_ranking_metrics": test_blk}
    return m


def build_manifest(training_dir: Optional[Path]) -> Dict[str, Any]:
    raw = _load_metrics_bundle(training_dir) if training_dir and (training_dir / "metrics.json").is_file() else {}
    m = _normalize_training_metrics(raw) if raw else {}
    manifest: Dict[str, Any] = {
        "project": PROJECT_DISPLAY_NAME,
        "version": "0.1",
        "dataset": "PROPEDIA canonical (leakage-free splits)",
        "training_dir": str(training_dir.relative_to(ROOT)) if training_dir and training_dir.exists() else None,
        "metrics": m if any(v is not None for v in m.values()) else None,
        "pages": {
            "viewer_demo": "embed/viewer-demo.html",
            "manifest": "data/manifest.json",
        },
    }
    return manifest


def _metrics_for_index(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Tek metrik satırı: yeni manifest `metrics`; eski anahtarlar için geriye dönük okuma."""
    m = manifest.get("metrics")
    if isinstance(m, dict) and any(v is not None for v in m.values()):
        return m
    for key in ("metrics_primary", "metrics_mlx", "metrics_secondary", "metrics_classical"):
        c = manifest.get(key)
        if isinstance(c, dict) and any(v is not None for v in c.values()):
            return c
    return {}


# (kaynak dosya adı, site altındaki dosya adı, img alt, figcaption)
_TRAINING_FIGURE_SPECS: List[tuple[str, str, str, str]] = [
    ("roc_curve.png", "roc_curve.png", "ROC eğrisi", "Test ROC — skorun pozitif/negatif ayrımı."),
    ("pr_curve.png", "pr_curve.png", "Precision–recall eğrisi", "Dengesiz sınıflarda AUROC’ye tamamlayıcı PR eğrisi."),
    ("calibration_curve.png", "calibration_curve.png", "Kalibrasyon eğrisi", "Tahmin olasılıklarının gözle uyumu (Brier ile birlikte yorumlanır)."),
    ("confusion_matrix.png", "confusion_matrix.png", "Karmaşıklık matrisi", "Seçilen eşikte TP/TN/FP/FN dağılımı."),
    ("validation_threshold_sweep.png", "validation_threshold_sweep.png", "Validasyon eşik taraması", "Eşik seçiminde metriklerin validasyon üzerindeki davranışı."),
    (
        "validation_score_histogram_pos_neg.png",
        "validation_score_histogram.png",
        "Validasyon skor histogramı",
        "Pozitif ve negatif örneklerin skor dağılımı (validasyon).",
    ),
    (
        "score_histogram_pos_neg.png",
        "test_score_histogram.png",
        "Test skor histogramı",
        "Test kümesinde pozitif/negatif skor dağılımı.",
    ),
]


def _copy_training_figures(training_out_dir: Optional[Path], img_root: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for src_name, dest_name, alt, caption in _TRAINING_FIGURE_SPECS:
        dest = img_root / dest_name
        src = training_out_dir / src_name if training_out_dir else None
        if src is not None and src.is_file():
            _copy_if(src, dest)
        elif not dest.is_file():
            write_placeholder_png(dest, alt, "Yerelde outputs/training içinden doldurulur")
        if dest.is_file():
            out.append({"href": f"assets/img/{dest_name}", "alt": alt, "caption": caption})
    return out


def _render_training_gallery_section(items: List[Dict[str, str]]) -> str:
    if not items:
        return """      <section class="block" id="egitim-gorselleri">
        <h2>Eğitim ve değerlendirme görselleri</h2>
        <p class="lead-in">
          ROC, PR, kalibrasyon, karmaşıklık matrisi ve skor histogramları, eğitim çıktı klasöründeki PNG dosyalarından otomatik kopyalanır.
          <strong>Şu an bu dosyalar bulunamadı</strong> (ör. GitHub Actions’ta <code>outputs/training/</code> yok veya henüz eğitim çalıştırılmadı).
          Yerelde eğitim tamamlayıp <code>python scripts/build_pages_site.py</code> ile siteyi yeniden üretin.
        </p>
      </section>"""
    cells = []
    for it in items:
        cells.append(
            f"""        <figure class="media">
          <img src="{html_module.escape(it["href"])}" alt="{html_module.escape(it["alt"])}" loading="lazy" />
          <figcaption>{html_module.escape(it["caption"])}</figcaption>
        </figure>"""
        )
    inner = "\n".join(cells)
    return f"""      <section class="block" id="egitim-gorselleri">
        <h2>Eğitim ve değerlendirme görselleri</h2>
        <p class="lead-in">
          Görseller mümkünse eğitim çıktı klasöründen (<code>metrics.json</code> ile aynı dizin) kopyalanır; GitHub Actions’ta klasör yoksa
          <strong>yer tutucu PNG</strong> üretilir (kırık resim ve boş URL görünmez). Gerçek eğriler için yerelde eğitim sonrası yeniden derleyin.
        </p>
        <div class="grid2 training-fig-grid">
{inner}
        </div>
      </section>"""


THEME_INLINE_HEAD = """  <script>
(function(){var k='pq-site-theme',s=localStorage.getItem(k),t=(s==='light'||s==='dark')?s:(window.matchMedia('(prefers-color-scheme: light)').matches?'light':'dark');document.documentElement.setAttribute('data-theme',t);})();
  </script>
"""


def write_theme_js(site: Path) -> None:
    js = site / "assets" / "js" / "site-theme.js"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(
        """(function () {
  var KEY = 'pq-site-theme';
  var root = document.documentElement;

  function applyLabels() {
    var lightOn = root.getAttribute('data-theme') === 'light';
    document.querySelectorAll('[data-theme-toggle]').forEach(function (btn) {
      btn.setAttribute('aria-pressed', lightOn ? 'true' : 'false');
      var el = btn.querySelector('.theme-toggle-text');
      if (el) el.textContent = lightOn ? 'Gece modu' : 'Gündüz modu';
      btn.setAttribute('aria-label', lightOn ? 'Koyu temaya geç' : 'Açık temaya geç');
    });
  }

  function toggle() {
    var t = root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    var next = t === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem(KEY, next); } catch (e) {}
    applyLabels();
  }

  function init() {
    applyLabels();
    document.querySelectorAll('[data-theme-toggle]').forEach(function (b) {
      b.addEventListener('click', toggle);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
""",
        encoding="utf-8",
    )


def write_demo_viewer(site: Path) -> None:
    """3Dmol demosu: pipeline viewer ile aynı mantıktaki kontroller + Türkçe açıklama panelleri."""
    demo_html = r"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>__PROJECT_NAME__ — 3D görüntüleyici demosu (1CRN)</title>
  <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
  <link rel="stylesheet" href="../assets/css/site.css" />
"""
    demo_html = (
        demo_html.rstrip()
        + "\n"
        + THEME_INLINE_HEAD
        + r"""  <style>
    .demo-page { max-width: 1600px; margin: 0 auto; padding: 0 1rem 2rem; width: 100%; box-sizing: border-box; }
    .demo-top {
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start;
      gap: 1rem; padding: 1.25rem 0; border-bottom: 1px solid var(--border);
    }
    .demo-top h1 { margin: 0 0 0.35rem; font-size: 1.35rem; }
    .demo-top .sub { color: var(--muted); font-size: 0.95rem; max-width: 42rem; margin: 0; }
    /* minmax(0,…) tüm izleri: taşmayı keser, 3Dmol tuvali sağa yapışmaz */
    .demo-grid {
      display: grid; gap: 1.25rem;
      grid-template-columns: minmax(0, 1fr);
      width: 100%;
    }
    @media (min-width: 900px) {
      .demo-grid { grid-template-columns: minmax(0, 260px) minmax(0, 1fr); align-items: start; }
    }
    @media (min-width: 1200px) {
      .demo-grid { grid-template-columns: minmax(0, 280px) minmax(0, 1fr) minmax(0, 280px); }
    }
    .demo-grid > * { min-width: 0; }
    .demo-viewer-stack {
      display: flex; flex-direction: column;
      width: 100%; max-width: 100%; min-width: 0;
    }
    .demo-panel {
      background: var(--card); border: 1px solid var(--border); border-radius: 10px;
      padding: 1rem 1.1rem; font-size: 0.88rem; line-height: 1.55;
    }
    .demo-panel h2 { margin: 0 0 0.6rem; font-size: 0.95rem; color: var(--accent-2); text-transform: uppercase; letter-spacing: 0.04em; }
    .demo-panel h3 { margin: 1rem 0 0.35rem; font-size: 0.9rem; }
    .demo-panel p, .demo-panel li { color: var(--text-dim); }
    .demo-panel ul { margin: 0.35rem 0 0 1.1rem; padding: 0; }
    .demo-panel dl { margin: 0; }
    .demo-panel dt { font-weight: 600; color: var(--text); margin-top: 0.65rem; }
    .demo-panel dd { margin: 0.2rem 0 0 0; color: var(--text-dim); }
    #viewer-demo {
      width: 100%; max-width: 100%; min-width: 0;
      height: min(62vh, 560px); min-height: 280px;
      background: #fff; border: 1px solid var(--border); border-radius: 8px;
      position: relative; overflow: hidden;
    }
    @media (min-width: 600px) {
      #viewer-demo { height: min(65vh, 620px); min-height: 360px; }
    }
    #viewer-demo canvas { display: block; max-width: 100%; vertical-align: middle; }
    .demo-controls {
      margin-top: 1rem; padding: 1rem; background: var(--card2); border: 1px solid var(--border);
      border-radius: 10px;
    }
    .demo-controls h3 { margin: 0 0 0.75rem; font-size: 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }
    .ctrl-row { margin-bottom: 0.85rem; }
    .ctrl-row label { display: block; font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.35rem; }
    .btn-group { display: flex; flex-wrap: wrap; gap: 0.4rem; }
    .demo-controls button {
      padding: 0.45rem 0.75rem; font-size: 0.85rem; border-radius: 6px; border: 1px solid var(--border);
      background: var(--card); color: var(--text); cursor: pointer;
    }
    .demo-controls button:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
    .demo-controls button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    .demo-controls button.primary:hover { filter: brightness(1.06); color: #fff; }
    .legend-demo {
      margin-top: 1rem; padding: 1rem; background: var(--card2); border-radius: 10px; border: 1px solid var(--border);
    }
    .legend-demo h3 { margin: 0 0 0.6rem; font-size: 0.95rem; }
    .legend-items { display: flex; flex-wrap: wrap; gap: 0.65rem 1.2rem; font-size: 0.82rem; color: var(--text-dim); }
    .lg { display: inline-flex; align-items: center; gap: 0.35rem; }
    .lg i { width: 14px; height: 14px; border-radius: 3px; border: 1px solid var(--border); }
    .hint-bar {
      margin-top: 0.75rem; padding: 0.65rem 0.85rem; background: rgba(47, 129, 247, 0.12);
      border: 1px solid rgba(47, 129, 247, 0.35); border-radius: 8px; font-size: 0.82rem; color: var(--text-dim);
    }
    code.inline { font-size: 0.84em; background: var(--card2); padding: 0.12rem 0.35rem; border-radius: 4px; }
    html[data-theme="light"] .hint-bar {
      background: rgba(9, 105, 218, 0.08);
      border: 1px solid rgba(9, 105, 218, 0.28);
    }
    @media (max-width: 720px) {
      .demo-top .sub, .demo-top h1 { padding-right: 4.5rem; }
    }
  </style>
</head>
<body class="site-body">
  <div class="site-toolbar">
    <button type="button" class="theme-toggle" data-theme-toggle aria-pressed="false" aria-label="Açık temaya geç">
      <span class="theme-toggle-text">Gündüz modu</span>
    </button>
  </div>
  <div class="demo-page">
    <div class="demo-top">
      <div>
        <h1>3D yapı görüntüleyici — teknoloji demosu</h1>
        <p class="sub">
          Aşağıdaki kontroller, yerel pipeline’ın ürettiği <code class="inline">viewer.html</code> ile aynı ailedendir
          (3Dmol.js, yüzey, gösterim modları, arka plan). Örnek yapı: <strong>RCSB 1CRN</strong> (crambin, tek zincir küçük protein).
        </p>
      </div>
      <a class="btn-pill" href="../index.html">← Ana dokümantasyon</a>
    </div>

    <div class="demo-grid">
      <aside class="demo-panel">
        <h2>Bu demo ne?</h2>
        <p>Tarayıcıda mmCIF okuma, WebGL ile çizim ve kullanıcı kontrollerinin çalıştığını doğrular. GitHub Pages’te <strong>aynı kökenden</strong> CIF dosyası yüklenir (<code class="inline">../assets/demo/1crn.cif</code>).</p>
        <h3>Üretim hattı ile fark</h3>
        <ul>
          <li>PROPEDIA çalıştırmasında protein ve peptit <strong>ayrı zincirler</strong> olarak boyanır (protein: cartoon + isteğe yüzey; peptit: stick).</li>
          <li>PLIP/Arpeggio’dan gelen çiftler <strong>silindir</strong> ile bağlanır; bu sayfada etkileşim listesi yoktur.</li>
          <li>Tam çıktı: <code class="inline">viewer_state.json</code> + <code class="inline">report.html</code>.</li>
        </ul>
      </aside>

      <div class="demo-viewer-stack">
        <div id="viewer-demo" role="img" aria-label="3D yapı görüntüsü"></div>
        <div class="demo-controls">
          <h3>Görünüm kontrolleri</h3>
          <div class="ctrl-row">
            <label>Temsil (tüm yapı — 1CRN tek zincir)</label>
            <div class="btn-group" id="rep-btn-group" role="group" aria-label="Gösterim modu">
              <button type="button" class="primary" data-rep="cartoon" aria-pressed="true" onclick="setRep('cartoon')">Cartoon</button>
              <button type="button" data-rep="stick" aria-pressed="false" onclick="setRep('stick')">Stick</button>
              <button type="button" data-rep="sphere" aria-pressed="false" onclick="setRep('sphere')">Sphere</button>
              <button type="button" data-rep="line" aria-pressed="false" onclick="setRep('line')">Line</button>
            </div>
          </div>
          <div class="ctrl-row">
            <label>Yüzey (VDW)</label>
            <div class="btn-group">
              <button type="button" onclick="toggleSurface()">Yüzey aç / kapat</button>
            </div>
          </div>
          <div class="ctrl-row">
            <label>Arka plan</label>
            <div class="btn-group">
              <button type="button" onclick="setBg('white')">Beyaz</button>
              <button type="button" onclick="setBg('black')">Siyah</button>
              <button type="button" onclick="setBg('#1a1a2e')">Lacivert</button>
            </div>
          </div>
          <div class="ctrl-row">
            <label>Navigasyon</label>
            <div class="btn-group">
              <button type="button" onclick="resetView()">Görünümü sıfırla</button>
            </div>
          </div>
          <div class="ctrl-row">
            <label>Etkileşim çizgileri (tam hatta)</label>
            <div class="btn-group">
              <button type="button" onclick="explainIx()" title="Bu demoda çizgi yok">Bilgi</button>
            </div>
          </div>
        </div>
        <div class="legend-demo">
          <h3>Etkileşim renkleri (üretim raporlarında)</h3>
          <p style="margin:0 0 0.5rem;font-size:0.82rem;color:var(--muted)">PLIP/Arpeggio birleşik çıktıda kullanılan tip → renk eşlemesi (referans).</p>
          <div class="legend-items">
            <span class="lg"><i style="background:#2E86AB"></i> H-bağı</span>
            <span class="lg"><i style="background:#A23B72"></i> Tuz köprüsü</span>
            <span class="lg"><i style="background:#F18F01"></i> Hidrofobik</span>
            <span class="lg"><i style="background:#C73E1D"></i> π-yığın</span>
            <span class="lg"><i style="background:#6A994E"></i> Katyon-π</span>
            <span class="lg"><i style="background:#CCCCCC"></i> vdW / fallback</span>
          </div>
        </div>
        <div class="hint-bar" id="ix-hint" style="display:none"></div>
      </div>

      <aside class="demo-panel">
        <h2>Kontrol sözlüğü</h2>
        <dl>
          <dt>Cartoon</dt>
          <dd>İkincil yapı şeridi; genel fold okuması için.</dd>
          <dt>Stick / Sphere / Line</dt>
          <dd>Atom bağları veya atom merkezleri; detay ve sunum için.</dd>
          <dt>Yüzey (VDW)</dt>
          <dd>Van der Waals yüzeyi — boşlukları ve yakın temas bölgelerini görmek için (performans maliyetli olabilir).</dd>
          <dt>Görünümü sıfırla</dt>
          <dd>Tüm yapıya zoom ve kamera reset.</dd>
        </dl>
        <h3>JSON ile ilişki</h3>
        <p><code class="inline">viewer_state.json</code> içinde <code class="inline">chains</code> (tip, renk, yüzey), <code class="inline">interactions</code> (çiftler, renk, mesafe) ve <code class="inline">view_config</code> üretimde doldurulur; bu HTML tek dosyada aynı motoru gösterir.</p>
      </aside>
    </div>
  </div>
  <script>
(function() {
  let viewer = null;
  let surfaceOn = false;
  let currentRep = 'cartoon';
  const el = document.getElementById('viewer-demo');

  function syncRepButtons(mode) {
    var grp = document.getElementById('rep-btn-group');
    if (!grp) return;
    var btns = grp.querySelectorAll('button[data-rep]');
    for (var i = 0; i < btns.length; i++) {
      var b = btns[i];
      var on = b.getAttribute('data-rep') === mode;
      b.classList.toggle('primary', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    }
  }

  /* 3Dmol kendi ResizeObserver + window resize ile resize() çağırır; burada yalnızca ilk boyama ve orientation yedeklenir */
  var resizeTimer = null;
  function resizeViewer() {
    if (!viewer || !el) return;
    try {
      if (typeof viewer.resize === 'function') viewer.resize();
    } catch (e) { /* noop */ }
    viewer.render();
  }
  function scheduleResizeViewer() {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function() {
      resizeTimer = null;
      resizeViewer();
    }, 80);
  }

  function initViewer(data) {
    viewer = $3Dmol.createViewer(el, { backgroundColor: 'white' });
    viewer.addModel(data, 'cif');
    currentRep = 'cartoon';
    syncRepButtons(currentRep);
    applyRepStyle(currentRep);
    viewer.zoomTo();
    viewer.render();
    requestAnimationFrame(function() {
      resizeViewer();
      requestAnimationFrame(resizeViewer);
    });
  }

  function applyRepStyle(mode) {
    if (!viewer) return;
    viewer.removeAllSurfaces();
    surfaceOn = false;
    var sel = {};
    if (mode === 'cartoon') viewer.setStyle(sel, { cartoon: { color: 'spectrum' } });
    else if (mode === 'stick') viewer.setStyle(sel, { stick: { radius: 0.18, colorscheme: 'Jmol' } });
    else if (mode === 'sphere') viewer.setStyle(sel, { sphere: { scale: 0.22, colorscheme: 'Jmol' } });
    else if (mode === 'line') viewer.setStyle(sel, { line: {} });
  }

  window.setRep = function(mode) {
    if (!viewer) return;
    currentRep = mode;
    syncRepButtons(mode);
    applyRepStyle(mode);
    viewer.render();
    requestAnimationFrame(resizeViewer);
  };

  window.toggleSurface = function() {
    if (!viewer) return;
    if (surfaceOn) {
      viewer.removeAllSurfaces();
      surfaceOn = false;
    } else {
      viewer.addSurface($3Dmol.SurfaceType.VDW, { opacity: 0.32, color: 'white' }, {});
      surfaceOn = true;
    }
    viewer.render();
  };

  window.setBg = function(c) {
    if (!viewer) return;
    viewer.setBackgroundColor(c);
    viewer.render();
  };

  window.resetView = function() {
    if (!viewer) return;
    viewer.zoomTo();
    viewer.render();
  };

  window.explainIx = function() {
    const h = document.getElementById('ix-hint');
    h.style.display = 'block';
    h.textContent = 'Bu demoda etkileşim listesi yok. Tam çalıştırmada PLIP/Arpeggio (veya geometrik fallback) çiftleri silindir olarak çizilir; viewer_state.json içindeki interactions dizisi bu çizgileri üretir.';
  };

  window.addEventListener('orientationchange', function() {
    setTimeout(function() { scheduleResizeViewer(); }, 200);
  });

  fetch(new URL('../assets/demo/1crn.cif', window.location.href))
    .then(function(r) { if (!r.ok) throw new Error('CIF'); return r.text(); })
    .then(initViewer)
    .catch(function() {
      el.innerHTML = '<p style="padding:1.5rem;color:#333;font-family:sans-serif">1crn.cif yüklenemedi. Yerelde: python scripts/build_pages_site.py</p>';
    });
})();
  </script>
  <script src="../assets/js/site-theme.js" defer></script>
</body>
</html>
"""
    )
    out = site / "embed" / "viewer-demo.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(demo_html.replace("__PROJECT_NAME__", PROJECT_DISPLAY_NAME), encoding="utf-8")


def write_site_css(site: Path) -> None:
    css = site / "assets" / "css" / "site.css"
    css.parent.mkdir(parents=True, exist_ok=True)
    css.write_text(
        """
/* Koyu (varsayılan tokenlar); html[data-theme="light"] ile gündüz */
:root, html[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0d1117;
  --card: #161b22;
  --card2: #21262d;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #c9d1d9;
  --muted: #8b949e;
  --accent: #2f81f7;
  --accent-2: #58a6ff;
  --ok: #3fb950;
  --warn: #d29922;
  --hero-mid: #1c2128;
  --hero-spot: rgba(47, 129, 247, 0.2);
  --pre-bg: #010409;
  --badge-bg: rgba(22, 27, 34, 0.88);
  --media-chrome: rgba(255, 255, 255, 0.04);
  --media-shadow: 0 0 0 1px rgba(255, 255, 255, 0.08);
  --toolbar-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}

html[data-theme="light"] {
  color-scheme: light;
  --bg: #eef1f7;
  --card: #ffffff;
  --card2: #f0f3f9;
  --border: #cfd7e1;
  --text: #1a1f26;
  --text-dim: #3a4450;
  --muted: #5c6670;
  --accent: #0969da;
  --accent-2: #0550ae;
  --hero-mid: #ffffff;
  --hero-spot: rgba(9, 105, 218, 0.12);
  --pre-bg: #f6f8fa;
  --badge-bg: rgba(255, 255, 255, 0.95);
  --media-chrome: #ffffff;
  --media-shadow: 0 4px 18px rgba(26, 31, 38, 0.1);
  --toolbar-shadow: 0 2px 12px rgba(26, 31, 38, 0.12);
}

* { box-sizing: border-box; }
.site-body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  font-size: 16px;
}
a { color: var(--accent-2); text-decoration: none; }
a:hover { text-decoration: underline; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.88em; }
code { background: var(--card2); padding: 0.15rem 0.4rem; border-radius: 4px; border: 1px solid var(--border); }

.site-shell {
  width: 100%;
  max-width: min(1180px, 100%);
  margin: 0 auto;
  padding: 0 clamp(0.5rem, 2.5vw, 1rem);
  padding-bottom: 2rem;
  box-sizing: border-box;
}
.shell-topbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 0.5rem;
  padding: 0.5rem 0 0.25rem;
  position: sticky;
  top: 0;
  z-index: 150;
  background: linear-gradient(180deg, var(--bg) 88%, transparent);
}
.shell-topbar .repo-link {
  font-size: 0.76rem;
  font-weight: 600;
  color: var(--muted);
  padding: 0.38rem 0.85rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--card);
  box-shadow: var(--toolbar-shadow);
}
.shell-topbar .repo-link:hover {
  color: var(--accent-2);
  border-color: var(--accent);
  text-decoration: none;
}
.theme-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.42rem 0.9rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--card);
  color: var(--text-dim);
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow: var(--toolbar-shadow);
  font-family: inherit;
}
.theme-toggle:hover {
  border-color: var(--accent);
  color: var(--accent-2);
}
.theme-toggle:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

.layout-wrap {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 0;
  width: 100%;
  max-width: 100%;
  margin: 0;
  padding: 0;
}
@media (min-width: 1100px) {
  .layout-wrap { grid-template-columns: minmax(0, 200px) minmax(0, 1fr); align-items: start; }
}
aside.sidenav {
  position: sticky;
  top: 0;
  z-index: 5;
  padding: clamp(0.85rem, 2vw, 1rem) clamp(0.75rem, 2vw, 1.25rem);
  background: var(--card);
  border-bottom: 1px solid var(--border);
  font-size: clamp(0.78rem, 2.2vw, 0.82rem);
}
@media (min-width: 1100px) {
  aside.sidenav {
    border-right: 1px solid var(--border);
    border-bottom: none;
    min-height: 100vh;
    align-self: start;
  }
}
aside.sidenav strong { display: block; margin-bottom: 0.65rem; color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
aside.sidenav a { display: block; padding: 0.35rem 0; color: var(--text-dim); border-left: 2px solid transparent; padding-left: 0.5rem; margin-left: -0.5rem; }
aside.sidenav a:hover { color: var(--accent-2); border-left-color: var(--accent); text-decoration: none; }

header.hero {
  padding: clamp(1.15rem, 3vw, 2rem) 0 clamp(1rem, 2.5vw, 1.65rem);
  border-bottom: 1px solid var(--border);
  border-radius: 0 0 12px 12px;
  background: radial-gradient(ellipse 120% 80% at 20% -20%, var(--hero-spot), transparent 55%),
              linear-gradient(180deg, var(--hero-mid) 0%, var(--bg) 100%);
}
header.hero .inner { max-width: min(52rem, 100%); margin: 0 auto; }
.badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
.badge {
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
  padding: 0.25rem 0.55rem; border-radius: 999px; border: 1px solid var(--border);
  color: var(--muted); background: var(--badge-bg);
}
.badge.accent { border-color: rgba(47, 129, 247, 0.45); color: var(--accent-2); }
html[data-theme="light"] .badge.accent { border-color: rgba(5, 80, 174, 0.35); }
header.hero h1 { margin: 0 0 0.65rem; font-size: clamp(1.6rem, 5vw, 2.35rem); font-weight: 700; letter-spacing: -0.02em; }
header.hero .lead { margin: 0 0 1.25rem; color: var(--text-dim); font-size: clamp(0.98rem, 2.8vw, 1.05rem); max-width: 48rem; }
.hero-cta { display: flex; flex-wrap: wrap; gap: 0.65rem; align-items: center; }
.btn-pill {
  display: inline-block; padding: 0.55rem 1.1rem; border-radius: 8px; font-weight: 600; font-size: 0.92rem;
  background: var(--accent); color: #fff !important; border: 1px solid transparent;
}
.btn-pill:hover { filter: brightness(1.08); text-decoration: none; }
.btn-pill.outline {
  background: transparent; color: var(--text) !important; border-color: var(--border);
}
.btn-pill.outline:hover { border-color: var(--accent); color: var(--accent-2) !important; }

nav.toc-inline {
  display: flex;
  flex-wrap: nowrap;
  gap: 0.4rem 0.65rem;
  row-gap: 0.45rem;
  padding: clamp(0.55rem, 1.8vw, 0.75rem) 0;
  margin: 0 0 0.35rem;
  background: transparent;
  border-bottom: 1px solid var(--border);
  font-size: clamp(0.76rem, 2.2vw, 0.85rem);
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  scrollbar-width: thin;
}
nav.toc-inline a {
  color: var(--muted);
  white-space: nowrap;
  flex-shrink: 0;
  padding: 0.2rem 0;
}
nav.toc-inline a:hover { color: var(--accent-2); }
@media (min-width: 960px) {
  nav.toc-inline {
    flex-wrap: wrap;
    overflow-x: visible;
    padding: clamp(0.65rem, 2vw, 0.8rem) 0;
  }
}

main.content {
  padding: clamp(0.85rem, 2.5vw, 1.35rem) 0 clamp(1rem, 3vw, 1.5rem);
  max-width: min(860px, 100%);
  width: 100%;
  min-width: 0;
  margin: 0 auto;
}

section.block {
  margin-bottom: clamp(1.25rem, 3vw, 2.25rem);
  padding: clamp(1rem, 3vw, 1.65rem);
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
}
section.block h2 {
  margin: 0 0 0.75rem;
  font-size: 1.2rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}
section.block h3 { margin: 1.35rem 0 0.5rem; font-size: 1.02rem; color: var(--text); }
section.block .lead-in { color: var(--text-dim); margin: 0 0 1rem; font-size: 0.98rem; }

.callout {
  margin: 1rem 0; padding: 0.85rem 1rem; border-radius: 8px; border: 1px solid var(--border);
  background: var(--card2); font-size: 0.9rem; color: var(--text-dim);
}
.callout strong { color: var(--text); }

.metrics-grid {
  display: grid; gap: 0.65rem;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 118px), 1fr));
  margin: 1rem 0;
}
.metric-card {
  padding: 0.75rem 0.85rem; border-radius: 8px; border: 1px solid var(--border); background: var(--card2);
}
.metric-card .k { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .v { font-size: 1.1rem; font-weight: 600; margin-top: 0.2rem; }
.metric-empty-hint { font-size: 0.84rem; color: var(--muted); margin: -0.25rem 0 1rem; line-height: 1.45; }
.training-fig-grid { margin-top: 0.35rem; }

table.data-table {
  width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 1rem 0;
}
table.data-table th, table.data-table td {
  border: 1px solid var(--border); padding: 0.55rem 0.65rem; text-align: left;
}
table.data-table th { background: var(--card2); color: var(--muted); font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
table.data-table td { color: var(--text-dim); }

.table-scroll {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  margin: 1rem 0;
  border-radius: 8px;
  border: 1px solid var(--border);
}
.table-scroll table.data-table { margin: 0; min-width: min(480px, 100%); }
@media (max-width: 600px) {
  .table-scroll { margin-left: 0; margin-right: 0; border-radius: 6px; }
  .table-scroll table.data-table { font-size: 0.8rem; min-width: 0; }
  .table-scroll table.data-table th,
  .table-scroll table.data-table td {
    padding: 0.45rem 0.5rem;
    word-break: break-word;
    hyphens: auto;
  }
}
.table-scroll table.data-table th:first-child,
.table-scroll table.data-table td:first-child { padding-left: 0.75rem; }
.table-scroll table.data-table th:last-child,
.table-scroll table.data-table td:last-child { padding-right: 0.75rem; }

.flow-steps {
  display: flex; flex-direction: column; gap: 0.5rem; margin: 1rem 0; counter-reset: step;
}
.flow-steps .step {
  display: flex; gap: 0.85rem; align-items: flex-start;
  padding: 0.65rem 0.85rem; border-radius: 8px; border: 1px solid var(--border); background: var(--card2);
  font-size: 0.9rem; color: var(--text-dim);
}
.flow-steps .step::before {
  counter-increment: step; content: counter(step);
  flex-shrink: 0; width: 1.65rem; height: 1.65rem; border-radius: 6px;
  background: var(--accent); color: #fff; font-weight: 700; font-size: 0.8rem;
  display: flex; align-items: center; justify-content: center;
}

details.expand {
  margin: 0.75rem 0; border: 1px solid var(--border); border-radius: 8px; background: var(--card2);
}
details.expand summary {
  cursor: pointer; padding: 0.65rem 1rem; font-weight: 600; color: var(--text-dim);
}
details.expand .inner { padding: 0 1rem 1rem; color: var(--text-dim); font-size: 0.9rem; }

.grid2 {
  display: grid;
  gap: clamp(0.85rem, 2vw, 1.25rem);
  grid-template-columns: minmax(0, 1fr);
}
@media (min-width: 700px) { .grid2 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

figure.media { margin: 1rem 0 0; }
figure.media img {
  display: block;
  max-width: 100%;
  width: 100%;
  height: auto;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--media-chrome);
  box-shadow: var(--media-shadow);
}
figure.media figcaption { font-size: clamp(0.78rem, 2vw, 0.82rem); color: var(--muted); margin-top: 0.45rem; line-height: 1.45; }

pre.json {
  background: var(--pre-bg);
  padding: clamp(0.75rem, 2vw, 1rem) clamp(0.85rem, 2vw, 1.15rem);
  border-radius: 8px;
  overflow: auto;
  font-size: clamp(0.72rem, 1.8vw, 0.78rem);
  line-height: 1.45;
  border: 1px solid var(--border);
  color: var(--text-dim);
  max-width: 100%;
}

.flow-diagram {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
  margin: 0.5rem 0 0;
}
.flow-row {
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  gap: 0.45rem 0.55rem;
}
.flow-node {
  flex: 1 1 140px;
  min-width: min(100%, 140px);
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.65rem 0.85rem;
  font-size: clamp(0.8rem, 2.2vw, 0.88rem);
  color: var(--text-dim);
  line-height: 1.45;
}
.flow-node strong { color: var(--accent-2); margin-right: 0.25rem; }
.flow-arr {
  color: var(--accent-2);
  font-weight: 700;
  align-self: center;
  user-select: none;
  padding: 0 0.15rem;
  flex-shrink: 0;
}
@media (max-width: 520px) {
  .flow-arr { display: none; }
  .flow-row { flex-direction: column; }
  .flow-node { min-width: 100%; }
}

footer.site-ft {
  padding: clamp(1.1rem, 2.5vw, 1.65rem) 0;
  margin-top: 1rem;
  color: var(--muted);
  font-size: clamp(0.76rem, 2vw, 0.82rem);
  text-align: center;
  border-top: 1px solid var(--border);
  background: transparent;
}
.footer-inner {
  display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; justify-content: center; align-items: center;
}

/* --- Topbar nav --- */
.topbar-nav { display: flex; gap: 0.5rem; margin-right: auto; margin-left: 0.75rem; }
.topbar-nav a {
  font-size: 0.76rem; font-weight: 600; color: var(--muted); padding: 0.3rem 0.6rem;
  border-radius: 6px; border: 1px solid transparent;
}
.topbar-nav a:hover { color: var(--accent-2); border-color: var(--border); text-decoration: none; }

/* --- Hero metrics strip --- */
.hero-metrics-strip {
  display: flex; flex-wrap: wrap; gap: 0.65rem; margin: 0 0 1.25rem;
}
.hm {
  display: flex; flex-direction: column; align-items: center; padding: 0.65rem 1rem;
  border-radius: 10px; background: var(--card); border: 1px solid var(--border);
  min-width: 80px;
}
.hm-val { font-size: 1.3rem; font-weight: 700; color: var(--accent-2); line-height: 1.2; }
.hm-key { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-top: 0.15rem; }

/* --- Pipeline grid --- */
.pipeline-grid {
  display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: stretch; margin: 1rem 0;
}
.pipe-node {
  flex: 1 1 120px; min-width: min(100%, 120px);
  display: flex; align-items: flex-start; gap: 0.55rem;
  padding: 0.65rem 0.75rem; border-radius: 10px;
  border: 1px solid var(--border); background: var(--card2);
}
.pipe-node.pipe-data { border-left: 3px solid var(--ok); }
.pipe-node.pipe-process { border-left: 3px solid var(--accent); }
.pipe-node.pipe-model { border-left: 3px solid var(--warn); }
.pipe-node.pipe-result { border-left: 3px solid #a371f7; }
.pipe-num {
  flex-shrink: 0; width: 1.5rem; height: 1.5rem; border-radius: 50%;
  background: var(--accent); color: #fff; font-weight: 700; font-size: 0.75rem;
  display: flex; align-items: center; justify-content: center;
}
.pipe-body { display: flex; flex-direction: column; gap: 0.1rem; }
.pipe-body strong { font-size: 0.82rem; color: var(--text); }
.pipe-body span { font-size: 0.75rem; color: var(--muted); line-height: 1.35; }
.pipe-arrow {
  flex-shrink: 0; width: 16px; align-self: center;
  background: linear-gradient(90deg, var(--accent) 40%, transparent 100%);
  height: 2px; border-radius: 1px;
}
@media (max-width: 600px) {
  .pipeline-grid { flex-direction: column; }
  .pipe-arrow { width: 2px; height: 12px; align-self: flex-start; margin-left: 1.3rem;
    background: linear-gradient(180deg, var(--accent) 40%, transparent 100%); }
}

/* --- Stats bar --- */
.stats-bar {
  display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0 0; justify-content: center;
}
.stat-item {
  display: flex; flex-direction: column; align-items: center;
  padding: 0.45rem 0.85rem; border-radius: 8px;
  background: var(--card2); border: 1px solid var(--border); min-width: 70px;
}
.stat-val { font-size: 1.05rem; font-weight: 700; color: var(--text); }
.stat-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }

/* --- Metrics grid 8 --- */
.metrics-grid-8 {
  display: grid; gap: 0.55rem;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 110px), 1fr));
  margin: 1rem 0;
}
.metrics-grid-8 .metric-card { text-align: center; }
.metrics-grid-8 .metric-card .desc { font-size: 0.68rem; color: var(--muted); margin-top: 0.15rem; }
.metric-card.primary { border-color: var(--accent); }
.metric-card.primary .v { color: var(--accent-2); }
.metric-card.highlight { border-color: var(--ok); background: rgba(63, 185, 80, 0.08); }
.metric-card.highlight .v { color: var(--ok); }
html[data-theme="light"] .metric-card.highlight { background: rgba(63, 185, 80, 0.06); }

/* --- Label chips --- */
.label-chip {
  display: inline-block; padding: 0.15rem 0.5rem; border-radius: 999px;
  font-size: 0.75rem; font-weight: 600;
}
.label-chip.pos { background: rgba(63, 185, 80, 0.15); color: var(--ok); border: 1px solid rgba(63, 185, 80, 0.3); }
.label-chip.neg { background: rgba(210, 153, 34, 0.12); color: var(--warn); border: 1px solid rgba(210, 153, 34, 0.3); }
html[data-theme="light"] .label-chip.pos { background: rgba(26, 127, 55, 0.1); color: #1a7f37; }
html[data-theme="light"] .label-chip.neg { background: rgba(154, 103, 0, 0.1); color: #9a6700; }

/* --- Ranked table --- */
.ranked-table td:nth-child(5) { font-variant-numeric: tabular-nums; }
.ranked-table code { font-size: 0.82em; }

/* --- Two col info --- */
.two-col-info { display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 1rem 0; }
@media (min-width: 600px) { .two-col-info { grid-template-columns: 1fr 1fr; } }
.two-col-info ul { margin: 0.35rem 0 0 1.1rem; padding: 0; font-size: 0.9rem; color: var(--text-dim); }

/* --- Viewer embed --- */
.viewer-embed-wrap {
  margin: 1rem 0; border-radius: 12px; overflow: hidden;
  border: 1px solid var(--border); background: #fff;
}
.viewer-iframe {
  width: 100%; height: 420px; border: none; display: block;
}
@media (min-width: 700px) { .viewer-iframe { height: 500px; } }

/* --- Accent callout --- */
.accent-callout { border-left: 3px solid var(--accent); }

/* --- Compact table --- */
table.data-table.compact { font-size: 0.84rem; }
table.data-table.compact td { padding: 0.4rem 0.55rem; }

/* --- Method diagram --- */
.method-diagram {
  display: flex; flex-direction: column; align-items: center; gap: 0; margin: 1.25rem 0;
}
.md-phase {
  width: 100%; max-width: 520px; border-radius: 12px; padding: 0.85rem 1rem;
  border: 1px solid var(--border); background: var(--card2); position: relative;
}
.md-phase[data-phase="veri"] { border-left: 3px solid var(--ok); }
.md-phase[data-phase="split"] { border-left: 3px solid var(--accent); }
.md-phase[data-phase="model"] { border-left: 3px solid var(--warn); }
.md-phase[data-phase="rapor"] { border-left: 3px solid #a371f7; }
.md-phase-title {
  font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); margin-bottom: 0.5rem;
}
.md-nodes { display: flex; flex-direction: column; gap: 0; }
.md-node {
  padding: 0.45rem 0.6rem; background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; display: flex; flex-direction: column; gap: 0.1rem;
}
.md-node strong { font-size: 0.88rem; color: var(--text); }
.md-node span { font-size: 0.78rem; color: var(--muted); line-height: 1.35; }
.md-node em { color: var(--accent-2); font-style: normal; font-weight: 600; }
.md-arrow-v {
  width: 2px; height: 14px; background: var(--accent); margin: 0 auto; border-radius: 1px;
}
.md-connector-v {
  width: 2px; height: 18px; background: var(--accent); margin: 0 auto; border-radius: 1px;
}

/* --- Lightbox --- */
.lightbox-overlay {
  position: fixed; inset: 0; z-index: 9999;
  background: rgba(0,0,0,0.88); display: flex; align-items: center; justify-content: center;
  padding: 1.5rem; cursor: zoom-out;
}
.lightbox-img {
  max-width: 95vw; max-height: 90vh; border-radius: 10px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.5); cursor: default;
  object-fit: contain;
}
.lightbox-close {
  position: absolute; top: 1rem; right: 1.5rem; font-size: 2.2rem; color: #fff;
  background: none; border: none; cursor: pointer; line-height: 1; z-index: 10000;
  opacity: 0.7;
}
.lightbox-close:hover { opacity: 1; }

/* --- Viewer responsive --- */
.viewer-responsive {
  position: relative; width: 100%; padding-bottom: 56.25%;
  border-radius: 12px; overflow: hidden; border: 1px solid var(--border);
  background: #fff; margin: 0.75rem 0;
}
.viewer-responsive iframe {
  position: absolute; inset: 0; width: 100%; height: 100%; border: none;
}
""",
        encoding="utf-8",
    )


def _fmt_metric(m: Dict[str, Any], k: str) -> str:
    v = m.get(k)
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def _extra_head_meta() -> str:
    base = (os.environ.get("PEPTIPROP_SITE_URL") or "").strip().rstrip("/")
    if not base:
        return ""
    esc = html_module.escape
    og_title = esc(f"{PROJECT_DISPLAY_NAME} — PROPEDIA skorlama")
    return (
        f'  <link rel="canonical" href="{esc(base + "/index.html")}" />\n'
        f'  <meta property="og:url" content="{esc(base + "/")}" />\n'
        f'  <meta property="og:title" content="{og_title}" />\n'
        f'  <meta name="twitter:card" content="summary" />\n'
    )


PIPELINE_DIAGRAM_HTML = """      <section class="block" id="akim">
        <h2>Proje akışı</h2>
        <p class="lead-in">
          <a href="__REPO_URL__">PeptiProp</a> deposundaki uçtan uca pipeline: PROPEDIA ham verisinden kanonik tablolara, sızıntısız sekans-küme split'lerine, aday kümesi + negatif çift üretimine, özellik ihracına, MLX model eğitimine ve son olarak 2D/3D raporlama ve bu statik siteye.
        </p>
        <div class="pipeline-grid" role="img" aria-label="Proje pipeline diyagramı">
          <div class="pipe-node pipe-data">
            <div class="pipe-num">1</div>
            <div class="pipe-body">
              <strong>PROPEDIA</strong>
              <span>Ham mmCIF / PDB kompleksleri</span>
            </div>
          </div>
          <div class="pipe-arrow"></div>
          <div class="pipe-node pipe-process">
            <div class="pipe-num">2</div>
            <div class="pipe-body">
              <strong>Kanonik tablolar</strong>
              <span>complexes, chains, residues + arayüz/pocket anotasyonu</span>
            </div>
          </div>
          <div class="pipe-arrow"></div>
          <div class="pipe-node pipe-process">
            <div class="pipe-num">3</div>
            <div class="pipe-body">
              <strong>Sekans-küme split</strong>
              <span>MMseqs2 (%30 kimlik); train / val / test</span>
            </div>
          </div>
          <div class="pipe-arrow"></div>
          <div class="pipe-node pipe-process">
            <div class="pipe-num">4</div>
            <div class="pipe-body">
              <strong>Negatif çiftler</strong>
              <span>Easy + Hard negatif ornekler</span>
            </div>
          </div>
          <div class="pipe-arrow"></div>
          <div class="pipe-node pipe-model">
            <div class="pipe-num">5</div>
            <div class="pipe-body">
              <strong>Özellik ihracı</strong>
              <span>Dense vektör: yapı + sekans + arayüz + yerel yoğunluk</span>
            </div>
          </div>
          <div class="pipe-arrow"></div>
          <div class="pipe-node pipe-model">
            <div class="pipe-num">6</div>
            <div class="pipe-body">
              <strong>MLX eğitim</strong>
              <span>BCE + pairwise ranking loss; erken durdurma</span>
            </div>
          </div>
          <div class="pipe-arrow"></div>
          <div class="pipe-node pipe-result">
            <div class="pipe-num">7</div>
            <div class="pipe-body">
              <strong>Metrikler &amp; rapor</strong>
              <span>AUROC, MRR, Hit@k + ROC/PR eğrileri + bu site</span>
            </div>
          </div>
        </div>
        <div class="stats-bar">
          <div class="stat-item"><span class="stat-val">__M_TRAIN_GROUPS__</span><span class="stat-label">Train grup</span></div>
          <div class="stat-item"><span class="stat-val">__M_VAL_GROUPS__</span><span class="stat-label">Val grup</span></div>
          <div class="stat-item"><span class="stat-val">__M_TEST_GROUPS__</span><span class="stat-label">Test grup</span></div>
          <div class="stat-item"><span class="stat-val">__M_EPOCHS__</span><span class="stat-label">Epoch</span></div>
          <div class="stat-item"><span class="stat-val">__M_THRESHOLD__</span><span class="stat-label">Eşik</span></div>
        </div>
      </section>
"""


def _build_top_ranked_table(training_dir: Optional[Path]) -> str:
    """top_ranked_examples.json'dan ilk 10 aday satırını HTML tablo olarak döndürür."""
    if not training_dir:
        return ""
    for candidate in (training_dir / "top_ranked_examples.json", ROOT / "publish" / "github_pages_training_bundle" / "top_ranked_examples.json"):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            preview = data.get("top_ranked_candidates_preview") or []
            if not preview:
                continue
            rows: List[str] = []
            for r in preview[:10]:
                pdb = html_module.escape(str(r.get("pdb_id", "—")))
                prot_ch = html_module.escape(str(r.get("protein_chain_id", "")))
                pep_ch = html_module.escape(str(r.get("peptide_chain_id", "")))
                plen = r.get("peptide_length", "—")
                score = float(r.get("score", 0))
                label = int(r.get("label", 0))
                rank = r.get("rank", "—")
                neg_type = str(r.get("negative_type", "—"))
                label_cls = "pos" if label == 1 else "neg"
                label_txt = "Pozitif" if label == 1 else neg_type.replace("_", " ").title()
                rows.append(
                    f"<tr>"
                    f'<td><code>{pdb}</code></td>'
                    f"<td>{prot_ch}</td><td>{pep_ch}</td>"
                    f"<td>{plen}</td>"
                    f'<td><strong>{score:.4f}</strong></td>'
                    f'<td><span class="label-chip {label_cls}">{label_txt}</span></td>'
                    f"<td>{rank}</td>"
                    f"</tr>"
                )
            return "\n            ".join(rows)
    return ""


def write_index(
    site: Path,
    manifest: Dict[str, Any],
    training_gallery_section: str,
    training_dir: Optional[Path] = None,
    site_extra_viz_section: str = "",
    peptide_2d_variants_section: str = "",
) -> None:
    m = _metrics_for_index(manifest)
    manifest_esc = html_module.escape(json.dumps(manifest, indent=2, ensure_ascii=False))

    top_ranked_rows = _build_top_ranked_table(training_dir)

    idx = _INDEX_HTML_TEMPLATE
    reps = {
        "__THEME_HEAD_SCRIPT__": THEME_INLINE_HEAD,
        "__EXTRA_HEAD_META__": _extra_head_meta(),
        "__PROJECT_NAME__": PROJECT_DISPLAY_NAME,
        "__REPO_URL__": REPO_URL,
        "__PIPELINE_DIAGRAM_BLOCK__": PIPELINE_DIAGRAM_HTML.replace("__REPO_URL__", html_module.escape(REPO_URL)),
        "__M_AUROC__": _fmt_metric(m, "test_auroc"),
        "__M_AUPRC__": _fmt_metric(m, "test_auprc"),
        "__M_F1__": _fmt_metric(m, "test_f1"),
        "__M_MCC__": _fmt_metric(m, "test_mcc"),
        "__M_MRR__": _fmt_metric(m, "test_mrr"),
        "__M_HIT1__": _fmt_metric(m, "test_hit1"),
        "__M_HIT3__": _fmt_metric(m, "test_hit3"),
        "__M_HIT5__": _fmt_metric(m, "test_hit5"),
        "__M_EPOCHS__": str(m.get("epochs") or "—"),
        "__M_THRESHOLD__": f'{m["threshold"]:.2f}' if isinstance(m.get("threshold"), (int, float)) else "—",
        "__M_TRAIN_GROUPS__": f'{m["train_groups"]:,}' if isinstance(m.get("train_groups"), (int, float)) else "—",
        "__M_VAL_GROUPS__": f'{m["val_groups"]:,}' if isinstance(m.get("val_groups"), (int, float)) else "—",
        "__M_TEST_GROUPS__": f'{m["test_groups"]:,}' if isinstance(m.get("test_groups"), (int, float)) else "—",
        "__TOP_RANKED_ROWS__": top_ranked_rows,
        "__MANIFEST_JSON__": manifest_esc,
        "__TRAINING_GALLERY_SECTION__": training_gallery_section,
        "__SITE_EXTRA_VIZ__": site_extra_viz_section,
        "__PEPTIDE_2D_VARIANTS__": peptide_2d_variants_section,
    }
    for k, v in reps.items():
        idx = idx.replace(k, v)
    (site / "index.html").write_text(idx, encoding="utf-8")


_INDEX_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="__PROJECT_NAME__ — PROPEDIA protein-peptid etkilesim skorlama, siralama ve 2D/3D gorsellestirme platformu" />
  <title>__PROJECT_NAME__ — Protein-Peptid Etkilesim Skorlama &amp; Reranking</title>
__EXTRA_HEAD_META__  <link rel="stylesheet" href="assets/css/site.css" />
__THEME_HEAD_SCRIPT__
</head>
<body class="site-body">
  <div class="site-shell">
  <div class="shell-topbar">
    <a class="repo-link" href="__REPO_URL__" target="_blank" rel="noopener noreferrer">GitHub: __PROJECT_NAME__</a>
    <nav class="topbar-nav">
      <a href="#metrikler">Metrikler</a>
      <a href="#gorsel-2d">2D</a>
      <a href="#gorsel-3d">3D</a>
    </nav>
    <button type="button" class="theme-toggle" data-theme-toggle aria-pressed="false" aria-label="Acik temaya gec">
      <span class="theme-toggle-text">Gunduz modu</span>
    </button>
  </div>

  <header class="hero">
    <div class="inner">
      <div class="badge-row">
        <span class="badge accent">PROPEDIA</span>
        <span class="badge">Leakage-free Split</span>
        <span class="badge">MLX / Apple Silicon</span>
        <span class="badge">Reranking</span>
        <span class="badge">2D + 3D</span>
      </div>
      <h1>__PROJECT_NAME__</h1>
      <p class="lead">
        Yapisal protein-peptid komplekslerinde <strong>skor uretimi</strong>, aday peptitler arasinda
        <strong>siralama (reranking)</strong> ve sonuclarin <strong>2D kimya + 3D yapi</strong> ile
        raporlanmasi. Sekans-kume tabanli sizintisiz split, arayuz/pocket anotasyonu ve
        BCE + pairwise ranking loss ile egitilmis MLP modeli.
      </p>
      <div class="hero-metrics-strip">
        <div class="hm"><span class="hm-val">__M_AUROC__</span><span class="hm-key">AUROC</span></div>
        <div class="hm"><span class="hm-val">__M_MRR__</span><span class="hm-key">MRR</span></div>
        <div class="hm"><span class="hm-val">__M_HIT3__</span><span class="hm-key">Hit@3</span></div>
        <div class="hm"><span class="hm-val">__M_HIT5__</span><span class="hm-key">Hit@5</span></div>
      </div>
      <div class="hero-cta">
        <a class="btn-pill" href="embed/viewer-demo.html">Interaktif 3D Demo</a>
        <a class="btn-pill outline" href="#metrikler">Tum Metrikler</a>
        <a class="btn-pill outline" href="data/manifest.json">manifest.json</a>
      </div>
    </div>
  </header>

  <nav class="toc-inline" aria-label="Sayfa ici">
    <a href="#akim">Pipeline</a>
    <a href="#amac">Amac</a>
    <a href="#yontem">Yontem</a>
    <a href="#veri">Veri</a>
    <a href="#metrikler">Metrikler</a>
    <a href="#top-ranked">Top Ranked</a>
    <a href="#egitim-gorselleri">Egitim Gorselleri</a>
    <a href="#ek-gorseller">Ek Gorseller</a>
    <a href="#gorsel-2d">2D Peptit</a>
    <a href="#peptit-skor-paneli">2D + Skor</a>
    <a href="#gorsel-3d">3D Yapi</a>
    <a href="#ciktilar">Dosyalar</a>
    <a href="#sss">SSS</a>
  </nav>

  <div class="layout-wrap">
    <aside class="sidenav" aria-label="Icindekiler">
      <strong>Icindekiler</strong>
      <a href="#akim">Pipeline akisi</a>
      <a href="#amac">Amac &amp; gorev</a>
      <a href="#yontem">Yontem</a>
      <a href="#veri">Veri katmani</a>
      <a href="#metrikler">Test metrikleri</a>
      <a href="#top-ranked">Top ranked ornekler</a>
      <a href="#egitim-gorselleri">Egitim gorselleri</a>
      <a href="#ek-gorseller">Ek gorseller</a>
      <a href="#gorsel-2d">2D peptit</a>
      <a href="#peptit-skor-paneli">2D + skor</a>
      <a href="#gorsel-3d">3D yapi</a>
      <a href="#ciktilar">Cikti dosyalari</a>
      <a href="#sss">SSS</a>
      <a href="#manifest">manifest.json</a>
    </aside>

    <main class="content">
__PIPELINE_DIAGRAM_BLOCK__

      <section class="block" id="amac">
        <h2>Amac ve gorev tanimi</h2>
        <p class="lead-in">
          Model yalnizca "bu cift baglanir mi?" sorusunu yanitmaz; her <strong>protein icin bir aday kumesi</strong>
          (genelde 1 gercek + birkac negatif) icinde pozitifi <strong>ust siralara</strong> tasir.
          Siralama icin <strong>MRR</strong> ve <strong>Hit@k</strong> kullanilirken, esik secimi icin
          <strong>AUROC / AUPRC / MCC</strong> gibi ikili siniflandirma metrikleri de raporlanir.
        </p>
        <div class="callout accent-callout">
          <strong>Ozet:</strong> Yapi + sekans ozelliklerinden skor &rarr; adaylar arasinda siralama &rarr;
          HTML / PNG / JSON ile izlenebilir, tekrarlanabilir rapor.
        </div>
      </section>

      <section class="block" id="yontem">
        <h2>Yontem — uctan uca akis</h2>
        <p class="lead-in">Her adimin ciktisi bir sonrakinin girdisidir. Ara urunler JSON/Parquet ile denetlenir.</p>
        <div class="method-diagram" role="img" aria-label="Yontem akis diyagrami">
          <div class="md-phase" data-phase="veri">
            <div class="md-phase-title">Veri Hazirligi</div>
            <div class="md-nodes">
              <div class="md-node"><strong>PROPEDIA mmCIF</strong><span>Ham yapisal dosyalar</span></div>
              <div class="md-arrow-v"></div>
              <div class="md-node"><strong>Kanonik tablolar</strong><span><code>complexes</code>, <code>chains</code>, <code>residues</code> Parquet</span></div>
              <div class="md-arrow-v"></div>
              <div class="md-node"><strong>Arayuz + Pocket</strong><span>5&#197; arayuz, 8&#197; pocket mesafe anotasyonu</span></div>
            </div>
          </div>
          <div class="md-connector-v"></div>
          <div class="md-phase" data-phase="split">
            <div class="md-phase-title">Split &amp; Negatif Uretimi</div>
            <div class="md-nodes">
              <div class="md-node"><strong>Sekans-kume split</strong><span>MMseqs2 %30 kimlik &rarr; train / val / test</span></div>
              <div class="md-arrow-v"></div>
              <div class="md-node"><strong>Negatif ciftler</strong><span><em>Easy:</em> rastgele peptit &middot; <em>Hard:</em> ayni protein ailesi</span></div>
            </div>
          </div>
          <div class="md-connector-v"></div>
          <div class="md-phase" data-phase="model">
            <div class="md-phase-title">Model Egitimi</div>
            <div class="md-nodes">
              <div class="md-node"><strong>Ozellik ihraci</strong><span>Yapi + sekans + arayuz + yerel yogunluk &rarr; dense vektor</span></div>
              <div class="md-arrow-v"></div>
              <div class="md-node"><strong>MLX MLP egitimi</strong><span>BCE + pairwise ranking loss; val MRR ile erken durdurma</span></div>
            </div>
          </div>
          <div class="md-connector-v"></div>
          <div class="md-phase" data-phase="rapor">
            <div class="md-phase-title">Degerlendirme &amp; Rapor</div>
            <div class="md-nodes">
              <div class="md-node"><strong>Test metrikleri</strong><span>AUROC, MRR, Hit@k, F1, MCC</span></div>
              <div class="md-arrow-v"></div>
              <div class="md-node"><strong>Rapor &amp; site</strong><span>ROC/PR egrileri, 2D peptit, 3D viewer, bu site</span></div>
            </div>
          </div>
        </div>
      </section>

      <section class="block" id="veri">
        <h2>Veri katmani</h2>
        <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th>Bilesen</th><th>Konum</th><th>Aciklama</th></tr></thead>
          <tbody>
            <tr><td>Kanonik kompleksler</td><td><code>data/canonical/complexes.parquet</code></td><td>Protein-peptid ciftlerinin ana tablosu</td></tr>
            <tr><td>Zincir bilgileri</td><td><code>data/canonical/chains.parquet</code></td><td>Sekans, uzunluk, zincir turu (protein/peptit)</td></tr>
            <tr><td>Rezidu detaylari</td><td><code>data/canonical/residues.parquet</code></td><td>Koordinat, is_interface, is_pocket, local_density</td></tr>
            <tr><td>Sekans-kume split</td><td><code>data/canonical/splits/*.txt</code></td><td>Kume-bazli train/val/test PDB ayirimi</td></tr>
            <tr><td>Ciftler + negatifler</td><td><code>data/canonical/pairs/*.parquet</code></td><td>Pozitif + easy / hard negatif ciftler</td></tr>
            <tr><td>Veri raporlari</td><td><code>pair_data_report.json</code></td><td>Dagilim ve butunluk kontrol raporlari</td></tr>
          </tbody>
        </table>
        </div>
        <div class="stats-bar">
          <div class="stat-item"><span class="stat-val">__M_TRAIN_GROUPS__</span><span class="stat-label">Train grup</span></div>
          <div class="stat-item"><span class="stat-val">__M_VAL_GROUPS__</span><span class="stat-label">Val grup</span></div>
          <div class="stat-item"><span class="stat-val">__M_TEST_GROUPS__</span><span class="stat-label">Test grup</span></div>
        </div>
      </section>

      <section class="block" id="metrikler">
        <h2>Test metrikleri</h2>
        <p class="lead-in">En iyi modelin <strong>test kumesi</strong> uzerindeki performansi.
          Kartlarda <strong>&mdash;</strong> gorunuyorsa build sirasinda <code>metrics.json</code> bulunamadi.</p>
        <div class="metrics-grid-8">
          <div class="metric-card primary"><div class="k">AUROC</div><div class="v">__M_AUROC__</div><div class="desc">Ikili ayirilabilirlik</div></div>
          <div class="metric-card"><div class="k">AUPRC</div><div class="v">__M_AUPRC__</div><div class="desc">Dengesiz sinif</div></div>
          <div class="metric-card primary"><div class="k">MRR</div><div class="v">__M_MRR__</div><div class="desc">Ort. ters sira</div></div>
          <div class="metric-card"><div class="k">Hit@1</div><div class="v">__M_HIT1__</div><div class="desc">Ilk sirada pozitif</div></div>
          <div class="metric-card highlight"><div class="k">Hit@3</div><div class="v">__M_HIT3__</div><div class="desc">Ilk 3te pozitif</div></div>
          <div class="metric-card"><div class="k">Hit@5</div><div class="v">__M_HIT5__</div><div class="desc">Ilk 5te pozitif</div></div>
          <div class="metric-card"><div class="k">F1</div><div class="v">__M_F1__</div><div class="desc">Esik-bazli F1</div></div>
          <div class="metric-card"><div class="k">MCC</div><div class="v">__M_MCC__</div><div class="desc">Matthews kor.</div></div>
        </div>
        <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th>Metrik</th><th>Ne olcer?</th><th>Neden onemli?</th></tr></thead>
          <tbody>
            <tr><td><strong>MRR</strong></td><td>Dogru adayin siralamadaki ters ortalamasidir</td><td>Reranking performansinin ana gostergesi</td></tr>
            <tr><td><strong>Hit@k</strong></td><td>Ilk k aday icinde pozitif var mi?</td><td>Pratik kisa liste basarisi</td></tr>
            <tr><td><strong>AUROC</strong></td><td>Skor ile ikili ayirilabilirlik (esik bagimsiz)</td><td>Genel siniflandirma gucu</td></tr>
            <tr><td><strong>AUPRC</strong></td><td>Precision-recall egrisi altindaki alan</td><td>Dengesiz sinif senaryolari icin tamamlayici</td></tr>
            <tr><td><strong>F1 / MCC</strong></td><td>Secilen esikte kesin siniflandirma</td><td>Operasyonel karar esigi basarisi</td></tr>
          </tbody>
        </table>
        </div>
      </section>

      <section class="block" id="top-ranked">
        <h2>Top ranked tahmin ornekleri</h2>
        <p class="lead-in">Test kumesinden aday gruplarin ilk siralamadaki ornekleri. <span class="label-chip pos">Pozitif</span> = native kristal cifti,
          <span class="label-chip neg">Negatif</span> = dekoy ornegi. Skor: model uretimi baglanti olasiligi (0-1).</p>
        <div class="table-scroll">
        <table class="data-table ranked-table">
          <thead>
            <tr>
              <th>PDB</th><th>Protein</th><th>Peptit</th><th>Uzunluk</th>
              <th>Skor</th><th>Etiket</th><th>Sira</th>
            </tr>
          </thead>
          <tbody>
            __TOP_RANKED_ROWS__
          </tbody>
        </table>
        </div>
        <p class="lead-in" style="margin-top:0.75rem;font-size:0.88rem">Tam liste: <code>top_ranked_examples.json</code> dosyasindaki <code>top_ranked_candidates_preview</code> alani.</p>
      </section>

__TRAINING_GALLERY_SECTION__

      <section class="block" id="ablation">
        <h2>Ablasyon ve model karsilastirma gorselleri</h2>
        <p class="lead-in">
          <strong>Isi haritasi:</strong> smoke asamasinda denenen kosullarin <strong>validation MRR</strong> degerleri.
          <strong>Model karsilastirma:</strong> tam egitimden sonra secilen en iyi kosullarin test metrikleri.
        </p>
        <div class="grid2">
          <figure class="media">
            <img src="assets/img/ablation_heatmap.png" alt="Ablasyon isi haritasi" loading="lazy" />
            <figcaption>Ablasyon isi haritasi (validation MRR)</figcaption>
          </figure>
          <figure class="media">
            <img src="assets/img/model_family_comparison.png" alt="Model ailesi karsilastirmasi" loading="lazy" />
            <figcaption>Test metrikleri karsilastirmasi (MRR, Hit@3, AUROC)</figcaption>
          </figure>
        </div>
      </section>

__SITE_EXTRA_VIZ__

      <section class="block" id="gorsel-2d">
        <h2>2D peptit gorselleri</h2>
        <p class="lead-in">
          RDKit ile tek harf aminoasit dizisinden turetilen <strong>2D bag yapisi</strong>.
          Her gorselde peptidin PDB kodu, zincir kimligi, sekans, model skoru ve etiket bilgisi yer alir.
        </p>
        <div class="two-col-info">
          <div>
            <h3>Amac</h3>
            <ul>
              <li>Peptidin <strong>kimyasal bag baglamini</strong> hizlica gostermek</li>
              <li>3D yapi dosyasi acilmadan <strong>hangi zincirin</strong> analiz edildigini hatirlatmak</li>
              <li>Farkli uzunluk ve skorlardaki peptitleri karsilastirmak</li>
            </ul>
          </div>
          <div>
            <h3>Gorsel icerigi</h3>
            <ul>
              <li><strong>pair_id:</strong> Benzersiz cift tanimlayicisi</li>
              <li><strong>PDB + zincir:</strong> Kaynak yapisal veri</li>
              <li><strong>Skor:</strong> Model uretimi baglanti olasiligi (0&ndash;1)</li>
              <li><strong>Etiket:</strong> Pozitif (native) veya negatif (decoy)</li>
            </ul>
          </div>
        </div>
        <figure class="media">
          <img src="assets/img/peptide_2d_example.png" alt="Ornek peptide 2D PNG" loading="lazy" />
          <figcaption>Genel ornek 2D peptit gorseli. Asagida farkli skor ve uzunluklarda ornekler bulunur.</figcaption>
        </figure>
__PEPTIDE_2D_VARIANTS__
      </section>

      <section class="block" id="gorsel-3d">
        <h2>3D yapi goruntuleme</h2>
        <div class="callout accent-callout">
          <strong>Demo yapisi:</strong> Asagidaki onizleme RCSB <strong>1CRN</strong> (Crambin, 46 aa, tek zincir kucuk protein) yapisini gosterir.
          Gercek pipeline ciktisinda her protein-peptit cifti icin ayri viewer uretilir; protein ve peptit farkli renklerde boyanir,
          etkilesim ciftleri silindir ile gosterilir.
        </div>
        <div class="viewer-responsive">
          <iframe src="embed/viewer-demo.html" loading="lazy" title="3D yapi demo goruntuleyici"></iframe>
        </div>
        <p class="lead-in" style="margin-top:0.75rem">
          <strong>3Dmol.js</strong> ile tarayici icinde canli 3D goruntuleme. Fare ile dondurme, zoom, ve kontrol panelini kullanabilirsiniz.
          Tam ekran icin asagidaki butona basin.
        </p>
        <div class="grid2" style="margin-top:1rem">
          <div>
            <h3>Pipeline cikti dosyalari</h3>
            <div class="table-scroll">
            <table class="data-table compact">
              <tbody>
                <tr><td><code>viewer.html</code></td><td>Tam ekran gorsel; cartoon/stick/sphere kontrolleri</td></tr>
                <tr><td><code>viewer_state.json</code></td><td>Kompleks kimlik, zincir bilgisi, etkilesim listesi</td></tr>
                <tr><td><code>interaction_provenance.json</code></td><td>PLIP/Arpeggio veya geometrik fallback bilgisi</td></tr>
                <tr><td><code>report.html</code></td><td>Ozet rapor: 2D gorsel + gomulu 3D viewer</td></tr>
              </tbody>
            </table>
            </div>
          </div>
          <div>
            <h3>Gorunum kontrolleri</h3>
            <ul>
              <li><strong>Cartoon:</strong> Ikincil yapi seridi (helix, sheet, loop)</li>
              <li><strong>Stick:</strong> Atom baglari; detay icin</li>
              <li><strong>Sphere:</strong> Atom merkezleri; VDW yaricapi</li>
              <li><strong>Yuzey:</strong> Van der Waals yuzey kaplamasiyla</li>
              <li><strong>Arka plan:</strong> Beyaz, siyah veya lacivert</li>
            </ul>
          </div>
        </div>
        <div class="hero-cta" style="margin-top:1rem">
          <a class="btn-pill" href="embed/viewer-demo.html">Tam ekran 3D demo</a>
        </div>
      </section>

      <section class="block" id="ciktilar">
        <h2>Tipik egitim ciktilari</h2>
        <div class="grid2">
          <div>
            <h3>Metrikler &amp; raporlar</h3>
            <ul>
              <li><code>metrics.json</code> — tum test/val metrikleri</li>
              <li><code>ranking_metrics.json</code> — MRR, Hit@k detaylari</li>
              <li><code>calibration_metrics.json</code> — Brier skoru</li>
              <li><code>selection_summary.json</code> — secim politikasi</li>
            </ul>
          </div>
          <div>
            <h3>Gorseller &amp; tablolar</h3>
            <ul>
              <li><code>roc_curve.png</code>, <code>pr_curve.png</code></li>
              <li><code>confusion_matrix.png</code>, <code>calibration_curve.png</code></li>
              <li><code>score_histogram_pos_neg.png</code></li>
              <li><code>train_log.csv</code>, <code>top_ranked_examples.json</code></li>
            </ul>
          </div>
        </div>
      </section>

      <section class="block" id="sss">
        <h2>Sik sorulanlar</h2>
        <details class="expand"><summary>GitHub Pages'te grafikler yer tutucu gorunuyor?</summary>
          <div class="inner">CI ortaminda <code>outputs/training/</code> olmayabilir. Derleme yer tutucu PNG uretir. Gercek gorseller icin yerelde egitimi tamamlayip <code>python scripts/build_pages_site.py</code> calistirin.</div>
        </details>
        <details class="expand"><summary>2D gorsel ile skor uretiliyor mu?</summary>
          <div class="inner">Hayir. 2D gorsel yalnizca raporlama icindir. Skorlar yapi/sekans ozelliklerinden gelen model ciktisidir.</div>
        </details>
        <details class="expand"><summary>Sizinti-siz split nasil calisiyor?</summary>
          <div class="inner">MMseqs2 ile protein sekanslari %30 kimlik esiginde kumelenir. Ayni kumedeki tum kompleksler ayni split'e atanir, boylece egitim ve test arasinda homoloji sizintisi onlenir.</div>
        </details>
        <details class="expand"><summary>PLIP / Arpeggio neden calismamis gorunuyor?</summary>
          <div class="inner">Pipeline yalnizca <code>plip</code> ve <code>arpeggio</code> komutlari PATH'te ve basarili ise calistirir. Aksi halde geometrik fallback kullanilir.</div>
        </details>
        <details class="expand"><summary>Etiket 1 (pozitif) ne anlama geliyor?</summary>
          <div class="inner">Etiket 1 = kristalde birlikte gozlenen (native/co-crystal) protein-peptit cifti. Model bu cifti tanimayi ve aday kumesi icinde ust siralara tasimayi ogrenir.</div>
        </details>
      </section>

      <section class="block" id="manifest">
        <h2>manifest.json</h2>
        <pre class="json">__MANIFEST_JSON__</pre>
      </section>
    </main>
  </div>

  <footer class="site-ft">
    <div class="footer-inner">
      <span>__PROJECT_NAME__ v0.1</span>
      <span>Statik site: <code>scripts/build_pages_site.py</code></span>
      <span><a href="__REPO_URL__">GitHub</a></span>
      <span><a href="https://atakan-emre.github.io/PeptiProp/">Pages</a></span>
    </div>
  </footer>
  </div>
  <div class="lightbox-overlay" id="lightbox" role="dialog" aria-modal="true" aria-label="Gorsel onizleme" style="display:none">
    <button class="lightbox-close" aria-label="Kapat">&times;</button>
    <img class="lightbox-img" id="lightbox-img" src="" alt="" />
  </div>
  <script src="assets/js/site-theme.js" defer></script>
  <script>
(function(){
  var overlay=document.getElementById('lightbox'),img=document.getElementById('lightbox-img');
  document.querySelectorAll('figure.media img').forEach(function(el){
    el.style.cursor='zoom-in';
    el.addEventListener('click',function(){
      img.src=el.src; img.alt=el.alt;
      overlay.style.display='flex';
      document.body.style.overflow='hidden';
    });
  });
  function closeLb(){overlay.style.display='none';document.body.style.overflow='';}
  overlay.addEventListener('click',function(e){if(e.target!==img)closeLb();});
  document.querySelector('.lightbox-close').addEventListener('click',closeLb);
  document.addEventListener('keydown',function(e){if(e.key==='Escape')closeLb();});
})();
  </script>
</body>
</html>
"""


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / ".nojekyll").touch()

    write_site_css(SITE)
    write_theme_js(SITE)
    training_dir = _resolve_primary_training_dir()
    if training_dir and training_dir.resolve() == PAGES_TRAINING_BUNDLE.resolve():
        print("[INFO] Eğitim kaynağı: publish/github_pages_training_bundle (CI / senkron paket)")
    manifest = build_manifest(training_dir)
    (SITE / "data").mkdir(parents=True, exist_ok=True)

    img = SITE / "assets" / "img"
    img.mkdir(parents=True, exist_ok=True)
    for fname, label in (
        ("ablation_heatmap.png", "Ablasyon ısı haritası (validation MRR)"),
        ("model_family_comparison.png", "Model ailesi karşılaştırması"),
    ):
        dest = img / fname
        src = training_dir / fname if training_dir else None
        if src is not None and src.is_file():
            _copy_if(src, dest)
        elif not dest.is_file():
            write_placeholder_png(dest, label, "Yerelde ablation / eğitim çıktısı gerekir")
    training_fig_items = _copy_training_figures(training_dir, img)
    manifest["training_figure_assets"] = [x["href"] for x in training_fig_items]
    gallery_section = _render_training_gallery_section(training_fig_items)

    p2d = _first_glob(
        [
            "outputs/**/peptide_2d.png",
            "outputs/analysis_propedia*/**/peptide_2d.png",
        ]
    )
    p2d_dest = img / "peptide_2d_example.png"
    if p2d:
        _copy_if(p2d, p2d_dest)
    elif not p2d_dest.is_file():
        write_placeholder_png(p2d_dest, "Örnek peptit 2D PNG", "Pipeline veya sanity çıktısı gerekir")

    site_extra_html = ""
    peptide_variants_html = ""
    try:
        from peptidquantum.visualization.plots.site_extras import generate_site_extra_assets

        extra_manifest = generate_site_extra_assets(ROOT, SITE)
        manifest["site_extra_figures"] = extra_manifest.get("site_extra_figures", [])
        manifest["site_extra_pages"] = extra_manifest.get("site_extra_pages", [])
    except Exception as exc:
        print(f"[WARN] Ek site görselleri atlandı: {exc}")
        manifest.setdefault("site_extra_figures", [])
        manifest.setdefault("site_extra_pages", [])

    for fname, t1, t2 in (
        ("peptide_length_histogram.png", "Peptit uzunluk dağılımı", "Kanonik parquet veya yer tutucu"),
        ("interaction_summary_panel.png", "Etkileşim özet paneli", "analysis_propedia çıktısı veya yer tutucu"),
    ):
        pth = img / fname
        if not pth.is_file():
            write_placeholder_png(pth, t1, t2)

    try:
        from peptidquantum.visualization.plots.site_extras import (
            html_extra_viz_section,
            html_peptide_2d_variants_section,
        )

        site_extra_html = html_extra_viz_section(SITE)
        peptide_variants_html = html_peptide_2d_variants_section(SITE)
    except Exception:
        site_extra_html = ""
        peptide_variants_html = ""

    extra_seen = set(manifest.get("site_extra_figures") or [])
    for fn in ("peptide_length_histogram.png", "interaction_summary_panel.png"):
        if (img / fn).is_file():
            extra_seen.add(f"assets/img/{fn}")
    for vp in sorted(img.glob("peptide_2d_v*.png")):
        extra_seen.add(f"assets/img/{vp.name}")
    manifest["site_extra_figures"] = sorted(extra_seen)

    demo_cif = SITE / "assets" / "demo" / "1crn.cif"
    if not demo_cif.is_file():
        download_demo_cif(demo_cif)

    with open(SITE / "data" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    write_demo_viewer(SITE)
    write_index(SITE, manifest, gallery_section, training_dir, site_extra_html, peptide_variants_html)
    print(f"[OK] GitHub Pages site: {SITE}")


if __name__ == "__main__":
    main()
