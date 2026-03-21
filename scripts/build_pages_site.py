#!/usr/bin/env python3
"""
GitHub Pages statik site üretir: manifest, görseller (varsa), 3D demo, index.

Çalıştır: python scripts/build_pages_site.py
Çıktı: site/ (workflow bu klasörü yayınlar)
"""
from __future__ import annotations

import html as html_module
import json
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
MLX_BEST = ROOT / "outputs" / "training" / "peptidquantum_v0_1_final_best_mlx_ablation"
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
    return {
        "test_auroc": first(raw.get("test_auroc"), tm.get("auroc")),
        "test_auprc": first(raw.get("test_auprc"), tm.get("auprc")),
        "test_mrr": first(raw.get("test_mrr"), tr.get("mrr")),
        "test_hit3": hit3,
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
    if not td.is_dir():
        return None
    any_m = [x for x in td.iterdir() if x.is_dir() and (x / "metrics.json").is_file()]
    return max(any_m, key=lambda p: (p / "metrics.json").stat().st_mtime) if any_m else None


def download_demo_cif(dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(DEMO_CIF_URL, timeout=60) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as exc:
        print(f"[WARN] Demo CIF indirilemedi: {exc}")
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
    if not training_out_dir:
        return out
    for src_name, dest_name, alt, caption in _TRAINING_FIGURE_SPECS:
        if _copy_if(training_out_dir / src_name, img_root / dest_name):
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
          Aşağıdaki görseller, site derlemesinde keşfedilen eğitim çıktı klasöründen (<code>metrics.json</code> ile aynı dizin) otomatik kopyalanmıştır.
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

  fetch('../assets/demo/1crn.cif')
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

.site-toolbar {
  position: fixed;
  top: max(0.5rem, env(safe-area-inset-top));
  right: max(0.5rem, env(safe-area-inset-right));
  z-index: 200;
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
  max-width: min(1320px, 100%);
  margin: 0 auto;
  padding: 0 clamp(0.5rem, 2.5vw, 1rem);
}
@media (min-width: 1100px) {
  .layout-wrap { grid-template-columns: minmax(0, 220px) minmax(0, 1fr); align-items: start; padding: 0 clamp(0.75rem, 2vw, 1.25rem); }
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
  padding: clamp(1.5rem, 4vw, 2.75rem) clamp(1rem, 4vw, 1.75rem) clamp(1.25rem, 3vw, 2.25rem);
  padding-right: clamp(1rem, 4vw, 1.75rem);
  border-bottom: 1px solid var(--border);
  background: radial-gradient(ellipse 120% 80% at 20% -20%, var(--hero-spot), transparent 55%),
              linear-gradient(180deg, var(--hero-mid) 0%, var(--bg) 100%);
}
header.hero .inner { max-width: 52rem; }
@media (max-width: 720px) {
  header.hero .inner { padding-right: 5.85rem; }
}
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
  display: flex; flex-wrap: wrap;
  gap: 0.45rem 0.85rem;
  row-gap: 0.5rem;
  padding: clamp(0.65rem, 2vw, 0.85rem) clamp(0.75rem, 3vw, 1.75rem);
  background: var(--card);
  border-bottom: 1px solid var(--border);
  font-size: clamp(0.8rem, 2.4vw, 0.88rem);
}
nav.toc-inline a { color: var(--muted); white-space: nowrap; }
nav.toc-inline a:hover { color: var(--accent-2); }

main.content {
  padding: clamp(1rem, 3.5vw, 1.75rem);
  max-width: min(920px, 100%);
  width: 100%;
  min-width: 0;
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
.table-scroll table.data-table { margin: 0; min-width: 480px; }
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

footer.site-ft {
  padding: clamp(1.25rem, 3vw, 2rem) clamp(1rem, 3vw, 1.75rem);
  color: var(--muted);
  font-size: clamp(0.78rem, 2vw, 0.82rem);
  text-align: center;
  border-top: 1px solid var(--border);
  background: var(--card);
}
""",
        encoding="utf-8",
    )


def _fmt_metric(m: Dict[str, Any], k: str) -> str:
    v = m.get(k)
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def write_index(
    site: Path,
    manifest: Dict[str, Any],
    training_gallery_section: str,
    site_extra_viz_section: str = "",
) -> None:
    m = _metrics_for_index(manifest)
    manifest_esc = html_module.escape(json.dumps(manifest, indent=2, ensure_ascii=False))

    idx = _INDEX_HTML_TEMPLATE
    reps = {
        "__THEME_HEAD_SCRIPT__": THEME_INLINE_HEAD,
        "__PROJECT_NAME__": PROJECT_DISPLAY_NAME,
        "__M_AUROC__": _fmt_metric(m, "test_auroc"),
        "__M_AUPRC__": _fmt_metric(m, "test_auprc"),
        "__M_MRR__": _fmt_metric(m, "test_mrr"),
        "__M_HIT3__": _fmt_metric(m, "test_hit3"),
        "__MANIFEST_JSON__": manifest_esc,
        "__TRAINING_GALLERY_SECTION__": training_gallery_section,
        "__SITE_EXTRA_VIZ__": site_extra_viz_section,
    }
    for k, v in reps.items():
        idx = idx.replace(k, v)
    (site / "index.html").write_text(idx, encoding="utf-8")


# Yer tutucular: __M_AUROC__ … __MANIFEST_JSON__
_INDEX_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="__PROJECT_NAME__ — PROPEDIA protein-peptid skorlama, ablasyon ve 2D/3D görselleştirme dokümantasyonu" />
  <title>__PROJECT_NAME__ — PROPEDIA skorlama &amp; görselleştirme</title>
  <link rel="stylesheet" href="assets/css/site.css" />
__THEME_HEAD_SCRIPT__
</head>
<body class="site-body">
  <div class="site-toolbar">
    <button type="button" class="theme-toggle" data-theme-toggle aria-pressed="false" aria-label="Açık temaya geç">
      <span class="theme-toggle-text">Gündüz modu</span>
    </button>
  </div>
  <header class="hero">
    <div class="inner">
      <div class="badge-row">
        <span class="badge accent">PROPEDIA</span>
        <span class="badge">Leakage-free split</span>
        <span class="badge">Reranking</span>
        <span class="badge">Model eğitimi</span>
      </div>
      <h1>__PROJECT_NAME__</h1>
      <p class="lead">
        Yapısal protein–peptid komplekslerinde <strong>skor üretimi</strong>, aynı protein için üretilmiş
        <strong>aday peptitler arasında sıralama</strong> ve sonuçların <strong>2D kimya + 3D yapı</strong> ile
        raporlanması. Bu sayfa projenin amacını, veri/metod özünü, metrikleri ve görsel çıktıların ne anlama geldiğini özetler.
      </p>
      <div class="hero-cta">
        <a class="btn-pill" href="embed/viewer-demo.html">İnteraktif 3D demosu</a>
        <a class="btn-pill outline" href="data/manifest.json">manifest.json</a>
      </div>
    </div>
  </header>
  <nav class="toc-inline" aria-label="Sayfa içi">
    <a href="#amac">Amaç</a>
    <a href="#yontem">Yöntem</a>
    <a href="#veri">Veri</a>
    <a href="#ablation">Ablasyon</a>
    <a href="#metrikler">Metrikler</a>
    <a href="#egitim-gorselleri">Eğitim PNG</a>
    <a href="#ek-gorseller">Ek görseller</a>
    <a href="#gorsel-2d">2D peptit</a>
    <a href="#gorsel-3d">3D yapı</a>
    <a href="#ciktilar">Dosyalar</a>
    <a href="#sss">SSS</a>
    <a href="#manifest">manifest</a>
  </nav>

  <div class="layout-wrap">
    <aside class="sidenav" aria-label="İçindekiler">
      <strong>İçindekiler</strong>
      <a href="#amac">Amaç &amp; görev</a>
      <a href="#yontem">İş akışı</a>
      <a href="#veri">Veri katmanı</a>
      <a href="#ablation">Ablasyon görselleri</a>
      <a href="#metrikler">Metrik sözlüğü</a>
      <a href="#egitim-gorselleri">Eğitim görselleri</a>
      <a href="#ek-gorseller">Ek görseller</a>
      <a href="#gorsel-2d">2D: ne, neden</a>
      <a href="#gorsel-3d">3D: ne, neden</a>
      <a href="#ciktilar">Çıktı dosyaları</a>
      <a href="#sss">SSS</a>
      <a href="#manifest">Ham manifest</a>
    </aside>

    <main class="content">
      <section class="block" id="amac">
        <h2>Amaç ve görev tanımı</h2>
        <p class="lead-in">
          Model yalnızca “bu çift bağlanır mı?” demekle kalmaz; her <strong>protein için bir aday kümesi</strong>
          (genelde 1 gerçek + birkaç negatif) içinde pozitifi <strong>üst sıralara</strong> taşımayı hedefler.
          Bu yüzden sıralama tarafında öne çıkan metrikler <strong>MRR</strong> ve <strong>Hit@k</strong> iken, eşik seçimi için
          <strong>AUROC / AUPRC / MCC / Brier</strong> gibi ikili sınıflandırma ve kalibrasyon metrikleri de raporlanır.
        </p>
        <div class="callout">
          <strong>Özet cümle:</strong> Yapı ve sekans özelliklerinden skor → adaylar arasında sıralama → HTML/PNG/JSON ile
          izlenebilir, tekrarlanabilir rapor.
        </div>
      </section>

      <section class="block" id="yontem">
        <h2>Yöntem — uçtan uca akış</h2>
        <p class="lead-in">Aşağıdaki adımlar sırayla çalıştırılır; ara ürünler JSON/Parquet ile denetlenir.</p>
        <div class="flow-steps">
          <div class="step">Ham PROPEDIA → kanonik tablolar (<code>complexes</code>, <code>chains</code>, yapı dosyaları)</div>
          <div class="step">PDB düzeyinde yapı-temelli train / val / test ayrımı (sızıntı kontrolü)</div>
          <div class="step">Split-içi negatif üretimi: easy, hard (ve isteğe bağlı structure-hard); oranlar split’e göre ayarlanabilir</div>
          <div class="step">Özellik ihracı (grafik veya dense tensör yolu — kullanılan eğitim yığınına bağlı)</div>
          <div class="step">Model eğitimi; <strong>validasyon</strong> metrikleri ile erken durdurma ve eşik seçimi</div>
          <div class="step">Test raporu, eğriler, top-k tabloları; isteğe bağlı görsel sanity (2D/3D)</div>
        </div>
        <h3>Eğitim</h3>
        <p class="lead-in">Aynı veri ve split üzerinde model eğitimi; çıktılar <code>metrics.json</code>, eğriler ve sıralama raporları ile sabitlenir. Depoda donanıma göre farklı eğitim betikleri olabilir; statik site yalnızca keşfedilen <strong>tek bir</strong> çıktı klasöründen özet üretir.</p>
      </section>

      <section class="block" id="veri">
        <h2>Veri</h2>
        <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th>Bileşen</th><th>Konum</th><th>Not</th></tr></thead>
          <tbody>
            <tr><td>Kanonik kompleksler</td><td><code>data/canonical/</code></td><td>Tek doğruluk kaynağı</td></tr>
            <tr><td>Split kimlikleri</td><td><code>data/canonical/splits/*.txt</code></td><td>Train/val/test PDB ayrımı</td></tr>
            <tr><td>Çiftler + negatifler</td><td><code>data/canonical/pairs/*.parquet</code></td><td>Rapor: <code>pair_data_report.json</code></td></tr>
            <tr><td>Aday seti raporu</td><td><code>candidate_set_report.json</code></td><td>Hard shortfall, oranlar</td></tr>
          </tbody>
        </table>
        </div>
        <p class="lead-in">Bu statik sitedeki özet: <a href="data/manifest.json"><code>data/manifest.json</code></a> (CI/yerel build üretir).</p>
      </section>

      <section class="block" id="ablation">
        <h2>Ablasyon ve model karşılaştırma görselleri</h2>
        <p class="lead-in">
          <strong>Isı haritası:</strong> duman (smoke) aşamasında denenen hücrelerin <strong>validation MRR</strong> değerlerini gösterir;
          eksenlerde model ailesi, özellik seti (F1/F2), kayıp (L1/L2) ve müfredat (C0/C1) kombinasyonları bulunur.
          <strong>Model family comparison:</strong> tam eğitimden sonra her aileden seçilen en iyi koşunun test metriklerini yan yana koyar.
        </p>
        <div class="grid2">
          <figure class="media">
            <img src="assets/img/ablation_heatmap.png" alt="Ablasyon ısı haritası (validation MRR)" loading="lazy" />
            <figcaption>Görsel yoksa: yerelde ablasyon betiğini (ör. <code>run_final_ablation_mlx.py</code>) tamamlayıp <code>build_pages_site.py</code> ile siteyi yeniden üretin.</figcaption>
          </figure>
          <figure class="media">
            <img src="assets/img/model_family_comparison.png" alt="Model ailesi karşılaştırması" loading="lazy" />
            <figcaption>Bar grafik: seçilen full koşuların test özetleri (MRR, Hit@3, AUROC).</figcaption>
          </figure>
        </div>
      </section>

      <section class="block" id="metrikler">
        <h2>Metrikler — hızlı özet ve anlam</h2>
        <div class="metrics-grid">
          <div class="metric-card"><div class="k">AUROC</div><div class="v">__M_AUROC__</div></div>
          <div class="metric-card"><div class="k">AUPRC</div><div class="v">__M_AUPRC__</div></div>
          <div class="metric-card"><div class="k">MRR</div><div class="v">__M_MRR__</div></div>
          <div class="metric-card"><div class="k">Hit@3</div><div class="v">__M_HIT3__</div></div>
        </div>
        <p class="metric-empty-hint">Kartlarda <strong>—</strong> görünüyorsa build sırasında okunabilir bir <code>metrics.json</code> yoktu (GitHub Actions’ta <code>outputs/training/</code> çoğu zaman depoda yoktur). Yerelde eğitim sonrası siteyi yeniden üretin.</p>
        <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th>Metrik</th><th>Ne ölçer?</th><th>Neden önemli?</th></tr></thead>
          <tbody>
            <tr><td>MRR</td><td>Doğru adayın sıralamadaki ters sırası (ortalama)</td><td>Reranking ana göstergesi</td></tr>
            <tr><td>Hit@k</td><td>İlk k aday içinde pozitif var mı?</td><td>Pratik “kısa liste” başarısı</td></tr>
            <tr><td>AUROC / AUPRC</td><td>Skor ile ikili ayırılabilirlik</td><td>Eşik seçimi ve dengesiz sınıflar</td></tr>
            <tr><td>MCC / F1</td><td>Kesin sınıflandırma (seçilen eşikte)</td><td>Operasyonel karar eşiği</td></tr>
            <tr><td>Brier</td><td>Olasılık kalibrasyonu</td><td>Skorların güvenilirliği</td></tr>
          </tbody>
        </table>
        </div>
        <p class="lead-in">Tam sayılar: <code>manifest.json</code> içindeki <code>metrics</code> ve kaynak <code>training_dir</code> altındaki <code>metrics.json</code> (gerekirse <code>ranking_metrics.json</code>). Eski manifest’lerde <code>metrics_primary</code> vb. anahtarlar olabilir.</p>
      </section>

__TRAINING_GALLERY_SECTION__
__SITE_EXTRA_VIZ__

      <section class="block" id="gorsel-2d">
        <h2>2D peptit görseli — ne, nasıl, neden?</h2>
        <p class="lead-in">
          <strong>Ne?</strong> RDKit ile tek harf aminoasit dizisinden türetilen <strong>2D bağ yapısı</strong> (keküle benzeri çizim).
          Çıktı dosyası genelde <code>figures/peptide_2d.png</code>.
        </p>
        <h3>Amaç</h3>
        <ul>
          <li>Peptidin <strong>kimyasal bağ bağlamını</strong> raporda hızlıca göstermek (özellikle sunum ve ek materyal).</li>
          <li>3D yapı dosyası açılmadan bile okuyucuya <strong>hangi zincirin</strong> analiz edildiğini hatırlatmak.</li>
        </ul>
        <h3>Başlık (title) satırı</h3>
        <p class="lead-in">Üretimde başlık şu bilgileri birleştirir: <code>complex_id | Protein &lt;zincir&gt; | Peptide &lt;zincir&gt;: &lt;Sekans&gt;</code>.
        Böylece görsel tek başına <strong>hangi kompleks ve hangi peptit zinciri</strong> için olduğu anlaşılır.</p>
        <h3>3D’den farkı</h3>
        <p class="lead-in">2D çizim <strong>yapısal konformasyonu veya bağ arayüzünü</strong> göstermez; bağlanma yüzeyi, hidrojen bağları vb. için
        <strong>3D viewer</strong> ve isteğe bağlı PLIP/Arpeggio tabanlı etkileşim çizgileri kullanılır.</p>
        <figure class="media">
          <img src="assets/img/peptide_2d_example.png" alt="Örnek peptide 2D PNG" loading="lazy" />
          <figcaption>Örnek: pipeline veya sanity çıktısından otomatik kopyalanır. Yoksa görsel eksiktir — yerelde sanity çalıştırıp siteyi yeniden üretin.</figcaption>
        </figure>
      </section>

      <section class="block" id="gorsel-3d">
        <h2>3D yapı görüntüleme — ne, nasıl, neden?</h2>
        <p class="lead-in">
          <strong>Ne?</strong> Tarayıcıda <strong>3Dmol.js</strong> ile mmCIF/PDB okuma, protein için cartoon (+isteğe VDW yüzeyi),
          peptit için stick ve (varsa) kalıntı çiftleri arasında <strong>silindir</strong> ile etkileşim gösterimi.
        </p>
        <h3>Dosyalar</h3>
        <div class="table-scroll">
        <table class="data-table">
          <thead><tr><th>Dosya</th><th>İçerik</th></tr></thead>
          <tbody>
            <tr><td><code>viewer.html</code></td><td>Tam ekran görüntüleyici; kontrol düğmeleri (görünüm, yüzey, arka plan)</td></tr>
            <tr><td><code>data/viewer_state.json</code></td><td><code>complex_id</code>, <code>structure_format</code> (pdb|cif), <code>structure_basename</code>, <code>chains[]</code>, <code>interactions[]</code>, <code>view_config</code></td></tr>
            <tr><td><code>data/interaction_provenance.json</code></td><td>PLIP/Arpeggio oranları, fallback bilgisi</td></tr>
            <tr><td><code>report.html</code></td><td>Özet + gömülü viewer + 2D şekil</td></tr>
          </tbody>
        </table>
        </div>
        <div class="callout">
          <strong>Canlı demo:</strong> Bu depoda Pages ile yayınlanan <a href="embed/viewer-demo.html">embed/viewer-demo.html</a>
          aynı kontrol ailesini (cartoon/stick/sphere/line, yüzey, arka plan, sıfırlama) gösterir; örnek yapı 1CRN’dir.
          Tam protein–peptit renk ayrımı ve etkileşim çizgileri için yerel pipeline çıktılarına bakın.
        </div>
        <a class="btn-pill" href="embed/viewer-demo.html">3D demosunu aç</a>
      </section>

      <section class="block" id="ciktilar">
        <h2>Tipik eğitim çıktıları</h2>
        <ul>
          <li><code>metrics.json</code>, <code>ranking_metrics.json</code>, <code>calibration_metrics.json</code></li>
          <li><code>train_log.csv</code>, ROC/PR/karmaşa/kalibrasyon PNG</li>
          <li><code>test_topk_positive_hits.csv</code>, <code>top_ranked_examples.json</code></li>
          <li><code>selection_summary.json</code> (validasyonla seçim politikası)</li>
        </ul>
      </section>

      <section class="block" id="sss">
        <h2>Sık sorulanlar</h2>
        <details class="expand"><summary>GitHub Pages’te görseller neden boş?</summary>
          <div class="inner">CI, repo içindeki <code>outputs/...</code> dosyalarını kopyalar; klasör yoksa sadece metin ve demo 3D çalışır. Yerelde eğitim/ablation sonrası <code>build_pages_site.py</code> çalıştırıp commit öncesi artifact’ı kontrol edin.</div>
        </details>
        <details class="expand"><summary>2D ile skor üretiliyor mu?</summary>
          <div class="inner">Hayır; 2D görsel raporlama içindir. Skorlar yapı/seq özelliklerinden gelen modele aittir.</div>
        </details>
        <details class="expand"><summary>viewer_state.json içinde yapı metni var mı?</summary>
          <div class="inner">Hayır; dosya yolu ve format bilgisi tutulur. Yapı metni <code>viewer.html</code> içinde güvenli şekilde gömülüdür.</div>
        </details>
        <details class="expand"><summary>PLIP / Arpeggio neden hiç çalışmamış gibi görünüyor?</summary>
          <div class="inner">Pipeline yalnızca <code>plip</code> ve <code>arpeggio</code> komutları <code>PATH</code>’te ve <code>--help</code> başarılı ise çalıştırır (<code>pipeline.py</code>). Aksi halde doğrudan <strong>geometrik fallback</strong> (Cα mesafesi) kullanılır; <code>interaction_provenance.json</code> içinde <code>tools_succeeded</code> boş kalır. Kurulum: <code>scripts/verify_external_tools.py</code> ve <code>EXTERNAL_TOOLS.md</code>.</div>
        </details>
      </section>

      <section class="block" id="manifest">
        <h2>manifest.json (kaçışlı)</h2>
        <pre class="json">__MANIFEST_JSON__</pre>
      </section>
    </main>
  </div>

  <footer class="site-ft">
    __PROJECT_NAME__ — statik site <code>scripts/build_pages_site.py</code> ile üretilir.
    Workflow: <code>.github/workflows/pages.yml</code>
  </footer>
  <script src="assets/js/site-theme.js" defer></script>
</body>
</html>
"""


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / ".nojekyll").touch()

    write_site_css(SITE)
    write_theme_js(SITE)
    training_dir = _resolve_primary_training_dir()
    manifest = build_manifest(training_dir)
    (SITE / "data").mkdir(parents=True, exist_ok=True)

    img = SITE / "assets" / "img"
    img.mkdir(parents=True, exist_ok=True)
    if training_dir:
        _copy_if(training_dir / "ablation_heatmap.png", img / "ablation_heatmap.png")
        _copy_if(training_dir / "model_family_comparison.png", img / "model_family_comparison.png")
    training_fig_items = _copy_training_figures(training_dir, img)
    manifest["training_figure_assets"] = [x["href"] for x in training_fig_items]
    gallery_section = _render_training_gallery_section(training_fig_items)

    p2d = _first_glob(
        [
            "outputs/**/peptide_2d.png",
            "outputs/analysis_propedia*/**/peptide_2d.png",
        ]
    )
    if p2d:
        _copy_if(p2d, img / "peptide_2d_example.png")

    site_extra_html = ""
    try:
        from peptidquantum.visualization.plots.site_extras import (
            generate_site_extra_assets,
            html_extra_viz_section,
        )

        extra_manifest = generate_site_extra_assets(ROOT, SITE)
        manifest["site_extra_figures"] = extra_manifest.get("site_extra_figures", [])
        manifest["site_extra_pages"] = extra_manifest.get("site_extra_pages", [])
        site_extra_html = html_extra_viz_section(SITE)
    except Exception as exc:
        print(f"[WARN] Ek site görselleri atlandı: {exc}")

    demo_cif = SITE / "assets" / "demo" / "1crn.cif"
    if not demo_cif.is_file():
        download_demo_cif(demo_cif)

    with open(SITE / "data" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    write_demo_viewer(SITE)
    write_index(SITE, manifest, gallery_section, site_extra_html)
    print(f"[OK] GitHub Pages site: {SITE}")


if __name__ == "__main__":
    main()
