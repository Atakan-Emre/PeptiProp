"""GitHub Pages için ek görseller: peptit uzunluğu, etkileşim paneli, kompleks kartları."""
from __future__ import annotations

import html as html_module
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

# Standart 3 harf → 1 harf (chains.sequence birleşik 3-harf kodları için)
_AA3_TO_1: Dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "XAA": "X",
}


def triseq_to_oneletter(seq3: str) -> str:
    s = (seq3 or "").strip().upper()
    if not s:
        return "—"
    if len(s) % 3 != 0:
        return s[:36] + ("…" if len(s) > 36 else "")
    return "".join(_AA3_TO_1.get(s[i : i + 3], "X") for i in range(0, len(s), 3))


def plot_peptide_length_histogram(complexes_parquet: Path, out_png: Path, dpi: int = 120) -> bool:
    if not complexes_parquet.is_file():
        return False
    df = pd.read_parquet(complexes_parquet, columns=["peptide_length"])
    lengths = df["peptide_length"].dropna().astype(int)
    if lengths.empty:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    ax.hist(lengths, bins=min(40, max(10, lengths.nunique())), color="#2f81f7", edgecolor="#1a1a2e", alpha=0.85)
    ax.set_xlabel("Peptit uzunluğu (aa)")
    ax.set_ylabel("Kompleks sayısı")
    ax.set_title("Kanonik set: peptit uzunluk dağılımı (PROPEDIA)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png.is_file()


def stitch_interaction_summary_panel(
    root: Path,
    out_png: Path,
    patterns: Tuple[str, ...] = (
        "outputs/analysis_propedia_batch_mlx/**/figures/interaction_summary.png",
        "outputs/analysis_propedia_top_ranked_batch_mlx/**/figures/interaction_summary.png",
    ),
    max_images: int = 4,
) -> bool:
    found: List[Path] = []
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                found.append(p)
    # Tek kompleks klasöründen bir kez
    by_complex: Dict[str, Path] = {}
    for p in sorted(found, key=lambda x: str(x)):
        parts = p.parts
        try:
            oi = parts.index("outputs")
            cid = parts[oi + 2] if len(parts) > oi + 2 else p.parent.parent.name
        except ValueError:
            cid = p.parent.parent.name
        if cid not in by_complex:
            by_complex[cid] = p
        if len(by_complex) >= max_images:
            break
    paths = list(by_complex.values())[:max_images]
    if len(paths) < 1:
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    n = len(paths)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.2 * rows), dpi=110)
    axes_flat = np.atleast_1d(axes).ravel()
    for i, pth in enumerate(paths):
        ax = axes_flat[i]
        try:
            img = mpimg.imread(str(pth))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Okunamadı", ha="center", va="center")
        ax.set_title(pth.parts[pth.parts.index("outputs") + 2] if "outputs" in pth.parts else pth.parent.parent.name, fontsize=9)
        ax.axis("off")
    for j in range(len(paths), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Örnek kompleksler: interaction_summary.png (pipeline çıktısı)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png.is_file()


def write_complex_cards_html(
    root: Path,
    site_embed: Path,
    complexes_parquet: Path,
    chains_parquet: Path,
    max_cards: int = 6,
) -> bool:
    """viewer_state.json yollarından örnek kart sayfası (sekans + temas sayısı + provenance)."""
    if not complexes_parquet.is_file() or not chains_parquet.is_file():
        return False
    chains = pd.read_parquet(chains_parquet)
    states: List[Path] = []
    for pat in (
        "outputs/analysis_propedia_batch_mlx/**/data/viewer_state.json",
        "outputs/analysis_propedia_top_ranked_batch_mlx/**/data/viewer_state.json",
    ):
        states.extend(root.glob(pat))
    seen: set[str] = set()
    uniq_states: List[Path] = []
    for p in sorted(states, key=lambda x: str(x)):
        cid = p.parts[p.parts.index("outputs") + 2] if "outputs" in p.parts else ""
        if cid in seen:
            continue
        seen.add(cid)
        uniq_states.append(p)
        if len(uniq_states) >= max_cards:
            break
    if not uniq_states:
        return False
    rows_html: List[str] = []
    for vp in uniq_states:
        try:
            data = json.loads(vp.read_text(encoding="utf-8"))
        except Exception:
            continue
        cid = str(data.get("complex_id") or vp.parent.parent.name)
        prov_path = vp.parent / "interaction_provenance.json"
        prov: Dict[str, Any] = {}
        if prov_path.is_file():
            try:
                prov = json.loads(prov_path.read_text(encoding="utf-8"))
            except Exception:
                prov = {}
        n_ix = len(data.get("interactions") or [])
        pep_chain = None
        for c in data.get("chains") or []:
            if c.get("type") == "peptide":
                pep_chain = str(c.get("chain_id"))
                break
        seq1 = "—"
        if pep_chain:
            sel = chains[(chains["complex_id"] == cid) & (chains["chain_id_auth"] == pep_chain)]
            if not sel.empty and pd.notna(sel.iloc[0].get("sequence")):
                seq1 = triseq_to_oneletter(str(sel.iloc[0]["sequence"]))
        mode = prov.get("extraction_mode") or "—"
        tool_f = prov.get("tool_based_interaction_fraction")
        tool_s = f"{float(tool_f):.0%}" if isinstance(tool_f, (int, float)) else "—"
        rows_html.append(
            f"""<tr>
<td><code>{html_module.escape(cid)}</code></td>
<td><code>{html_module.escape(seq1[:80])}</code>{'…' if len(seq1) > 80 else ''}</td>
<td>{n_ix}</td>
<td><code>{html_module.escape(str(mode))}</code></td>
<td>{html_module.escape(tool_s)}</td>
</tr>"""
        )
    if not rows_html:
        return False
    site_embed.mkdir(parents=True, exist_ok=True)
    body = f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Örnek kompleks kartları</title>
  <link rel="stylesheet" href="../assets/css/site.css" />
</head>
<body class="site-body">
  <div class="demo-page" style="max-width:960px;margin:0 auto;padding:1rem 1.25rem 2rem">
    <p><a class="btn-pill outline" href="../index.html">← Ana sayfa</a></p>
    <h1 style="margin-top:1rem">Örnek kompleks kartları</h1>
    <p class="lead-in">Kanonik <code>chains.parquet</code> sekansı (1 harf) ve <code>viewer_state.json</code> temas sayısı.
    PLIP/Arpeggio kullanılmadıysa <code>extraction_mode</code> genelde <code>geometric_fallback</code> olur; ayrıntı için <code>docs/VERI_VE_GORSEL_GERCEK_TR.md</code>.</p>
    <div class="table-scroll">
    <table class="data-table">
      <thead><tr><th>Kompleks</th><th>Peptit (1 harf)</th><th>Temas sayısı</th><th>Mod</th><th>Araç oranı</th></tr></thead>
      <tbody>
      {''.join(rows_html)}
      </tbody>
    </table>
    </div>
  </div>
</body>
</html>
"""
    (site_embed / "complex-cards.html").write_text(body, encoding="utf-8")
    return True


def generate_site_extra_assets(root: Path, site: Path) -> Dict[str, Any]:
    """
    site/assets/img altına PNG üretir; site/embed/complex-cards.html yazar.
    Dönüş: manifest'e eklenecek özet alanlar.
    """
    img_dir = site / "assets" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    cx = root / "data" / "canonical" / "complexes.parquet"
    ch = root / "data" / "canonical" / "chains.parquet"
    out: Dict[str, Any] = {"site_extra_figures": [], "site_extra_pages": []}

    hist = img_dir / "peptide_length_histogram.png"
    if plot_peptide_length_histogram(cx, hist):
        out["site_extra_figures"].append("assets/img/peptide_length_histogram.png")

    panel = img_dir / "interaction_summary_panel.png"
    if stitch_interaction_summary_panel(root, panel):
        out["site_extra_figures"].append("assets/img/interaction_summary_panel.png")

    if write_complex_cards_html(root, site / "embed", cx, ch):
        out["site_extra_pages"].append("embed/complex-cards.html")

    return out


def html_extra_viz_section(site: Path) -> str:
    """index.html içine gömülecek blok (dosya yoksa boş)."""
    parts: List[str] = []
    hist = site / "assets" / "img" / "peptide_length_histogram.png"
    panel = site / "assets" / "img" / "interaction_summary_panel.png"
    cards = site / "embed" / "complex-cards.html"
    if not any(p.is_file() for p in (hist, panel, cards)):
        return ""
    parts.append('      <section class="block" id="ek-gorseller">')
    parts.append("        <h2>Ek veri görselleri</h2>")
    parts.append(
        '        <p class="lead-in">Kanonik tablolardan ve yerel pipeline çıktılarından otomatik üretilir '
        "(<code>python scripts/build_pages_site.py</code>). PLIP/Arpeggio devreye girmemiş çalıştırmalarda "
        "temaslar <strong>geometrik fallback</strong> ile sayılır.</p>"
    )
    if hist.is_file():
        parts.append(
            '        <figure class="media">\n'
            f'          <img src="assets/img/peptide_length_histogram.png" alt="Peptit uzunluk histogramı" loading="lazy" />\n'
            "          <figcaption>Kanonik komplekslerde peptit uzunluğu dağılımı.</figcaption>\n"
            "        </figure>"
        )
    if panel.is_file():
        parts.append(
            '        <figure class="media">\n'
            f'          <img src="assets/img/interaction_summary_panel.png" alt="Etkileşim özet paneli" loading="lazy" />\n'
            "          <figcaption>Birden fazla komplexin <code>interaction_summary.png</code> birleşik görünümü (çıktı klasörü varsa).</figcaption>\n"
            "        </figure>"
        )
    if cards.is_file():
        parts.append(
            '        <p class="lead-in">Örnek kompleks tablosu (PDB kimliği, peptit sekansı, temas sayısı, etkileşim modu): '
            f'<a class="btn-pill outline" href="embed/complex-cards.html">complex-cards.html</a></p>'
        )
    parts.append("      </section>")
    return "\n".join(parts) + "\n"
