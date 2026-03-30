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


def resolve_top_ranked_examples_path(root: Path) -> Optional[Path]:
    """Yerel eğitim veya Pages bundle içinde top_ranked_examples.json."""
    bundle = root / "publish" / "github_pages_training_bundle" / "top_ranked_examples.json"
    if bundle.is_file():
        return bundle
    found = list(root.glob("outputs/training/**/top_ranked_examples.json"))
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def _row_label_eval(rec: Dict[str, Any]) -> int:
    if rec.get("label_eval") is not None:
        return int(rec["label_eval"])
    return int(rec.get("label", 0))


def _peptide_one_letter_sequence(
    chains_df: pd.DataFrame, peptide_complex_id: str, peptide_chain_id: str
) -> Optional[str]:
    cid, pid = str(peptide_complex_id), str(peptide_chain_id)
    for col in ("chain_id_auth", "chain_id"):
        if col not in chains_df.columns:
            continue
        sel = chains_df[(chains_df["complex_id"].astype(str) == cid) & (chains_df[col].astype(str) == pid)]
        if sel.empty:
            continue
        seq = sel.iloc[0].get("sequence")
        if pd.isna(seq):
            continue
        return triseq_to_oneletter(str(seq))
    return None


def _pick_ranked_variant_rows(preview: List[Dict[str, Any]], max_n: int = 5) -> List[Dict[str, Any]]:
    if not preview:
        return []
    pos = [r for r in preview if _row_label_eval(r) == 1]
    neg = [r for r in preview if _row_label_eval(r) == 0]
    picked: List[Dict[str, Any]] = []
    used: set[str] = set()

    def add(r: Optional[Dict[str, Any]]) -> None:
        if r is None or len(picked) >= max_n:
            return
        pid = str(r.get("pair_id", ""))
        if not pid or pid in used:
            return
        picked.append(r)
        used.add(pid)

    if pos:
        add(max(pos, key=lambda r: float(r.get("score", 0))))
    if pos and len(picked) < max_n:
        rest = [r for r in pos if str(r.get("pair_id", "")) not in used]
        if rest:
            add(min(rest, key=lambda r: float(r.get("score", 1))))
    if neg and len(picked) < max_n:
        rest = [r for r in neg if str(r.get("pair_id", "")) not in used]
        if rest:
            add(max(rest, key=lambda r: float(r.get("score", 0))))
    rest_all = [r for r in preview if str(r.get("pair_id", "")) not in used]
    if rest_all and len(picked) < max_n:
        add(max(rest_all, key=lambda r: int(r.get("peptide_length", 0) or 0)))
    rest_all = [r for r in preview if str(r.get("pair_id", "")) not in used]
    if rest_all and len(picked) < max_n:
        shortest = min(rest_all, key=lambda r: int(r.get("peptide_length", 999) or 999))
        if int(shortest.get("peptide_length", 0) or 0) >= 2:
            add(shortest)
    rest_all = [r for r in preview if str(r.get("pair_id", "")) not in used]
    if rest_all and len(picked) < max_n:
        scores = sorted(rest_all, key=lambda r: float(r.get("score", 0)))
        mid = scores[len(scores) // 2]
        add(mid)
    return picked[:max_n]


def generate_peptide_2d_variant_assets(root: Path, site: Path) -> Dict[str, Any]:
    """top_ranked_examples + chains → peptide_2d_v*.png + data/peptide_2d_variants.json (RDKit gerekir)."""
    out: Dict[str, Any] = {"peptide_2d_variants": [], "files": []}
    tr_path = resolve_top_ranked_examples_path(root)
    ch_path = root / "data" / "canonical" / "chains.parquet"
    if not tr_path or not ch_path.is_file():
        return out
    try:
        from peptidquantum.visualization.chemistry.peptide_2d import Peptide2DRenderer
    except ImportError:
        return out

    try:
        raw = json.loads(tr_path.read_text(encoding="utf-8"))
        preview = raw.get("top_ranked_candidates_preview") or []
        rows = _pick_ranked_variant_rows(preview, 5)
        if not rows:
            return out
        chains_df = pd.read_parquet(ch_path)
        img_dir = site / "assets" / "img"
        data_dir = site / "data"
        img_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        renderer = Peptide2DRenderer(img_size=(720, 380))
        variants: List[Dict[str, Any]] = []
        for i, row in enumerate(rows, start=1):
            pcx = str(row.get("peptide_complex_id", ""))
            pch = str(row.get("peptide_chain_id", ""))
            seq = _peptide_one_letter_sequence(chains_df, pcx, pch)
            if not seq:
                continue
            score = float(row.get("score", 0))
            lab = _row_label_eval(row)
            plen = int(row.get("peptide_length", len(seq)) or len(seq))
            pair_id = str(row.get("pair_id", ""))
            pdb = str(row.get("pdb_id", ""))
            neg_t = str(row.get("negative_type", ""))
            fname = f"peptide_2d_v{i}.png"
            dest = img_dir / fname
            seq_short = seq[:45] + ("…" if len(seq) > 45 else "")
            title = (
                f"{pair_id} | PDB {pdb} chain {pch} | len {plen} | "
                f"score={score:.4f} label={lab} | {seq_short}"
            )
            renderer.from_sequence(seq, dest, title=title)
            if not dest.is_file():
                continue
            href = f"assets/img/{fname}"
            cap = (
                f"<code>{html_module.escape(pair_id)}</code> · PDB {html_module.escape(pdb)} · "
                f"peptit {html_module.escape(pch)} · uzunluk {plen} · "
                f"model skoru <strong>{score:.4f}</strong> · etiket <strong>{lab}</strong> "
                f"({'yapısal pozitif' if lab == 1 else html_module.escape(neg_t or 'negatif')}) · "
                f"<code>{html_module.escape(seq[:40])}{'…' if len(seq) > 40 else ''}</code>"
            )
            variants.append({"file": href, "caption_html": cap, "alt": f"Peptit 2D {pair_id}"})
            out["files"].append(href)
        if variants:
            (data_dir / "peptide_2d_variants.json").write_text(
                json.dumps({"variants": variants}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            out["peptide_2d_variants"] = variants
    except Exception:
        return out
    return out


def html_peptide_2d_variants_section(site: Path) -> str:
    man_path = site / "data" / "peptide_2d_variants.json"
    if not man_path.is_file():
        return ""
    try:
        payload = json.loads(man_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    variants = payload.get("variants") or []
    if not variants:
        return ""
    doc_url = "https://github.com/Atakan-Emre/PeptiProp/blob/main/docs/POZITIF_PEPTIT_VE_SKOR_TR.md"
    lines: List[str] = [
        '        <h3 id="peptit-skor-paneli">Test örnekleri: farklı skor, etiket ve uzunlukta 2D peptitler</h3>',
        '        <p class="lead-in"><strong>Etiket 1</strong> = kristalde birlikte görünen (native) peptit; '
        "<strong>skor</strong> = model çıktısı. "
        "<code>top_ranked_examples.json</code> + <code>chains.parquet</code> gerekir (RDKit ile site derlemesinde üretilir). "
        f'<a href="{html_module.escape(doc_url)}">Pozitif peptit ve skor (TR)</a></p>',
        '        <div class="grid2 training-fig-grid">',
    ]
    for v in variants:
        href = html_module.escape(v.get("file", ""))
        alt = html_module.escape(v.get("alt", "Peptit 2D"))
        cap = v.get("caption_html", "")
        lines.append('        <figure class="media">')
        lines.append(f'          <img src="{href}" alt="{alt}" loading="lazy" />')
        lines.append(f"          <figcaption>{cap}</figcaption>")
        lines.append("        </figure>")
    lines.append("        </div>")
    return "\n".join(lines) + "\n"


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
        "outputs/analysis_propedia_batch_gnn/**/figures/interaction_summary.png",
        "outputs/analysis_propedia_top_ranked_batch_gnn/**/figures/interaction_summary.png",
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
        "outputs/analysis_propedia_batch_gnn/**/data/viewer_state.json",
        "outputs/analysis_propedia_top_ranked_batch_gnn/**/data/viewer_state.json",
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
    Final aktif hatta temas çizgileri <code>geometric_fallback</code> ile üretilir; ayrıntı için <code>docs/VERI_VE_GORSEL_GERCEK_TR.md</code>.</p>
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

    vart = generate_peptide_2d_variant_assets(root, site)
    for rel in vart.get("files") or []:
        if rel not in out["site_extra_figures"]:
            out["site_extra_figures"].append(rel)

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
        "(<code>python scripts/build_pages_site.py</code>). Final aktif hatta "
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
