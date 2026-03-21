#!/usr/bin/env python3
"""
Yerel eğitim çıktısından GitHub Pages için repoya girecek dosyaları kopyalar.

outputs/ .gitignore’da olduğu için CI metrik/görsel göremez; bu klasör commit edilir.

Kullanım:
  python scripts/sync_pages_training_bundle.py
  python scripts/sync_pages_training_bundle.py path/to/training_folder
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "publish" / "github_pages_training_bundle"
DEFAULT_SRC = ROOT / "outputs" / "training" / "peptidquantum_v0_1_final_best_mlx_ablation"

# Site derlemesi + manifest birleştirme için yeterli; gereksiz ağır dosya kopyalanmaz.
ALLOW_JSON = frozenset({"metrics.json", "ranking_metrics.json", "top_ranked_examples.json"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Pages yayın paketini eğitim klasöründen doldurur.")
    ap.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SRC,
        help=f"metrics.json içeren eğitim dizini (varsayılan: {DEFAULT_SRC})",
    )
    args = ap.parse_args()
    src: Path = args.source.expanduser().resolve()
    if not (src / "metrics.json").is_file():
        print(f"[HATA] metrics.json yok: {src}", file=sys.stderr)
        sys.exit(1)

    DEST.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() == ".png" or p.name in ALLOW_JSON:
            shutil.copy2(p, DEST / p.name)
            n += 1

    print(f"[OK] {n} dosya → {DEST}")
    print("     Commit: git add publish/github_pages_training_bundle && git commit -m 'Pages: eğitim bundle güncelle'")


if __name__ == "__main__":
    main()
