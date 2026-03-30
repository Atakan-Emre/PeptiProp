#!/usr/bin/env python3
"""
Copy a lightweight training artifact bundle into publish/github_pages_training_bundle.

GitHub Actions cannot see outputs/, so a commit-friendly snapshot is kept here.
The active default source is the GNN final run; MLX outputs are fallback candidates.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "publish" / "github_pages_training_bundle"
DEFAULT_CANDIDATES = [
    ROOT / "outputs" / "training" / "peptiprop_v0_2_gnn_esm2",
    ROOT / "outputs" / "training" / "peptidquantum_v0_1_final_best_mlx_ablation",
    ROOT / "outputs" / "training" / "peptidquantum_v0_1_final_mlx_m4",
]
ALLOW_SUFFIXES = {".png", ".json", ".csv", ".txt"}


def resolve_default_source() -> Path:
    for candidate in DEFAULT_CANDIDATES:
        if (candidate / "metrics.json").is_file():
            return candidate
    return DEFAULT_CANDIDATES[0]


def copy_surface(src: Path, dest: Path) -> int:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    seen = set()
    roots = [src / "figures", src]
    for base in roots:
        if not base.is_dir():
            continue
        for path in sorted(base.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ALLOW_SUFFIXES:
                continue
            if path.name in seen:
                continue
            shutil.copy2(path, dest / path.name)
            seen.add(path.name)
            copied += 1

    # Optional CI fallback assets produced by build_pages_site.py
    site_img = ROOT / "site" / "assets" / "img"
    site_data = ROOT / "site" / "data"
    for path in sorted(site_img.glob("peptide_2d_v*.png")):
        if path.name in seen:
            continue
        shutil.copy2(path, dest / path.name)
        seen.add(path.name)
        copied += 1
    for path in (site_data / "peptide_2d_variants.json",):
        if not path.is_file() or path.name in seen:
            continue
        shutil.copy2(path, dest / path.name)
        seen.add(path.name)
        copied += 1
    return copied


def main() -> None:
    default_src = resolve_default_source()
    parser = argparse.ArgumentParser(description="Sync the Pages training bundle from a training directory.")
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=default_src,
        help=f"Training directory containing metrics.json (default: {default_src})",
    )
    args = parser.parse_args()

    src = args.source.expanduser().resolve()
    if not (src / "metrics.json").is_file():
        print(f"[FAIL] metrics.json not found: {src}", file=sys.stderr)
        sys.exit(1)

    copied = copy_surface(src, DEST)
    print(f"[OK] {copied} files -> {DEST}")
    print(f"     source={src}")
    print("     Commit: git add publish/github_pages_training_bundle && git commit -m 'Pages: refresh training bundle'")


if __name__ == "__main__":
    main()
