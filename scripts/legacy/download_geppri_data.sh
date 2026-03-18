#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/data/raw/GEPPRI"
mkdir -p "$OUT_DIR"

BASE_URL="https://raw.githubusercontent.com/shima403shafiee9513/GEPPRI.method-at-Bioinformatics/main"
FILES=(Train1.txt Test1.txt Train2.txt Test2.txt README.md)

for f in "${FILES[@]}"; do
  echo "[download] $f"
  curl -fsSL "$BASE_URL/$f" -o "$OUT_DIR/$f"
done

echo "[ok] data saved under $OUT_DIR"
