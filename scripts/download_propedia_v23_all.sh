#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/data/raw/propedia"
mkdir -p "${OUT_DIR}"

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is required. Install with: brew install aria2" >&2
  exit 1
fi

BASE_URL="https://bioinfo.dcc.ufmg.br/propedia2/public/download"
FILES=(
  "peptides_signature.zip"
  "sequences2_3.zip"
  "peptides2_3.zip"
  "interfaces2_3.zip"
  "receptor2_3.zip"
  "complex2_3.zip"
)

echo "PROPEDIA v2.3 full download"
echo "Output: ${OUT_DIR}"

for file in "${FILES[@]}"; do
  echo "===== ${file} ====="
  aria2c \
    --check-certificate=false \
    --continue=true \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=1M \
    --summary-interval=5 \
    --file-allocation=none \
    --dir="${OUT_DIR}" \
    --out="${file}" \
    "${BASE_URL}/${file}"
  ls -lh "${OUT_DIR}/${file}"
done

echo "All PROPEDIA v2.3 files downloaded."
