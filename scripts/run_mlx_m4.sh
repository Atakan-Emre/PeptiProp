#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="configs/train_v0_1_scoring_mlx_m4.yaml"

echo "[1/2] Export MLX features"
python scripts/export_mlx_features.py --config "${CONFIG}"

echo "[2/2] Train MLX scoring model"
python scripts/train_scoring_mlx.py --config "${CONFIG}"

echo "Done. Check outputs/training/peptidquantum_v0_1_final_mlx_m4/"

