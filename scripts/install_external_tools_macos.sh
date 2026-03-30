#!/usr/bin/env bash
# PeptidQuantum: Arpeggio kurulumu (macOS / Homebrew, .venv-mlx).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv-mlx"
ARP_DIR="${ROOT}/third_party/arpeggio"
if [[ ! -d "$VENV" ]]; then
  echo "Beklenen sanal ortam yok: $VENV"
  exit 1
fi

echo "==> pip: biopython"
"$VENV/bin/pip" install --upgrade biopython

echo "==> Arpeggio (klon + venv içinde 'arpeggio' başlatıcısı)"
mkdir -p "${ROOT}/third_party"
if [[ ! -f "${ARP_DIR}/arpeggio.py" ]]; then
  git clone --depth 1 https://github.com/harryjubb/arpeggio.git "$ARP_DIR"
fi
cat > "${VENV}/bin/arpeggio" <<EOF
#!/usr/bin/env bash
exec "\$(dirname "\$0")/python3" "${ARP_DIR}/arpeggio.py" "\$@"
EOF
chmod +x "${VENV}/bin/arpeggio"

echo "==> Doğrulama"
export PATH="${VENV}/bin:/opt/homebrew/bin:/usr/local/bin:\$PATH"
arpeggio --help | head -5

echo ""
echo "Kurulum bitti. Ortam: source ${VENV}/bin/activate"
