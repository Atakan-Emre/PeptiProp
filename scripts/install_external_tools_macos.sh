#!/usr/bin/env bash
# PeptidQuantum: Arpeggio + PLIP + Open Babel (macOS / Homebrew, .venv-mlx).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv-mlx"
ARP_DIR="${ROOT}/third_party/arpeggio"
OB_PREFIX="$(brew --prefix open-babel 2>/dev/null || true)"

if [[ ! -d "$VENV" ]]; then
  echo "Beklenen sanal ortam yok: $VENV"
  exit 1
fi

echo "==> Homebrew: open-babel, swig, pkgconf"
brew install open-babel swig pkgconf

echo "==> Python openbabel (Homebrew başlık/lib ile derleme)"
# pip paketi /usr/local yoluna bakıyor; macOS'ta açık -I/-L gerekir.
TMP_OB="$(mktemp -d)"
trap 'rm -rf "$TMP_OB"' EXIT
"$VENV/bin/pip" download openbabel -d "$TMP_OB" --no-deps
tar xzf "$TMP_OB"/openbabel-*.tar.gz -C "$TMP_OB"
OB_SRC="$(echo "$TMP_OB"/openbabel-*)"
(
  cd "$OB_SRC"
  "$VENV/bin/python" setup.py build_ext \
    -I"${OB_PREFIX}/include/openbabel3" \
    -L"${OB_PREFIX}/lib"
  "$VENV/bin/pip" install . --no-build-isolation
)

echo "==> pip: plip, biopython"
"$VENV/bin/pip" install --upgrade plip biopython

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
plip --help | head -8
obabel -H 2>&1 | head -3 || true

echo ""
echo "Kurulum bitti. Ortam: source ${VENV}/bin/activate"
