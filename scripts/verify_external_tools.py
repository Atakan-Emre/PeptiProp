"""Hızlı kontrol: Arpeggio CLI ve wrapper is_available."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peptidquantum.interaction.extractors.arpeggio_wrapper import ArpeggioWrapper


def run_help(cmd: list[str], name: str, timeout: int = 25) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        ok = r.returncode == 0
        print(f"  {name}: {'OK' if ok else 'FAIL'} (exit {r.returncode})")
        return ok
    except Exception as exc:
        print(f"  {name}: FAIL ({exc})")
        return False


def main() -> int:
    print("PeptidQuantum harici araç doğrulaması\n")
    ok = True
    ok &= run_help(["arpeggio", "--help"], "arpeggio --help")

    a = ArpeggioWrapper()
    print(f"  ArpeggioWrapper.is_available: {a.is_available()}")
    ok &= a.is_available()

    print("\nSonuç:", "TAMAM" if ok else "EKSİK — scripts/install_external_tools_macos.sh bakın")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
