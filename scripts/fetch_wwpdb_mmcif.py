#!/usr/bin/env python3
"""
Resmî PDB girişi mmCIF indir (bilimsel kaynak).

Koordinatlar wwPDB arşivindedir; RCSB PDB (https://www.rcsb.org/) bu arşivi
HTTPS üzerinden `files.rcsb.org/download/<PDB_ID>.cif` ile dağıtır — PDB ID
için yapısal “altın standart” dosyadır. Yerel PROPEDIA/raw aynası boş veya
bozuksa bu betikle aynı PDB kimliği için arşiv kopyasını yazın.

Örnek:
  python scripts/fetch_wwpdb_mmcif.py --pdb-id 1ABT \\
    --output data/raw/propedia/complexes/1ABT.cif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

# RCSB, wwPDB ile anlaşmalı ayna; tek PDB kimliği = tek arşiv girişi.
PRIMARY_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
FALLBACK_URL = "https://files.wwpdb.org/download/{pdb_id}.cif"


def fetch_mmcif(pdb_id: str, timeout: int = 60) -> str:
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4 or not pdb_id[0].isdigit():
        raise ValueError(f"Geçersiz PDB kimliği: {pdb_id!r}")

    last_err: Exception | None = None
    for template in (PRIMARY_URL, FALLBACK_URL):
        url = template.format(pdb_id=pdb_id)
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            text = r.text
        except requests.RequestException as exc:
            last_err = exc
            continue
        if not text or len(text.strip()) < 200:
            last_err = RuntimeError(f"Boş veya çok kısa yanıt: {url}")
            continue
        if not text.lstrip().startswith("data_"):
            last_err = RuntimeError(f"mmCIF başlığı beklenmiyor (data_ yok): {url}")
            continue
        return text

    raise RuntimeError(f"İndirme başarısız ({pdb_id}): {last_err}")


def main() -> int:
    p = argparse.ArgumentParser(description="wwPDB arşivi mmCIF indir (RCSB aynası).")
    p.add_argument("--pdb-id", required=True, help="Dört karakter PDB kimliği, örn. 1ABT")
    p.add_argument(
        "--output",
        type=Path,
        help="Çıktı yolu (varsayılan: data/raw/propedia/complexes/<PDB>.cif)",
    )
    args = p.parse_args()
    out = args.output
    if out is None:
        root = Path(__file__).resolve().parents[1]
        out = root / "data" / "raw" / "propedia" / "complexes" / f"{args.pdb_id.strip().upper()}.cif"

    out.parent.mkdir(parents=True, exist_ok=True)
    text = fetch_mmcif(args.pdb_id)
    out.write_text(text, encoding="utf-8")
    n = out.stat().st_size
    print(f"[OK] {out} ({n} byte) — kaynak: wwPDB arşivi (RCSB/wwpdb.org indirme uçları)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as e:
        print(f"[HATA] {e}", file=sys.stderr)
        raise SystemExit(1)
