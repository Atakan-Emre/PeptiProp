"""Arpeggio / PLIP için tek-model, sütun-uyumlu PDB üretimi (mmCIF veya çok modelli PDB)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from Bio.PDB import MMCIFParser, PDBIO, PDBParser


def export_single_model_pdb(
    structure_path: Path,
    output_pdb: Path,
    *,
    model_index: int = 0,
    min_bytes: int = 200,
) -> Optional[Path]:
    """
    wwPDB mmCIF veya PDB oku; yalnızca seçilen modeli PDBIO ile yazar.

    Arpeggio (third_party) girişi BioPython PDBParser ile okur; ham mmCIF vermek
    sütun kaymasına yol açar. NMR gibi çok modelli girişlerde model 0 varsayılan
    makale prosedürüyle uyumludur.
    """
    structure_path = Path(structure_path)
    output_pdb = Path(output_pdb)
    ext = structure_path.suffix.lower()

    try:
        if ext in {".cif", ".mmcif"}:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("exp", str(structure_path))
        elif ext == ".pdb":
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("exp", str(structure_path))
        else:
            return None
    except Exception:
        return None

    try:
        models = list(structure)
        if not models or model_index >= len(models):
            return None
        model = models[model_index]
    except Exception:
        return None

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    try:
        io = PDBIO()
        io.set_structure(model)
        io.save(str(output_pdb))
    except Exception:
        return None

    if not output_pdb.is_file() or output_pdb.stat().st_size < min_bytes:
        return None
    return output_pdb
