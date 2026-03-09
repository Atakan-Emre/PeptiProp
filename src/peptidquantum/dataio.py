from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


AA_SET = set("ACDEFGHIKLMNPQRSTVWYX")


@dataclass
class SequenceEntry:
    sequence: str
    labels: list[int]


class DatasetFormatError(ValueError):
    pass


def _clean_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    bad = [aa for aa in seq if aa not in AA_SET]
    if bad:
        raise DatasetFormatError(f"Sequence contains unsupported residues: {sorted(set(bad))}")
    return seq


def _clean_labels(lbl: str) -> list[int]:
    lbl = lbl.strip()
    if any(ch not in {"0", "1"} for ch in lbl):
        raise DatasetFormatError("Label string must contain only 0/1")
    return [int(ch) for ch in lbl]


def parse_geppri_file(path: str | Path) -> list[SequenceEntry]:
    """Parse GEPPRI-like files with columns: seq, label."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if len(lines) < 3:
        raise DatasetFormatError("File is too short")

    # Skip contact line and header line.
    data_lines = lines[2:]
    entries: list[SequenceEntry] = []
    for idx, ln in enumerate(data_lines, start=3):
        parts = ln.split()
        if len(parts) < 2:
            raise DatasetFormatError(f"Line {idx}: expected <seq> <label>")
        seq = _clean_sequence(parts[0])
        labels = _clean_labels(parts[1])
        if len(seq) != len(labels):
            raise DatasetFormatError(f"Line {idx}: sequence/label length mismatch ({len(seq)} != {len(labels)})")
        entries.append(SequenceEntry(sequence=seq, labels=labels))

    if not entries:
        raise DatasetFormatError("No entries found")
    return entries


def flatten_entries(entries: Iterable[SequenceEntry]) -> tuple[list[str], list[list[int]]]:
    seqs: list[str] = []
    labels: list[list[int]] = []
    for item in entries:
        seqs.append(item.sequence)
        labels.append(item.labels)
    return seqs, labels
