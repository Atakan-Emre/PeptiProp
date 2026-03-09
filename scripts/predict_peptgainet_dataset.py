#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from peptidquantum.peptgainet.dataset import PairGraphSample, collate_pair_graphs
from peptidquantum.peptgainet.graph import build_residue_graph
from peptidquantum.peptgainet.model import PeptGAINET


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch prediction on pair jsonl with PeptGAINET")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pairs-jsonl", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for i, ln in enumerate(f, start=1):
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if "pair_id" not in obj:
                obj["pair_id"] = f"pair_{i}"
            rows.append(obj)
    return rows


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PeptGAINET(node_dim=ckpt["node_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    records = _read_jsonl(args.pairs_jsonl)

    out_rows = []
    with torch.no_grad():
        for r in records:
            protein_seq = str(r.get("protein_seq", "")).upper()
            peptide_seq = str(r.get("peptide_seq", "")).upper()
            if not protein_seq or not peptide_seq:
                continue

            sample = PairGraphSample(
                pair_id=str(r.get("pair_id")),
                protein_graph=build_residue_graph(protein_seq),
                peptide_graph=build_residue_graph(peptide_seq),
                label=int(r.get("label", 0)),
            )
            batch = collate_pair_graphs([sample])
            pred = model(batch)
            prob = float(pred["prob"].item())

            row = dict(r)
            row["pred_prob"] = prob
            row["pred_label"] = int(prob >= args.threshold)
            row["actual"] = int(r.get("label", row.get("actual", 0)))
            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[saved] {out}")
    print(f"[rows] {len(df)}")


if __name__ == "__main__":
    main()
