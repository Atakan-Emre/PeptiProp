#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from peptidquantum.peptgainet import PairGraphDataset, TrainConfig, load_pairs_jsonl, train_peptgainet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PeptGAINET on pair jsonl")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--valid-jsonl")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", default="models/peptgainet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_records = load_pairs_jsonl(args.train_jsonl)
    valid_records = load_pairs_jsonl(args.valid_jsonl) if args.valid_jsonl else None

    train_ds = PairGraphDataset(train_records)
    valid_ds = PairGraphDataset(valid_records) if valid_records else None

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    model, history = train_peptgainet(train_ds, valid_ds, cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "state_dict": model.state_dict(),
        "node_dim": train_ds[0].protein_graph.x.shape[1],
        "history": history,
    }
    torch.save(ckpt, out_dir / "peptgainet.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"[saved] {out_dir / 'peptgainet.pt'}")
    print(f"[saved] {out_dir / 'history.json'}")


if __name__ == "__main__":
    main()
