#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from peptidquantum.peptgainet import (
    ImprovedTrainConfig,
    PairGraphDataset,
    load_pairs_jsonl,
    train_improved_peptgainet,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Improved PeptGAINET on pair jsonl")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--valid-jsonl")
    p.add_argument("--test-jsonl", help="Optional test set for final evaluation")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--model-type", choices=["improved", "v2"], default="improved")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--out-dir", default="models/peptgainet_improved")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Model type: {args.model_type}")
    
    # Load datasets
    train_records = load_pairs_jsonl(args.train_jsonl)
    valid_records = load_pairs_jsonl(args.valid_jsonl) if args.valid_jsonl else None
    test_records = load_pairs_jsonl(args.test_jsonl) if args.test_jsonl else None
    
    print(f"Training samples: {len(train_records)}")
    if valid_records:
        print(f"Validation samples: {len(valid_records)}")
    if test_records:
        print(f"Test samples: {len(test_records)}")

    # Create datasets
    train_ds = PairGraphDataset(train_records)
    valid_ds = PairGraphDataset(valid_records) if valid_records else None
    test_ds = PairGraphDataset(test_records) if test_records else None

    # Configure training
    cfg = ImprovedTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        model_type=args.model_type,
        patience=args.patience,
        save_best=True,
    )

    # Train model
    model, history = train_improved_peptgainet(
        train_ds, 
        valid_ds, 
        cfg,
        save_dir=args.out_dir
    )
    
    # Save training history
    out_dir = Path(args.out_dir)
    history_path = out_dir / f"history_{args.model_type}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")
    
    # Evaluate on test set if provided
    if test_records is not None:
        from peptidquantum.peptgainet import PairGraphDataset, evaluate_with_threshold
        
        test_ds = PairGraphDataset(test_records)
        metrics = evaluate_with_threshold(model, test_ds, args.device)
        
        # Save test metrics
        metrics_path = out_dir / f"test_metrics_{args.model_type}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("\n=== Test Set Evaluation ===")
        print(f"Optimal threshold: {metrics['threshold']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Test metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
