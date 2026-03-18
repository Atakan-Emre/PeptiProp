#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from peptidquantum.peptgainet import (
    ImprovedPeptGAINET,
    PairGraphDataset,
    PeptGAINETV2,
    load_pairs_jsonl,
)


def load_checkpoint(checkpoint_path: str, device: str):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    node_dim = checkpoint["node_dim"]
    config = checkpoint["config"]
    
    if config.model_type == "improved":
        model = ImprovedPeptGAINET(node_dim=node_dim)
    elif config.model_type == "v2":
        model = PeptGAINETV2(node_dim=node_dim)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def predict_interactions(
    model,
    pairs_jsonl: str,
    checkpoint_path: str,
    device: str,
    threshold: float = None,
    batch_size: int = 32
) -> pd.DataFrame:
    """Predict interactions for all pairs in jsonl file"""
    
    # Load checkpoint to get optimal threshold if not provided
    if threshold is None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "history" in checkpoint and checkpoint["history"]["valid"]:
            # Use threshold from best validation epoch
            valid_metrics = checkpoint["history"]["valid"][-1]
            threshold = valid_metrics.get("optimal_threshold", 0.5)
            print(f"Using optimal threshold from validation: {threshold:.4f}")
        else:
            threshold = 0.5
            print(f"No optimal threshold found, using default: {threshold}")
    
    # Load data
    records = load_pairs_jsonl(pairs_jsonl)
    dataset = PairGraphDataset(records)
    
    # Create dataframe with all info
    results = []
    
    # Process in batches
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    
    with torch.no_grad():
        for batch in loader:
            # Get batch info
            batch_size_actual = len(batch["labels"])
            pair_ids = batch["pair_id"]
            labels = batch["labels"].cpu().numpy()
            
            # Move to device
            t_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Predict
            out = model(t_batch)
            probs = out["prob"].cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            # Store results
            for i in range(batch_size_actual):
                record = records[i]
                results.append({
                    "pair_id": record.pair_id,
                    "protein_name": getattr(record, 'protein_name', f"prot_{record.pair_id.split('_')[0]}"),
                    "peptide_name": getattr(record, 'peptide_name', f"pep_{record.pair_id.split('_')[1]}"),
                    "protein_seq": record.protein_seq,
                    "peptide_seq": record.peptide_seq,
                    "label": record.label,
                    "pred_prob": float(probs[i]),
                    "pred_label": int(preds[i]),
                    "threshold": threshold,
                    "actual": int(labels[i]),
                    "correct": int(preds[i] == labels[i])
                })
    
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with Improved PeptGAINET")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--pairs-jsonl", required=True, help="Path to pairs jsonl file")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for classification")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.checkpoint}")
    print(f"Using device: {args.device}")
    
    # Load model
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    print(f"Loaded {checkpoint['config'].model_type} model")
    print(f"Best validation F1: {checkpoint.get('best_f1', 'N/A'):.4f}")
    
    # Predict
    print(f"Predicting on {args.pairs_jsonl}")
    df = predict_interactions(
        model,
        args.pairs_jsonl,
        args.checkpoint,
        args.device,
        threshold=args.threshold,
        batch_size=args.batch_size
    )
    
    # Save results
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\nResults saved to {output_path}")
    print(f"Total predictions: {len(df)}")
    print(f"Accuracy: {df['correct'].mean():.4f}")
    print(f"Positive predictions: {df['pred_label'].sum()}/{len(df)} ({df['pred_label'].mean():.2%})")
    
    if args.threshold:
        print(f"Used threshold: {args.threshold}")
    else:
        print(f"Used optimal threshold: {df['threshold'].iloc[0]:.4f}")
    
    # Print confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(df['actual'], df['pred_label'])
    print("\nConfusion Matrix:")
    print("    Pred=0  Pred=1")
    print(f"True=0  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"True=1  {cm[1,0]:6d}  {cm[1,1]:6d}")


if __name__ == "__main__":
    main()
