#!/usr/bin/env python3
"""
PeptidQuantum - Unified Training Script
Complete training pipeline with all features
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import torch
from torch.utils.data import DataLoader

from peptidquantum.peptgainet import load_pairs_jsonl, PairGraphDataset, collate_pair_graphs
from peptidquantum.peptgainet.model_fixed import ImprovedPeptGAINET
from peptidquantum.utils import stratified_split
from peptidquantum.training import Trainer, TrainingConfig, AblationStudy, AblationConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train PeptidQuantum Model")
    
    # Data
    parser.add_argument("--data", required=True, help="Path to combined data JSONL")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=15)
    
    # Model
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--message-steps", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dense-units", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # Checkpointing
    parser.add_argument("--output-dir", default="models/trained")
    parser.add_argument("--resume", help="Resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=5)
    
    # Ablation study
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--ablation-mode", choices=["grid", "one_at_a_time", "random"], 
                       default="one_at_a_time")
    
    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()


def prepare_data(args):
    """Load and split data"""
    print("\n" + "="*60)
    print("Loading and splitting data")
    print("="*60)
    
    # Load all data
    all_records = load_pairs_jsonl(args.data)
    labels = [r.label for r in all_records]
    
    print(f"Total samples: {len(all_records)}")
    print(f"Positive samples: {sum(labels)}")
    print(f"Negative samples: {len(labels) - sum(labels)}")
    
    # Stratified split
    train_records, val_records, test_records = stratified_split(
        all_records, labels,
        args.train_ratio, args.val_ratio, args.test_ratio
    )
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_records)} ({args.train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_records)} ({args.val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_records)} ({args.test_ratio*100:.0f}%)")
    
    # Create datasets
    train_ds = PairGraphDataset(train_records)
    val_ds = PairGraphDataset(val_records)
    test_ds = PairGraphDataset(test_records)
    
    return train_ds, val_ds, test_ds


def create_model(node_dim: int, config: TrainingConfig):
    """Create model with given configuration"""
    model = ImprovedPeptGAINET(
        node_dim=node_dim,
        hidden_dim=config.hidden_dim,
        message_steps=config.message_steps,
        num_attention_heads=config.num_heads,
        dense_units=config.dense_units,
        dropout=config.dropout
    )
    return model


def train_single(args, train_ds, val_ds):
    """Train single model"""
    print("\n" + "="*60)
    print("Training Single Model")
    print("="*60)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pair_graphs,
        num_workers=2,
        pin_memory=True if args.device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pair_graphs,
        num_workers=2,
        pin_memory=True if args.device == "cuda" else False
    )
    
    # Get node dimension
    node_dim = train_ds[0].protein_graph.x.shape[1]
    
    # Create config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        patience=args.patience,
        save_every=args.save_every,
        hidden_dim=args.hidden_dim,
        message_steps=args.message_steps,
        num_heads=args.num_heads,
        dense_units=args.dense_units,
        dropout=args.dropout
    )
    
    # Create model
    model = create_model(node_dim, config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=args.output_dir
    )
    
    # Train
    history = trainer.train(resume_from=args.resume)
    
    return trainer, history


def run_ablation(args, train_ds, val_ds):
    """Run ablation study"""
    print("\n" + "="*60)
    print("Running Ablation Study")
    print("="*60)
    
    # Create ablation config
    ablation_config = AblationConfig()
    ablation_study = AblationStudy(
        config=ablation_config,
        output_dir=Path(args.output_dir) / "ablation"
    )
    
    # Generate experiments
    experiments = ablation_study.generate_experiments(mode=args.ablation_mode)
    print(f"\nTotal experiments: {len(experiments)}")
    
    # Get node dimension
    node_dim = train_ds[0].protein_graph.x.shape[1]
    
    # Run each experiment
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(experiments)}")
        print(f"{'='*60}")
        print("Configuration:")
        for k, v in exp.items():
            print(f"  {k}: {v}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_ds,
            batch_size=exp['batch_size'],
            shuffle=True,
            collate_fn=collate_pair_graphs,
            num_workers=2,
            pin_memory=True if args.device == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=exp['batch_size'],
            shuffle=False,
            collate_fn=collate_pair_graphs,
            num_workers=2,
            pin_memory=True if args.device == "cuda" else False
        )
        
        # Create config
        config = TrainingConfig(
            epochs=min(args.epochs, 30),  # Shorter for ablation
            batch_size=exp['batch_size'],
            lr=exp['learning_rate'],
            weight_decay=exp['weight_decay'],
            device=args.device,
            patience=10,
            save_every=10,
            hidden_dim=exp['hidden_dim'],
            message_steps=exp['message_steps'],
            num_heads=exp['num_heads'],
            dense_units=exp['dense_units'],
            dropout=exp['dropout']
        )
        
        # Create model
        model = create_model(node_dim, config)
        
        # Create trainer
        exp_dir = Path(args.output_dir) / "ablation" / f"exp_{i}"
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=exp_dir
        )
        
        # Train
        try:
            history = trainer.train()
            
            # Get final metrics
            final_metrics = history["val_metrics"][-1] if history["val_metrics"] else {}
            ablation_study.add_result(exp, final_metrics)
            
        except Exception as e:
            print(f"Experiment {i} failed: {e}")
            continue
    
    # Get best config
    best_config = ablation_study.get_best_config()
    print(f"\n{'='*60}")
    print("Best Configuration Found:")
    print('='*60)
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    
    return ablation_study


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("PeptidQuantum Training Pipeline")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    
    # Prepare data
    train_ds, val_ds, test_ds = prepare_data(args)
    
    # Run training or ablation
    if args.ablation:
        ablation_study = run_ablation(args, train_ds, val_ds)
    else:
        trainer, history = train_single(args, train_ds, val_ds)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
