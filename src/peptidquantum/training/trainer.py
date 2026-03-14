"""Advanced training system with early stopping, checkpointing, and resume"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-5
    device: str = "cuda"
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Checkpointing
    save_every: int = 5
    save_best: bool = True
    
    # Regularization
    dropout: float = 0.2
    grad_clip: float = 1.0
    
    # Model architecture
    hidden_dim: int = 64
    message_steps: int = 6
    num_heads: int = 8
    dense_units: int = 512


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Find optimal classification threshold"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return float(thresholds[optimal_idx])


def evaluate_model(model, loader: DataLoader, device: str) -> dict:
    """Evaluate model on dataset"""
    model.eval()
    ys, ps = [], []
    
    with torch.no_grad():
        for batch in loader:
            y = batch["labels"].to(device)
            t_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(t_batch)
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(out["prob"].cpu().numpy().tolist())
    
    y = np.array(ys, dtype=int)
    p = np.array(ps, dtype=float)
    
    optimal_threshold = find_optimal_threshold(y, p)
    pred_optimal = (p >= optimal_threshold).astype(int)
    pred_fixed = (p >= 0.5).astype(int)
    
    metrics = {
        "f1@0.5": float(f1_score(y, pred_fixed, zero_division=0)),
        "f1@optimal": float(f1_score(y, pred_optimal, zero_division=0)),
        "optimal_threshold": optimal_threshold,
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0,
    }
    
    return metrics


class Trainer:
    """Advanced trainer with all features"""
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
        output_dir: str | Path
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.criterion = torch.nn.BCELoss()
        self.early_stopping = EarlyStopping(config.patience, mode='max')
        
        self.history = {
            "train_loss": [],
            "val_metrics": [],
            "lr": [],
            "epoch_time": []
        }
        
        self.start_epoch = 0
        self.best_f1 = 0.0
        self.best_epoch = 0
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": asdict(self.config),
            "history": self.history,
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
        # Save latest
        latest_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.history = checkpoint["history"]
        self.best_f1 = checkpoint["best_f1"]
        self.best_epoch = checkpoint["best_epoch"]
        
        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Best F1: {self.best_f1:.4f} at epoch {self.best_epoch}")
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        losses = []
        
        for batch in self.train_loader:
            y = batch["labels"].float().to(self.device)
            t_batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                      for k, v in batch.items()}
            
            out = self.model(t_batch)
            loss = self.criterion(out["prob"], y)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.grad_clip
            )
            
            self.optimizer.step()
            losses.append(loss.item())
        
        return np.mean(losses)
    
    def train(self, resume_from: Optional[str | Path] = None):
        """Main training loop"""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print(f"\nStarting training from epoch {self.start_epoch + 1}")
        print(f"Total epochs: {self.config.epochs}")
        print(f"Device: {self.device}")
        print("="*60)
        
        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]['lr'])
            
            # Validate
            val_metrics = None
            if self.val_loader:
                val_metrics = evaluate_model(self.model, self.val_loader, str(self.device))
                self.history["val_metrics"].append(val_metrics)
                
                # Update scheduler
                self.scheduler.step(val_metrics["f1@optimal"])
                
                # Check for best model
                if val_metrics["f1@optimal"] > self.best_f1:
                    self.best_f1 = val_metrics["f1@optimal"]
                    self.best_epoch = epoch + 1
                    self.save_checkpoint(epoch + 1, is_best=True)
                
                # Early stopping
                if self.early_stopping(val_metrics["f1@optimal"]):
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
            
            # Epoch time
            epoch_time = time.time() - epoch_start
            self.history["epoch_time"].append(epoch_time)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{self.config.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                  f"Time: {epoch_time:.1f}s")
            
            if val_metrics:
                print(f"  Val: F1@0.5={val_metrics['f1@0.5']:.4f} | "
                      f"F1@opt={val_metrics['f1@optimal']:.4f} | "
                      f"AUPRC={val_metrics['auprc']:.4f} | "
                      f"ROC-AUC={val_metrics['roc_auc']:.4f}")
            
            # Save periodic checkpoints
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        # Save final checkpoint
        self.save_checkpoint(self.config.epochs)
        
        # Save history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "="*60)
        print(f"Training completed!")
        print(f"Best F1: {self.best_f1:.4f} at epoch {self.best_epoch}")
        print(f"Model saved to: {self.output_dir}")
        
        return self.history
