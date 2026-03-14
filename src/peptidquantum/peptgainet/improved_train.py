from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)
from torch.utils.data import DataLoader

from .dataset import PairGraphDataset, collate_pair_graphs
from .improved_model import ImprovedPeptGAINET, PeptGAINETV2


@dataclass
class ImprovedTrainConfig:
    epochs: int = 50  # Increased from 12
    batch_size: int = 32  # Increased from 16
    lr: float = 5e-4  # Reduced from 1e-3 (as in COVID project)
    weight_decay: float = 1e-5
    device: str = "cpu"
    model_type: str = "improved"  # "improved" or "v2"
    patience: int = 10  # Early stopping patience
    save_best: bool = True


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Find optimal threshold using ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # Youden's J statistic: J = Sensitivity + Specificity - 1
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return thresholds[optimal_idx]


def _run_eval(model, loader: DataLoader, device: str, threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["labels"].to(device)
            t_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(t_batch)
            ys.extend(y.detach().cpu().numpy().tolist())
            ps.extend(out["prob"].detach().cpu().numpy().tolist())

    y = np.asarray(ys, dtype=int)
    p = np.asarray(ps, dtype=float)
    
    # Find optimal threshold on validation set
    optimal_threshold = find_optimal_threshold(y, p)
    pred_optimal = (p >= optimal_threshold).astype(int)
    pred_fixed = (p >= threshold).astype(int)

    metrics = {
        "f1@0.5": float(f1_score(y, pred_fixed, zero_division=0)),
        "f1@optimal": float(f1_score(y, pred_optimal, zero_division=0)),
        "optimal_threshold": float(optimal_threshold),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }
    return metrics


def train_improved_peptgainet(
    train_ds: PairGraphDataset,
    valid_ds: PairGraphDataset | None = None,
    config: ImprovedTrainConfig | None = None,
    save_dir: str | Path | None = None
) -> tuple[ImprovedPeptGAINET | PeptGAINETV2, dict]:
    if config is None:
        config = ImprovedTrainConfig()

    device = torch.device(config.device)
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_pair_graphs,
        num_workers=2,  # Parallel data loading
        pin_memory=True if device.type == "cuda" else False
    )
    
    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(
            valid_ds, 
            batch_size=config.batch_size, 
            shuffle=False, 
            collate_fn=collate_pair_graphs,
            num_workers=2,
            pin_memory=True if device.type == "cuda" else False
        )

    node_dim = train_ds[0].protein_graph.x.shape[1]
    
    # Initialize model based on config
    if config.model_type == "improved":
        model = ImprovedPeptGAINET(node_dim=node_dim).to(device)
    elif config.model_type == "v2":
        model = PeptGAINETV2(node_dim=node_dim).to(device)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Loss function
    criterion = torch.nn.BCELoss()  # Direct probability loss

    history = {"train_loss": [], "valid": [], "lr": []}
    best_f1 = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            y = batch["labels"].float().to(device)  # Float for BCE
            t_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            out = model(t_batch)
            loss = criterion(out["prob"], y)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            losses.append(float(loss.item()))

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history["train_loss"].append(epoch_loss)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        if valid_loader is not None:
            metrics = _run_eval(model, valid_loader, str(device))
            history["valid"].append(metrics)
            
            # Update scheduler based on F1 score
            scheduler.step(metrics["f1@optimal"])
            
            # Save best model
            if config.save_best and metrics["f1@optimal"] > best_f1:
                best_f1 = metrics["f1@optimal"]
                best_epoch = epoch
                best_state = model.state_dict().copy()
            
            print(f"[epoch {epoch}] loss={epoch_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")
            print(f"  valid: F1@0.5={metrics['f1@0.5']:.4f} F1@optimal={metrics['f1@optimal']:.4f}")
            print(f"  optimal_threshold={metrics['optimal_threshold']:.4f} AUPRC={metrics['auprc']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
        else:
            print(f"[epoch {epoch}] loss={epoch_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if valid_loader is not None and epoch - best_epoch >= config.patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    # Load best model if saved
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model from epoch {best_epoch} with F1={best_f1:.4f}")

    # Save model if directory provided
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "node_dim": node_dim,
            "history": history,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
        }
        
        model_name = f"peptgainet_{config.model_type}_v2.pt"
        torch.save(checkpoint, save_path / model_name)
        print(f"Model saved to {save_path / model_name}")

    return model, history


def evaluate_with_threshold(
    model, 
    test_ds: PairGraphDataset, 
    device: str,
    threshold: float = None
) -> dict[str, float]:
    """Evaluate model with specific or optimal threshold"""
    loader = DataLoader(
        test_ds, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_pair_graphs
    )
    
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["labels"].to(device)
            t_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(t_batch)
            ys.extend(y.detach().cpu().numpy().tolist())
            ps.extend(out["prob"].detach().cpu().numpy().tolist())

    y = np.asarray(ys, dtype=int)
    p = np.asarray(ps, dtype=float)
    
    if threshold is None:
        threshold = find_optimal_threshold(y, p)
    
    pred = (p >= threshold).astype(int)
    
    # Detailed metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    return {
        "threshold": threshold,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(np.sum(y & pred) / np.sum(pred) if np.sum(pred) > 0 else 0),
        "recall": float(np.sum(y & pred) / np.sum(y) if np.sum(y) > 0 else 0),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0)
    }
