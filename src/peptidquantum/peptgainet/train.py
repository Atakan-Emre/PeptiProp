from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import PairGraphDataset, collate_pair_graphs
from .model import PeptGAINET


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"


def _run_eval(model: PeptGAINET, loader: DataLoader, device: str) -> dict[str, float]:
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
    pred = (p >= 0.5).astype(int)

    metrics = {
        "f1@0.5": float(f1_score(y, pred, zero_division=0)),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
    }
    return metrics


def train_peptgainet(
    train_ds: PairGraphDataset,
    valid_ds: PairGraphDataset | None = None,
    config: TrainConfig | None = None,
) -> tuple[PeptGAINET, dict]:
    if config is None:
        config = TrainConfig()

    device = torch.device(config.device)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_pair_graphs)
    valid_loader = None
    if valid_ds is not None:
        valid_loader = DataLoader(valid_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_pair_graphs)

    node_dim = train_ds[0].protein_graph.x.shape[1]
    model = PeptGAINET(node_dim=node_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "valid": []}
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            y = batch["labels"].to(device)
            t_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            out = model(t_batch)
            loss = criterion(out["logit"], y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history["train_loss"].append(epoch_loss)

        if valid_loader is not None:
            metrics = _run_eval(model, valid_loader, str(device))
            history["valid"].append(metrics)
            print(f"[epoch {epoch}] loss={epoch_loss:.4f} valid={metrics}")
        else:
            print(f"[epoch {epoch}] loss={epoch_loss:.4f}")

    return model, history
