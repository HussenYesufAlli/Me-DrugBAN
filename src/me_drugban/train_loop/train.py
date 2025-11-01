"""Simple training loop helper (high-level placeholder)."""

from typing import Any

def train_one_epoch(model: Any, dataloader, optimizer, loss_fn):
    """Run one epoch over dataloader. This is a minimal example."""
    # In a minimal placeholder we just iterate
    total_loss = 0.0
    count = 0
    for batch in dataloader:
        features = batch.get("features")
        label = batch.get("label")
        if features is None or label is None:
            continue
        count += 1
    return total_loss if count > 0 else 0.0