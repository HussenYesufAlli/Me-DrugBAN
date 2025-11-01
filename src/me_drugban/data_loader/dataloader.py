"""Minimal dataset placeholder for molecular data."""

from typing import Any, Dict
import random

class MoleculeDataset:
    """Tiny placeholder dataset. Replace with real loader (RDKit, CSV, etc.)."""

    def __init__(self, size: int = 100):
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Replace with real molecule features/labels
        random.seed(idx)
        return {
            "id": idx,
            "features": [random.random() for _ in range(10)],
            "label": random.randint(0, 1),
        }