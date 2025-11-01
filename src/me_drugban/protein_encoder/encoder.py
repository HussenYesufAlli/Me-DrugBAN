"""Placeholder protein encoder.

Replace with real sequence encoder (e.g., transformer, pretrained embeddings).
"""

class ProteinEncoder:
    def __init__(self, output_dim: int = 128):
        self.output_dim = output_dim

    def encode(self, sequence: str):
        """Return a fixed-size vector for a given sequence (placeholder)."""
        # Return deterministic simple encoding (truncated/padded)
        vec = [ord(c) % 256 for c in sequence]
        if len(vec) >= self.output_dim:
            return vec[: self.output_dim]
        return vec + [0] * (self.output_dim - len(vec))