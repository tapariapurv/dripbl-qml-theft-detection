"""Batch CNN embedding: (N, 64, 64, 3) float32 RGB in [0, 1] -> (N, 128) latents.

Reuses the same LightweightCNN the live edge pipeline runs (edge_pipeline.py)
so offline-trained artifacts and the on-device model share one architecture.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from edge_pipeline import LightweightCNN

torch.set_num_threads(1)


def load_cnn(weights_path: str | None = None, in_channels: int = 3) -> LightweightCNN:
    cnn = LightweightCNN(latent_dim=128, in_channels=in_channels).eval()
    if weights_path and Path(weights_path).is_file():
        cnn.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return cnn


def embed_batch(frames_rgb01: np.ndarray, cnn: LightweightCNN) -> np.ndarray:
    """frames_rgb01: (N, 64, 64, 3) float32 in [0, 1] -> (N, 128) float32 latents."""
    tensor = torch.from_numpy(frames_rgb01).permute(0, 3, 1, 2).contiguous()
    with torch.inference_mode():
        return cnn(tensor).numpy()
