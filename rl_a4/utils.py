"""
utils.py – Reproducibility helpers and miscellaneous utilities.

Reproducibility is critical for fair comparison across seeds and algorithms.
We seed: Python random, NumPy, PyTorch (CPU and CUDA), and the Gym environment.
Note: full determinism on GPU requires additional CUDA flags and is not
guaranteed across PyTorch versions; we default to CPU only.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set global seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For CPU reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env(env_id: str = "CartPole-v1", seed: int = 0):
    """Return a callable that creates a seeded Gym environment."""
    def _thunk():
        env = __import__("gymnasium").make(env_id)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
        return env
    return _thunk


def save_config(config: Dict[str, Any], path: Path) -> None:
    """Serialise config dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON config file."""
    with open(path) as f:
        return json.load(f)


def get_device(prefer_gpu: bool = False) -> torch.device:
    """Return torch device (GPU if available and preferred, else CPU)."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
