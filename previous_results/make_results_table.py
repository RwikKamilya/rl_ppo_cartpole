#!/usr/bin/env python3

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MAX_STEPS = 1_000_000
SMOOTH_WINDOW = 25
SOLVE_THRESHOLD = 475.0
N_GRID = 1000

FILES = {
    "REINFORCE": "reinforce.npz",
    "AC": "ac.npz",
    "A2C": "a2c.npz",
    "DQN A2": "linear_basic_training.npz",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

@dataclass
class MethodStats:
    final_mean: float
    final_std: float
    aulc_mean: float
    aulc_std: float
    t475_mean: Optional[float]
    t475_std: Optional[float]
    solved: int
    total: int


def moving_average(values: np.ndarray, window: int = 25) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)

    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = values[start : i + 1].mean()

    return out


def ensure_cumulative_steps(steps: np.ndarray) -> np.ndarray:
    steps = np.asarray(steps, dtype=np.float64)

    if len(steps) == 0:
        return steps

    # If steps are not monotonic, assume they are per-episode lengths.
    if np.any(np.diff(steps) < 0):
        return np.cumsum(steps)

    return steps


def load_raw_runs(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    keys = set(data.keys())

    if not {"rewards", "steps"}.issubset(keys):
        raise ValueError(
            f"{path} does not contain raw-run keys 'rewards' and 'steps'. "
            f"Found keys: {sorted(keys)}"
        )

    rewards = [np.asarray(x, dtype=np.float64) for x in data["rewards"]]
    steps = [ensure_cumulative_steps(np.asarray(x, dtype=np.float64)) for x in data["steps"]]

    if len(rewards) != len(steps):
        raise ValueError(f"{path}: number of reward runs and step runs differ.")

    return rewards, steps


def first_step_to_threshold(
    steps: np.ndarray,
    smoothed_returns: np.ndarray,
    threshold: float = SOLVE_THRESHOLD,
) -> Optional[float]:
    """First environment step where smoothed return reaches threshold."""
    hits = np.where(smoothed_returns >= threshold)[0]
    if len(hits) == 0:
        return None
    return float(steps[hits[0]])


def compute_method_stats(path: str) -> MethodStats:
    rewards_runs, steps_runs = load_raw_runs(path)

    final_returns = []
    aulcs = []
    t475s = []

    grid = np.linspace(0, MAX_STEPS, N_GRID)

    for rewards, steps in zip(rewards_runs, steps_runs):
        if len(rewards) == 0 or len(steps) == 0:
            continue

        # Align just in case one array is longer than the other.
        n = min(len(rewards), len(steps))
        rewards = rewards[:n]
        steps = steps[:n]

        smoothed = moving_average(rewards, window=SMOOTH_WINDOW)

        # Final return: mean raw episode return over last 25 episodes.
        final_returns.append(float(np.mean(rewards[-SMOOTH_WINDOW:])))

        # AULC: average smoothed return over a fixed 0..1e6 step grid.
        # If a run ends before MAX_STEPS, np.interp holds its final value.
        curve = np.interp(grid, steps, smoothed)
        aulcs.append(float(np.mean(curve)))

        # Time-to-solve: first step where smoothed curve crosses threshold.
        t_hit = first_step_to_threshold(steps, smoothed, SOLVE_THRESHOLD)
        if t_hit is not None:
            t475s.append(t_hit)

    total = len(rewards_runs)
    solved = len(t475s)

    final_returns = np.asarray(final_returns, dtype=np.float64)
    aulcs = np.asarray(aulcs, dtype=np.float64)
    t475s_arr = np.asarray(t475s, dtype=np.float64)

    return MethodStats(
        final_mean=float(np.mean(final_returns)),
        final_std=float(np.std(final_returns)),
        aulc_mean=float(np.mean(aulcs)),
        aulc_std=float(np.std(aulcs)),
        t475_mean=float(np.mean(t475s_arr)) if solved > 0 else None,
        t475_std=float(np.std(t475s_arr)) if solved > 0 else None,
        solved=solved,
        total=total,
    )


def fmt_mean_std(mean: float, std: float, decimals: int = 1) -> str:
    return rf"${mean:.{decimals}f}{{\pm}}{std:.{decimals}f}$"


def fmt_t475(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None or std is None:
        return "N.A."

    mean_k = mean / 1000.0
    std_k = std / 1000.0
    return rf"${mean_k:.0f}{{\pm}}{std_k:.0f}$k"


def main() -> None:
    stats: Dict[str, MethodStats] = {}

    for method, filename in FILES.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Missing {filename}. Put this script in the same folder as the .npz files."
            )

        stats[method] = compute_method_stats(filename)

    print(stats)


if __name__ == "__main__":
    main()