"""
metrics.py – Compute summary metrics from PPO and baseline result CSVs/npz files.

Metrics computed per algorithm (aggregated over seeds):
  - final_eval_return_mean ± std : mean of last 5 eval checkpoints per seed,
                                    then mean ± std over seeds.
  - final_train_return_mean ± std: mean of last 100 training episodes per seed.
  - AULC_1M   : area under eval learning curve [0, 1e6], normalised by 500.
  - AULC_200k : area under eval learning curve [0, 200k], normalised by 500.
  - t_475     : first env step at which eval_mean ≥ 475 for 3 consecutive
                checkpoints (per seed); aggregated as mean ± std over seeds.
  - solve_rate: fraction of seeds with finite t_475.
  - post_solve_min_return: minimum eval return across seeds/checkpoints after
                            t_475 (stability measure).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_ppo_eval_seeds(results_dir: Path) -> List[pd.DataFrame]:
    """Load eval CSVs for all seeds of one PPO variant."""
    dfs = []
    for seed_dir in sorted(results_dir.glob("seed_*")):
        eval_csv = seed_dir / "eval.csv"
        if eval_csv.exists():
            dfs.append(pd.read_csv(eval_csv))
    return dfs


def _load_ppo_train_seeds(results_dir: Path) -> List[pd.DataFrame]:
    """Load train CSVs for all seeds of one PPO variant."""
    dfs = []
    for seed_dir in sorted(results_dir.glob("seed_*")):
        train_csv = seed_dir / "train.csv"
        if train_csv.exists():
            dfs.append(pd.read_csv(train_csv))
    return dfs


def _aulc(steps: np.ndarray, returns: np.ndarray,
          max_step: float, norm: float = 500.0) -> float:
    """Area under learning curve up to max_step, normalised by norm*max_step."""
    mask = steps <= max_step
    s = steps[mask]
    r = returns[mask]
    if len(s) < 2:
        return float("nan")
    # Prepend 0,0 if first point > 0
    if s[0] > 0:
        s = np.concatenate([[0.0], s])
        r = np.concatenate([[0.0], r])
    area = float(np.trapz(r, s))
    return area / (norm * max_step)


def _t_475_per_seed(steps: np.ndarray, means: np.ndarray,
                    threshold: float = 475.0, consec: int = 3) -> float:
    """Return first step at which eval_mean >= threshold for `consec` checkpoints."""
    above = means >= threshold
    for i in range(len(above) - consec + 1):
        if above[i: i + consec].all():
            return float(steps[i])
    return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# Per-algorithm metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_ppo_metrics(results_dir: Path) -> Dict[str, float]:
    """Compute all metrics for one PPO variant (multiple seed sub-dirs)."""
    eval_dfs = _load_ppo_eval_seeds(results_dir)
    train_dfs = _load_ppo_train_seeds(results_dir)

    if not eval_dfs:
        return {}

    # ── final_eval_return ────────────────────────────────────────────────────
    final_evals = []
    for df in eval_dfs:
        last5 = df["mean_return"].values[-5:]
        final_evals.append(last5.mean())
    final_eval_mean = float(np.mean(final_evals))
    final_eval_std = float(np.std(final_evals))

    # ── final_train_return ───────────────────────────────────────────────────
    final_trains = []
    for df in train_dfs:
        last100 = df["return"].values[-100:]
        final_trains.append(last100.mean())
    final_train_mean = float(np.mean(final_trains)) if final_trains else float("nan")
    final_train_std = float(np.std(final_trains)) if final_trains else float("nan")

    # ── AULC ─────────────────────────────────────────────────────────────────
    aulc_1m_list, aulc_200k_list = [], []
    for df in eval_dfs:
        steps = df["step"].values.astype(float)
        means = df["mean_return"].values.astype(float)
        aulc_1m_list.append(_aulc(steps, means, 1e6))
        aulc_200k_list.append(_aulc(steps, means, 2e5))
    aulc_1m = float(np.nanmean(aulc_1m_list))
    aulc_200k = float(np.nanmean(aulc_200k_list))

    # ── t_475 ─────────────────────────────────────────────────────────────────
    t475_list = []
    post_min_returns = []
    for df in eval_dfs:
        steps = df["step"].values.astype(float)
        means = df["mean_return"].values.astype(float)
        t = _t_475_per_seed(steps, means)
        t475_list.append(t)
        if np.isfinite(t):
            post = means[steps >= t]
            if len(post) > 0:
                post_min_returns.append(float(post.min()))

    finite_t = [t for t in t475_list if np.isfinite(t)]
    t475_mean = float(np.mean(finite_t)) if finite_t else float("nan")
    t475_std = float(np.std(finite_t)) if finite_t else float("nan")
    solve_rate = len(finite_t) / len(t475_list)
    post_min = float(np.mean(post_min_returns)) if post_min_returns else float("nan")

    return {
        "final_eval_return_mean": final_eval_mean,
        "final_eval_return_std": final_eval_std,
        "final_train_return_mean": final_train_mean,
        "final_train_return_std": final_train_std,
        "AULC_1M": aulc_1m,
        "AULC_200k": aulc_200k,
        "t_475_mean": t475_mean,
        "t_475_std": t475_std,
        "solve_rate": solve_rate,
        "post_solve_min_return": post_min,
        "n_seeds": len(eval_dfs),
    }


def compute_npz_metrics(npz_path: Path) -> Dict[str, float]:
    """Compute metrics from a previous-assignment .npz file.

    NPZ format:  rewards[i] = 1-D array of per-episode returns for seed i
                 steps[i]   = 1-D array of cumulative env steps for seed i
    """
    data = np.load(npz_path, allow_pickle=True)
    all_rewards = data["rewards"]   # shape (n_seeds,) of object arrays
    all_steps = data["steps"]       # shape (n_seeds,) of object arrays
    n_seeds = len(all_rewards)

    final_trains, aulc_1m_list, aulc_200k_list, t475_list, post_min_returns = [], [], [], [], []

    for i in range(n_seeds):
        rewards = np.array(all_rewards[i], dtype=np.float32)
        steps = np.array(all_steps[i], dtype=np.float32)

        # final_train_return: last 100 episodes
        last100 = rewards[-100:]
        final_trains.append(last100.mean())

        # AULC: use episode returns as the "eval" signal (no separate eval)
        aulc_1m_list.append(_aulc(steps, rewards, 1e6))
        aulc_200k_list.append(_aulc(steps, rewards, 2e5))

        # t_475: smoothed with window=25 before checking threshold
        window = 25
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            smooth_steps = steps[window - 1:]
            t = _t_475_per_seed(smooth_steps, smoothed, consec=3)
        else:
            t = float("nan")
        t475_list.append(t)
        if np.isfinite(t):
            post = rewards[steps >= t]
            if len(post) > 0:
                post_min_returns.append(float(post.min()))

    finite_t = [t for t in t475_list if np.isfinite(t)]

    return {
        "final_eval_return_mean": float(np.mean(final_trains)),
        "final_eval_return_std": float(np.std(final_trains)),
        "final_train_return_mean": float(np.mean(final_trains)),
        "final_train_return_std": float(np.std(final_trains)),
        "AULC_1M": float(np.nanmean(aulc_1m_list)),
        "AULC_200k": float(np.nanmean(aulc_200k_list)),
        "t_475_mean": float(np.mean(finite_t)) if finite_t else float("nan"),
        "t_475_std": float(np.std(finite_t)) if finite_t else float("nan"),
        "solve_rate": len(finite_t) / n_seeds,
        "post_solve_min_return": float(np.mean(post_min_returns)) if post_min_returns else float("nan"),
        "n_seeds": n_seeds,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX table helpers
# ──────────────────────────────────────────────────────────────────────────────

def metrics_to_latex(df: pd.DataFrame) -> str:
    """Convert a metrics DataFrame to a LaTeX table string."""
    cols_display = {
        "algorithm": "Algorithm",
        "final_eval_return_mean": r"Final Eval $\bar{R}$",
        "final_eval_return_std": r"$\sigma$",
        "AULC_1M": "AULC$_{1M}$",
        "AULC_200k": "AULC$_{200k}$",
        "t_475_mean": r"$t_{475}$ (steps)",
        "solve_rate": "Solve Rate",
    }
    sub = df[[c for c in cols_display if c in df.columns]].copy()
    sub.rename(columns=cols_display, inplace=True)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Performance metrics on CartPole-v1 over 5 seeds}",
        r"\label{tab:metrics}",
        r"\small",
        sub.to_latex(index=False, float_format="%.3f", escape=False),
        r"\end{table}",
    ]
    return "\n".join(lines)
