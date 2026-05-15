"""
plotting.py – Publication-ready figure generation.

Generates three figures:
  1. figures/main_comparison.{pdf,png}
     DQN vs REINFORCE vs AC vs A2C vs PPO_full on CartPole-v1.
  2. figures/ppo_ablation.{pdf,png}
     PPO_full vs PPO_no_clip vs PPO_lambda0 vs PPO_no_adv_norm.
  3. figures/ppo_eval.pdf
     Deterministic evaluation return for PPO_full over env steps.

Plot style:
  - x-axis: environment steps, 0 to 1e6
  - y-axis: return, 0 to 520
  - horizontal dotted line at y=500 (CartPole max)
  - smoothed mean curve (moving average window=25) with ±1 std deviation shade
  - For PPO eval curves: raw eval mean ± std (already smooth, no MA needed)
  - label: "mean ± 1 std dev over N seeds"

Previous results are loaded from .npz files (A2/A3 format):
  - rewards[i], steps[i]: episode returns and step counts for seed i
  - Smoothing applied: MA window=25 before shading.
  - Missing npz files print a warning and are skipped gracefully.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "DQN": "#1f77b4",
    "REINFORCE": "#ff7f0e",
    "AC": "#2ca02c",
    "A2C": "#d62728",
    "PPO_full": "#9467bd",
    "PPO_no_clip": "#8c564b",
    "PPO_lambda0": "#e377c2",
    "PPO_no_adv_norm": "#7f7f7f",
}

LINESTYLES = {
    "PPO_full": "-",
    "PPO_no_clip": "--",
    "PPO_lambda0": "-.",
    "PPO_no_adv_norm": ":",
}

MA_WINDOW = 25
STEP_MAX = 1_000_000
RETURN_MAX = 520
SOLVE_LINE = 500


# ──────────────────────────────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────────────────────────────

def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """1-D moving average (valid mode)."""
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def load_npz_curves(
    npz_path: Path,
    smooth_window: int = MA_WINDOW,
    max_step: float = STEP_MAX,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load a previous-assignment npz and return (common_steps, mean, std).

    Returns None if file is missing.
    """
    if not npz_path.exists():
        print(f"  [WARNING] Previous result not found: {npz_path}")
        return None

    data = np.load(npz_path, allow_pickle=True)
    all_rewards = data["rewards"]
    all_steps = data["steps"]
    n_seeds = len(all_rewards)

    # Interpolate each seed onto a common step grid
    common_steps = np.linspace(0, max_step, 500)
    interped = []
    for i in range(n_seeds):
        steps = np.array(all_steps[i], dtype=float)
        rewards = np.array(all_rewards[i], dtype=float)
        # Smooth first
        if len(rewards) >= smooth_window:
            smoothed = _moving_average(rewards, smooth_window)
            s = steps[smooth_window - 1:]
        else:
            smoothed = rewards
            s = steps
        # Clip to max_step
        mask = s <= max_step
        if mask.sum() < 2:
            continue
        # Interpolate onto common grid
        interp_fn = np.interp(common_steps, s[mask], smoothed[mask],
                              left=smoothed[mask][0], right=smoothed[mask][-1])
        interped.append(interp_fn)

    if not interped:
        return None

    arr = np.stack(interped, axis=0)  # (n_seeds, n_points)
    return common_steps, arr.mean(axis=0), arr.std(axis=0)


def load_ppo_eval_curves(
    results_dir: Path,
    max_step: float = STEP_MAX,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load PPO eval CSVs and return (common_steps, mean_curve, std_curve).

    Each seed's eval curve is interpolated onto a common grid before averaging.
    """
    seed_dirs = sorted(results_dir.glob("seed_*"))
    if not seed_dirs:
        return None

    common_steps = np.linspace(0, max_step, 500)
    interped = []
    for sd in seed_dirs:
        csv = sd / "eval.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        steps = df["step"].values.astype(float)
        means = df["mean_return"].values.astype(float)
        mask = steps <= max_step
        if mask.sum() < 2:
            continue
        interp_vals = np.interp(common_steps, steps[mask], means[mask],
                                left=means[mask][0], right=means[mask][-1])
        interped.append(interp_vals)

    if not interped:
        return None

    arr = np.stack(interped, axis=0)
    return common_steps, arr.mean(axis=0), arr.std(axis=0)


def load_ppo_train_curves(
    results_dir: Path,
    smooth_window: int = MA_WINDOW,
    max_step: float = STEP_MAX,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load PPO train CSVs, smooth and interpolate onto common grid."""
    seed_dirs = sorted(results_dir.glob("seed_*"))
    if not seed_dirs:
        return None

    common_steps = np.linspace(0, max_step, 500)
    interped = []
    for sd in seed_dirs:
        csv = sd / "train.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        steps = df["step"].values.astype(float)
        returns = df["return"].values.astype(float)
        if len(returns) >= smooth_window:
            smoothed = _moving_average(returns, smooth_window)
            s = steps[smooth_window - 1:]
        else:
            smoothed = returns
            s = steps
        mask = s <= max_step
        if mask.sum() < 2:
            continue
        interp_vals = np.interp(common_steps, s[mask], smoothed[mask],
                                left=smoothed[mask][0], right=smoothed[mask][-1])
        interped.append(interp_vals)

    if not interped:
        return None

    arr = np.stack(interped, axis=0)
    return common_steps, arr.mean(axis=0), arr.std(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _setup_axes(ax: plt.Axes, title: str = "", n_seeds: int = 5) -> None:
    ax.set_xlim(0, STEP_MAX)
    ax.set_ylim(0, RETURN_MAX)
    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.axhline(SOLVE_LINE, color="black", linestyle=":", linewidth=1.0,
               label="_nolegend_")
    ax.text(STEP_MAX * 0.01, SOLVE_LINE + 4, "max (500)", fontsize=8, color="black")
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_curve(ax: plt.Axes, steps, mean, std, label, color,
               linestyle="-", alpha_shade=0.15) -> None:
    ax.plot(steps, mean, color=color, linestyle=linestyle,
            linewidth=1.8, label=label)
    ax.fill_between(steps, mean - std, mean + std,
                    color=color, alpha=alpha_shade)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: main comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_main_comparison(
    results_dir: Path,
    prev_results_dir: Path,
    figures_dir: Path,
) -> None:
    """DQN vs REINFORCE vs AC vs A2C vs PPO_full."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    n_seeds_found = 0

    # ── Previous baselines ────────────────────────────────────────────────────
    prev_algorithms = {
        "DQN": prev_results_dir / "linear_basic_training.npz",
        "REINFORCE": prev_results_dir / "reinforce.npz",
        "AC": prev_results_dir / "ac.npz",
        "A2C": prev_results_dir / "a2c.npz",
    }
    for name, npz_path in prev_algorithms.items():
        result = load_npz_curves(npz_path)
        if result is None:
            continue
        steps, mean, std = result
        color = COLORS.get(name, "#333333")
        _add_curve(ax, steps, mean, std, label=name, color=color)
        n_seeds_found = max(n_seeds_found, 5)

    # ── PPO_full (eval curves) ─────────────────────────────────────────────
    ppo_dir = results_dir / "ppo_full"
    ppo_result = load_ppo_eval_curves(ppo_dir)
    if ppo_result is None:
        print("  [WARNING] PPO_full eval results not found. Trying train curves.")
        ppo_result = load_ppo_train_curves(ppo_dir)
    if ppo_result is not None:
        steps, mean, std = ppo_result
        n_ppo_seeds = len(list(ppo_dir.glob("seed_*")))
        n_seeds_found = max(n_seeds_found, n_ppo_seeds)
        _add_curve(ax, steps, mean, std, label="PPO",
                   color=COLORS["PPO_full"], linestyle="-")
    else:
        print("  [WARNING] No PPO results found. Plotting baselines only.")

    _setup_axes(ax, title="CartPole-v1: Algorithm Comparison")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", fontsize=10, framealpha=0.85)
    ax.text(0.01, 0.01,
            f"Shaded: ±1 std dev | Smoothed: MA({MA_WINDOW}) | {n_seeds_found} seeds",
            transform=ax.transAxes, fontsize=7.5, color="gray",
            verticalalignment="bottom")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = figures_dir / f"main_comparison.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: PPO ablation
# ──────────────────────────────────────────────────────────────────────────────

def plot_ppo_ablation(results_dir: Path, figures_dir: Path) -> None:
    """PPO_full vs PPO_no_clip vs PPO_lambda0 vs PPO_no_adv_norm."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    variants = ["PPO_full", "PPO_no_clip", "PPO_lambda0", "PPO_no_adv_norm"]
    labels = {
        "PPO_full": "PPO (full)",
        "PPO_no_clip": "PPO (no clip)",
        "PPO_lambda0": r"PPO ($\lambda$=0)",
        "PPO_no_adv_norm": "PPO (no adv norm)",
    }
    found_any = False
    for variant in variants:
        vdir = results_dir / variant.lower()
        result = load_ppo_eval_curves(vdir)
        if result is None:
            result = load_ppo_train_curves(vdir)
        if result is None:
            print(f"  [WARNING] {variant} results not found, skipping.")
            continue
        steps, mean, std = result
        color = COLORS.get(variant, "#333333")
        ls = LINESTYLES.get(variant, "-")
        _add_curve(ax, steps, mean, std,
                   label=labels[variant], color=color, linestyle=ls)
        found_any = True

    if not found_any:
        print("  [WARNING] No ablation results found.")
        plt.close(fig)
        return

    _setup_axes(ax, title="CartPole-v1: PPO Ablation Study")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.85)
    ax.text(0.01, 0.01,
            f"Shaded: ±1 std dev | Eval curves | 5 seeds",
            transform=ax.transAxes, fontsize=7.5, color="gray",
            verticalalignment="bottom")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = figures_dir / f"ppo_ablation.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: PPO eval only
# ──────────────────────────────────────────────────────────────────────────────

def plot_ppo_eval(results_dir: Path, figures_dir: Path) -> None:
    """Deterministic evaluation curve for PPO_full."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ppo_dir = results_dir / "ppo_full"
    result = load_ppo_eval_curves(ppo_dir)
    if result is None:
        print("  [WARNING] PPO_full eval results not found.")
        plt.close(fig)
        return

    steps, mean, std = result
    n_seeds = len(list(ppo_dir.glob("seed_*")))
    _add_curve(ax, steps, mean, std, label=f"PPO (eval, {n_seeds} seeds)",
               color=COLORS["PPO_full"])

    _setup_axes(ax, title="CartPole-v1: PPO Deterministic Evaluation")
    ax.legend(loc="lower right", fontsize=10)
    ax.text(0.01, 0.01,
            "Action = argmax(logits) | 20 episodes per checkpoint | ±1 std dev",
            transform=ax.transAxes, fontsize=7.5, color="gray",
            verticalalignment="bottom")

    fig.tight_layout()
    out = figures_dir / "ppo_eval.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)
