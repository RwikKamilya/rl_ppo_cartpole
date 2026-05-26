"""
Plotting utilities for the final PPO CartPole-v1 study.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

_cache_root = Path(os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache")))
_cache_root.mkdir(parents=True, exist_ok=True)
_mpl_cache_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib")))
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_a4.metrics import (
    BASELINE_METHODS,
    PPO_ABLATION_METHODS,
    PPO_FINAL_METHODS,
    PPO_SELECTED_METHODS,
    compute_ppo_per_seed_df,
)


COLORS = {
    "PPO_final": "#1b4d89",
    "PPO_selected": "#111111",
    "PPO_no_clip": "#b04a4a",
    "PPO_lambda0": "#db8f2f",
    "PPO_no_entropy": "#6a8f2a",
    "PPO_adv_norm_on": "#7b5794",
    "PPO_single_epoch": "#2f8f8f",
    "DQN": "#4c78a8",
    "REINFORCE": "#f58518",
    "AC": "#54a24b",
    "A2C": "#e45756",
}

LINESTYLES = {
    "PPO_final": "-",
    "PPO_selected": (0, (7, 2)),
    "PPO_no_clip": "--",
    "PPO_lambda0": "-.",
    "PPO_no_entropy": ":",
    "PPO_adv_norm_on": (0, (5, 2)),
    "PPO_single_epoch": (0, (3, 1, 1, 1)),
}


def _moving_average(values: np.ndarray, window: int = 25) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _save_pdf_png(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=160, bbox_inches="tight")


def _style_return_axis(ax: plt.Axes, title: str, max_step: float) -> None:
    ax.set_title(title)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Return")
    ax.set_xlim(0, max_step)
    ax.set_ylim(0, 520)
    ax.axhline(500, color="black", linestyle=":", linewidth=1)
    ax.grid(alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_mean_std(
    ax: plt.Axes,
    steps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    label: str,
    color: str,
    linestyle="-",
) -> None:
    ax.plot(steps, mean, color=color, linestyle=linestyle, linewidth=1.8, label=label)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16)


def _common_grid(max_step: float, points: int = 500) -> np.ndarray:
    return np.linspace(0.0, float(max_step), points)


def _interpolate_curves(curves: Iterable[Tuple[np.ndarray, np.ndarray]], max_step: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grid = _common_grid(max_step)
    values = []
    for steps, returns in curves:
        if len(steps) < 2 or len(returns) < 2:
            continue
        mask = steps <= max_step
        if np.sum(mask) < 2:
            continue
        s = steps[mask].astype(float)
        r = returns[mask].astype(float)
        values.append(np.interp(grid, s, r, left=r[0], right=r[-1]))
    if not values:
        return None
    arr = np.stack(values, axis=0)
    return grid, arr.mean(axis=0), arr.std(axis=0)


def load_ppo_eval_curves(results_dir: Path, max_step: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    curves = []
    for seed_dir in sorted(results_dir.glob("seed_*")):
        csv_path = seed_dir / "eval.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        curves.append((df["step"].values.astype(float), df["mean_return"].values.astype(float)))
    return _interpolate_curves(curves, max_step)


def load_npz_curves(npz_path: Path, max_step: float, smooth_window: int = 25) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    curves = []
    for rewards, steps in zip(data["rewards"], data["steps"]):
        rewards = np.asarray(rewards, dtype=float)
        steps = np.asarray(steps, dtype=float)
        if len(steps) > 1 and np.any(np.diff(steps) < 0):
            steps = np.cumsum(steps)
        n = min(len(rewards), len(steps))
        rewards = rewards[:n]
        steps = steps[:n]
        smooth = _moving_average(rewards, smooth_window)
        smooth_steps = steps[-len(smooth):]
        curves.append((smooth_steps, smooth))
    return _interpolate_curves(curves, max_step)


def load_update_metric_curves(results_dir: Path, column: str, max_step: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    curves = []
    for seed_dir in sorted(results_dir.glob("seed_*")):
        csv_path = seed_dir / "update_log.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if column not in df.columns:
            continue
        curves.append((df["step"].values.astype(float), df[column].values.astype(float)))
    return _interpolate_curves(curves, max_step)


def _max_observed_step(variant_dirs: Dict[str, Path], default: float) -> float:
    max_step = float(default)
    for variant_dir in variant_dirs.values():
        for csv_path in variant_dir.glob("seed_*/eval.csv"):
            df = pd.read_csv(csv_path, usecols=["step"])
            if not df.empty:
                max_step = max(max_step, float(df["step"].max()))
    return max_step


def plot_final_learning_curves(
    study_root: Path,
    previous_results_dir: Path,
    output_dir: Path,
    max_step: float = 1_000_000.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    for method, (display, rel_path) in {**PPO_FINAL_METHODS, **PPO_SELECTED_METHODS}.items():
        ppo_dir = study_root / rel_path
        result = load_ppo_eval_curves(ppo_dir, max_step)
        if result is not None:
            steps, mean, std = result
            _plot_mean_std(
                ax,
                steps,
                mean,
                std,
                display,
                COLORS.get(method, "#333333"),
                LINESTYLES.get(method, "-"),
            )

    for method, (display, filename) in BASELINE_METHODS.items():
        result = load_npz_curves(previous_results_dir / filename, max_step)
        if result is None:
            continue
        steps, mean, std = result
        _plot_mean_std(ax, steps, mean, std, display, COLORS[method])

    _style_return_axis(ax, "CartPole-v1 final comparison", max_step)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(
        0.01,
        0.02,
        "PPO: deterministic greedy evaluation. Baselines: saved Assignment 2/3 traces.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    _save_pdf_png(fig, output_dir, "final_learning_curves_comparison")
    plt.close(fig)


def _available_variant_dirs(study_root: Path) -> Dict[str, Path]:
    dirs: Dict[str, Path] = {}
    for method, (_, rel_path) in {**PPO_FINAL_METHODS, **PPO_ABLATION_METHODS}.items():
        variant_dir = study_root / rel_path
        if variant_dir.exists() and any(variant_dir.glob("seed_*/eval.csv")):
            dirs[method] = variant_dir
    return dirs


def plot_ppo_ablation_curves(study_root: Path, output_dir: Path) -> None:
    variant_dirs = _available_variant_dirs(study_root)
    max_step = _max_observed_step(variant_dirs, 500_000.0)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    for method, variant_dir in variant_dirs.items():
        display = {**PPO_FINAL_METHODS, **PPO_ABLATION_METHODS}[method][0]
        result = load_ppo_eval_curves(variant_dir, max_step)
        if result is None:
            continue
        steps, mean, std = result
        _plot_mean_std(
            ax,
            steps,
            mean,
            std,
            display,
            COLORS.get(method, "#333333"),
            LINESTYLES.get(method, "-"),
        )

    _style_return_axis(ax, "PPO ablation curves", max_step)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save_pdf_png(fig, output_dir, "ppo_ablation_curves")
    plt.close(fig)


def plot_ppo_diagnostics(study_root: Path, output_dir: Path) -> None:
    variant_dirs = _available_variant_dirs(study_root)
    max_step = _max_observed_step(variant_dirs, 500_000.0)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    diagnostics = [
        ("approx_kl", "Approximate KL", "KL"),
        ("clip_fraction", "Clip fraction", "Fraction"),
        ("entropy", "Policy entropy", "Entropy"),
        ("value_loss", "Value loss", "Loss"),
    ]

    for ax, (column, title, ylabel) in zip(axes.flat, diagnostics):
        for method, variant_dir in variant_dirs.items():
            result = load_update_metric_curves(variant_dir, column, max_step)
            if result is None:
                continue
            steps, mean, std = result
            display = {**PPO_FINAL_METHODS, **PPO_ABLATION_METHODS}[method][0]
            _plot_mean_std(
                ax,
                steps,
                mean,
                std,
                display,
                COLORS.get(method, "#333333"),
                LINESTYLES.get(method, "-"),
            )
        ax.set_title(title)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_pdf_png(fig, output_dir, "ppo_diagnostics")
    plt.close(fig)


def plot_ppo_stability_summary(study_root: Path, output_dir: Path) -> None:
    variant_dirs = _available_variant_dirs(study_root)
    metrics = [
        ("t_475", "Solve time t_475", "Steps"),
        ("final_eval_return", "Final return", "Return"),
        ("post_solve_worst_return", "Post-solve worst", "Return"),
        ("late_100k_worst_return", "Late-100k worst", "Return"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.5))

    labels = []
    data_by_metric = {key: [] for key, _, _ in metrics}
    colors = []
    for method, variant_dir in variant_dirs.items():
        display = {**PPO_FINAL_METHODS, **PPO_ABLATION_METHODS}[method][0]
        try:
            df = compute_ppo_per_seed_df(variant_dir, method, display)
        except Exception:
            continue
        if df.empty:
            continue
        labels.append(display.replace("PPO ", ""))
        colors.append(COLORS.get(method, "#333333"))
        for key, _, _ in metrics:
            vals = pd.to_numeric(df[key], errors="coerce").dropna().values
            data_by_metric[key].append(vals)

    for ax, (key, title, ylabel) in zip(axes, metrics):
        data = [vals for vals in data_by_metric[key] if len(vals)]
        positions = [idx + 1 for idx, vals in enumerate(data_by_metric[key]) if len(vals)]
        if data:
            box = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
            for patch, pos in zip(box["boxes"], positions):
                patch.set_facecolor(colors[pos - 1])
                patch.set_alpha(0.35)
            for pos, vals in zip(positions, data):
                ax.scatter([pos] * len(vals), vals, color=colors[pos - 1], s=22, alpha=0.75)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_pdf_png(fig, output_dir, "ppo_stability_summary")
    plt.close(fig)


def write_figure_captions(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    captions = {
        "final_learning_curves_comparison": (
            "PPO_final and, when present, PPO_selected compared with DQN, REINFORCE, AC and A2C on CartPole-v1. "
            "PPO curves use deterministic greedy evaluation checkpoints; baseline "
            "curves use saved Assignment 2/3 training-return traces."
        ),
        "ppo_ablation_curves": (
            "PPO_final and ablations under the same tuned hyperparameter family. "
            "Curves show deterministic evaluation return over environment steps."
        ),
        "ppo_diagnostics": (
            "Training diagnostics from PPO update logs: approximate KL, clip "
            "fraction, policy entropy and value loss."
        ),
        "ppo_stability_summary": (
            "Seed-level stability metrics: solve time, final deterministic return, "
            "post-solve worst return and worst return in the final 100k-step window."
        ),
    }
    with (output_dir / "figure_captions.txt").open("w") as f:
        for name, caption in captions.items():
            f.write(f"{name}\n")
            f.write("-" * len(name) + "\n")
            f.write(caption + "\n\n")


def generate_all_plots(study_root: Path, previous_results_dir: Path, output_dir: Path) -> None:
    plot_final_learning_curves(study_root, previous_results_dir, output_dir)
    plot_ppo_ablation_curves(study_root, output_dir)
    plot_ppo_diagnostics(study_root, output_dir)
    plot_ppo_stability_summary(study_root, output_dir)
    write_figure_captions(output_dir)

