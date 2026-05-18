"""
plotting.py – Publication-ready figure generation.

Generates multiple figures:
  1. figures/main_comparison.{pdf,png}
     DQN vs REINFORCE vs AC vs A2C vs PPO_full on CartPole-v1.
  2. figures/ppo_ablation.{pdf,png}
     PPO_full vs PPO_no_clip vs PPO_lambda0 vs PPO_no_adv_norm vs PPO_single_epoch.
  3. figures/ppo_eval.pdf
     Deterministic evaluation return for PPO_full over env steps.
  4. figures/ppo_diagnostics.{pdf,png}
     PPO update diagnostics: KL, clip fraction, entropy, value loss.
  5. figures/ppo_stability.{pdf,png}
     Per-seed robustness summaries across PPO ablations.

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
    "PPO_tuned": "#2f4b7c",
    "PPO_no_clip": "#8c564b",
    "PPO_lambda0": "#e377c2",
    "PPO_no_adv_norm": "#7f7f7f",
    "PPO_single_epoch": "#17becf",
}

LINESTYLES = {
    "PPO_full": "-",
    "PPO_no_clip": "--",
    "PPO_lambda0": "-.",
    "PPO_no_adv_norm": ":",
    "PPO_single_epoch": (0, (3, 1, 1, 1)),
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


def _moving_average_same(arr: np.ndarray, window: int) -> np.ndarray:
    """1-D moving average with same-length output for plotting."""
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


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


def load_ppo_update_metric_curves(
    results_dir: Path,
    column: str,
    max_step: float = STEP_MAX,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load PPO update diagnostics and interpolate them onto a common step grid."""
    seed_dirs = sorted(results_dir.glob("seed_*"))
    if not seed_dirs:
        return None

    common_steps = np.linspace(0, max_step, 500)
    interped = []
    for sd in seed_dirs:
        csv = sd / "update_log.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if column not in df.columns:
            continue
        steps = df["step"].values.astype(float)
        values = df[column].values.astype(float)
        mask = steps <= max_step
        if mask.sum() < 2:
            continue
        interp_vals = np.interp(
            common_steps,
            steps[mask],
            values[mask],
            left=values[mask][0],
            right=values[mask][-1],
        )
        interped.append(interp_vals)

    if not interped:
        return None

    arr = np.stack(interped, axis=0)
    return common_steps, arr.mean(axis=0), arr.std(axis=0)


def load_ppo_eval_seed_stats(results_dir: Path) -> Optional[Dict[str, List[float]]]:
    """Compute per-seed stability statistics from PPO eval logs."""
    seed_dirs = sorted(results_dir.glob("seed_*"))
    if not seed_dirs:
        return None

    solve_times = []
    final_returns = []
    post_solve_mins = []

    for sd in seed_dirs:
        csv = sd / "eval.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        steps = df["step"].values.astype(float)
        means = df["mean_return"].values.astype(float)
        if len(means) == 0:
            continue

        final_returns.append(float(means[-5:].mean()))

        solve_step = float("nan")
        for i in range(len(means) - 2):
            if np.all(means[i:i + 3] >= 475.0):
                solve_step = float(steps[i])
                break
        solve_times.append(solve_step)

        if np.isfinite(solve_step):
            post = means[steps >= solve_step]
            post_solve_mins.append(float(post.min()) if len(post) else float("nan"))
        else:
            post_solve_mins.append(float("nan"))

    if not final_returns:
        return None

    return {
        "t_475": solve_times,
        "final_eval_return": final_returns,
        "post_solve_min_return": post_solve_mins,
    }


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


def _setup_diag_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_xlim(0, STEP_MAX)
    ax.set_xlabel("Environment Steps", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _variant_dir_name(variant: str) -> str:
    return variant.lower()


def _count_seed_dirs(results_dir: Path) -> int:
    return len(list(results_dir.glob("seed_*")))


def _smooth_plot_curves(
    steps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window <= 1:
        return steps, mean, std
    return (
        steps,
        _moving_average_same(mean, window),
        _moving_average_same(std, window),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: main comparison
# ──────────────────────────────────────────────────────────────────────────────

def _latest_ppo_dir(results_dir: Path) -> Optional[Tuple[str, Path]]:
    """Pick the best available PPO directory from a stage result folder."""
    candidates = [
        ("PPO (best tuned)", results_dir / "ppo_tuned"),
        ("PPO (baseline)", results_dir / "ppo_full"),
    ]
    for label, path in candidates:
        if path.exists() and list(path.glob("seed_*/eval.csv")):
            return label, path
    return None


def plot_main_comparison(
    results_dir: Path,
    prev_results_dir: Path,
    figures_dir: Path,
) -> None:
    """DQN vs REINFORCE vs AC vs A2C vs latest PPO."""
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

    # ── Latest PPO (eval curves) ───────────────────────────────────────────
    latest_ppo = _latest_ppo_dir(results_dir)
    if latest_ppo is None:
        print("  [WARNING] No staged PPO results found. Plotting baselines only.")
        latest_label = "PPO"
        ppo_dir = results_dir / "ppo_full"
    else:
        latest_label, ppo_dir = latest_ppo
    ppo_result = load_ppo_eval_curves(ppo_dir)
    if ppo_result is None:
        print(f"  [WARNING] {latest_label} eval results not found. Trying train curves.")
        ppo_result = load_ppo_train_curves(ppo_dir)
    if ppo_result is not None:
        steps, mean, std = ppo_result
        n_ppo_seeds = len(list(ppo_dir.glob("seed_*")))
        n_seeds_found = max(n_seeds_found, n_ppo_seeds)
        ppo_color_key = "PPO_tuned" if ppo_dir.name == "ppo_tuned" else "PPO_full"
        _add_curve(ax, steps, mean, std, label=latest_label,
                   color=COLORS.get(ppo_color_key, COLORS["PPO_full"]), linestyle="-")
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
    """PPO_full vs PPO_no_clip vs PPO_lambda0 vs PPO_no_adv_norm vs PPO_single_epoch."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    eval_smooth_window = 9

    variants = ["PPO_full", "PPO_no_clip", "PPO_lambda0", "PPO_no_adv_norm", "PPO_single_epoch"]
    labels = {
        "PPO_full": "PPO (full)",
        "PPO_no_clip": "PPO (no clip)",
        "PPO_lambda0": r"PPO ($\lambda$=0)",
        "PPO_no_adv_norm": "PPO (no adv norm)",
        "PPO_single_epoch": "PPO (single epoch)",
    }
    found_any = False
    seed_counts = []
    for variant in variants:
        vdir = results_dir / _variant_dir_name(variant)
        result = load_ppo_eval_curves(vdir)
        if result is None:
            result = load_ppo_train_curves(vdir)
        if result is None:
            print(f"  [WARNING] {variant} results not found, skipping.")
            continue
        steps, mean, std = result
        steps, mean, std = _smooth_plot_curves(steps, mean, std, eval_smooth_window)
        color = COLORS.get(variant, "#333333")
        ls = LINESTYLES.get(variant, "-")
        _add_curve(ax, steps, mean, std,
                   label=labels[variant], color=color, linestyle=ls, alpha_shade=0.10)
        found_any = True
        seed_counts.append(_count_seed_dirs(vdir))

    if not found_any:
        print("  [WARNING] No ablation results found.")
        plt.close(fig)
        return

    _setup_axes(ax, title="CartPole-v1: PPO Ablation Study")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.85)
    n_seeds = max(seed_counts) if seed_counts else 0
    ax.text(0.01, 0.01,
            f"Shaded: ±1 std dev | Deterministic eval | Smoothed: MA({eval_smooth_window}) | up to {n_seeds} seeds",
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
    """Deterministic evaluation curve for the latest PPO run."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    latest_ppo = _latest_ppo_dir(results_dir)
    if latest_ppo is None:
        print("  [WARNING] No staged PPO results found for eval plot.")
        plt.close(fig)
        return
    latest_label, ppo_dir = latest_ppo
    result = load_ppo_eval_curves(ppo_dir)
    if result is None:
        print(f"  [WARNING] {latest_label} eval results not found.")
        plt.close(fig)
        return

    steps, mean, std = result
    n_seeds = len(list(ppo_dir.glob("seed_*")))
    ppo_color_key = "PPO_tuned" if ppo_dir.name == "ppo_tuned" else "PPO_full"
    _add_curve(ax, steps, mean, std, label=f"{latest_label} (eval, {n_seeds} seeds)",
               color=COLORS.get(ppo_color_key, COLORS["PPO_full"]))

    _setup_axes(ax, title=f"CartPole-v1: {latest_label} Deterministic Evaluation")
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


def plot_ppo_sweep_summary(sweep_dir: Path, figures_dir: Path) -> None:
    """Plot one-factor sweep results in a compact, report-friendly format."""
    summary_csv = sweep_dir / "summary.csv"
    if not summary_csv.exists():
        print(f"  [WARNING] Sweep summary not found: {summary_csv}")
        return

    df = pd.read_csv(summary_csv)
    if df.empty:
        print("  [WARNING] Sweep summary is empty.")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))

    factor_specs = [
        ("clip_", "Clip Coefficient", "clip_coef"),
        ("gae_", "GAE Lambda", "gae_lambda"),
        ("rollout_", "Rollout Steps", "rollout_steps"),
        ("epochs_", "Update Epochs", "update_epochs"),
    ]

    def _factor_value(name: str, prefix: str):
        raw = name[len(prefix):]
        if prefix in ("clip_", "gae_"):
            return float(raw.replace("p", "."))
        return int(raw)

    for ax, (prefix, title, xlabel) in zip(axes.flat, factor_specs):
        sub = df[df["variant"].str.startswith(prefix)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub["factor_value"] = sub["variant"].apply(lambda name: _factor_value(name, prefix))
        sub.sort_values("factor_value", inplace=True)

        x = np.arange(len(sub))
        bars = ax.bar(x, sub["AULC_200k"], color="#4c78a8", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["factor_value"].tolist())
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("AULC_200k", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

        best_idx = int(sub["AULC_200k"].idxmax())
        for rect, (_, row) in zip(bars, sub.iterrows()):
            label = f"{row['final_eval_return_mean']:.0f}"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.01,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )
        best_row = sub.loc[best_idx]
        ax.axvline(sub.index.get_loc(best_idx), color="#d62728", linestyle=":", linewidth=1.0)
        ax.text(
            0.02,
            0.98,
            f"Best: {best_row['factor_value']} | final eval {best_row['final_eval_return_mean']:.0f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#555555",
        )

    fig.suptitle("PPO One-Factor Sweep Summary", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Bars show early-learning performance (AULC_200k); labels show final deterministic evaluation return.",
        ha="center",
        fontsize=9,
        color="gray",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    for ext in ("pdf", "png"):
        out = figures_dir / f"ppo_sweep_summary.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


def plot_ppo_diagnostics(results_dir: Path, figures_dir: Path) -> None:
    """Plot PPO training diagnostics that explain ablation behavior."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    variants = ["PPO_full", "PPO_no_clip", "PPO_lambda0", "PPO_no_adv_norm", "PPO_single_epoch"]
    labels = {
        "PPO_full": "PPO (full)",
        "PPO_no_clip": "PPO (no clip)",
        "PPO_lambda0": r"PPO ($\lambda$=0)",
        "PPO_no_adv_norm": "PPO (no adv norm)",
        "PPO_single_epoch": "PPO (single epoch)",
    }
    diagnostics = [
        ("approx_kl", "Approximate KL", "Approx. KL"),
        ("clip_fraction", "Clip Fraction", "Clip Fraction"),
        ("entropy", "Policy Entropy", "Entropy"),
        ("value_loss", "Critic Loss", "Value Loss"),
    ]

    found_any = False
    for ax, (column, title, ylabel) in zip(axes.flat, diagnostics):
        for variant in variants:
            vdir = results_dir / _variant_dir_name(variant)
            result = load_ppo_update_metric_curves(vdir, column)
            if result is None:
                continue
            steps, mean, std = result
            _add_curve(
                ax,
                steps,
                mean,
                std,
                label=labels[variant],
                color=COLORS.get(variant, "#333333"),
                linestyle=LINESTYLES.get(variant, "-"),
                alpha_shade=0.12,
            )
            found_any = True
        _setup_diag_axis(ax, title, ylabel)

    if not found_any:
        print("  [WARNING] No PPO diagnostic logs found.")
        plt.close(fig)
        return

    handles, labels_list = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="upper center", ncol=5, framealpha=0.9, fontsize=10)
    fig.text(
        0.5,
        0.01,
        "These plots show whether each trick improves trust-region control, exploration, and critic stability.",
        ha="center",
        fontsize=9,
        color="gray",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    for ext in ("pdf", "png"):
        out = figures_dir / f"ppo_diagnostics.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


def plot_ppo_stability(results_dir: Path, figures_dir: Path) -> None:
    """Plot per-seed stability and robustness summaries across PPO variants."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    variants = ["PPO_full", "PPO_no_clip", "PPO_lambda0", "PPO_no_adv_norm", "PPO_single_epoch"]
    labels = ["full", "no_clip", "lambda0", "no_adv_norm", "single_epoch"]
    colors = [COLORS.get(v, "#333333") for v in variants]
    metrics = [
        ("t_475", r"Solve Time $t_{475}$", "Steps"),
        ("final_eval_return", "Final Eval Return", "Return"),
        ("post_solve_min_return", "Post-Solve Worst Return", "Return"),
    ]

    variant_stats = []
    for variant in variants:
        stats = load_ppo_eval_seed_stats(results_dir / _variant_dir_name(variant))
        variant_stats.append(stats)

    if not any(stats is not None for stats in variant_stats):
        print("  [WARNING] No PPO eval logs found for stability plot.")
        plt.close(fig)
        return

    for ax, (metric_key, title, ylabel) in zip(axes, metrics):
        data = []
        positions = []
        scatter_x = []
        scatter_y = []
        scatter_c = []

        for idx, stats in enumerate(variant_stats, start=1):
            if stats is None:
                continue
            vals = np.array(stats[metric_key], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            data.append(vals)
            positions.append(idx)
            scatter_x.extend([idx] * len(vals))
            scatter_y.extend(vals.tolist())
            scatter_c.extend([colors[idx - 1]] * len(vals))

        if not data:
            ax.set_visible(False)
            continue

        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(box["boxes"], [colors[p - 1] for p in positions]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)

        ax.scatter(scatter_x, scatter_y, s=26, c=scatter_c, alpha=0.75, zorder=3)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("PPO Ablation Robustness Across Seeds", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "Boxes summarize seed variation; points show individual seeds. This is useful for discussing stability, not just mean reward.",
        ha="center",
        fontsize=9,
        color="gray",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])

    for ext in ("pdf", "png"):
        out = figures_dir / f"ppo_stability.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)
