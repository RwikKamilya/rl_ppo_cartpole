"""
Metric utilities for the final PPO CartPole-v1 study.

PPO logs are read from the final study layout:
  results/final_study/final/ppo_final/seed_*
  results/final_study/ablations/<variant>/seed_*

Previous Assignment 2/3 baselines are read from previous_results/*.npz. Those
baseline files contain saved training-return traces, not deterministic PPO-style
evaluation logs, so baseline comparisons are contextual.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


THRESHOLD_RETURN = 475.0
MAX_RETURN = 500.0
AULC_MAX_STEP = 200_000.0
LATE_WINDOW_STEPS = 100_000.0


PPO_FINAL_METHODS = {
    "PPO_final": ("PPO Final", Path("final") / "ppo_final"),
}

PPO_SELECTED_METHODS = {
    "PPO_selected": ("PPO Selected", Path("selected") / "ppo_selected"),
}

PPO_ABLATION_METHODS = {
    "PPO_no_clip": ("PPO No Clip", Path("ablations") / "ppo_no_clip"),
    "PPO_lambda0": ("PPO lambda=0", Path("ablations") / "ppo_lambda0"),
    "PPO_no_entropy": ("PPO No Entropy", Path("ablations") / "ppo_no_entropy"),
    "PPO_adv_norm_on": ("PPO Adv. Norm On", Path("ablations") / "ppo_adv_norm_on"),
    "PPO_single_epoch": ("PPO Single Epoch", Path("ablations") / "ppo_single_epoch"),
}

BASELINE_METHODS = {
    "DQN": ("DQN", "linear_basic_training.npz"),
    "REINFORCE": ("REINFORCE", "reinforce.npz"),
    "AC": ("AC", "ac.npz"),
    "A2C": ("A2C", "a2c.npz"),
}

METHOD_ORDER = [
    "PPO_final",
    "PPO_selected",
    "PPO_no_clip",
    "PPO_lambda0",
    "PPO_no_entropy",
    "PPO_adv_norm_on",
    "PPO_single_epoch",
    "DQN",
    "REINFORCE",
    "AC",
    "A2C",
]


def _seed_id_from_dir(seed_dir: Path) -> int:
    try:
        return int(seed_dir.name.split("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid seed directory name: {seed_dir}") from exc


def _load_ppo_seed_csv_items(
    results_dir: Path,
    filename: str,
    required_seeds: Optional[Sequence[int]] = None,
) -> List[Tuple[int, pd.DataFrame]]:
    seed_dirs = {
        _seed_id_from_dir(seed_dir): seed_dir
        for seed_dir in sorted(results_dir.glob("seed_*"))
        if seed_dir.is_dir()
    }

    if required_seeds is not None:
        missing = [
            seed
            for seed in required_seeds
            if seed not in seed_dirs or not (seed_dirs[seed] / filename).exists()
        ]
        if missing:
            expected = [f"seed_{seed}/{filename}" for seed in required_seeds]
            raise FileNotFoundError(
                f"Missing required {filename} for seeds {missing} in {results_dir}. "
                f"Expected files: {expected}"
            )

    items: List[Tuple[int, pd.DataFrame]] = []
    for seed, seed_dir in sorted(seed_dirs.items()):
        csv_path = seed_dir / filename
        if csv_path.exists():
            items.append((seed, pd.read_csv(csv_path)))
    return items


def _aulc_200k(steps: np.ndarray, returns: np.ndarray) -> float:
    mask = steps <= AULC_MAX_STEP
    s = steps[mask].astype(float)
    r = returns[mask].astype(float)
    if len(s) < 2:
        return float("nan")
    if s[0] > 0:
        s = np.concatenate([[0.0], s])
        r = np.concatenate([[0.0], r])
    area = float(np.trapz(r, s))
    return area / (MAX_RETURN * AULC_MAX_STEP)


def _first_solve_step(steps: np.ndarray, returns: np.ndarray) -> float:
    hits = np.where(returns >= THRESHOLD_RETURN)[0]
    if len(hits) == 0:
        return float("nan")
    return float(steps[hits[0]])


def _retention(values: np.ndarray, threshold: float = THRESHOLD_RETURN) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values >= threshold))


def _curve_metrics(
    steps: np.ndarray,
    returns: np.ndarray,
    final_return: Optional[float] = None,
    eval_return_std_last: float = float("nan"),
) -> Dict[str, float | bool]:
    steps = np.asarray(steps, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if len(steps) == 0 or len(returns) == 0:
        raise ValueError("Cannot compute metrics from an empty curve.")
    if len(steps) != len(returns):
        n = min(len(steps), len(returns))
        steps = steps[:n]
        returns = returns[:n]
    if len(steps) > 1 and np.any(np.diff(steps) < 0):
        raise ValueError("Metric curves require non-decreasing environment steps.")

    t475 = _first_solve_step(steps, returns)
    solved = bool(np.isfinite(t475))

    if solved:
        post = returns[steps >= t475]
        post_worst = float(np.min(post)) if len(post) else float("nan")
        post_retention = _retention(post)
    else:
        post_worst = float("nan")
        post_retention = 0.0

    late_start = max(0.0, float(np.max(steps)) - LATE_WINDOW_STEPS)
    late = returns[steps >= late_start]
    late_worst = float(np.min(late)) if len(late) else float("nan")
    late_retention = _retention(late)

    return {
        "aulc_200k": _aulc_200k(steps, returns),
        "t_475": t475,
        "solved": solved,
        "final_eval_return": float(returns[-1] if final_return is None else final_return),
        "post_solve_worst_return": post_worst,
        "post_solve_retention": post_retention,
        "late_100k_worst_return": late_worst,
        "late_100k_retention": late_retention,
        "eval_return_std_last": eval_return_std_last,
    }


def _diagnostic_metrics(update_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    out = {
        "approx_kl_mean": float("nan"),
        "approx_kl_max": float("nan"),
        "clip_fraction_mean": float("nan"),
        "clip_fraction_max": float("nan"),
        "entropy_final": float("nan"),
        "value_loss_max": float("nan"),
    }
    if update_df is None or update_df.empty:
        return out

    if "approx_kl" in update_df.columns:
        vals = pd.to_numeric(update_df["approx_kl"], errors="coerce").dropna().values
        if len(vals):
            out["approx_kl_mean"] = float(np.mean(vals))
            out["approx_kl_max"] = float(np.max(vals))
    if "clip_fraction" in update_df.columns:
        vals = pd.to_numeric(update_df["clip_fraction"], errors="coerce").dropna().values
        if len(vals):
            out["clip_fraction_mean"] = float(np.mean(vals))
            out["clip_fraction_max"] = float(np.max(vals))
    if "entropy" in update_df.columns:
        vals = pd.to_numeric(update_df["entropy"], errors="coerce").dropna().values
        if len(vals):
            out["entropy_final"] = float(vals[-1])
    if "value_loss" in update_df.columns:
        vals = pd.to_numeric(update_df["value_loss"], errors="coerce").dropna().values
        if len(vals):
            out["value_loss_max"] = float(np.max(vals))
    return out


def compute_ppo_per_seed_df(
    results_dir: Path,
    method: str,
    display_name: str,
    required_seeds: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    eval_items = _load_ppo_seed_csv_items(results_dir, "eval.csv", required_seeds)
    update_items = dict(_load_ppo_seed_csv_items(results_dir, "update_log.csv"))

    rows: List[Dict[str, object]] = []
    for seed, eval_df in eval_items:
        if eval_df.empty:
            continue
        steps = pd.to_numeric(eval_df["step"], errors="raise").values.astype(float)
        returns = pd.to_numeric(eval_df["mean_return"], errors="raise").values.astype(float)
        std_last = (
            float(pd.to_numeric(eval_df["std_return"], errors="coerce").dropna().values[-1])
            if "std_return" in eval_df.columns and len(eval_df["std_return"].dropna())
            else float("nan")
        )
        row: Dict[str, object] = {
            "method": method,
            "display_name": display_name,
            "seed": seed,
            **_curve_metrics(steps, returns, eval_return_std_last=std_last),
            **_diagnostic_metrics(update_items.get(seed)),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _moving_average(values: np.ndarray, window: int = 25) -> np.ndarray:
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def compute_npz_per_seed_df(
    npz_path: Path,
    method: str,
    display_name: str,
    required_seeds: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing baseline trace: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    rewards_runs = data["rewards"]
    step_runs = data["steps"]
    n_seeds = len(rewards_runs)

    if required_seeds is not None and len(required_seeds) != n_seeds:
        raise ValueError(
            f"{npz_path} contains {n_seeds} seeds, but --required-seeds has "
            f"{len(required_seeds)} entries: {list(required_seeds)}"
        )

    rows: List[Dict[str, object]] = []
    for idx in range(n_seeds):
        seed = int(required_seeds[idx]) if required_seeds is not None else idx
        rewards = np.asarray(rewards_runs[idx], dtype=float)
        steps = np.asarray(step_runs[idx], dtype=float)
        if len(steps) > 1 and np.any(np.diff(steps) < 0):
            steps = np.cumsum(steps)
        n = min(len(rewards), len(steps))
        rewards = rewards[:n]
        steps = steps[:n]
        if n == 0:
            continue

        smooth = _moving_average(rewards, 25)
        smooth_steps = steps[-len(smooth):]
        final_return = float(np.mean(rewards[-100:]))
        row: Dict[str, object] = {
            "method": method,
            "display_name": display_name,
            "seed": seed,
            **_curve_metrics(smooth_steps, smooth, final_return=final_return),
            **_diagnostic_metrics(None),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_per_seed_to_summary(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        raise ValueError("Cannot aggregate an empty per-seed DataFrame.")

    def mean(col: str) -> float:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        return float(np.mean(vals)) if len(vals) else float("nan")

    def std(col: str) -> float:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        return float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0

    method = str(df["method"].iloc[0])
    display_name = str(df["display_name"].iloc[0])
    finite_t = pd.to_numeric(df["t_475"], errors="coerce").dropna().values

    summary: Dict[str, object] = {
        "method": method,
        "display_name": display_name,
        "n_seeds": int(len(df)),
        "aulc_200k_mean": mean("aulc_200k"),
        "aulc_200k_std": std("aulc_200k"),
        "t_475_mean": float(np.mean(finite_t)) if len(finite_t) else float("nan"),
        "t_475_std": float(np.std(finite_t, ddof=0)) if len(finite_t) > 1 else 0.0,
        "solve_rate": float(np.mean(df["solved"].astype(bool).values)),
        "final_eval_return_mean": mean("final_eval_return"),
        "final_eval_return_std": std("final_eval_return"),
        "post_solve_worst_return_mean": mean("post_solve_worst_return"),
        "post_solve_worst_return_std": std("post_solve_worst_return"),
        "post_solve_retention_mean": mean("post_solve_retention"),
        "post_solve_retention_std": std("post_solve_retention"),
        "late_100k_worst_return_mean": mean("late_100k_worst_return"),
        "late_100k_worst_return_std": std("late_100k_worst_return"),
        "late_100k_retention_mean": mean("late_100k_retention"),
        "late_100k_retention_std": std("late_100k_retention"),
        "eval_return_std_last_mean": mean("eval_return_std_last"),
    }

    for col in [
        "approx_kl_mean",
        "approx_kl_max",
        "clip_fraction_mean",
        "clip_fraction_max",
        "entropy_final",
        "value_loss_max",
    ]:
        summary[f"diag_{col}"] = mean(col)

    return summary


def order_methods(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    order = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    out = df.copy()
    out["_order"] = out["method"].map(lambda method: order.get(method, len(order)))
    sort_cols = ["_order"]
    if "seed" in out.columns:
        sort_cols.append("seed")
    out.sort_values(sort_cols, inplace=True)
    out.drop(columns=["_order"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def metrics_to_markdown(summary_df: pd.DataFrame) -> str:
    columns = [
        ("display_name", "Method"),
        ("n_seeds", "n"),
        ("aulc_200k_mean", "AULC_200k"),
        ("t_475_mean", "t_475"),
        ("solve_rate", "Solve"),
        ("final_eval_return_mean", "Final"),
        ("post_solve_worst_return_mean", "Post-worst"),
        ("late_100k_worst_return_mean", "Late100k-worst"),
    ]
    lines = [
        "| " + " | ".join(header for _, header in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in summary_df.iterrows():
        vals = []
        for key, _ in columns:
            val = row.get(key, "")
            if key == "display_name":
                vals.append(str(val))
            elif key == "n_seeds":
                vals.append(str(int(val)))
            elif key == "solve_rate":
                vals.append("--" if not np.isfinite(float(val)) else f"{float(val) * 100:.0f}%")
            else:
                vals.append("--" if not np.isfinite(float(val)) else f"{float(val):.3f}")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"

