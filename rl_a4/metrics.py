"""
metrics.py – Compute summary metrics from PPO and baseline result CSVs/npz files.

Public API (backward-compatible):
  compute_ppo_metrics(results_dir)       → dict  (original, unchanged)
  compute_npz_metrics(npz_path)          → dict  (original, unchanged)
  metrics_to_latex(df)                   → str   (original, unchanged)

Extended API (new):
  compute_ppo_per_seed_df(results_dir)   → DataFrame, one row per seed
  compute_npz_per_seed_df(npz_path)      → DataFrame, one row per seed
  aggregate_per_seed_to_summary(df, ...)→ dict, one-row summary
  metrics_to_markdown(summary_df)        → str, Markdown table
  summary_df_to_latex(summary_df)        → str, LaTeX tabular

Per-seed columns returned by compute_ppo_per_seed_df / compute_npz_per_seed_df:
  seed, aulc_200k, t_475, solved, final_eval_return,
  post_solve_worst_return, post_solve_retention_rate, eval_return_std_last,
  approx_kl_mean, approx_kl_max, clip_fraction_mean, clip_fraction_max,
  entropy_mean, entropy_final, value_loss_mean, value_loss_max.

t_475 uses consec=1 (first single checkpoint where mean_return >= 475).
AULC is normalised so that constant return=500 over 200k steps gives 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────────

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
            seed for seed in required_seeds
            if seed not in seed_dirs or not (seed_dirs[seed] / filename).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {filename} for seeds {missing} in {results_dir}. "
                f"Expected seed directories: {[f'seed_{s}' for s in required_seeds]}"
            )

    items = []
    for seed, seed_dir in sorted(seed_dirs.items()):
        csv_path = seed_dir / filename
        if csv_path.exists():
            items.append((seed, pd.read_csv(csv_path)))
    return items


def _load_ppo_eval_seed_items(
    results_dir: Path,
    required_seeds: Optional[Sequence[int]] = None,
) -> List[Tuple[int, pd.DataFrame]]:
    """Load eval CSVs with their seed ids for one PPO variant."""
    return _load_ppo_seed_csv_items(results_dir, "eval.csv", required_seeds)


def _load_ppo_eval_seeds(results_dir: Path) -> List[pd.DataFrame]:
    """Load eval CSVs for all seeds of one PPO variant."""
    return [df for _, df in _load_ppo_eval_seed_items(results_dir)]


def _load_ppo_train_seeds(results_dir: Path) -> List[pd.DataFrame]:
    """Load train CSVs for all seeds of one PPO variant."""
    return [df for _, df in _load_ppo_seed_csv_items(results_dir, "train.csv")]


def _load_ppo_update_seed_items(results_dir: Path) -> List[Tuple[int, pd.DataFrame]]:
    """Load update_log CSVs with their seed ids for one PPO variant."""
    return _load_ppo_seed_csv_items(results_dir, "update_log.csv")


def _load_ppo_update_seeds(results_dir: Path) -> List[pd.DataFrame]:
    """Load update_log CSVs for all seeds of one PPO variant."""
    return [df for _, df in _load_ppo_update_seed_items(results_dir)]


def _aulc(steps: np.ndarray, returns: np.ndarray,
          max_step: float, norm: float = 500.0) -> float:
    """Area under learning curve up to max_step, normalised by norm*max_step.

    If the first checkpoint is not at step 0, a virtual (0, 0) point is
    prepended, giving a conservative (lower-bound) AULC for early performance.
    """
    mask = steps <= max_step
    s = steps[mask]
    r = returns[mask]
    if len(s) < 2:
        return float("nan")
    if s[0] > 0:
        s = np.concatenate([[0.0], s])
        r = np.concatenate([[0.0], r])
    area = float(np.trapz(r, s))
    return area / (norm * max_step)


def _t_475_per_seed(steps: np.ndarray, means: np.ndarray,
                    threshold: float = 475.0, consec: int = 3) -> float:
    """Return first step at which eval_mean >= threshold for `consec` consecutive
    checkpoints.  Returns NaN if threshold is never reached."""
    above = means >= threshold
    for i in range(len(above) - consec + 1):
        if above[i: i + consec].all():
            return float(steps[i])
    return float("nan")


def _retention_rate_per_seed(steps: np.ndarray, means: np.ndarray,
                              t475: float, threshold: float = 475.0) -> float:
    """Fraction of eval checkpoints at or after t475 where mean_return >= threshold.

    Returns 0.0 if t475 is NaN (never solved), so the metric is always finite
    and directly interpretable as a solve-retention probability.
    """
    if not np.isfinite(t475):
        return 0.0
    post = means[steps >= t475]
    if len(post) == 0:
        return 0.0
    return float((post >= threshold).sum() / len(post))


# ──────────────────────────────────────────────────────────────────────────────
# Per-seed DataFrames (new extended API)
# ──────────────────────────────────────────────────────────────────────────────

def compute_ppo_per_seed_df(
    results_dir: Path,
    required_seeds: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Compute per-seed metrics for one PPO variant.

    Returns a DataFrame with one row per seed.

    t_475 uses consec=1: first single eval checkpoint where mean_return >= 475,
    per the spec definition of "first environment step where return >= 475."
    AULC is normalised so that a constant return of 500 over 200k steps = 1.0.
    If the first eval checkpoint is not at step 0, a virtual (0, 0) start is
    inserted (conservative lower bound).
    """
    eval_items = _load_ppo_eval_seed_items(results_dir, required_seeds=required_seeds)
    update_by_seed = dict(_load_ppo_update_seed_items(results_dir))

    diag_spec = [
        # (csv_col, mean_out_key, second_out_key, second_is_final)
        ("approx_kl",     "approx_kl_mean",      "approx_kl_max",       False),
        ("clip_fraction", "clip_fraction_mean",   "clip_fraction_max",   False),
        ("entropy",       "entropy_mean",         "entropy_final",       True),
        ("value_loss",    "value_loss_mean",       "value_loss_max",      False),
    ]

    rows: List[Dict] = []
    for seed, df in eval_items:
        steps = df["step"].values.astype(float)
        means = df["mean_return"].values.astype(float)

        aulc_200k = _aulc(steps, means, 2e5)

        assert np.all(np.diff(steps) > 0), f"Non-increasing eval steps in {results_dir}"

        # consec=1: first checkpoint where mean_return >= 475
        t475 = _t_475_per_seed(steps, means, consec=1)
        solved = bool(np.isfinite(t475))

        # Final eval: last raw deterministic evaluation checkpoint
        final_eval = float(means[-1])

        if solved:
            post_mask = steps >= t475
            post_vals = means[post_mask]
            post_worst = float(post_vals.min()) if len(post_vals) > 0 else float("nan")
            retention  = _retention_rate_per_seed(steps, means, t475)
            post_window_len = int(len(post_vals))
        else:
            post_worst = float("nan")
            retention  = 0.0
            post_window_len = 0

        assert np.isfinite(aulc_200k) and 0.0 <= aulc_200k <= 1.05, (
            f"AULC_200k out of range in {results_dir}: {aulc_200k}"
        )
        assert 0.0 <= final_eval <= 500.0, (
            f"final_eval_return out of range in {results_dir}: {final_eval}"
        )
        if np.isfinite(post_worst):
            assert 0.0 <= post_worst <= 500.0, (
                f"post_solve_worst_return out of range in {results_dir}: {post_worst}"
            )
        assert 0.0 <= retention <= 1.0, (
            f"post_solve_retention_rate out of range in {results_dir}: {retention}"
        )

        # std_return at last eval checkpoint (logged in eval.csv)
        eval_std_last = (
            float(df["std_return"].values[-1])
            if "std_return" in df.columns and len(df) > 0
            else float("nan")
        )

        row: Dict = {
            "seed": seed,
            "aulc_200k":                 aulc_200k,
            "t_475":                     t475,
            "solved":                    solved,
            "final_eval_return":         final_eval,
            "post_solve_worst_return":   post_worst,
            "post_solve_retention_rate": retention,
            "post_solve_window_len":     post_window_len,
            "eval_return_std_last":      eval_std_last,
        }

        # Diagnostics from update_log
        if seed in update_by_seed:
            udf = update_by_seed[seed]
            for csv_col, mean_key, second_key, second_is_final in diag_spec:
                if csv_col in udf.columns:
                    vals = udf[csv_col].dropna().values.astype(float)
                    row[mean_key]   = float(np.nanmean(vals))
                    row[second_key] = float(vals[-1]) if second_is_final else float(np.nanmax(vals))
                else:
                    row[mean_key]   = float("nan")
                    row[second_key] = float("nan")
        else:
            for _, mean_key, second_key, _ in diag_spec:
                row[mean_key]   = float("nan")
                row[second_key] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_npz_per_seed_df(
    npz_path: Path,
    required_seeds: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Compute per-seed metrics from a previous-assignment .npz baseline.

    These files contain training-return traces only (no separate deterministic
    evaluation). AULC is computed on the raw training-return trace. t_475 and
    the post-solve metrics are computed on a smoothed training curve
    (MA window = 25, consistent with the existing plotting code). The
    final_eval_return field is the mean of the last 100 raw training episodes,
    so this contextual baseline is not directly protocol-matched to PPO eval.csv.
    All diagnostic columns (KL, clip fraction, entropy, value loss) are NaN.
    eval_return_std_last is NaN (no per-episode std is logged).
    """
    data = np.load(npz_path, allow_pickle=True)
    all_rewards = data["rewards"]
    all_steps   = data["steps"]
    n_seeds     = len(all_rewards)
    if required_seeds is not None and n_seeds != len(required_seeds):
        raise ValueError(
            f"{npz_path} contains {n_seeds} seeds, expected {len(required_seeds)} "
            f"seeds {list(required_seeds)}."
        )

    SMOOTH    = 25
    THRESHOLD = 475.0
    NAN_DIAG  = {
        "approx_kl_mean": float("nan"), "approx_kl_max":      float("nan"),
        "clip_fraction_mean": float("nan"), "clip_fraction_max": float("nan"),
        "entropy_mean":   float("nan"), "entropy_final":       float("nan"),
        "value_loss_mean":float("nan"), "value_loss_max":      float("nan"),
    }

    rows: List[Dict] = []
    for i in range(n_seeds):
        seed = int(required_seeds[i]) if required_seeds is not None else i
        rewards = np.array(all_rewards[i], dtype=np.float32)
        steps   = np.array(all_steps[i],   dtype=np.float32)

        aulc_200k = _aulc(steps, rewards, 2e5)

        if len(rewards) >= SMOOTH:
            smoothed = np.convolve(rewards, np.ones(SMOOTH) / SMOOTH, mode="valid")
            s_steps  = steps[SMOOTH - 1:]
        else:
            smoothed = rewards
            s_steps  = steps

        t475   = _t_475_per_seed(s_steps, smoothed, consec=1)
        solved = bool(np.isfinite(t475))

        # final_eval: last 100 raw training episodes
        final_eval = float(rewards[-100:].mean())

        if solved:
            post_mask  = s_steps >= t475
            post_vals  = smoothed[post_mask]
            post_worst = float(post_vals.min()) if len(post_vals) > 0 else float("nan")
            retention  = _retention_rate_per_seed(s_steps, smoothed, t475)
        else:
            post_worst = float("nan")
            retention  = 0.0

        row = {
            "seed":                      seed,
            "aulc_200k":                 aulc_200k,
            "t_475":                     t475,
            "solved":                    solved,
            "final_eval_return":         final_eval,
            "post_solve_worst_return":   post_worst,
            "post_solve_retention_rate": retention,
            "eval_return_std_last":      float("nan"),
            **NAN_DIAG,
        }
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_per_seed_to_summary(df: pd.DataFrame,
                                   method_key: str,
                                   display_name: str) -> Dict:
    """Aggregate a per-seed DataFrame to a one-row summary dict.

    NaN per-seed values are excluded from mean/std but counted in n_seeds.
    solve_rate = fraction of seeds where solved == True.
    t_475_std uses ddof=0 (population std, consistent with reporting convention).
    """
    if df.empty:
        return {"method": method_key, "display_name": display_name, "n_seeds": 0}

    n = len(df)

    def _mean(col: str) -> float:
        if col not in df.columns:
            return float("nan")
        v = pd.to_numeric(df[col], errors="coerce").dropna().values
        return float(np.mean(v)) if len(v) > 0 else float("nan")

    def _std(col: str) -> float:
        if col not in df.columns:
            return float("nan")
        v = pd.to_numeric(df[col], errors="coerce").dropna().values
        return float(np.std(v, ddof=0)) if len(v) > 1 else 0.0

    finite_t = pd.to_numeric(df["t_475"], errors="coerce").dropna().values

    summary: Dict = {
        "method":       method_key,
        "display_name": display_name,
        "n_seeds":      n,
        # Sample efficiency
        "aulc_200k_mean": _mean("aulc_200k"),
        "aulc_200k_std":  _std("aulc_200k"),
        "t_475_mean":     float(np.mean(finite_t))          if len(finite_t) > 0 else float("nan"),
        "t_475_std":      float(np.std(finite_t, ddof=0))   if len(finite_t) > 1 else 0.0,
        "solve_rate":     float(df["solved"].sum() / n),
        # Final performance
        "final_eval_return_mean": _mean("final_eval_return"),
        "final_eval_return_std":  _std("final_eval_return"),
        # Stability
        "post_solve_worst_return_mean": _mean("post_solve_worst_return"),
        "post_solve_worst_return_std":  _std("post_solve_worst_return"),
        "post_solve_retention_rate_mean": _mean("post_solve_retention_rate"),
        "post_solve_retention_rate_std":  _std("post_solve_retention_rate"),
        # Eval noise at final checkpoint
        "eval_return_std_last_mean": _mean("eval_return_std_last"),
    }

    # Diagnostic aggregates (mean across seeds of per-seed means/maxes)
    for col in ["approx_kl_mean", "approx_kl_max",
                "clip_fraction_mean", "clip_fraction_max",
                "entropy_mean", "entropy_final",
                "value_loss_mean", "value_loss_max"]:
        summary[f"diag_{col}"] = _mean(col)

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Table formatters (extended API)
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_nan(val, fmt: str) -> str:
    """Format val with fmt; return '--' if NaN or inf."""
    try:
        f = float(val)
        if not np.isfinite(f):
            return "--"
        return fmt % f
    except (TypeError, ValueError):
        return "--"


def metrics_to_markdown(summary_df: pd.DataFrame) -> str:
    """Format a summary metrics DataFrame as a Markdown table.

    Columns: Method | AULC_200k ↑ | t_475 ↓ | Solve rate ↑ | Final eval ↑ |
             Post-solve worst ↑ | Retention ↑
    """
    header = (
        "| Method | AULC_200k ↑ | t_475 ↓ | Solve rate ↑ | "
        "Final eval ↑ | Post-solve worst ↑ | Retention ↑ |"
    )
    sep = (
        "|--------|------------|---------|-------------|"
        "-------------|-------------------|------------|"
    )
    lines = [header, sep]

    for _, r in summary_df.iterrows():
        name  = str(r.get("display_name", r.get("method", "?")))
        aulc  = _fmt_nan(r.get("aulc_200k_mean"), "%.3f")

        t_m = r.get("t_475_mean", float("nan"))
        t_s = r.get("t_475_std",  float("nan"))
        if np.isfinite(float(t_m)) if isinstance(t_m, (int, float)) else False:
            t475 = f"{int(round(float(t_m)))} ± {int(round(float(t_s)))}"
        else:
            t475 = "--"

        sr  = r.get("solve_rate", float("nan"))
        solve = f"{float(sr)*100:.0f}%" if np.isfinite(float(sr)) else "--"

        fe_m = r.get("final_eval_return_mean", float("nan"))
        fe_s = r.get("final_eval_return_std",  float("nan"))
        final = (
            f"{float(fe_m):.1f} ± {float(fe_s):.1f}"
            if np.isfinite(float(fe_m)) else "--"
        )

        pw_m = r.get("post_solve_worst_return_mean", float("nan"))
        pw_s = r.get("post_solve_worst_return_std",  float("nan"))
        worst = (
            f"{float(pw_m):.1f} ± {float(pw_s):.1f}"
            if np.isfinite(float(pw_m)) else "--"
        )

        rr = r.get("post_solve_retention_rate_mean", float("nan"))
        ret = f"{float(rr)*100:.0f}%" if np.isfinite(float(rr)) else "--"

        lines.append(
            f"| {name} | {aulc} | {t475} | {solve} | {final} | {worst} | {ret} |"
        )

    return "\n".join(lines) + "\n"


def summary_df_to_latex(summary_df: pd.DataFrame) -> str:
    """Convert a summary metrics DataFrame to a LaTeX table string.

    Produces the report-ready table:
      Method | AULC200k | t475 | Solve | Final eval | Post-solve worst | Retention
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Performance on CartPole-v1 under a $10^6$ environment-step budget."
        r" $\uparrow$ higher is better; $\downarrow$ lower is better."
        r" $t_{475}$: mean $\pm$ std of first solve step across seeds (unsolved seeds excluded)."
        r" Post-solve worst: minimum deterministic evaluation return after first reaching 475."
        r" Retention: fraction of post-solve checkpoints $\geq 475$.}",
        r"\label{tab:ppo_metrics}",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        (r"Method & $n$ & AULC$_{200k}$ $\uparrow$ & $t_{475}$ $\downarrow$ "
         r"& Solve $\uparrow$ & Final eval $\uparrow$ & Post-worst $\uparrow$ "
         r"& Retention $\uparrow$ \\"),
        r"\midrule",
    ]

    for _, r in summary_df.iterrows():
        name = str(r.get("display_name", r.get("method", "?")))
        n    = int(r.get("n_seeds", 0))

        aulc = _fmt_nan(r.get("aulc_200k_mean"), "%.3f")

        t_m = r.get("t_475_mean", float("nan"))
        t_s = r.get("t_475_std",  float("nan"))
        try:
            t475 = (f"{int(round(float(t_m)))} $\\pm$ {int(round(float(t_s)))}"
                    if np.isfinite(float(t_m)) else "--")
        except (TypeError, ValueError):
            t475 = "--"

        sr   = r.get("solve_rate", float("nan"))
        solve = _fmt_nan(sr, "%.0f%%") if not _fmt_nan(sr, "%.0f%%").startswith("-") else "--"
        try:
            solve = f"{float(sr)*100:.0f}\\%" if np.isfinite(float(sr)) else "--"
        except (TypeError, ValueError):
            solve = "--"

        fe_m = r.get("final_eval_return_mean", float("nan"))
        fe_s = r.get("final_eval_return_std",  float("nan"))
        try:
            final = (f"{float(fe_m):.1f} $\\pm$ {float(fe_s):.1f}"
                     if np.isfinite(float(fe_m)) else "--")
        except (TypeError, ValueError):
            final = "--"

        pw_m = r.get("post_solve_worst_return_mean", float("nan"))
        pw_s = r.get("post_solve_worst_return_std",  float("nan"))
        try:
            worst = (f"{float(pw_m):.1f} $\\pm$ {float(pw_s):.1f}"
                     if np.isfinite(float(pw_m)) else "--")
        except (TypeError, ValueError):
            worst = "--"

        rr = r.get("post_solve_retention_rate_mean", float("nan"))
        try:
            ret = f"{float(rr)*100:.0f}\\%" if np.isfinite(float(rr)) else "--"
        except (TypeError, ValueError):
            ret = "--"

        lines.append(
            f"{name} & {n} & {aulc} & {t475} & {solve} & {final} & {worst} & {ret} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Original API – preserved unchanged for backward compatibility
# ──────────────────────────────────────────────────────────────────────────────

def compute_ppo_metrics(results_dir: Path) -> Dict:
    """Compute all metrics for one PPO variant (multiple seed sub-dirs).

    NOTE: This original function uses consec=3 for t_475 and returns a flat
    dict.  New code should use compute_ppo_per_seed_df + aggregate_per_seed_to_summary
    instead, which uses consec=1 and exposes per-seed data.
    """
    eval_dfs  = _load_ppo_eval_seeds(results_dir)
    train_dfs = _load_ppo_train_seeds(results_dir)

    if not eval_dfs:
        return {}

    final_evals = [df["mean_return"].values[-5:].mean() for df in eval_dfs]
    final_eval_mean = float(np.mean(final_evals))
    final_eval_std  = float(np.std(final_evals))

    final_trains = []
    for df in train_dfs:
        final_trains.append(df["return"].values[-100:].mean())
    final_train_mean = float(np.mean(final_trains)) if final_trains else float("nan")
    final_train_std  = float(np.std(final_trains))  if final_trains else float("nan")

    aulc_1m_list, aulc_200k_list = [], []
    for df in eval_dfs:
        s = df["step"].values.astype(float)
        m = df["mean_return"].values.astype(float)
        aulc_1m_list.append(_aulc(s, m, 1e6))
        aulc_200k_list.append(_aulc(s, m, 2e5))

    t475_list, post_min_returns = [], []
    for df in eval_dfs:
        s = df["step"].values.astype(float)
        m = df["mean_return"].values.astype(float)
        t = _t_475_per_seed(s, m, consec=3)
        t475_list.append(t)
        if np.isfinite(t):
            post = m[s >= t]
            if len(post) > 0:
                post_min_returns.append(float(post.min()))

    finite_t = [t for t in t475_list if np.isfinite(t)]

    return {
        "final_eval_return_mean": final_eval_mean,
        "final_eval_return_std":  final_eval_std,
        "final_train_return_mean": final_train_mean,
        "final_train_return_std":  final_train_std,
        "AULC_1M":    float(np.nanmean(aulc_1m_list)),
        "AULC_200k":  float(np.nanmean(aulc_200k_list)),
        "t_475_mean": float(np.mean(finite_t))         if finite_t else float("nan"),
        "t_475_std":  float(np.std(finite_t))          if finite_t else float("nan"),
        "solve_rate": len(finite_t) / len(t475_list),
        "post_solve_min_return": float(np.mean(post_min_returns)) if post_min_returns else float("nan"),
        "n_seeds":    len(eval_dfs),
    }


def compute_npz_metrics(npz_path: Path) -> Dict:
    """Compute metrics from a previous-assignment .npz file (original API).

    NOTE: New code should use compute_npz_per_seed_df + aggregate_per_seed_to_summary.
    """
    data = np.load(npz_path, allow_pickle=True)
    all_rewards = data["rewards"]
    all_steps   = data["steps"]
    n_seeds = len(all_rewards)

    final_trains, aulc_1m_list, aulc_200k_list, t475_list, post_min_returns = \
        [], [], [], [], []

    for i in range(n_seeds):
        rewards = np.array(all_rewards[i], dtype=np.float32)
        steps   = np.array(all_steps[i],   dtype=np.float32)

        final_trains.append(rewards[-100:].mean())
        aulc_1m_list.append(_aulc(steps, rewards, 1e6))
        aulc_200k_list.append(_aulc(steps, rewards, 2e5))

        window = 25
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            s = steps[window - 1:]
            t = _t_475_per_seed(s, smoothed, consec=3)
        else:
            t = float("nan")
        t475_list.append(t)
        if np.isfinite(t):
            post = rewards[steps >= t]
            if len(post) > 0:
                post_min_returns.append(float(post.min()))

    finite_t = [t for t in t475_list if np.isfinite(t)]
    return {
        "final_eval_return_mean":  float(np.mean(final_trains)),
        "final_eval_return_std":   float(np.std(final_trains)),
        "final_train_return_mean": float(np.mean(final_trains)),
        "final_train_return_std":  float(np.std(final_trains)),
        "AULC_1M":    float(np.nanmean(aulc_1m_list)),
        "AULC_200k":  float(np.nanmean(aulc_200k_list)),
        "t_475_mean": float(np.mean(finite_t)) if finite_t else float("nan"),
        "t_475_std":  float(np.std(finite_t))  if finite_t else float("nan"),
        "solve_rate": len(finite_t) / n_seeds,
        "post_solve_min_return": float(np.mean(post_min_returns)) if post_min_returns else float("nan"),
        "n_seeds":    n_seeds,
    }


def metrics_to_latex(df: pd.DataFrame) -> str:
    """Convert a metrics DataFrame to a LaTeX table string (original API)."""
    cols_display = {
        "algorithm":              "Algorithm",
        "n_seeds":                "Seeds",
        "final_eval_return_mean": r"Final Eval $\bar{R}$",
        "final_eval_return_std":  r"$\sigma$",
        "AULC_1M":                "AULC$_{1M}$",
        "AULC_200k":              "AULC$_{200k}$",
        "t_475_mean":             r"$t_{475}$ (steps)",
        "solve_rate":             "Solve Rate",
    }
    sub = df[[c for c in cols_display if c in df.columns]].copy()
    sub.rename(columns=cols_display, inplace=True)

    return "\n".join([
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Performance metrics on CartPole-v1.}",
        r"\label{tab:metrics}",
        r"\small",
        sub.to_latex(index=False, float_format="%.3f", escape=False),
        r"\end{table}",
    ])
