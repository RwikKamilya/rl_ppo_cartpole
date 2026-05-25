"""
compute_metrics.py – Compute comprehensive per-seed and summary metrics for the
PPO CartPole study and save report-ready output files.

Usage:
    python scripts/compute_metrics.py
    python scripts/compute_metrics.py --study-root results/ablation_study/full

Output files (under results/metrics/ — never overwrites results/metrics.csv):
  ppo_per_seed_metrics.csv    – one row per (method, seed)
  ppo_summary_metrics.csv     – one row per method, mean ± std aggregates
  metrics_summary.csv/json    – aliases for final report reproducibility
  ppo_summary_metrics.md      – Markdown table (copy-paste into report notes)
  ppo_latex_table.txt         – LaTeX tabular for the paper

Default PPO data source:
  results/ablation_study/full
    final/ppo_full, final/ppo_tuned
    ablations/ppo_no_clip, ppo_lambda0, ppo_no_adv_norm, ppo_single_epoch

Use --study-root to point at a different study directory explicitly.
Baselines are still read from previous_results/<name>.npz.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from rl_a4.metrics import (
    compute_ppo_per_seed_df,
    compute_npz_per_seed_df,
    aggregate_per_seed_to_summary,
    metrics_to_markdown,
    summary_df_to_latex,
)


# ── Display names ────────────────────────────────────────────────────────────
DISPLAY_NAMES = {
    "PPO_final":        "PPO Final",
    "PPO_full":         "PPO Full",
    "PPO_tuned":        "PPO Tuned",
    "PPO_no_clip":      "PPO No Clip",
    "PPO_lambda0":      "PPO λ=0",
    "PPO_no_adv_norm":  "PPO No Adv. Norm.",
    "PPO_single_epoch": "PPO Single Epoch",
    "DQN":              "DQN",
    "REINFORCE":        "REINFORCE",
    "AC":               "AC",
    "A2C":              "A2C",
}

METHOD_ORDER = [
    "PPO_final", "PPO_full", "PPO_tuned",
    "PPO_no_clip", "PPO_lambda0", "PPO_no_adv_norm", "PPO_single_epoch",
    "DQN", "REINFORCE", "AC", "A2C",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_latest_study_root(studies_dir: Path) -> Optional[Path]:
    """Return the most recently modified study folder, or None if none found."""
    candidates = [
        c for c in studies_dir.iterdir()
        if c.is_dir() and ((c / "ablations").exists() or (c / "final").exists())
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _has_eval_csvs(d: Path) -> bool:
    """True if d contains at least one seed_*/eval.csv."""
    if not d.exists():
        return False
    return any(True for _ in d.glob("seed_*/eval.csv"))


def parse_args():
    parser = argparse.ArgumentParser(description="Compute PPO study metrics")
    parser.add_argument(
        "--study-root",
        type=str,
        default="results/ablation_study/full",
        help="Study directory containing final/ and ablations/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/metrics",
        help="Directory for generated metric files.",
    )
    parser.add_argument(
        "--required-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional exact seed list. Aggregation fails if any listed seed is missing.",
    )
    return parser.parse_args()


def _resolve_study_root(root: Path, study_root_arg: str) -> Path:
    study_root = Path(study_root_arg)
    if not study_root.is_absolute():
        study_root = root / study_root
    if study_root.exists():
        return study_root

    fallback = _find_latest_study_root(root / "results" / "ablation_study")
    if fallback is None:
        raise FileNotFoundError(
            f"Study root not found: {study_root}. No fallback study directory exists."
        )
    print(f"[WARNING] Requested study root not found; falling back to {fallback}")
    return fallback


def _resolve_ppo_dir(study_root: Path, method_key: str) -> Optional[Path]:
    """Resolve a PPO variant strictly within one study root."""
    mapping = {
        "PPO_final": study_root / "final" / "ppo_final",
        "PPO_full": study_root / "final" / "ppo_full",
        "PPO_tuned": study_root / "final" / "ppo_tuned",
        "PPO_no_clip": study_root / "ablations" / "ppo_no_clip",
        "PPO_lambda0": study_root / "ablations" / "ppo_lambda0",
        "PPO_no_adv_norm": study_root / "ablations" / "ppo_no_adv_norm",
        "PPO_single_epoch": study_root / "ablations" / "ppo_single_epoch",
    }
    candidate = mapping[method_key]
    return candidate if _has_eval_csvs(candidate) else None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    study_root = _resolve_study_root(root, args.study_root)
    prev_dir = root / "previous_results"
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _display_path(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    print(f"Using study root: {study_root}")

    per_seed_rows = []   # list of per-seed DataFrames (with 'method' column added)
    summary_rows  = []   # list of summary dicts

    # ── PPO variants ──────────────────────────────────────────────────────────
    ppo_methods = [
        "PPO_final", "PPO_full", "PPO_tuned",
        "PPO_no_clip", "PPO_lambda0", "PPO_no_adv_norm", "PPO_single_epoch",
    ]

    for method_key in ppo_methods:
        vdir = _resolve_ppo_dir(study_root, method_key)
        if vdir is None:
            print(f"  [WARNING] {method_key}: no results found, skipping.")
            continue

        per_seed_df = compute_ppo_per_seed_df(vdir, required_seeds=args.required_seeds)
        if per_seed_df.empty:
            print(f"  [WARNING] {method_key}: empty per-seed DataFrame at {vdir}.")
            continue

        display = DISPLAY_NAMES.get(method_key, method_key)
        per_seed_df.insert(0, "method", method_key)
        per_seed_df.insert(1, "display_name", display)
        per_seed_rows.append(per_seed_df)

        summary = aggregate_per_seed_to_summary(per_seed_df, method_key, display)
        summary_rows.append(summary)

        n     = summary["n_seeds"]
        aulc  = summary.get("aulc_200k_mean", float("nan"))
        t_m   = summary.get("t_475_mean",     float("nan"))
        t_s   = summary.get("t_475_std",      float("nan"))
        solve = summary.get("solve_rate",     float("nan"))
        t_str = f"{int(t_m)}±{int(t_s)}" if np.isfinite(t_m) else "nan"
        print(f"  {display:<22}  n={n}  AULC_200k={aulc:.3f}  "
              f"t_475={t_str}  solve={solve:.0%}  ({vdir.relative_to(root)})")

    # ── Baselines (npz) ───────────────────────────────────────────────────────
    baselines = [
        ("DQN",       prev_dir / "linear_basic_training.npz"),
        ("REINFORCE", prev_dir / "reinforce.npz"),
        ("AC",        prev_dir / "ac.npz"),
        ("A2C",       prev_dir / "a2c.npz"),
    ]

    for method_key, npz_path in baselines:
        if not npz_path.exists():
            print(f"  [WARNING] {method_key}: {npz_path.name} not found, skipping.")
            continue

        per_seed_df = compute_npz_per_seed_df(npz_path, required_seeds=args.required_seeds)
        if per_seed_df.empty:
            print(f"  [WARNING] {method_key}: empty per-seed DataFrame.")
            continue

        display = DISPLAY_NAMES.get(method_key, method_key)
        per_seed_df.insert(0, "method", method_key)
        per_seed_df.insert(1, "display_name", display)
        per_seed_rows.append(per_seed_df)

        summary = aggregate_per_seed_to_summary(per_seed_df, method_key, display)
        summary_rows.append(summary)

        n     = summary["n_seeds"]
        aulc  = summary.get("aulc_200k_mean", float("nan"))
        t_m   = summary.get("t_475_mean",     float("nan"))
        t_s   = summary.get("t_475_std",      float("nan"))
        solve = summary.get("solve_rate",     float("nan"))
        t_str = f"{int(t_m)}±{int(t_s)}" if np.isfinite(t_m) else "nan"
        print(f"  {display:<22}  n={n}  AULC_200k={aulc:.3f}  "
              f"t_475={t_str}  solve={solve:.0%}")

    if not summary_rows:
        print("[ERROR] No results found. Run experiments first.")
        return

    # ── Build DataFrames ──────────────────────────────────────────────────────
    per_seed_all = pd.concat(per_seed_rows, ignore_index=True)
    summary_df   = pd.DataFrame(summary_rows)

    # Reorder to canonical display order
    order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    summary_df["_order"] = summary_df["method"].map(
        lambda m: order_map.get(m, len(METHOD_ORDER))
    )
    summary_df.sort_values("_order", inplace=True)
    summary_df.drop(columns=["_order"], inplace=True)
    summary_df.reset_index(drop=True, inplace=True)

    # Apply same order to per_seed
    per_seed_all["_order"] = per_seed_all["method"].map(
        lambda m: order_map.get(m, len(METHOD_ORDER))
    )
    per_seed_all.sort_values(["_order", "seed"], inplace=True)
    per_seed_all.drop(columns=["_order"], inplace=True)
    per_seed_all.reset_index(drop=True, inplace=True)

    # ── Save outputs ──────────────────────────────────────────────────────────
    per_seed_path = out_dir / "ppo_per_seed_metrics.csv"
    per_seed_all.to_csv(per_seed_path, index=False, float_format="%.6f")
    print(f"\nPer-seed metrics  → {_display_path(per_seed_path)}")

    summary_csv_path = out_dir / "ppo_summary_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False, float_format="%.6f")
    print(f"Summary CSV       → {_display_path(summary_csv_path)}")

    metrics_summary_csv = out_dir / "metrics_summary.csv"
    summary_df.to_csv(metrics_summary_csv, index=False, float_format="%.6f")
    print(f"Summary alias CSV → {_display_path(metrics_summary_csv)}")

    metrics_summary_json = out_dir / "metrics_summary.json"
    metrics_summary_json.write_text(summary_df.to_json(orient="records", indent=2))
    print(f"Summary JSON      → {_display_path(metrics_summary_json)}")

    md_path = out_dir / "ppo_summary_metrics.md"
    md_path.write_text(metrics_to_markdown(summary_df))
    print(f"Markdown table    → {_display_path(md_path)}")

    latex_path = out_dir / "ppo_latex_table.txt"
    latex_path.write_text(summary_df_to_latex(summary_df))
    print(f"LaTeX table       → {_display_path(latex_path)}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    print(metrics_to_markdown(summary_df))

    # Highlight best in class
    if "aulc_200k_mean" in summary_df.columns:
        ppo_mask = summary_df["method"].str.startswith("PPO_")
        if ppo_mask.any():
            best_aulc = summary_df.loc[
                summary_df.loc[ppo_mask, "aulc_200k_mean"].idxmax(), "display_name"
            ]
            print(f"  Best AULC_200k (PPO variants): {best_aulc}")

    if "t_475_mean" in summary_df.columns:
        finite_t = summary_df["t_475_mean"].apply(
            lambda x: float(x) if np.isfinite(float(x) if x == x else float("nan")) else float("nan")
        )
        if finite_t.notna().any():
            best_t = summary_df.loc[finite_t.idxmin(), "display_name"]
            print(f"  Fastest solve (all methods):   {best_t}")

    print("=" * 72)


if __name__ == "__main__":
    main()
