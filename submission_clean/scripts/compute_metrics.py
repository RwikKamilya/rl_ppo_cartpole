"""
Compute final PPO CartPole-v1 metrics from the clean study layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from rl_a4.metrics import (
    BASELINE_METHODS,
    METHOD_ORDER,
    PPO_ABLATION_METHODS,
    PPO_FINAL_METHODS,
    PPO_SELECTED_METHODS,
    aggregate_per_seed_to_summary,
    compute_npz_per_seed_df,
    compute_ppo_per_seed_df,
    metrics_to_markdown,
    order_methods,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute final PPO study metrics")
    parser.add_argument("--study-root", type=str, default="results/final_study")
    parser.add_argument("--output-dir", type=str, default="results/final_study/metrics")
    parser.add_argument("--required-seeds", type=int, nargs="+", default=None)
    return parser.parse_args()


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _write_outputs(per_seed_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_df.to_csv(output_dir / "per_seed_metrics.csv", index=False, float_format="%.6f")
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False, float_format="%.6f")
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False, float_format="%.6f")
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary_df.to_dict(orient="records"), indent=2)
    )
    (output_dir / "metrics_summary.md").write_text(metrics_to_markdown(summary_df))


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    study_root = _resolve(project_root, args.study_root)
    output_dir = _resolve(project_root, args.output_dir)
    previous_results_dir = project_root / "previous_results"

    per_seed_frames = []
    summary_rows = []

    for method, (display, rel_path) in PPO_FINAL_METHODS.items():
        variant_dir = study_root / rel_path
        df = compute_ppo_per_seed_df(variant_dir, method, display, args.required_seeds)
        if df.empty:
            raise FileNotFoundError(f"No PPO final eval logs found in {variant_dir}")
        per_seed_frames.append(df)
        summary_rows.append(aggregate_per_seed_to_summary(df))
        print(f"{display}: loaded {len(df)} seeds from {variant_dir}")

    for method, (display, rel_path) in PPO_SELECTED_METHODS.items():
        variant_dir = study_root / rel_path
        if not variant_dir.exists():
            print(f"[skip] {display}: {variant_dir} not found")
            continue
        df = compute_ppo_per_seed_df(variant_dir, method, display, args.required_seeds)
        if df.empty:
            print(f"[skip] {display}: no eval logs in {variant_dir}")
            continue
        per_seed_frames.append(df)
        summary_rows.append(aggregate_per_seed_to_summary(df))
        print(f"{display}: loaded {len(df)} seeds from {variant_dir}")

    for method, (display, rel_path) in PPO_ABLATION_METHODS.items():
        variant_dir = study_root / rel_path
        if not variant_dir.exists():
            print(f"[skip] {display}: {variant_dir} not found")
            continue
        df = compute_ppo_per_seed_df(variant_dir, method, display, args.required_seeds)
        if df.empty:
            print(f"[skip] {display}: no eval logs in {variant_dir}")
            continue
        per_seed_frames.append(df)
        summary_rows.append(aggregate_per_seed_to_summary(df))
        print(f"{display}: loaded {len(df)} seeds from {variant_dir}")

    for method, (display, filename) in BASELINE_METHODS.items():
        df = compute_npz_per_seed_df(previous_results_dir / filename, method, display, args.required_seeds)
        if df.empty:
            raise FileNotFoundError(f"No baseline rows loaded from {filename}")
        per_seed_frames.append(df)
        summary_rows.append(aggregate_per_seed_to_summary(df))
        print(f"{display}: loaded {len(df)} saved baseline seeds")

    per_seed_df = order_methods(pd.concat(per_seed_frames, ignore_index=True))
    summary_df = order_methods(pd.DataFrame(summary_rows))
    summary_df["method"] = pd.Categorical(summary_df["method"], METHOD_ORDER, ordered=True)
    summary_df.sort_values("method", inplace=True)
    summary_df["method"] = summary_df["method"].astype(str)
    summary_df.reset_index(drop=True, inplace=True)

    _write_outputs(per_seed_df, summary_df, output_dir)
    print(f"Metrics written to {output_dir}")
    print(metrics_to_markdown(summary_df))


if __name__ == "__main__":
    main()

