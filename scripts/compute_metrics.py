"""
compute_metrics.py – Compute summary metrics for the latest PPO study and save
                     metrics.csv and metrics_latex.tex.

Usage:
    python scripts/compute_metrics.py

Output:
    results/metrics.csv
    results/metrics_latex.tex

Algorithm sources:
    PPO variants:  latest results/ablation_study/<study>/{ablations,final}/...
    Baselines:     previous_results/<name>.npz  (A2/A3 format)

The script ignores stale top-level PPO folders and uses only the latest staged
study outputs. Missing sources are skipped with a warning.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from rl_a4.metrics import (
    compute_ppo_metrics,
    compute_npz_metrics,
    metrics_to_latex,
)


def find_latest_study_root(studies_dir: Path) -> Path:
    candidates = []
    for child in studies_dir.iterdir():
        if not child.is_dir():
            continue
        if (child / "ablations").exists() or (child / "final").exists():
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(
            f"No staged study found under {studies_dir}. "
            "Run `python scripts/run_ablation_study.py --stage all` first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    studies_dir = results_dir / "ablation_study"
    study_root = find_latest_study_root(studies_dir)
    ablations_dir = study_root / "ablations"
    final_dir = study_root / "final"
    prev_dir = root / "previous_results"

    rows = []
    print(f"Using latest study: {study_root}")

    # ── PPO variants ────────────────────────────────────────────────────────
    ppo_variants = [
        ("PPO_full", final_dir / "ppo_full"),
        ("PPO_tuned", final_dir / "ppo_tuned"),
        ("PPO_no_clip", ablations_dir / "ppo_no_clip"),
        ("PPO_lambda0", ablations_dir / "ppo_lambda0"),
        ("PPO_no_adv_norm", ablations_dir / "ppo_no_adv_norm"),
        ("PPO_single_epoch", ablations_dir / "ppo_single_epoch"),
    ]
    for name, vdir in ppo_variants:
        if not vdir.exists() or not list(vdir.glob("seed_*/eval.csv")):
            print(f"  [WARNING] {name}: no results found at {vdir}, skipping.")
            continue
        m = compute_ppo_metrics(vdir)
        m["algorithm"] = name
        rows.append(m)
        print(f"  {name}: AULC_200k={m['AULC_200k']:.3f}  "
              f"t_475={m['t_475_mean']:.0f}±{m['t_475_std']:.0f}  "
              f"solve={m['solve_rate']:.1%}")

    # ── Previous baselines (npz) ────────────────────────────────────────────
    baselines = [
        ("A2C",      prev_dir / "a2c.npz"),
        ("REINFORCE", prev_dir / "reinforce.npz"),
        ("AC",       prev_dir / "ac.npz"),
        ("DQN",      prev_dir / "linear_basic_training.npz"),
    ]
    for name, npz_path in baselines:
        if not npz_path.exists():
            print(f"  [WARNING] {name}: {npz_path} not found, skipping.")
            continue
        m = compute_npz_metrics(npz_path)
        m["algorithm"] = name
        rows.append(m)
        print(f"  {name}: AULC_200k={m['AULC_200k']:.3f}  "
              f"t_475={m['t_475_mean']:.0f}±{m['t_475_std']:.0f}  "
              f"solve={m['solve_rate']:.1%}")

    if not rows:
        print("[ERROR] No results found. Run experiments first.")
        return

    # ── Build DataFrame ─────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Re-order columns for readability
    col_order = [
        "algorithm", "n_seeds",
        "final_eval_return_mean", "final_eval_return_std",
        "final_train_return_mean", "final_train_return_std",
        "AULC_1M", "AULC_200k",
        "t_475_mean", "t_475_std",
        "solve_rate", "post_solve_min_return",
    ]
    df = df.reindex(columns=[c for c in col_order if c in df.columns])

    # Save CSV
    csv_path = results_dir / "metrics.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\nMetrics saved → {csv_path}")

    # Save LaTeX
    tex_path = results_dir / "metrics_latex.tex"
    tex_str = metrics_to_latex(df)
    tex_path.write_text(tex_str)
    print(f"LaTeX table  → {tex_path}")

    # ── Report summary ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    ppo_row = df[df["algorithm"] == "PPO_full"]
    tuned_row = df[df["algorithm"] == "PPO_tuned"]
    if not ppo_row.empty:
        r = ppo_row.iloc[0]
        t_mean = r.get("t_475_mean", float("nan"))
        t_std  = r.get("t_475_std",  float("nan"))
        solve  = r.get("solve_rate", float("nan"))
        n      = int(r.get("n_seeds", 0))
        print(f"\nPPO_full:")
        print(f"  t_475        = {t_mean:.0f} ± {t_std:.0f} steps")
        print(f"  solve_rate   = {solve:.1%}  ({int(round(solve*n))}/{n} seeds)")
        print(f"  AULC_200k    = {r.get('AULC_200k', float('nan')):.3f}")
        print(f"  final_eval_R = {r.get('final_eval_return_mean', float('nan')):.1f} "
              f"± {r.get('final_eval_return_std', float('nan')):.1f}")
        print()
        print(f"  Sentence: \"PPO reaches the 475 threshold after "
              f"{t_mean:.0f}±{t_std:.0f} steps and solves CartPole "
              f"in {int(round(solve*n))}/{n} seeds.\"")

    if not tuned_row.empty:
        r = tuned_row.iloc[0]
        print(f"\nPPO_tuned:")
        print(f"  t_475        = {r.get('t_475_mean', float('nan')):.0f} ± "
              f"{r.get('t_475_std', float('nan')):.0f} steps")
        print(f"  solve_rate   = {r.get('solve_rate', float('nan')):.1%}")
        print(f"  AULC_200k    = {r.get('AULC_200k', float('nan')):.3f}")
        print(f"  final_eval_R = {r.get('final_eval_return_mean', float('nan')):.1f} "
              f"± {r.get('final_eval_return_std', float('nan')):.1f}")

    # Best by AULC_200k
    if "AULC_200k" in df.columns:
        best_early = df.loc[df["AULC_200k"].idxmax(), "algorithm"]
        print(f"\n  Best AULC_200k (early learning):  {best_early}")

    if "final_eval_return_mean" in df.columns:
        best_final = df.loc[df["final_eval_return_mean"].idxmax(), "algorithm"]
        print(f"  Best final_eval_return:           {best_final}")

    # PPO vs A2C comparison
    a2c_row = df[df["algorithm"] == "A2C"]
    if not ppo_row.empty and not a2c_row.empty:
        ppo_aulc = ppo_row.iloc[0].get("AULC_200k", float("nan"))
        a2c_aulc = a2c_row.iloc[0].get("AULC_200k", float("nan"))
        if ppo_aulc > a2c_aulc:
            direction = "faster"
        elif ppo_aulc < a2c_aulc:
            direction = "slower"
        else:
            direction = "similar"
        print(f"\n  PPO vs A2C early learning (AULC_200k): "
              f"PPO={ppo_aulc:.3f}  A2C={a2c_aulc:.3f} → PPO is {direction}")

    print("="*60)


if __name__ == "__main__":
    main()
