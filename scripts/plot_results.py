"""
plot_results.py – Generate all publication-quality figures from the latest PPO study.

Usage:
    python scripts/plot_results.py

Outputs (in figures/):
    main_comparison.pdf / .png
    ppo_ablation.pdf / .png
    ppo_eval.pdf
    ppo_diagnostics.pdf / .png
    ppo_stability.pdf / .png
    ppo_sweep_summary.pdf / .png

Previous-assignment results are expected at:
    previous_results/linear_basic_training.npz   (DQN)
    previous_results/reinforce.npz
    previous_results/ac.npz
    previous_results/a2c.npz

The script reads PPO results from the most recent staged study under
`results/ablation_study/` and ignores stale top-level PPO folders.
Missing files are skipped with a warning; the script never crashes.
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_a4.plotting import (
    plot_main_comparison,
    plot_ppo_ablation,
    plot_ppo_diagnostics,
    plot_ppo_eval,
    plot_ppo_sweep_summary,
    plot_ppo_stability,
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
    studies_dir = root / "results" / "ablation_study"
    study_root = find_latest_study_root(studies_dir)
    ablations_dir = study_root / "ablations"
    final_dir = study_root / "final"
    prev_results_dir = root / "previous_results"
    figures_dir = root / "figures"

    sweep_dir = study_root / "sweep"

    print(f"Generating figures from latest study: {study_root}")

    print("\n[1/6] Main comparison (DQN / REINFORCE / AC / A2C / best PPO)")
    plot_main_comparison(final_dir, prev_results_dir, figures_dir)

    print("\n[2/6] PPO ablation study")
    plot_ppo_ablation(ablations_dir, figures_dir)

    print("\n[3/6] PPO deterministic evaluation curve")
    plot_ppo_eval(final_dir, figures_dir)

    print("\n[4/6] PPO update diagnostics")
    plot_ppo_diagnostics(ablations_dir, figures_dir)

    print("\n[5/6] PPO stability across seeds")
    plot_ppo_stability(ablations_dir, figures_dir)

    print("\n[6/6] PPO sweep summary")
    plot_ppo_sweep_summary(sweep_dir, figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
