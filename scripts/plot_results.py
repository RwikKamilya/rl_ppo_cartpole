"""
plot_results.py – Generate all publication-quality figures.

Usage:
    python scripts/plot_results.py

Outputs (in figures/):
    main_comparison.pdf / .png
    ppo_ablation.pdf / .png
    ppo_eval.pdf

Previous-assignment results are expected at:
    previous_results/linear_basic_training.npz   (DQN)
    previous_results/reinforce.npz
    previous_results/ac.npz
    previous_results/a2c.npz

Missing files are skipped with a warning; the script never crashes.
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_a4.plotting import (
    plot_main_comparison,
    plot_ppo_ablation,
    plot_ppo_eval,
)


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    prev_results_dir = root / "previous_results"
    figures_dir = root / "figures"

    print("Generating figures...")

    print("\n[1/3] Main comparison (DQN / REINFORCE / AC / A2C / PPO)")
    plot_main_comparison(results_dir, prev_results_dir, figures_dir)

    print("\n[2/3] PPO ablation study")
    plot_ppo_ablation(results_dir, figures_dir)

    print("\n[3/3] PPO deterministic evaluation curve")
    plot_ppo_eval(results_dir, figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
