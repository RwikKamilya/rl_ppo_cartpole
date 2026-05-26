"""
Generate final report figures from the clean study layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_a4.plotting import generate_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final PPO study plots")
    parser.add_argument("--study-root", type=str, default="results/final_study")
    parser.add_argument("--output-dir", type=str, default="results/final_study/plots")
    return parser.parse_args()


def _resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    study_root = _resolve(project_root, args.study_root)
    output_dir = _resolve(project_root, args.output_dir)
    previous_results_dir = project_root / "previous_results"

    generate_all_plots(study_root, previous_results_dir, output_dir)
    print(f"Plots written to {output_dir}")


if __name__ == "__main__":
    main()

