"""
plot_results.py – Generate final report-ready figures from one PPO study root.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --study-root results/ablation_study/full

Outputs (under results/plots/ — does NOT touch the old figures/ folder):
    main_comparison.png / .pdf
    final_learning_curves_comparison.png / .pdf
    ppo_ablation.png / .pdf
    ppo_stability.png / .pdf
    ppo_diagnostics.png / .pdf
    ppo_pareto.png / .pdf
    figure_captions.txt

Previous-assignment baselines:
    previous_results/linear_basic_training.npz  (DQN)
    previous_results/reinforce.npz
    previous_results/ac.npz
    previous_results/a2c.npz
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_a4.plotting import (
    plot_main_comparison,
    plot_ppo_ablation,
    plot_ppo_diagnostics,
    plot_ppo_eval,
    plot_ppo_pareto,
    plot_ppo_stability,
    plot_ppo_sweep_summary,
)


# ── Captions ──────────────────────────────────────────────────────────────────
CAPTIONS = {
    "main_comparison": (
        "Contextual comparison between PPO and previous Assignment 2/3 agents "
        "on CartPole-v1. Lines show mean return over independent seeds and shaded "
        "regions show ±1 standard deviation. Curves are smoothed with a moving "
        "average of 25 evaluation points where applicable. PPO is evaluated "
        "deterministically every 10k environment steps over 20 episodes; DQN, "
        "REINFORCE, AC and A2C are reconstructed from saved previous-assignment "
        "training-return traces and are therefore contextual rather than strictly "
        "protocol-matched baselines. The dotted horizontal line marks the maximum "
        "CartPole return of 500."
    ),
    "ppo_ablation": (
        "Controlled PPO ablation study on CartPole-v1. Lines show deterministic "
        "evaluation return averaged over independent seeds, with shaded regions "
        "denoting ±1 standard deviation. Evaluation is performed every 10k "
        "environment steps over 20 greedy episodes and curves are smoothed with a "
        "moving average of 9 checkpoints. The ablations isolate clipping, GAE depth, "
        "advantage normalization and repeated minibatch epochs; all other settings "
        "follow the relevant PPO configuration unless stated otherwise."
    ),
    "ppo_stability": (
        "Seed-level robustness diagnostics for PPO variants. Boxes summarize "
        "variation across independent seeds and points show individual seeds. "
        "t_475 is the first evaluation checkpoint at which mean deterministic "
        "return reaches 475. Final evaluation return measures end-of-training "
        "performance. Post-solve worst return is the minimum deterministic "
        "evaluation return after the first solve checkpoint, so higher values "
        "indicate better retention of the solved policy. This plot separates "
        "sample efficiency from post-solve stability."
    ),
    "ppo_diagnostics": (
        "PPO training diagnostics averaged across seeds. Approximate KL and the "
        "fraction of policy ratios outside [1−ε,1+ε] measure policy drift relative "
        "to the rollout policy; entropy measures policy stochasticity; critic loss "
        "measures value-target fitting. Curves are smoothed for readability and "
        "shaded regions show seed variability. For the no-clipping ablation, the "
        "ratio fraction is logged diagnostically but is not used to constrain the "
        "loss."
    ),
    "ppo_pareto": (
        "Sample efficiency versus post-solve stability for PPO variants. The x-axis "
        "shows mean first-solve step t_475 (lower is faster). The y-axis shows the "
        "mean minimum deterministic evaluation return after first reaching 475 "
        "(higher indicates more stable policies). Marker size is proportional to "
        "AULC_200k (area under the learning curve over the first 200k steps), "
        "encoding early-learning performance in a single compact figure."
    ),
}


# ── Data-source helpers ───────────────────────────────────────────────────────

def _find_latest_study_root(studies_dir: Path) -> Optional[Path]:
    candidates = [
        c for c in studies_dir.iterdir()
        if c.is_dir() and ((c / "ablations").exists() or (c / "final").exists())
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _has_eval_csvs(d: Path) -> bool:
    return d.exists() and any(True for _ in d.glob("seed_*/eval.csv"))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PPO study plots")
    parser.add_argument(
        "--study-root",
        type=str,
        default="results/ablation_study/full",
        help="Study directory containing final/ and ablations/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory for generated figure files.",
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


def _resolve_variant_dirs(study_root: Path) -> Dict[str, Path]:
    """Return an explicit PPO variant → results directory mapping for one study."""
    resolved = {
        "PPO_final": study_root / "final" / "ppo_final",
        "PPO_full": study_root / "final" / "ppo_full",
        "PPO_tuned": study_root / "final" / "ppo_tuned",
        "PPO_no_clip": study_root / "ablations" / "ppo_no_clip",
        "PPO_lambda0": study_root / "ablations" / "ppo_lambda0",
        "PPO_no_adv_norm": study_root / "ablations" / "ppo_no_adv_norm",
        "PPO_single_epoch": study_root / "ablations" / "ppo_single_epoch",
    }
    return {
        key: path for key, path in resolved.items() if _has_eval_csvs(path)
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    study_root = _resolve_study_root(root, args.study_root)
    prev_dir = root / "previous_results"
    figures_dir = Path(args.output_dir)
    if not figures_dir.is_absolute():
        figures_dir = root / figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    def _display_path(path: Path) -> str:
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    print(f"Using study root: {study_root}")

    variant_dirs = _resolve_variant_dirs(study_root)
    print(f"Resolved {len(variant_dirs)} PPO variant directories:")
    for k, v in variant_dirs.items():
        n_seeds = len(list(v.glob("seed_*")))
        print(f"  {k:<22} → {v.relative_to(root)}  ({n_seeds} seeds)")

    # The ablations dir for plot functions that take a results_dir fallback.
    # We pass variant_dirs explicitly, so this is only used for variants not in
    # the dict (graceful fallback).
    ablations_dir = study_root / "ablations" if study_root else root / "results"
    final_dir     = study_root / "final"     if study_root else root / "results"

    print(f"\n[1/6] Main comparison (DQN / REINFORCE / AC / A2C / PPO)")
    plot_main_comparison(final_dir, prev_dir, figures_dir)

    print(f"\n[2/6] PPO ablation study")
    plot_ppo_ablation(ablations_dir, figures_dir, variant_dirs=variant_dirs)

    print(f"\n[3/6] PPO stability / robustness")
    plot_ppo_stability(ablations_dir, figures_dir, variant_dirs=variant_dirs)

    print(f"\n[4/6] PPO training diagnostics")
    plot_ppo_diagnostics(ablations_dir, figures_dir, variant_dirs=variant_dirs)

    print(f"\n[5/6] Pareto: sample efficiency vs. stability")
    plot_ppo_pareto(ablations_dir, figures_dir, variant_dirs=variant_dirs)

    # Optional: sweep summary (only if sweep data exists)
    sweep_dir = study_root / "sweep" if study_root else None
    if sweep_dir and (sweep_dir / "summary.csv").exists():
        print(f"\n[6/6] PPO sweep summary")
        plot_ppo_sweep_summary(sweep_dir, figures_dir)
    else:
        print(f"\n[6/6] PPO sweep summary – skipped (no summary.csv found)")

    # ── Write captions file ────────────────────────────────────────────────────
    captions_path = figures_dir / "figure_captions.txt"
    with captions_path.open("w") as f:
        f.write("Figure Captions for RL Assignment 4 Report\n")
        f.write("=" * 60 + "\n\n")
        for key, caption in CAPTIONS.items():
            f.write(f"Figure: {key}\n")
            f.write("-" * 40 + "\n")
            f.write(caption + "\n\n")
    print(f"\nFigure captions → {_display_path(captions_path)}")
    print(f"All outputs saved to {_display_path(figures_dir)}/")


if __name__ == "__main__":
    main()
