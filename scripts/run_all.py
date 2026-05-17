"""
run_all.py – Launch all PPO experiments (full + ablations) sequentially.

Usage:
    python scripts/run_all.py                  # all 4 variants, 5 seeds each
    python scripts/run_all.py --seeds 0 1 2    # subset of seeds
    python scripts/run_all.py --smoke          # quick smoke test (5000 steps)

Experiments:
    ppo_full        – baseline PPO-clip with all tricks
    ppo_no_clip     – unclipped surrogate (tests clip importance)
    ppo_lambda0     – gae_lambda=0.0 (tests GAE importance)
    ppo_no_adv_norm – no advantage normalisation
"""

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = [
    # ("configs/ppo_full.json",        "ppo_full"),
    # ("configs/ppo_no_clip.json",     "ppo_no_clip"),
    # ("configs/ppo_lambda0.json",     "ppo_lambda0"),
    # ("configs/ppo_no_adv_norm.json", "ppo_no_adv_norm"),
    ("configs/ppo_tuned.json", "ppo_tuned"),
]

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def parse_args():
    p = argparse.ArgumentParser(description="Run all PPO experiments")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--smoke", action="store_true",
                   help="Run smoke test with 5000 steps only.")
    p.add_argument("--exp", type=str, nargs="+", default=None,
                   help="Subset of experiments to run (by exp_name).")
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    experiments = EXPERIMENTS
    if args.exp is not None:
        experiments = [(c, n) for c, n in EXPERIMENTS if n in args.exp]
        if not experiments:
            print(f"[ERROR] No matching experiments for: {args.exp}")
            sys.exit(1)

    total_runs = len(experiments) * len(args.seeds)
    run_idx = 0

    for config_path, exp_name in experiments:
        for seed in args.seeds:
            run_idx += 1
            print(f"\n{'='*60}")
            print(f"[{run_idx}/{total_runs}] exp={exp_name}  seed={seed}")
            print(f"{'='*60}")

            cmd = [
                sys.executable,
                str(project_root / "scripts" / "train_ppo.py"),
                "--config", str(project_root / config_path),
                "--seed", str(seed),
            ]
            if args.smoke:
                cmd += ["--total-env-steps", "5000"]

            result = subprocess.run(cmd, cwd=str(project_root))
            if result.returncode != 0:
                print(f"[ERROR] Run failed: {exp_name} seed={seed}")
                sys.exit(result.returncode)

    print(f"\n{'='*60}")
    print(f"All {total_runs} runs completed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
