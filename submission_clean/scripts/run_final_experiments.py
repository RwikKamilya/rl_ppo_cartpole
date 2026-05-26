"""
Run PPO CartPole-v1 experiment stages.

Stages:
  smoke     -> 20k-step ppo_final seed 0 check
  final     -> 1M-step ppo_final seeds 0..4 by default
  selected  -> 1M-step ppo_selected seeds 0..4 by default
  ablations -> 500k-step PPO ablations seeds 0..2 by default
  all       -> final and ablations
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


FINAL_CONFIG = "ppo_final.json"
SELECTED_CONFIG = "ppo_selected.json"
ABLATION_CONFIGS = [
    "ppo_final.json",
    "ppo_no_clip.json",
    "ppo_lambda0.json",
    "ppo_no_entropy.json",
    "ppo_adv_norm_on.json",
    "ppo_single_epoch.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO CartPole-v1 experiments")
    parser.add_argument("--stage", choices=["smoke", "final", "selected", "ablations", "all"], default="smoke")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ablation-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--ablation-total-env-steps", type=int, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--study-name", type=str, default="final_study")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--force", action="store_true", help="Rerun even when complete seed logs exist.")
    return parser.parse_args()


def _resolve_python_bin(python_bin: str, invocation_cwd: Path) -> str:
    path = Path(python_bin)
    if path.is_absolute():
        return str(path)
    if path.parent != Path("."):
        # Keep virtualenv symlinks intact. Path.resolve() dereferences
        # ./venv/bin/python to /usr/bin/python and drops venv site-packages.
        return str(invocation_cwd / path)
    return python_bin


def _study_root(project_root: Path, results_root_arg: str, study_name: str) -> Path:
    results_root = Path(results_root_arg)
    if not results_root.is_absolute():
        results_root = project_root / results_root
    if results_root.name == study_name:
        return results_root
    return results_root / study_name


def _eval_has_two_data_rows(eval_csv: Path) -> bool:
    try:
        with eval_csv.open(newline="") as f:
            rows = list(csv.reader(f))
    except OSError:
        return False
    return max(0, len(rows) - 1) >= 2


def _run_complete(results_dir: Path, exp_name: str, seed: int) -> bool:
    seed_dir = results_dir / exp_name / f"seed_{seed}"
    required = ["train.csv", "eval.csv", "update_log.csv", "config.json"]
    for filename in required:
        path = seed_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            return False
    return _eval_has_two_data_rows(seed_dir / "eval.csv")


def _exp_name_from_config(config_name: str) -> str:
    return config_name.removesuffix(".json")


def _run_one(
    project_root: Path,
    python_bin: str,
    config_name: str,
    seed: int,
    results_dir: Path,
    total_env_steps: int | None,
    skip_existing: bool,
    force: bool,
) -> None:
    exp_name = _exp_name_from_config(config_name)
    seed_dir = results_dir / exp_name / f"seed_{seed}"
    if not force and skip_existing and _run_complete(results_dir, exp_name, seed):
        print(f"[SKIP] existing complete run: {seed_dir}")
        return
    if force:
        print(f"[RUN] forced rerun: {seed_dir}")
    else:
        print(f"[RUN] missing/incomplete run: {seed_dir}")

    cmd = [
        python_bin,
        str(project_root / "scripts" / "train_ppo.py"),
        "--config",
        str(project_root / "configs" / config_name),
        "--seed",
        str(seed),
        "--results-dir",
        str(results_dir),
    ]
    if total_env_steps is not None:
        cmd += ["--total-env-steps", str(total_env_steps)]

    print("[CMD] " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        raise RuntimeError(f"Failed: {exp_name} seed={seed}")


def run_smoke(args: argparse.Namespace, project_root: Path, python_bin: str) -> None:
    steps = args.total_env_steps if args.total_env_steps is not None else 20_000
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = project_root / results_root
    results_dir = results_root / "smoke" / "final"
    _run_one(project_root, python_bin, FINAL_CONFIG, 0, results_dir, steps, args.skip_existing, args.force)


def run_final(args: argparse.Namespace, project_root: Path, python_bin: str) -> None:
    results_dir = _study_root(project_root, args.results_root, args.study_name) / "final"
    for seed in args.seeds:
        _run_one(project_root, python_bin, FINAL_CONFIG, seed, results_dir, args.total_env_steps, args.skip_existing, args.force)


def run_selected(args: argparse.Namespace, project_root: Path, python_bin: str) -> None:
    results_dir = _study_root(project_root, args.results_root, args.study_name) / "selected"
    for seed in args.seeds:
        _run_one(project_root, python_bin, SELECTED_CONFIG, seed, results_dir, args.total_env_steps, args.skip_existing, args.force)


def run_ablations(args: argparse.Namespace, project_root: Path, python_bin: str) -> None:
    steps = args.ablation_total_env_steps if args.ablation_total_env_steps is not None else 500_000
    results_dir = _study_root(project_root, args.results_root, args.study_name) / "ablations"
    for config_name in ABLATION_CONFIGS:
        for seed in args.ablation_seeds:
            _run_one(project_root, python_bin, config_name, seed, results_dir, steps, args.skip_existing, args.force)


def main() -> None:
    args = parse_args()
    invocation_cwd = Path.cwd()
    project_root = Path(__file__).resolve().parent.parent
    python_bin = _resolve_python_bin(args.python_bin, invocation_cwd)

    if args.stage == "smoke":
        run_smoke(args, project_root, python_bin)
    elif args.stage == "final":
        run_final(args, project_root, python_bin)
    elif args.stage == "selected":
        run_selected(args, project_root, python_bin)
    elif args.stage == "ablations":
        run_ablations(args, project_root, python_bin)
    elif args.stage == "all":
        run_final(args, project_root, python_bin)
        run_ablations(args, project_root, python_bin)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
