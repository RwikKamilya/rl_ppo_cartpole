"""
run_ablation_study.py – compact PPO study runner for the assignment report.

Study design
------------
1. `ablations`
   baseline, no_clip, lambda0, no_adv_norm, single_epoch
2. `sweep`
   one-factor tuning variants for clip_coef, gae_lambda, rollout_steps,
   and update_epochs (12 total variants)
3. `final`
   baseline PPO vs the final tuned PPO config

Default seed counts follow the recommended report setup:
  - ablations: 3 seeds
  - sweep: 3 seeds
  - final: 5 seeds

Usage examples
--------------
python scripts/run_ablation_study.py --stage ablations
python scripts/run_ablation_study.py --stage sweep --smoke
python scripts/run_ablation_study.py --stage final --final-config configs/ppo_tuned.json
python scripts/run_ablation_study.py --stage all --summarize-only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_a4.metrics import compute_ppo_metrics
from rl_a4.utils import load_config


ABLATION_DEFAULT_SEEDS = [0, 1, 2]
SWEEP_DEFAULT_SEEDS = [0, 1, 2]
FINAL_DEFAULT_SEEDS = [0, 1, 2, 3, 4]


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    overrides: Dict[str, object]


def _variant(name: str, description: str, **overrides) -> Variant:
    return Variant(name=name, description=description, overrides=overrides)


STAGES: Dict[str, List[Variant]] = {
    "ablations": [
        _variant("baseline", "Reference PPO configuration"),
        _variant("no_clip", "Disable PPO clipping", use_clip=False),
        _variant("lambda0", "Disable GAE; one-step TD advantages", gae_lambda=0.0),
        _variant("no_adv_norm", "Disable per-rollout advantage normalization", normalize_advantages=False),
        _variant("single_epoch", "Single update epoch per rollout", update_epochs=1),
    ],
    "sweep": [
        _variant("clip_0p1", "Clip coefficient sweep: clip_coef=0.1", clip_coef=0.1),
        _variant("clip_0p2", "Clip coefficient sweep: clip_coef=0.2", clip_coef=0.2),
        _variant("clip_0p3", "Clip coefficient sweep: clip_coef=0.3", clip_coef=0.3),
        _variant("gae_0p9", "GAE lambda sweep: gae_lambda=0.9", gae_lambda=0.9),
        _variant("gae_0p95", "GAE lambda sweep: gae_lambda=0.95", gae_lambda=0.95),
        _variant("gae_0p98", "GAE lambda sweep: gae_lambda=0.98", gae_lambda=0.98),
        _variant("rollout_512", "Rollout length sweep: rollout_steps=512", rollout_steps=512),
        _variant("rollout_1024", "Rollout length sweep: rollout_steps=1024", rollout_steps=1024),
        _variant("rollout_2048", "Rollout length sweep: rollout_steps=2048", rollout_steps=2048),
        _variant("epochs_2", "Update epoch sweep: update_epochs=2", update_epochs=2),
        _variant("epochs_4", "Update epoch sweep: update_epochs=4", update_epochs=4),
        _variant("epochs_8", "Update epoch sweep: update_epochs=8", update_epochs=8),
    ],
    "final": [
        _variant("baseline", "Reference PPO configuration"),
        _variant("final_tuned", "Final tuned PPO configuration"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run staged PPO ablations")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all"] + sorted(STAGES.keys()),
        help="Which study stage to run.",
    )
    parser.add_argument("--base-config", type=str, default="configs/ppo_full.json")
    parser.add_argument("--final-config", type=str, default="configs/ppo_tuned.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seeds for every stage.")
    parser.add_argument("--ablation-seeds", type=int, nargs="+", default=ABLATION_DEFAULT_SEEDS)
    parser.add_argument("--sweep-seeds", type=int, nargs="+", default=SWEEP_DEFAULT_SEEDS)
    parser.add_argument("--final-seeds", type=int, nargs="+", default=FINAL_DEFAULT_SEEDS)
    parser.add_argument("--smoke", action="store_true", help="Run with 50k steps instead of 1M.")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--results-root", type=str, default="results/ablation_study")
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional suffix for the ablation results directory.",
    )
    return parser.parse_args()


def _selected_stages(stage_name: str) -> List[str]:
    if stage_name == "all":
        return ["ablations", "sweep", "final"]
    return [stage_name]


def _seeds_for_stage(args, stage: str) -> List[int]:
    if args.seeds is not None:
        return args.seeds
    if stage == "ablations":
        return args.ablation_seeds
    if stage == "sweep":
        return args.sweep_seeds
    if stage == "final":
        return args.final_seeds
    raise ValueError(f"Unknown stage: {stage}")


def _load_base_config(project_root: Path, path: str) -> dict:
    cfg = load_config(project_root / path)
    cfg.setdefault("use_orthogonal_init", True)
    return cfg


def _study_root(project_root: Path, results_root: str, study_name: str | None) -> Path:
    root = project_root / results_root
    if study_name:
        root = root / study_name
    return root


def _build_config(base_config: dict, stage: str, variant: Variant, stage_root: Path) -> dict:
    config = dict(base_config)
    config.update(variant.overrides)
    config["exp_name"] = _exp_name_for_variant(stage, variant)
    config["results_dir"] = str(stage_root)
    config.setdefault("use_orthogonal_init", True)
    if config.get("minibatch_size", 0) > config["rollout_steps"]:
        raise ValueError(
            f"{stage}/{variant.name}: minibatch_size={config['minibatch_size']} "
            f"cannot exceed rollout_steps={config['rollout_steps']}"
        )
    return config


def _exp_name_for_variant(stage: str, variant: Variant) -> str:
    if stage == "final" and variant.name == "baseline":
        return "ppo_full"
    if stage == "final" and variant.name == "final_tuned":
        return "ppo_tuned"
    if stage == "ablations":
        return {
            "baseline": "ppo_full",
            "no_clip": "ppo_no_clip",
            "lambda0": "ppo_lambda0",
            "no_adv_norm": "ppo_no_adv_norm",
            "single_epoch": "ppo_single_epoch",
        }[variant.name]
    return variant.name


def _run_variant(
    project_root: Path,
    python_bin: str,
    config_path: Path,
    seed: int,
    total_env_steps: int | None,
) -> None:
    cmd = [
        python_bin,
        str(project_root / "scripts" / "train_ppo.py"),
        "--config",
        str(config_path),
        "--seed",
        str(seed),
    ]
    if total_env_steps is not None:
        cmd += ["--total-env-steps", str(total_env_steps)]
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for {config_path.name} seed={seed}")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _stage_manifest_rows(stage: str, variants: List[Variant], stage_root: Path) -> List[dict]:
    rows = []
    for variant in variants:
        rows.append(
            {
                "stage": stage,
                "variant": variant.name,
                "exp_name": _exp_name_for_variant(stage, variant),
                "description": variant.description,
                "result_dir": str(stage_root / _exp_name_for_variant(stage, variant)),
                "overrides": variant.overrides,
            }
        )
    return rows


def _summarize_stage(stage: str, variants: List[Variant], stage_root: Path) -> List[dict]:
    rows = []
    for variant in variants:
        variant_dir = stage_root / _exp_name_for_variant(stage, variant)
        if not variant_dir.exists():
            continue
        metrics = compute_ppo_metrics(variant_dir)
        if not metrics:
            continue
        row = {
            "stage": stage,
            "variant": variant.name,
            "description": variant.description,
        }
        row.update(metrics)
        rows.append(row)
    return rows


def _write_summary_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "stage",
        "variant",
        "description",
        "n_seeds",
        "final_eval_return_mean",
        "final_eval_return_std",
        "final_train_return_mean",
        "final_train_return_std",
        "AULC_1M",
        "AULC_200k",
        "t_475_mean",
        "t_475_std",
        "solve_rate",
        "post_solve_min_return",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_top_variants(rows: List[dict], stage: str) -> None:
    if not rows:
        print(f"[{stage}] No completed results found.")
        return
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("AULC_200k", float("-inf")),
            row.get("final_eval_return_mean", float("-inf")),
        ),
        reverse=True,
    )
    print(f"\n[{stage}] Top variants by AULC_200k")
    for row in ranked[:5]:
        print(
            f"  {row['variant']:<28} "
            f"AULC_200k={row['AULC_200k']:.3f}  "
            f"final_eval={row['final_eval_return_mean']:.1f}  "
            f"t_475={row['t_475_mean']:.0f}"
        )


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    base_config = _load_base_config(project_root, args.base_config)
    final_config = _load_base_config(project_root, args.final_config)
    stage_names = _selected_stages(args.stage)

    study_name = args.study_name
    if study_name is None:
        study_name = "smoke" if args.smoke else "full"

    total_env_steps = 50_000 if args.smoke else None
    root = _study_root(project_root, args.results_root, study_name)
    root.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []

    for stage in stage_names:
        variants = STAGES[stage]
        seeds = _seeds_for_stage(args, stage)
        stage_root = root / stage
        stage_root.mkdir(parents=True, exist_ok=True)

        manifest_rows = _stage_manifest_rows(stage, variants, stage_root)
        _write_json(stage_root / "manifest.json", manifest_rows)

        if not args.summarize_only:
            for variant in variants:
                base = final_config if stage == "final" and variant.name == "final_tuned" else base_config
                config = _build_config(base, stage, variant, stage_root)
                config_path = stage_root / f"{variant.name}.json"
                _write_json(config_path, config)

                print(f"\n[{stage}] {variant.name}")
                print(f"  {variant.description}")
                print(f"  overrides={variant.overrides if variant.overrides else '{}'}")

                for seed in seeds:
                    _run_variant(
                        project_root=project_root,
                        python_bin=args.python_bin,
                        config_path=config_path,
                        seed=seed,
                        total_env_steps=total_env_steps,
                    )

        stage_rows = _summarize_stage(stage, variants, stage_root)
        _write_summary_csv(stage_root / "summary.csv", stage_rows)
        _print_top_variants(stage_rows, stage)
        all_summary_rows.extend(stage_rows)

    _write_summary_csv(root / "summary_all.csv", all_summary_rows)
    print(f"\nStudy outputs saved to {root}")


if __name__ == "__main__":
    main()
