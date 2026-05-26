"""
Check whether required final, selected, and ablation PPO seed logs are complete.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


FINAL_VARIANTS = {"final/ppo_final": [0, 1, 2, 3, 4]}
SELECTED_VARIANTS = {"selected/ppo_selected": [0, 1, 2, 3, 4]}
ABLATION_VARIANTS = {
    "ablations/ppo_final": [0, 1, 2, 3, 4],
    "ablations/ppo_no_clip": [0, 1, 2, 3, 4],
    "ablations/ppo_lambda0": [0, 1, 2, 3, 4],
    "ablations/ppo_no_entropy": [0, 1, 2, 3, 4],
    "ablations/ppo_adv_norm_on": [0, 1, 2, 3, 4],
    "ablations/ppo_single_epoch": [0, 1, 2, 3, 4],
}
REQUIRED_FILES = ["train.csv", "eval.csv", "update_log.csv", "config.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PPO result completeness")
    parser.add_argument("--study-root", type=str, default="results/final_study")
    parser.add_argument("--include-selected", action="store_true", help="Also report selected PPO seeds 0..4.")
    return parser.parse_args()


def _resolve(project_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _eval_has_two_data_rows(eval_csv: Path) -> bool:
    try:
        with eval_csv.open(newline="") as f:
            rows = list(csv.reader(f))
    except OSError:
        return False
    return max(0, len(rows) - 1) >= 2


def is_complete(seed_dir: Path) -> bool:
    for filename in REQUIRED_FILES:
        path = seed_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            return False
    return _eval_has_two_data_rows(seed_dir / "eval.csv")


def _check_group(study_root: Path, variants: dict[str, list[int]]) -> tuple[list[dict[str, str]], bool]:
    rows = []
    all_complete = True
    for rel_variant, seeds in variants.items():
        complete = []
        missing = []
        for seed in seeds:
            seed_dir = study_root / rel_variant / f"seed_{seed}"
            if is_complete(seed_dir):
                complete.append(str(seed))
            else:
                missing.append(str(seed))
                all_complete = False
        rows.append(
            {
                "variant": rel_variant,
                "expected": ",".join(str(seed) for seed in seeds),
                "complete": ",".join(complete) if complete else "-",
                "missing": ",".join(missing) if missing else "-",
            }
        )
    return rows, all_complete


def _print_table(rows: list[dict[str, str]]) -> None:
    widths = {
        "variant": max(len("variant"), *(len(row["variant"]) for row in rows)),
        "expected": max(len("expected seeds"), *(len(row["expected"]) for row in rows)),
        "complete": max(len("found complete"), *(len(row["complete"]) for row in rows)),
        "missing": max(len("missing/incomplete"), *(len(row["missing"]) for row in rows)),
    }
    print(
        f"{'variant':<{widths['variant']}}  "
        f"{'expected seeds':<{widths['expected']}}  "
        f"{'found complete':<{widths['complete']}}  "
        f"{'missing/incomplete':<{widths['missing']}}"
    )
    print("-" * (sum(widths.values()) + 6))
    for row in rows:
        print(
            f"{row['variant']:<{widths['variant']}}  "
            f"{row['expected']:<{widths['expected']}}  "
            f"{row['complete']:<{widths['complete']}}  "
            f"{row['missing']:<{widths['missing']}}"
        )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    study_root = _resolve(project_root, args.study_root)

    final_rows, final_ok = _check_group(study_root, FINAL_VARIANTS)
    ablation_rows, ablations_ok = _check_group(study_root, ABLATION_VARIANTS)
    selected_rows: list[dict[str, str]] = []
    selected_ok = True
    if args.include_selected:
        selected_rows, selected_ok = _check_group(study_root, SELECTED_VARIANTS)

    print(f"study_root={study_root}")
    _print_table(final_rows + selected_rows + ablation_rows)

    if args.include_selected and not selected_ok:
        print("[WARN] one or more selected PPO seeds are missing or incomplete")
    if not ablations_ok:
        print("[WARN] one or more ablation seeds are missing or incomplete")
    if not final_ok:
        print("[ERROR] required final PPO seeds are missing or incomplete")
        sys.exit(1)


if __name__ == "__main__":
    main()
