# Code Audit Findings

Ranked by importance for report correctness and grading.

## 1. Latest staged study is incomplete

- Severity: high for reporting workflow
- Evidence:
  - `results/ablation_study/full/ablations/summary.csv` exists and is complete.
  - `results/ablation_study/full/sweep/` contains only `clip_0p1` with seeds `0` and `1`.
  - `results/ablation_study/full/final/` is absent.
- Why it matters:
  - `scripts/compute_metrics.py` and `scripts/plot_results.py` now read the latest staged study.
  - Without `final/`, `PPO_full` and `PPO_tuned` are omitted from current repo-level metrics and some plots cannot be regenerated as intended.
- Recommendation:
  - Finish staged `sweep` and `final`, or temporarily point reporting scripts at the complete top-level `results/ppo_*` folders.

## 2. Evaluation seeding is weaker than training seeding

- Severity: medium
- Evidence:
  - Training seeds the environment at first reset: `scripts/train_ppo.py:155`.
  - Evaluation creates fresh envs and calls `env.reset()` without a seed: `rl_a4/evaluate.py:48-49`.
  - `rl_a4/utils.py:33-40` seeds action and observation spaces, but not the env RNG via `reset(seed=...)`.
- Why it matters:
  - Evaluation may not be strictly reproducible across runs even with fixed top-level seeds.
- Recommendation:
  - If code changes are allowed later, seed evaluation resets explicitly.
  - In the report, avoid claiming full bitwise reproducibility.

## 3. PPO vs previous baselines is not a perfectly matched evaluation protocol

- Severity: medium
- Evidence:
  - PPO uses deterministic evaluation checkpoints from `eval.csv`.
  - Previous baselines are loaded from `.npz` training reward traces and processed as comparison curves in `rl_a4/metrics.py` and `rl_a4/plotting.py`.
  - `t_475` for baselines is computed on a smoothed moving average of training return, not deterministic eval.
- Why it matters:
  - Main-comparison plots and AULC/timing metrics are informative but not strictly apples-to-apples.
- Recommendation:
  - State this explicitly as a limitation in the Experiments/Discussion section.

## 4. Plot annotation is outdated for current staged ablations

- Severity: low
- Evidence:
  - `rl_a4/plotting.py` ablation figure text says `5 seeds`.
  - The latest staged ablation folder contains only seeds `0,1,2`.
- Why it matters:
  - The figure annotation is factually incorrect if regenerated from staged ablations.
- Recommendation:
  - Fix before final figure generation, or disclose that current staged ablations use 3 seeds.

## 5. README is out of sync with the current experiment workflow

- Severity: low-medium for grading
- Evidence:
  - `README.md:20` still describes “4 variants × 5 seeds”.
  - `README.md:50-54` omits `ppo_single_epoch.json` and `ppo_tuned.json`.
  - `README.md` does not document `scripts/run_ablation_study.py`.
- Why it matters:
  - The assignment explicitly requires reproducible rerun instructions.
- Recommendation:
  - Update README before submission.

## 6. Main comparison and repo-level metrics currently depend on “latest study” detection by directory mtime

- Severity: low
- Evidence:
  - `find_latest_study_root()` in both `scripts/compute_metrics.py` and `scripts/plot_results.py` selects the most recently modified study folder.
- Why it matters:
  - An incomplete or stale study directory can become the active reporting source.
- Recommendation:
  - For the final submission, either keep only one clean study folder or document the intended `study_name`.

## 7. No target-KL early stopping, no checkpointing, no explained variance

- Severity: low
- Evidence:
  - No target-KL stopping code found.
  - No model checkpoint saves found.
  - No explained variance logging found.
- Why it matters:
  - These are optional engineering features, not correctness bugs.
- Recommendation:
  - Report them as absent rather than implying they were used.

## 8. Core PPO implementation itself looks structurally correct

- Severity: positive finding
- Evidence:
  - correct old-log-prob storage and reuse
  - correct ratio computation
  - correct terminated vs truncated handling for bootstrap and GAE continuation
  - correct return construction `R = A + V`
  - deterministic evaluation isolated from training
- Recommendation:
  - Highlight these implementation details in the report; they strengthen credibility.

