# RL Assignment 4: PPO CartPole-v1

This is the clean submission repository for the final PPO CartPole-v1 experiments. It contains the audited PPO implementation, final configs, experiment scripts, metrics/plotting code, and saved baseline traces from previous assignments.

## Environment Setup

Run from this directory on a Linux machine:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If this directory is inside the original repository, `run_all.sh` automatically uses `../venv/bin/python` when that environment exists; otherwise it uses the active `python`.

## Correct Results Layout

The expected study layout is:

```text
results/final_study/final/ppo_final/seed_*/
results/final_study/ablations/<variant>/seed_*/
results/final_study/metrics/
results/final_study/plots/
```

The old incorrect nested layout was:

```text
results/final_study/final_study/final/ppo_final/seed_*/
results/final_study/final_study/ablations/<variant>/seed_*/
```

Fix it manually with:

```bash
python scripts/fix_results_layout.py --study-root results/final_study
```

The fixer is conservative and idempotent: identical files are skipped, non-identical conflicts fail clearly, and the old nested directory is renamed to `results/final_study_nested_backup` instead of being deleted.

## One-Command Final Workflow

Use this command from now on:

```bash
bash ./run_all.sh
```

`run_all.sh` is idempotent. It first fixes/validates the layout, skips completed seed runs, checks result completeness, recomputes metrics, and regenerates plots. Metrics and plots are regenerated every time; completed training is not rerun unless forced.

For background execution:

```bash
nohup bash ./run_all.sh > logs/nohup_launcher.log 2>&1 &
```

## Smoke Test

Runs one 20k-step `ppo_final` seed and writes logs under `results/smoke/final/ppo_final/seed_0/`.

```bash
python scripts/run_final_experiments.py --stage smoke
```

## Final 5-Seed PPO Comparison

Runs the final tuned PPO config for seeds `0 1 2 3 4` with the default 1,000,000 environment-step budget.

```bash
python scripts/run_final_experiments.py --stage final --seeds 0 1 2 3 4 --results-root results --study-name final_study --skip-existing
```

Expected PPO output:

```text
results/final_study/final/ppo_final/seed_0/
results/final_study/final/ppo_final/seed_1/
results/final_study/final/ppo_final/seed_2/
results/final_study/final/ppo_final/seed_3/
results/final_study/final/ppo_final/seed_4/
```

Each seed directory contains `train.csv`, `eval.csv`, `update_log.csv`, and `config.json`. A run is considered complete only if all four files are non-empty and `eval.csv` has at least two data rows.

## PPO Selected

`ppo_selected` is a post-ablation tuned candidate. It keeps PPO clipping and GAE, keeps the tuned rollout and epoch settings, and removes entropy regularization based on the ablation result. It writes to `results/final_study/selected/ppo_selected/seed_*/`.

```bash
python scripts/run_final_experiments.py --stage selected --seeds 0 1 2 3 4 --results-root results --study-name final_study --skip-existing
```

## PPO Ablations

Runs `ppo_final`, `ppo_no_clip`, `ppo_lambda0`, `ppo_no_entropy`, `ppo_adv_norm_on`, and `ppo_single_epoch` for seeds `0 1 2 3 4` with the default 500,000 environment-step ablation budget. Existing complete seeds are skipped, so this extends the current 3-seed ablations by running only missing seeds 3 and 4.

```bash
python scripts/run_final_experiments.py --stage ablations --ablation-seeds 0 1 2 3 4 --results-root results --study-name final_study --skip-existing
```

Expected ablation output:

```text
results/final_study/ablations/<exp_name>/seed_*/
```

To recompute metrics and regenerate plots after selected or extended ablation runs, use the Metrics and Plots commands below.

To force reruns even when seed outputs already exist:

```bash
python scripts/run_final_experiments.py --stage all --results-root results --study-name final_study --force
```

`--force` wins over `--skip-existing`.

## Metrics

Compute final metrics from the study layout and saved previous-assignment baselines. If `results/final_study/selected/ppo_selected/seed_*` exists, `PPO Selected` is included alongside `PPO Final`:

```bash
python scripts/compute_metrics.py --study-root results/final_study --output-dir results/final_study/metrics
```

Expected metric outputs:

```text
results/final_study/metrics/per_seed_metrics.csv
results/final_study/metrics/summary_metrics.csv
results/final_study/metrics/metrics_summary.csv
results/final_study/metrics/metrics_summary.json
results/final_study/metrics/metrics_summary.md
```

## Plots

Generate report figures. The main comparison includes `PPO Selected` when selected logs exist:

```bash
python scripts/plot_results.py --study-root results/final_study --output-dir results/final_study/plots
```

Expected plot outputs:

```text
results/final_study/plots/final_learning_curves_comparison.pdf
results/final_study/plots/final_learning_curves_comparison.png
results/final_study/plots/ppo_ablation_curves.pdf
results/final_study/plots/ppo_ablation_curves.png
results/final_study/plots/ppo_diagnostics.pdf
results/final_study/plots/ppo_diagnostics.png
results/final_study/plots/ppo_stability_summary.pdf
results/final_study/plots/ppo_stability_summary.png
results/final_study/plots/figure_captions.txt
```

## Notes

PPO training actions are stochastic samples from a categorical policy. PPO evaluation is deterministic greedy `argmax(logits)` under `torch.no_grad()`.

DQN, REINFORCE, AC, and A2C are not retrained in this repository. They are loaded from previous Assignment 2/3 `.npz` traces in `previous_results/`, so those curves are contextual saved-trace baselines rather than newly rerun deterministic evaluations.
