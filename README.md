# Leiden RL Assignment 4: PPO CartPole

This repository contains the PPO-Clipped implementation, staged ablation runner, metric scripts, and report assets for CartPole-v1. The audited final study root is `results/ablation_study/full`.

## Python And Setup

Expected Python version: Python 3.10 or newer. The code uses Python 3.10 syntax and was verified with the local `.venv` Python 3.10.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If the system does not provide `python` before activation, use `python3 -m venv .venv` for the first command. After activation, the commands below assume `python` resolves to `.venv/bin/python`.

## Reproduce Final Comparison

Run each command from the repository root after activating the virtual environment.

Report-ready PPO final comparison, exactly seeds `[0, 1, 2, 3, 4]`:

```bash
python scripts/run_final_comparison.py
```

This trains `configs/ppo_final.json`, validates that all five seed logs exist, loads the existing five-seed Assignment 2/3 baseline traces from `previous_results/`, aggregates metrics, and writes a new timestamped output folder:

- `results/final_comparison/ppo_final_<timestamp>/logs/ppo_final/seed_*/`
- `results/final_comparison/ppo_final_<timestamp>/metrics/metrics_summary.csv`
- `results/final_comparison/ppo_final_<timestamp>/metrics/metrics_summary.json`
- `results/final_comparison/ppo_final_<timestamp>/plots/final_learning_curves_comparison.png`
- `results/final_comparison/ppo_final_<timestamp>/plots/final_learning_curves_comparison.pdf`
- `results/final_comparison/ppo_final_<timestamp>/plots/ppo_ablation_or_variants.png`
- `results/final_comparison/ppo_final_<timestamp>/plots/ppo_ablation_or_variants.pdf`

Quick smoke test of the full pipeline:

```bash
python scripts/run_final_comparison.py --run-name smoke_protocol --total-env-steps 3000 --eval-interval 1000 --n-eval-episodes 2
```

The smoke command is not for report numbers; it only checks training, deterministic evaluation, seed-log validation, aggregation, and plotting.

## Optional PPO Ablations

The older staged PPO ablation workflow is still available. It is useful for analysis, but the final comparison should use the five-seed command above.

```bash
python scripts/run_ablation_study.py --stage ablations --study-name full
```

Full final study, including ablations, tuning sweep, and final baseline-vs-tuned runs:

```bash
python scripts/run_ablation_study.py --stage all --study-name full
```

The `--stage all` command regenerates the older audited study tree in one command.

For a quick non-report smoke run:

```bash
python scripts/run_ablation_study.py --stage all --smoke
```

## Metrics And Plots

Compute report-ready metrics from the audited final study root:

```bash
python scripts/compute_metrics.py
```

Regenerate report-ready plots from the same audited study root:

```bash
python scripts/plot_results.py
```

For a custom output folder:

```bash
python scripts/compute_metrics.py --study-root results/ablation_study/full --output-dir results/metrics
python scripts/plot_results.py --study-root results/ablation_study/full --output-dir results/plots
```

Expected output locations:

- `results/metrics/`
- `results/plots/`

The metric and plotting scripts default to `results/ablation_study/full`. To use a different completed study, pass `--study-root path/to/study`.

## Baseline Notes

PPO metrics and PPO plots use deterministic evaluation logs from each `seed_*/eval.csv`, with diagnostics from `seed_*/update_log.csv`. PPO training remains stochastic: rollout actions are sampled from `Categorical(logits=logits)`, while evaluation uses `argmax(logits)` under `torch.no_grad()`.

Previous baselines in `previous_results/*.npz` are existing five-seed saved training-return traces from earlier assignments. This Assignment 4 repository does not include checkpointed AC/A2C/REINFORCE/DQN trainers for deterministic reevaluation, so the final comparison loads those logs instead of rerunning them. The shared comparison x-axis is environment steps and the environment budget is 1,000,000 steps.

Stale top-level PPO folders such as `results/ppo_full`, `results/ppo_tuned`, and `results/ppo_no_clip` must not be used for final metrics or final report figures. They are old mixed-run outputs and differ from the audited final study root.

## Key Files

- Source package: `rl_a4/`
- Entry-point scripts: `scripts/`
- Training configs: `configs/`
- Final comparison config: `configs/ppo_final.json`
- Contextual baseline traces: `previous_results/*.npz`
- Audited final PPO logs: `results/ablation_study/full/`
- Final generated metrics: `results/metrics/`
- Final generated plots: `results/plots/`
- Submission shortlist: `SUBMISSION_MANIFEST.md`
