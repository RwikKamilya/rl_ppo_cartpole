# Results Manifest

This manifest lists result and plot artifacts found during repository inspection.

## 1. Previous-assignment baseline artifacts

| File path | Algorithm | Seeds | Data present | Usable for report? | Caveat |
|---|---|---|---|---|---|
| `previous_results/a2c.npz` | A2C | 5 | `rewards`, `steps` | yes | no deterministic eval logs |
| `previous_results/ac.npz` | AC | 5 | `rewards`, `steps` | yes | no deterministic eval logs |
| `previous_results/reinforce.npz` | REINFORCE | 5 | `rewards`, `steps` | yes | no deterministic eval logs |
| `previous_results/linear_basic_training.npz` | DQN | 5 | `rewards`, `steps` | yes | no deterministic eval logs |
| `previous_results/a2c.png` | A2C plot | n/a | figure | optional | legacy artifact |
| `previous_results/ac.png` | AC plot | n/a | figure | optional | legacy artifact |
| `previous_results/reinforce.png` | REINFORCE plot | n/a | figure | optional | legacy artifact |
| `previous_results/comparison_pg.png` | prior comparison plot | n/a | figure | optional | legacy artifact |

## 2. Top-level PPO run artifacts

Each top-level PPO folder contains `seed_<seed>/config.json`, `seed_<seed>/train.csv`, `seed_<seed>/eval.csv`, and `seed_<seed>/update_log.csv`.

| Folder path | Experiment | Seeds present | Total steps | Key metrics present | Usable for report? | Caveat |
|---|---|---|---:|---|---|---|
| `results/ppo_full/` | baseline PPO | 0,1,2,3,4 | 1,000,000 | train return, eval mean/std, policy loss, value loss, entropy, approx KL, clip fraction | yes | older generation than staged study |
| `results/ppo_no_clip/` | no-clip ablation | 0,1,2,3,4 | 1,000,000 | same | yes | older generation than staged study |
| `results/ppo_lambda0/` | `lambda=0` ablation | 0,1,2,3,4 | 1,000,000 | same | yes | older generation than staged study |
| `results/ppo_no_adv_norm/` | no advantage normalization | 0,1,2,3,4 | 1,000,000 | same | yes | older generation than staged study |
| `results/ppo_tuned/` | tuned PPO | 0,1,2,3,4 | 1,000,000 | same | yes | not mirrored in staged `final/` yet |

## 3. Latest staged-study artifacts

### 3.1 Complete staged ablations

| Path | Experiment | Seeds present | Total steps | Usable for report? | Caveat |
|---|---|---|---:|---|---|
| `results/ablation_study/full/ablations/ppo_full/` | staged baseline | 0,1,2 | 1,000,000 | yes | 3 seeds only |
| `results/ablation_study/full/ablations/ppo_no_clip/` | staged no-clip | 0,1,2 | 1,000,000 | yes | 3 seeds only |
| `results/ablation_study/full/ablations/ppo_lambda0/` | staged `lambda=0` | 0,1,2 | 1,000,000 | yes | 3 seeds only |
| `results/ablation_study/full/ablations/ppo_no_adv_norm/` | staged no-adv-norm | 0,1,2 | 1,000,000 | yes | 3 seeds only |
| `results/ablation_study/full/ablations/ppo_single_epoch/` | staged single-epoch | 0,1,2 | 1,000,000 | yes | 3 seeds only |
| `results/ablation_study/full/ablations/summary.csv` | aggregated ablation summary | 3 seeds each | 1,000,000 | yes | strongest current latest-study summary |
| `results/ablation_study/full/ablations/manifest.json` | ablation manifest | n/a | n/a | yes | metadata only |

### 3.2 Partial staged sweep

| Path | Experiment | Seeds present | Total steps | Usable for report? | Caveat |
|---|---|---|---:|---|---|
| `results/ablation_study/full/sweep/clip_0p1/` | `clip_coef=0.1` sweep point | 0,1 | 1,000,000 | not yet | incomplete seeds and no stage summary |
| `results/ablation_study/full/sweep/clip_0p1.json` | config for partial sweep point | n/a | n/a | metadata only | partial stage |
| `results/ablation_study/full/sweep/manifest.json` | sweep manifest | n/a | n/a | metadata only | stage incomplete |

### 3.3 Missing staged final comparison

| Path | Status | Report use |
|---|---|---|
| `results/ablation_study/full/final/` | absent | not usable yet |

### 3.4 Stale staged artifacts

| Path | Status | Caveat |
|---|---|---|
| `results/ablation_study/full/core_tricks/` | older study layout | not the current `ablations/sweep/final` design |
| `results/ablation_study/dry_run/core_tricks/` | dry-run artifact | not final-report data |

## 4. Metric artifacts

| File path | Contents | Usable for report? | Caveat |
|---|---|---|---|
| `results/metrics.csv` | aggregated metrics table from latest staged study plus previous baselines | partially | current file omits `PPO_full` and `PPO_tuned` because latest staged `final/` is absent |
| `results/metrics_latex.tex` | LaTeX table generated from `metrics.csv` | partially | same caveat |

## 5. Figure artifacts

| File path | Figure | Usable for report? | Caveat |
|---|---|---|---|
| `figures/main_comparison.pdf` | main comparison | uncertain | may predate current plotting logic |
| `figures/main_comparison.png` | main comparison | uncertain | same caveat |
| `figures/ppo_ablation.pdf` | PPO ablation | yes if regenerated | latest code reads staged ablations |
| `figures/ppo_ablation.png` | PPO ablation | yes if regenerated | latest code reads staged ablations |
| `figures/ppo_diagnostics.pdf` | PPO diagnostics | yes if regenerated | latest code reads staged ablations |
| `figures/ppo_diagnostics.png` | PPO diagnostics | yes if regenerated | latest code reads staged ablations |
| `figures/ppo_eval.pdf` | latest PPO eval curve | not yet | latest code expects staged `final/`, which is absent |
| `figures/ppo_stability.pdf` | PPO stability | yes if regenerated | latest code reads staged ablations |
| `figures/ppo_stability.png` | PPO stability | yes if regenerated | latest code reads staged ablations |

## 6. Summary of what is currently final-report ready

- Ready now:
  - previous baseline `.npz` comparisons
  - top-level 5-seed PPO folders
  - staged ablation 3-seed folders
  - staged ablation `summary.csv`
- Not ready now:
  - staged `final/` comparison
  - full staged sweep summary
  - any figure that depends on completed staged `final/`

