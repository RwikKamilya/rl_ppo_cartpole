# Context

## Project Purpose

This repository is the clean submission version of an RL Assignment 4 PPO study on `CartPole-v1`. It keeps the audited PPO implementation intact and provides a reproducible workflow for final five-seed PPO evaluation, ablations, metrics, and plots.

## Assignment Requirements Satisfied

- Implements PPO-Clip for a discrete-action Gymnasium environment.
- Uses stochastic policy sampling during training.
- Uses deterministic greedy policy evaluation.
- Runs independent random seeds.
- Reports final performance, sample efficiency, solve rate, and stability.
- Compares PPO to saved DQN, REINFORCE, AC, and A2C traces from earlier assignments.
- Includes ablations for clipping, GAE, entropy, advantage normalization, and repeated PPO epochs.
- Adds `ppo_selected`, a post-ablation tuned PPO candidate that keeps clipping and GAE but removes entropy regularization.

## PPO Implementation Checklist

The PPO core is in `rl_a4/` and should not be rewritten for final runs.

- Stochastic training actions: `ActorCriticNet.get_action_and_log_prob` samples from `Categorical(logits=logits)`.
- Old log probabilities: rollout collection stores old log probabilities in `RolloutBuffer.log_probs_old`.
- PPO clipped loss: `PPOAgent.update` uses the minimum of unclipped and clipped policy surrogate losses.
- Value loss: critic loss is `0.5 * mean((return - value)^2)`.
- GAE: `RolloutBuffer.compute_gae_and_returns` computes generalized advantage estimates.
- Terminal handling: true terminal states stop bootstrapping; time-limit truncations bootstrap but do not carry GAE across reset.
- Entropy bonus: included in the total loss as `-ent_coef * entropy`.
- Minibatching: update data are shuffled with `torch.randperm`.
- Multiple epochs: `update_epochs` controls repeated passes over one rollout.
- Gradient clipping: `clip_grad_norm_` uses `max_grad_norm`.
- Evaluation: `evaluate_policy` uses greedy `argmax(logits)` under `torch.no_grad()`.
- Environment separation: evaluation uses a separate seed offset and fresh evaluation environments.

## Final Config Explanation

The final config is `configs/ppo_final.json`. It is based on the tuned configuration that performed well in diagnostics:

- `rollout_steps: 512`: shorter rollouts than the older 2048-step full config, giving more frequent updates.
- `update_epochs: 4`: fewer epochs per rollout than the older 10-epoch config, reducing over-updating.
- `minibatch_size: 64`: eight minibatches per rollout.
- `learning_rate: 0.0003`: shared Adam learning rate.
- `clip_coef: 0.2`: standard PPO clipping range.
- `gamma: 0.99`: CartPole discount factor.
- `gae_lambda: 0.95`: standard GAE bias-variance tradeoff.
- `vf_coef: 0.5`: critic loss weight.
- `ent_coef: 0.001`: small entropy bonus for exploration without excessive late stochasticity.
- `max_grad_norm: 0.5`: standard gradient clipping threshold.
- `hidden_dim: 64`: two-layer shared actor-critic MLP width.
- `use_orthogonal_init: true`: standard PPO initialization.
- `normalize_advantages: false`: tuned final setting.
- `use_clip: true`: PPO-Clip enabled.
- `eval_interval: 10000`: deterministic evaluation checkpoint interval.
- `n_eval_episodes: 20`: evaluation episodes per checkpoint.
- `eval_seed_offset: 9999`: keeps evaluation seeds separate from training seeds.
- `prefer_gpu: false`: CPU default for reproducibility on university Linux machines.

## Correct Results Layout

The correct final study layout is:

```text
results/final_study/final/ppo_final/seed_*/
results/final_study/ablations/<variant>/seed_*/
results/final_study/metrics/
results/final_study/plots/
```

A previous runner version accidentally created this nested layout:

```text
results/final_study/final_study/final/ppo_final/seed_*/
results/final_study/final_study/ablations/<variant>/seed_*/
```

The fixer is:

```bash
python scripts/fix_results_layout.py --study-root results/final_study
```

It is safe to run repeatedly. It merges nested files into the correct layout, refuses non-identical overwrites, skips identical files, and renames the old nested directory to `results/final_study_nested_backup` instead of deleting it.

## Experiment Design

Smoke:

- Config: `ppo_final`
- Seeds: `0`
- Steps: `20000`
- Output: `results/smoke/final/ppo_final/seed_0/`

Final:

- Config: `ppo_final`
- Seeds: `0 1 2 3 4`
- Steps: `1000000`
- Output: `results/final_study/final/ppo_final/seed_*/`

Selected:

- Config: `ppo_selected`
- Seeds: `0 1 2 3 4`
- Steps: `1000000`
- Output: `results/final_study/selected/ppo_selected/seed_*/`
- Rationale: post-ablation candidate that keeps PPO clipping and GAE but removes entropy regularization.

Ablations:

- Configs: `ppo_final`, `ppo_no_clip`, `ppo_lambda0`, `ppo_no_entropy`, `ppo_adv_norm_on`, `ppo_single_epoch`
- Seeds: `0 1 2 3 4`
- Steps: `500000`
- Output: `results/final_study/ablations/<exp_name>/seed_*/`

Baselines:

- `previous_results/linear_basic_training.npz`: DQN
- `previous_results/reinforce.npz`: REINFORCE
- `previous_results/ac.npz`: AC
- `previous_results/a2c.npz`: A2C

These baseline files are saved traces from previous assignments. They are not retrained here.

## Idempotent Workflow

`run_all.sh` remains the canonical final-plus-ablation launcher. `ppo_selected` is run explicitly with `--stage selected` so it can be launched and compared after reviewing ablation results. It:

1. Fixes the results layout.
2. Runs the final and ablation experiment plan with skip-existing enabled.
3. Checks completion of required seeds.
4. Recomputes metrics.
5. Regenerates plots.

Completed seed runs are skipped when all four files exist and are non-empty: `train.csv`, `eval.csv`, `update_log.csv`, and `config.json`. `eval.csv` must contain at least two data rows. Metrics and plots are regenerated on every `run_all.sh` execution.

Force reruns with:

```bash
python scripts/run_final_experiments.py --stage all --results-root results --study-name final_study --force
```

## Selected PPO Commands

Extend ablations to five seeds without rerunning existing complete seeds:

```bash
python scripts/run_final_experiments.py --stage ablations --ablation-seeds 0 1 2 3 4 --results-root results --study-name final_study --skip-existing
```

Run PPO Selected for five seeds:

```bash
python scripts/run_final_experiments.py --stage selected --seeds 0 1 2 3 4 --results-root results --study-name final_study --skip-existing
```

Recompute metrics and regenerate plots:

```bash
python scripts/compute_metrics.py --study-root results/final_study --output-dir results/final_study/metrics
python scripts/plot_results.py --study-root results/final_study --output-dir results/final_study/plots
```

## Metrics Definitions

- `AULC_200k`: area under the return curve up to 200k environment steps, normalized by `500 * 200000`.
- `t_475`: first environment step where return reaches at least 475.
- `solve_rate`: fraction of seeds with finite `t_475`.
- `final_eval_return`: final PPO deterministic evaluation return; for baselines, mean of the final saved training returns.
- `post_solve_worst_return`: worst return after first reaching 475.
- `post_solve_retention`: fraction of post-solve checkpoints with return at least 475.
- `late_100k_worst_return`: worst return in the final 100k observed environment steps.
- `late_100k_retention`: fraction of final-100k checkpoints with return at least 475.
- Diagnostics: mean/max approximate KL, mean/max clip fraction, final entropy, and max value loss.

Known caveat: PPO can show transient post-solve dips on CartPole. For that reason, both `post_solve_worst_return` and late-window stability metrics are reported rather than relying only on final return.

## Plot List

- `final_learning_curves_comparison.pdf/png`: PPO_final versus DQN, REINFORCE, AC, and A2C.
- `ppo_ablation_curves.pdf/png`: PPO_final versus PPO ablations.
- `ppo_diagnostics.pdf/png`: approximate KL, clip fraction, entropy, and value loss.
- `ppo_stability_summary.pdf/png`: solve time, final return, post-solve worst return, and late-100k worst return.
- `figure_captions.txt`: self-contained captions for all figures.

## Exact Current State Of The Clean Repo

Retained source/data files:

- `README.md`
- `requirements.txt`
- `CONTEXT.md`
- `configs/*.json` for the final config family
- `rl_a4/*.py`
- `scripts/train_ppo.py`
- `scripts/run_final_experiments.py`
- `scripts/fix_results_layout.py`
- `scripts/check_results_complete.py`
- `scripts/compute_metrics.py`
- `scripts/plot_results.py`
- `previous_results/*.npz`
- `results/.gitkeep`
- `run_all.sh`
- `logs/.gitkeep`

Intentionally excluded from the clean submission state:

- Old figures from the parent repo.
- Old broad runner scripts.
- Unused `main.py`.
- Duplicate configs not used in final experiments.
- Extra markdown/report files outside `README.md` and `CONTEXT.md`.
- Python cache directories.
