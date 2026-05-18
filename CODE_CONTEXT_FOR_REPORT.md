# Code Context For Report

This document is tied only to code and artifacts present in this repository on inspection. If a claim is uncertain, the uncertainty is stated explicitly.

## A. Repository overview

### Important file tree

```text
rl_ppo_cartpole/
├── configs/
│   ├── ppo_full.json
│   ├── ppo_no_clip.json
│   ├── ppo_lambda0.json
│   ├── ppo_no_adv_norm.json
│   ├── ppo_single_epoch.json
│   └── ppo_tuned.json
├── rl_a4/
│   ├── buffers.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── networks.py
│   ├── plotting.py
│   ├── ppo.py
│   └── utils.py
├── scripts/
│   ├── compute_metrics.py
│   ├── plot_results.py
│   ├── run_ablation_study.py
│   ├── run_all.py
│   └── train_ppo.py
├── previous_results/
│   ├── a2c.npz
│   ├── ac.npz
│   ├── linear_basic_training.npz
│   └── reinforce.npz
├── results/
│   ├── ablation_study/
│   ├── metrics.csv
│   ├── metrics_latex.tex
│   ├── ppo_full/
│   ├── ppo_lambda0/
│   ├── ppo_no_adv_norm/
│   ├── ppo_no_clip/
│   └── ppo_tuned/
├── figures/
│   ├── main_comparison.pdf/.png
│   ├── ppo_ablation.pdf/.png
│   ├── ppo_diagnostics.pdf/.png
│   ├── ppo_eval.pdf
│   └── ppo_stability.pdf/.png
├── README.md
├── notes.md
├── requirements.txt
└── RL_assignment_4.pdf
```

### What each important file does

- `scripts/train_ppo.py`: main training entry point; loads config, seeds RNGs, trains PPO, logs train/eval/update CSVs.
- `scripts/run_all.py`: sequential runner for top-level PPO variants.
- `scripts/run_ablation_study.py`: staged study runner for `ablations`, `sweep`, and `final`; also writes per-stage summaries.
- `scripts/compute_metrics.py`: computes `metrics.csv` and `metrics_latex.tex` from the latest staged study plus previous Assignment 2/3 `.npz` baselines.
- `scripts/plot_results.py`: regenerates figures from the latest staged study plus previous Assignment 2/3 `.npz` baselines.
- `rl_a4/ppo.py`: PPO update rule and optimizer step.
- `rl_a4/networks.py`: shared-backbone actor-critic network and action distribution.
- `rl_a4/buffers.py`: rollout storage, GAE, return construction, minibatch iterator.
- `rl_a4/evaluate.py`: deterministic greedy evaluation.
- `rl_a4/metrics.py`: AULC, `t_475`, solve rate, and LaTeX table formatting.
- `rl_a4/plotting.py`: figure generation and seed aggregation logic.
- `rl_a4/utils.py`: config I/O, device selection, seeding, env factory.
- `configs/*.json`: experiment hyperparameters and ablation flags.
- `notes.md`: human-written explanation of implemented PPO engineering tricks.
- `previous_results/*.npz`: saved Assignment 2/3 baseline results used for comparison; no training code for those baselines was inspected in this repo.

### Entry points

- Training: `scripts/train_ppo.py`
- Evaluation during training: `rl_a4.evaluate.evaluate_policy`
- Experiment runners: `scripts/run_all.py`, `scripts/run_ablation_study.py`
- Metrics computation: `scripts/compute_metrics.py`
- Plot generation: `scripts/plot_results.py`
- Config loading: `rl_a4.utils.load_config`

### Where outputs are saved

- Per-run PPO logs: `results/<exp_name>/seed_<seed>/`
- Latest staged study logs: `results/ablation_study/<study_name>/<stage>/<exp_name>/seed_<seed>/`
- Per-seed files:
  - `train.csv`
  - `eval.csv`
  - `update_log.csv`
  - `config.json`
- Stage summaries:
  - `results/ablation_study/<study_name>/<stage>/summary.csv`
  - `results/ablation_study/<study_name>/summary_all.csv`
- Repo-level metric outputs:
  - `results/metrics.csv`
  - `results/metrics_latex.tex`
- Figures:
  - `figures/main_comparison.pdf`, `figures/main_comparison.png`
  - `figures/ppo_ablation.pdf`, `figures/ppo_ablation.png`
  - `figures/ppo_diagnostics.pdf`, `figures/ppo_diagnostics.png`
  - `figures/ppo_eval.pdf`
  - `figures/ppo_stability.pdf`, `figures/ppo_stability.png`
- Checkpoints: absent. No model weights are saved anywhere in inspected code.

## B. Algorithm choice and exact implementation

### What algorithm is implemented?

- Implemented algorithm: PPO only.
- SAC: absent.
- Exact PPO variant: PPO-Clip.

Evidence:

- `rl_a4/ppo.py` documents and implements the clipped surrogate objective.
- No replay buffer, target critics, soft-Q losses, or entropy temperature tuning code were found anywhere in `rl_a4/` or `scripts/`.

### Network structure

- Actor and critic share a two-layer MLP backbone with separate output heads.
- Shared body:
  - `rl_a4/networks.py:65-71`
- Actor head:
  - `rl_a4/networks.py:73-75`
- Critic head:
  - `rl_a4/networks.py:76-77`
- Activation: `Tanh`
- Hidden width: `64` by default
- Action distribution for CartPole: `Categorical(logits=logits)`
  - `rl_a4/networks.py:99-105`, `113-118`

### Exact mathematical losses

Let:

- `s_t` be the observation
- `a_t` be the sampled action
- `A_t` be the advantage from GAE
- `R_t` be the return target
- `V_\theta(s_t)` be the critic prediction
- `\log \pi_\theta(a_t|s_t)` be the current log probability
- `\log \pi_{\text{old}}(a_t|s_t)` be the stored rollout log probability
- `r_t(\theta) = \exp(\log \pi_\theta(a_t|s_t) - \log \pi_{\text{old}}(a_t|s_t))`

Policy loss:

\[
L_{\text{policy}} =
\begin{cases}
-\mathbb{E}\left[\min\left(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right], & \text{if } use\_clip=True \\
-\mathbb{E}\left[r_t(\theta) A_t\right], & \text{if } use\_clip=False
\end{cases}
\]

Code mapping:

- `log_prob_new`: `rl_a4/ppo.py:146`
- `lp_old_mb`: minibatch of stored old log-probs from `RolloutBuffer`
- `log_ratio = log_prob_new - lp_old_mb`: `rl_a4/ppo.py:149`
- `ratio = log_ratio.exp()`: `rl_a4/ppo.py:150`
- `surr1 = ratio * adv_mb`: `rl_a4/ppo.py:158`
- `surr2 = clamp(ratio, 1-eps, 1+eps) * adv_mb`: `rl_a4/ppo.py:161`
- `policy_loss = -min(surr1, surr2).mean()`: `rl_a4/ppo.py:162`
- unclipped ablation: `rl_a4/ppo.py:163-165`

Critic loss:

\[
L_{\text{value}} = \frac{1}{2}\mathbb{E}\left[(R_t - V_\theta(s_t))^2\right]
\]

Code mapping:

- `ret_mb`: return target minibatch
- `value_new`: critic prediction minibatch
- `value_loss = 0.5 * ((ret_mb - value_new) ** 2).mean()`: `rl_a4/ppo.py:167-169`

Entropy term:

\[
H = \mathbb{E}\left[\mathcal{H}(\pi_\theta(\cdot|s_t))\right]
\]

Code mapping:

- `entropy = dist.entropy()`: `rl_a4/networks.py:104`, `117`
- `entropy_bonus = entropy.mean()`: `rl_a4/ppo.py:171-172`

Full combined optimization objective actually minimized:

\[
L_{\text{total}} = L_{\text{policy}} + c_v L_{\text{value}} - c_e H
\]

with:

- `c_v = vf_coef`
- `c_e = ent_coef`

Code:

- `loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus`
  - `rl_a4/ppo.py:175-179`

### How clipping is implemented

- PPO clipping is applied to `ratio`, not directly to logits.
- Clip interval: `[1 - clip_coef, 1 + clip_coef]`
- Default `clip_coef = 0.2`
- Code: `torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)`
  - `rl_a4/ppo.py:161`

### KL / clip diagnostics / early stopping

- Approximate KL is computed and logged:
  - `approx_kl = ((ratio - 1) - log_ratio).mean().item()`
  - `rl_a4/ppo.py:153-155`
- Clip fraction is computed and logged:
  - `clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()`
  - `rl_a4/ppo.py:153-155`
- Target-KL early stopping: absent.
- Explained variance: absent.

### Old log-prob handling

- Old log-probs are collected under `@torch.no_grad()` and stored as Python floats in the rollout buffer:
  - `rl_a4/ppo.py:87-103`
  - `rl_a4/buffers.py:84-105`
- During update they are converted to tensors and treated as constants:
  - `rl_a4/buffers.py:146-152`
- This is correct for PPO; the old policy is not recomputed using updated weights.

## C. Relation to A2C / Assignment 3

### What stays the same as A2C

- Shared actor-critic structure with a policy head and value head.
- Advantage-based policy gradient update.
- Critic trained as a value baseline.
- Entropy bonus for exploration.

### What changes relative to basic A2C

- PPO replaces the plain policy-gradient objective with a clipped surrogate objective.
- PPO reuses one rollout for multiple epochs, whereas basic A2C typically updates once per batch.
- PPO shuffles rollout data into minibatches before each update.
- PPO uses GAE explicitly with configurable `\lambda`.
- PPO logs trust-region-style diagnostics (`approx_kl`, `clip_fraction`) that basic AC/A2C implementations often omit.

### Report-ready sentences

- “Our implementation keeps the standard actor-critic decomposition from Assignment 3, but replaces the plain advantage-weighted policy update with PPO’s clipped surrogate objective.”
- “Relative to A2C, PPO improves data efficiency by taking multiple minibatch SGD passes over the same on-policy rollout while constraining policy drift through probability-ratio clipping.”
- “The critic still provides a value baseline, but the advantage target is upgraded from a simple TD estimate to generalized advantage estimation, which trades variance against bias through the parameter `\lambda`.”
- “In that sense, PPO can be viewed as a stabilized, multi-epoch actor-critic method rather than a completely different family of algorithms.”

## D. Pseudocode

```text
for each rollout:
    collect rollout_steps transitions with sampled Categorical actions; store obs, action, reward, done, terminated, old_log_prob, value, next_value
    compute GAE advantages backward using terminated for bootstrap masking and done for recursion stopping
    set returns = advantages + stored_values
    for update_epochs passes:
        shuffle rollout and iterate minibatches
        recompute current log_prob, entropy, and value for stored actions
        compute ratio = exp(log_prob_new - log_prob_old)
        compute policy_loss = -mean(min(ratio*A, clamp(ratio,1-eps,1+eps)*A)) or -mean(ratio*A) if no_clip
        minimize policy_loss + vf_coef*0.5*MSE(return, value) - ent_coef*mean(entropy); clip gradients; log KL and clip fraction
```

## E. Engineering tricks audit

See also the standalone `ENGINEERING_TRICKS_TABLE.md`. Summary:

| Trick | Present? | Where in code | Exact hyperparameter/value | Why it is used | How to describe it in report | Reference needed? |
|---|---|---|---|---|---|---|
| Minibatching | yes | `rl_a4/buffers.py:155-179` | default `256` | lower-variance SGD and data reuse | rollout is shuffled and split into minibatches each epoch | yes |
| Multiple epochs on same rollout | yes | `rl_a4/ppo.py:141-144` | default `10` | improves sample efficiency | same on-policy batch is reused for several passes | yes |
| GAE | yes | `rl_a4/buffers.py:108-130` | default `0.95` | bias-variance tradeoff for advantage estimates | backward recursion over TD residuals | yes |
| Lambda sweep / ablation | yes | `configs/ppo_lambda0.json`, `scripts/run_ablation_study.py` | `0.0`, `0.9`, `0.95`, `0.98` in study runner | test GAE sensitivity | compare one-step TD against standard GAE | no |
| Advantage normalization | yes | `rl_a4/buffers.py:142-145` | on by default, off in `ppo_no_adv_norm` and `ppo_tuned` | stabilizes policy gradient scale | normalize per rollout, not globally | yes |
| Reward normalization/scaling/clipping | absent | no code found | absent | not used | explicitly state absent in from-scratch CartPole setup | no |
| Observation normalization | absent | no code found | absent | not used | explicitly state absent | no |
| Entropy regularization | yes | `rl_a4/ppo.py:171-179` | default `0.01`, tuned `0.001` | prevents premature deterministic collapse | subtract `ent_coef * entropy` from total loss | yes |
| Value-function loss coefficient | yes | `rl_a4/ppo.py:175-179` | `vf_coef = 0.5` | balances policy and critic terms | weighted MSE critic loss | yes |
| Gradient clipping | yes | `rl_a4/ppo.py:183-188` | `max_grad_norm = 0.5` | guards against unstable updates | clip global gradient norm before Adam step | yes |
| Learning-rate scheduling | absent | no scheduler found | fixed `3e-4` | not used | state fixed Adam learning rate | no |
| Rollout horizon | yes | `scripts/train_ppo.py:55-73`, configs | default `2048`, tuned `512` | controls on-policy batch size | one rollout collected before each PPO update phase | no |
| Number of environments / vectorization | absent | no vector env code found | `1` env | simple single-env from-scratch setup | state that collection is single-threaded and sequential | no |
| Time-limit / truncation handling | yes | `scripts/train_ppo.py:176-189`, `rl_a4/buffers.py:117-126` | bootstrap through truncation, stop recursion across reset | distinguishes MDP terminal from time-limit boundary | terminated controls bootstrap; done controls recursion boundary | yes |
| Bootstrap value for truncated transitions | yes | `scripts/train_ppo.py:178` | `next_value = agent.get_value(next_obs)` unless `terminated` | preserves bootstrap at truncation | only true terminals zero out bootstrap | yes |
| Seeding | partial | `rl_a4/utils.py:22-40`, `scripts/train_ppo.py:104`, `155` | seeds Python, NumPy, Torch, action space, observation space, initial train reset | reproducibility | training seed path exists; evaluation seeding is weaker, see risks | no |
| Deterministic evaluation | yes | `rl_a4/evaluate.py:43-66` | greedy `argmax`, `20` episodes, every `10,000` steps | cleaner measurement than stochastic train return | training is stochastic, evaluation is greedy | yes |
| Smoothing in plots | yes | `rl_a4/plotting.py:66`, `75-115`, `156-190` | moving average window `25` for train/baseline curves | reduce visual noise | only train-like curves are smoothed; PPO eval curves are interpolated raw | no |
| Averaging over seeds | yes | `rl_a4/plotting.py`, `rl_a4/metrics.py` | 3 or 5 seeds depending on study | robustness and uncertainty | mean curves with ±1 std, or box plots for per-seed stability | no |
| Target KL / early stopping | absent | no code found | absent | not used | report as not implemented | no |
| Checkpointing weights | absent | no code found | absent | not used | config and CSV logs only; no model snapshots | no |
| Device selection | yes | `rl_a4/utils.py:56-60` | CPU default, GPU if `prefer_gpu` and available | optional acceleration | simple runtime device choice | no |
| Dtype handling | yes | `float32` obs/returns/advantages/log-probs, `int64` actions | `rl_a4/buffers.py:61-68`, `146-150`, `rl_a4/ppo.py:97`, `109`, `rl_a4/evaluate.py:54` | standard Torch PPO types | mention only if space allows | no |

## F. Hyperparameters and config table

### Defaults and variants

| Algorithm/config | total steps | seeds | gamma | lambda | lr | rollout | minibatch | epochs | clip eps | ent coef | vf coef | max grad norm | architecture | activation | optimizer | eval frequency | eval episodes | smoothing | flags |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---|
| `ppo_full` | 1,000,000 | `0-4` top-level, `0-2` staged ablation | 0.99 | 0.95 | 3e-4 | 2048 | 256 | 10 | 0.2 | 0.01 | 0.5 | 0.5 | shared 2x64 MLP + actor/critic heads | `Tanh` | Adam `eps=1e-5` | 10,000 steps | 20 | 25 in plots | `use_clip=True`, `normalize_advantages=True`, `use_orthogonal_init=True` |
| `ppo_no_clip` | 1,000,000 | `0-4` top-level, `0-2` staged ablation | 0.99 | 0.95 | 3e-4 | 2048 | 256 | 10 | 0.2 | 0.01 | 0.5 | 0.5 | same | `Tanh` | Adam | 10,000 | 20 | 25 | `use_clip=False` |
| `ppo_lambda0` | 1,000,000 | `0-4` top-level, `0-2` staged ablation | 0.99 | 0.0 | 3e-4 | 2048 | 256 | 10 | 0.2 | 0.01 | 0.5 | 0.5 | same | `Tanh` | Adam | 10,000 | 20 | 25 | one-step TD-style advantages |
| `ppo_no_adv_norm` | 1,000,000 | `0-4` top-level, `0-2` staged ablation | 0.99 | 0.95 | 3e-4 | 2048 | 256 | 10 | 0.2 | 0.01 | 0.5 | 0.5 | same | `Tanh` | Adam | 10,000 | 20 | 25 | `normalize_advantages=False` |
| `ppo_single_epoch` | 1,000,000 | `0-2` staged ablation | 0.99 | 0.95 | 3e-4 | 2048 | 256 | 1 | 0.2 | 0.01 | 0.5 | 0.5 | same | `Tanh` | Adam | 10,000 | 20 | 25 | single PPO pass |
| `ppo_tuned` | 1,000,000 | `0-4` top-level | 0.99 | 0.95 | 3e-4 | 512 | 64 | 4 | 0.2 | 0.001 | 0.5 | 0.5 | same | `Tanh` | Adam | 10,000 | 20 | 25 | `normalize_advantages=False` |

### CLI arguments

`scripts/train_ppo.py`

- `--config`
- `--seed`
- `--total-env-steps`
- `--exp-name`
- `--results-dir`

`scripts/run_all.py`

- `--seeds`
- `--smoke`
- `--exp`

`scripts/run_ablation_study.py`

- `--stage`
- `--base-config`
- `--final-config`
- `--seeds`
- `--ablation-seeds`
- `--sweep-seeds`
- `--final-seeds`
- `--smoke`
- `--summarize-only`
- `--python-bin`
- `--results-root`
- `--study-name`

### Config file options actually read by training

- `exp_name`
- `env_id`
- `total_env_steps`
- `rollout_steps`
- `update_epochs`
- `minibatch_size`
- `learning_rate`
- `clip_coef`
- `gamma`
- `gae_lambda`
- `vf_coef`
- `ent_coef`
- `max_grad_norm`
- `hidden_dim`
- `use_orthogonal_init`
- `normalize_advantages`
- `use_clip`
- `eval_interval`
- `n_eval_episodes`
- `prefer_gpu`
- `results_dir`

## G. Experiment design actually implemented

### What can be run now

- Single PPO run via `scripts/train_ppo.py`
- Top-level batch of main variants via `scripts/run_all.py`
- Staged study via `scripts/run_ablation_study.py`
  - `ablations`: baseline, no_clip, lambda0, no_adv_norm, single_epoch
  - `sweep`: one-factor sweep over clip coefficient, GAE lambda, rollout steps, update epochs
  - `final`: baseline vs tuned config

### What has already produced results

- Complete top-level 5-seed folders:
  - `results/ppo_full`
  - `results/ppo_no_clip`
  - `results/ppo_lambda0`
  - `results/ppo_no_adv_norm`
  - `results/ppo_tuned`
- Complete staged ablation results:
  - `results/ablation_study/full/ablations/*`
- Partial staged sweep:
  - only `results/ablation_study/full/sweep/clip_0p1/seed_0`
  - and `results/ablation_study/full/sweep/clip_0p1/seed_1`
- Staged final comparison:
  - absent
- Stale older staged folders:
  - `results/ablation_study/full/core_tricks`
  - `results/ablation_study/dry_run/core_tricks`

### Fairness of PPO vs previous baselines

- The intended comparison budget is matched at 1,000,000 environment steps.
- However, the comparison is not perfectly apples-to-apples:
  - PPO metrics use deterministic evaluation CSVs.
  - previous baselines are loaded from `.npz` episode-return traces and treated as “eval-like” in `rl_a4/metrics.py`.
  - `t_475` for baselines is computed on a 25-episode moving average of training return, not on deterministic evaluation checkpoints.
- This is usable for a student report, but should be disclosed as a limitation.

### Logged metrics

Per run:

- training episodic return
- deterministic evaluation mean return
- deterministic evaluation std return
- policy loss
- value loss
- entropy
- approximate KL
- clip fraction
- wall-clock SPS printed to terminal only, not saved

Aggregated metrics:

- `final_eval_return_mean`
- `final_eval_return_std`
- `final_train_return_mean`
- `final_train_return_std`
- `AULC_1M`
- `AULC_200k`
- `t_475_mean`
- `t_475_std`
- `solve_rate`
- `post_solve_min_return`

Missing but potentially useful:

- explained variance
- clipped value loss diagnostics
- gradient norm logging
- wall-clock time saved to file
- confidence intervals or standard error

### Generated plots

- `main_comparison`: previous baselines plus latest PPO
- `ppo_ablation`: five PPO ablations
- `ppo_eval`: latest PPO deterministic eval curve
- `ppo_diagnostics`: KL, clip fraction, entropy, value loss
- `ppo_stability`: per-seed box/scatter stability summaries

### X-axis and uncertainty

- PPO curves are over environment steps.
- Previous baseline curves are also plotted over environment steps from saved `.npz` cumulative step arrays.
- PPO uncertainty: mean ± 1 standard deviation over seeds.
- Previous baselines: mean ± 1 standard deviation over interpolated seed curves.

## H. Results inventory

### Current usable artifacts

| Path / pattern | Experiment | Config / algorithm | Seeds present | Total steps | Metrics/data present | Usable for final report? | Caveats |
|---|---|---|---|---:|---|---|---|
| `results/ppo_full/seed_{0..4}/` | top-level baseline PPO | `ppo_full` | 0,1,2,3,4 | 1,000,000 | `train.csv`, `eval.csv`, `update_log.csv`, `config.json` | yes | older output generation than staged study; not the “latest study” path |
| `results/ppo_no_clip/seed_{0..4}/` | top-level no-clip ablation | `ppo_no_clip` | 0,1,2,3,4 | 1,000,000 | same | yes | same caveat |
| `results/ppo_lambda0/seed_{0..4}/` | top-level lambda0 ablation | `ppo_lambda0` | 0,1,2,3,4 | 1,000,000 | same | yes | same caveat |
| `results/ppo_no_adv_norm/seed_{0..4}/` | top-level no-adv-norm ablation | `ppo_no_adv_norm` | 0,1,2,3,4 | 1,000,000 | same | yes | same caveat |
| `results/ppo_tuned/seed_{0..4}/` | top-level tuned PPO | `ppo_tuned` | 0,1,2,3,4 | 1,000,000 | same | yes | not part of completed staged `final/` folder |
| `results/ablation_study/full/ablations/ppo_full/seed_{0..2}/` | staged ablation baseline | `ppo_full` | 0,1,2 | 1,000,000 | same | yes | latest staged ablation baseline |
| `results/ablation_study/full/ablations/ppo_no_clip/seed_{0..2}/` | staged ablation no-clip | `ppo_no_clip` | 0,1,2 | 1,000,000 | same | yes | latest staged ablation |
| `results/ablation_study/full/ablations/ppo_lambda0/seed_{0..2}/` | staged ablation lambda0 | `ppo_lambda0` | 0,1,2 | 1,000,000 | same | yes | latest staged ablation |
| `results/ablation_study/full/ablations/ppo_no_adv_norm/seed_{0..2}/` | staged ablation no-adv-norm | `ppo_no_adv_norm` | 0,1,2 | 1,000,000 | same | yes | latest staged ablation |
| `results/ablation_study/full/ablations/ppo_single_epoch/seed_{0..2}/` | staged ablation single epoch | `ppo_single_epoch` | 0,1,2 | 1,000,000 | same | yes | latest staged ablation |
| `results/ablation_study/full/ablations/summary.csv` | staged ablation summary | aggregated PPO ablations | 3 seeds each | 1,000,000 | AULC, `t_475`, solve rate, final returns | yes | strongest current summary table |
| `results/ablation_study/full/sweep/clip_0p1/seed_{0,1}/` | staged sweep partial | `clip_coef=0.1` | 0,1 | 1,000,000 | same | not yet | incomplete variant and no stage summary |
| `results/ablation_study/full/final/` | staged final comparison | baseline vs tuned | absent | n/a | absent | no | latest staged final comparison not completed |
| `results/metrics.csv` | repo-level metrics table | latest staged study + previous baselines | mixed | mixed | aggregated table | partially | current file omits `PPO_full` and `PPO_tuned` because latest staged `final/` is absent |
| `results/metrics_latex.tex` | LaTeX table | same as above | mixed | mixed | LaTeX tabular | partially | same caveat |
| `figures/main_comparison.*` | main comparison plot | baselines + latest PPO | n/a | mixed | figure only | uncertain | may predate the current “latest study only” plotting logic; file timestamp not inspected here |
| `figures/ppo_ablation.*` | ablation plot | five PPO ablations | n/a | 1,000,000 | figure only | yes if regenerated | current code expects latest staged ablations |
| `figures/ppo_diagnostics.*` | diagnostics plot | five PPO ablations | n/a | 1,000,000 | figure only | yes if regenerated | current code expects latest staged ablations |
| `figures/ppo_eval.pdf` | latest PPO eval curve | latest staged final PPO | n/a | mixed | figure only | not yet | current code expects staged `final/`; absent final means this may not regenerate now |
| `figures/ppo_stability.*` | stability plot | five PPO ablations | n/a | 1,000,000 | figure only | yes if regenerated | current code expects latest staged ablations |
| `previous_results/a2c.npz` | Assignment 3 baseline | A2C | 5 | saved step arrays | rewards, steps | yes | no deterministic eval logs |
| `previous_results/ac.npz` | Assignment 3 baseline | AC | 5 | saved step arrays | rewards, steps | yes | no deterministic eval logs |
| `previous_results/reinforce.npz` | Assignment 2/3 baseline | REINFORCE | 5 | saved step arrays | rewards, steps | yes | no deterministic eval logs |
| `previous_results/linear_basic_training.npz` | Assignment 2 baseline | DQN | 5 | saved step arrays | rewards, steps | yes | no deterministic eval logs |

### Result artifact caveats

- There are stale outputs from an older study layout (`core_tricks`) alongside the new `ablations/sweep/final` layout.
- The latest staged study is incomplete: `final/` is missing and `sweep/` is incomplete.
- `metrics.csv` currently reflects that incomplete latest study, so it is not a full final-report table.

## I. Reproducibility

### Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Single main PPO run:

```bash
python scripts/train_ppo.py --config configs/ppo_full.json --seed 0
```

Quick smoke test:

```bash
python scripts/train_ppo.py --config configs/ppo_full.json --seed 0 --total-env-steps 5000
```

Top-level PPO suite:

```bash
python scripts/run_all.py
```

Staged ablations only:

```bash
python scripts/run_ablation_study.py --stage ablations
```

Staged sweep only:

```bash
python scripts/run_ablation_study.py --stage sweep
```

Staged final comparison:

```bash
python scripts/run_ablation_study.py --stage final --final-config configs/ppo_tuned.json
```

Full staged study:

```bash
python scripts/run_ablation_study.py --stage all
```

Recompute metrics:

```bash
python scripts/compute_metrics.py
```

Regenerate plots:

```bash
python scripts/plot_results.py
```

### Missing or fragile README instructions

- `README.md:20` still says “4 variants × 5 seeds”, but `run_all.py` now includes six variants including `ppo_single_epoch` and `ppo_tuned`.
- `README.md:50-54` omits `ppo_single_epoch.json` and `ppo_tuned.json`.
- `README.md:87-92` omits the newer ablation and tuned variants.
- `README.md` does not explain the staged study runner `scripts/run_ablation_study.py`.
- Previous baselines are only available as `.npz` result files here; commands to rerun DQN/REINFORCE/AC/A2C were not found in this repo.

## J. Code correctness and risks

| Risk | Status | File/function | Evidence | Recommendation |
|---|---|---|---|---|
| PPO ratio uses stored old log-probs, not updated policy | OK | `rl_a4/ppo.py:update`, `rl_a4/buffers.py:add` | old log-probs stored at collection time, reused in update | describe this explicitly in report |
| GAE recursion with terminated vs truncated | OK | `rl_a4/buffers.py:108-130`, `scripts/train_ppo.py:176-189` | bootstrap mask uses `terminated`; continuation mask uses `done` | strong implementation detail to highlight |
| Bootstrap target at true terminal only | OK | `scripts/train_ppo.py:178` | `next_value = 0.0 if terminated else agent.get_value(next_obs)` | mention this as correct Gymnasium handling |
| Training env seeding | OK | `scripts/train_ppo.py:155`, `rl_a4/utils.py:22-40` | first `env.reset(seed=seed)` plus global Torch/NumPy seeds | acceptable |
| Evaluation env reproducibility | questionable | `rl_a4/evaluate.py:48-49`, `rl_a4/utils.py:33-40` | fresh envs call `reset()` without a seed; `make_env` seeds spaces but not env RNG | seed `reset(seed=...)` in evaluation if strict reproducibility is needed |
| Training/eval contamination | OK | `rl_a4/evaluate.py:43-66` | separate env instances, `torch.no_grad()`, no training ops | no issue found |
| Eval mode handling | OK | `rl_a4/evaluate.py:43`, `63` | `net.eval()` then `net.train()` | correct, though no dropout/bn are present |
| Old advantages detached from graph | OK | `rl_a4/buffers.py:142-150` | advantages computed in NumPy before Torch tensors | no issue found |
| Off-by-one in environment step budget | OK | `scripts/train_ppo.py:171-173`, `191` | step count increments once per env step and loop breaks at budget | no issue found |
| Plot text says 5 seeds for ablation plot | questionable | `rl_a4/plotting.py:457-459` | hardcoded “5 seeds” text, latest staged ablations have 3 seeds | update text before final submission |
| Latest-study plot/metrics logic with incomplete `final/` | questionable | `scripts/compute_metrics.py`, `scripts/plot_results.py` | current latest study lacks `final/`, so `PPO_full` and `PPO_tuned` are omitted or not plotted | finish `final/` stage or point scripts to complete top-level results for reporting |
| Fairness of PPO vs previous baselines | questionable | `rl_a4/metrics.py`, `rl_a4/plotting.py` | PPO uses deterministic eval logs; baselines use smoothed training returns from `.npz` | disclose as limitation in report |
| Equal step budgets across compared PPO variants | OK | configs and stored `config.json` | all inspected PPO result folders use `1,000,000` total steps | no issue found |
| Vectorized environment support | absent | no code found | single env only | acceptable for from-scratch CartPole |
| Reward/observation normalization | absent | no code found | absent | acceptable if explicitly stated |

## K. Report positioning

### Possible research questions

1. “Does PPO-Clip improve policy-update stability over a comparable actor-critic baseline by constraining the policy ratio while reusing on-policy rollouts?”
2. “Which PPO engineering choices matter most on CartPole-v1: clipping, GAE, advantage normalization, or repeated update epochs?”
3. “Can a lightly tuned PPO configuration improve early sample efficiency over the default PPO setup and over previous Assignment 2/3 baselines under a fixed 1M-step budget?”

### Strongest question given current code and artifacts

Question 2 is the strongest right now because:

- the code and staged ablation runner are centered on engineering-trick ablations;
- `results/ablation_study/full/ablations/summary.csv` is complete and already supports the claim;
- the staged `final/` tuned comparison is not yet available in the latest study checkout.

If you complete the staged `final/` folder, Question 3 becomes the strongest end-to-end paper question.

## L. Suggested report structure under 4 pages

### Section plan

- Abstract: 120-150 words
- Introduction: 300-400 words
- Theory: 500-650 words
- Experiments: 700-900 words
- Discussion: 250-350 words
- Conclusion: 120-180 words

### Recommended figures/tables

- Include:
  - one main comparison figure: previous baselines vs PPO
  - one PPO ablation figure
  - one compact metrics table with `AULC_200k`, `t_475`, solve rate, final return
  - optionally one compact diagnostics figure if space remains
- Omit:
  - too many overlapping sweep learning curves
  - redundant train and eval versions of the same PPO curve
  - any figure whose source is incomplete in the latest staged study

## M. Draft report bullets

### Introduction motivation

- PPO is a modern actor-critic method designed to stabilize policy-gradient updates without the complexity of second-order trust-region methods.
- Relative to Assignment 3 actor-critic methods, PPO keeps the actor-critic structure but adds a clipped surrogate objective that constrains how far the new policy may move from the behavior policy used to collect data.
- The implementation goal in this repo is to study whether those stabilizing changes improve sample efficiency and robustness on CartPole-v1 under the same 1M-step budget used for earlier assignments.

### Theory explanation

- The actor is trained with the PPO-Clip objective, which uses the probability ratio between the current policy and the stored behavior policy.
- The critic is trained by mean-squared regression to return targets built from generalized advantage estimation.
- Entropy regularization is included in the total loss to prevent premature collapse to a deterministic policy.
- Compared with basic A2C, PPO performs several minibatch SGD passes over the same rollout and relies on clipping to keep those repeated updates stable.

### Experiments setup

- Environment: `CartPole-v1`.
- Network: shared two-layer MLP with 64 hidden units and `Tanh` activations.
- Default PPO hyperparameters: `[gamma]`, `[lambda]`, `[clip epsilon]`, `[rollout steps]`, `[minibatch size]`, `[update epochs]`.
- Main ablations: baseline, no clipping, `lambda=0`, no advantage normalization, single update epoch.
- Comparison baselines: DQN, REINFORCE, AC, and A2C loaded from saved Assignment 2/3 result files.

### Results interpretation placeholders

- PPO reached `[final_eval_return]` with an early-learning score of `[AULC_200k]`.
- The no-clip ablation changed `[approx_kl / clip_fraction / stability]`, supporting the interpretation that PPO’s clipping improves update stability.
- The `lambda=0` ablation affected `[sample efficiency / variance / solve time]`, highlighting the role of GAE in bias-variance control.
- The single-epoch ablation reduced data reuse and led to `[slower solve time / lower AULC]`.
- Compared with Assignment 2/3 baselines, PPO was `[faster/slower]` on `[AULC_200k]` and `[more/less]` stable on `[solve rate / post_solve_min_return]`.

### Discussion limitations

- Previous baselines are compared using saved training-return traces rather than deterministic evaluation checkpoints, so the comparison is informative but not perfectly matched.
- The repo currently contains incomplete staged `sweep` and missing staged `final` outputs, so only completed artifacts should be used in the final report.
- CartPole-v1 is a simple benchmark, so conclusions about PPO’s engineering tricks may not fully transfer to harder continuous-control tasks.

