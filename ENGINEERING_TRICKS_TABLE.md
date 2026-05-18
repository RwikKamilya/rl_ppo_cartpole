# Engineering Tricks Table

This table lists every engineering trick I could verify in code, plus key absent tricks that may need to be stated explicitly in the report.

| Trick | Present? | Where in code | Exact hyperparameter/value | Why it is used | How to describe it in the report | Reference needed? |
|---|---|---|---|---|---|---|
| Clipped surrogate objective | yes | `rl_a4/ppo.py:157-165` | `clip_coef=0.2` by default | constrains policy drift during repeated updates | PPO uses probability-ratio clipping around 1 to create a first-order trust region | yes |
| Minibatching | yes | `rl_a4/buffers.py:155-179` | `minibatch_size=256` default, `64` tuned | improves SGD efficiency and reduces gradient noise | each rollout is shuffled and split into minibatches | yes |
| Multiple epochs over same rollout | yes | `rl_a4/ppo.py:141-144` | `update_epochs=10` default, `1` single-epoch ablation, `4` tuned | better sample efficiency from on-policy data reuse | PPO reuses the same rollout for several SGD passes | yes |
| Generalized Advantage Estimation | yes | `rl_a4/buffers.py:108-130` | `gae_lambda=0.95` default | trades variance against bias in advantage estimation | backward recursion over TD residuals | yes |
| Lambda ablation | yes | `configs/ppo_lambda0.json` | `gae_lambda=0.0` | tests one-step TD-like advantages | compare standard GAE to `\lambda=0` | no |
| Advantage normalization | yes | `rl_a4/buffers.py:142-145` | enabled by default, disabled in `ppo_no_adv_norm` and `ppo_tuned` | stabilizes gradient scale within a rollout | normalize advantages per rollout before PPO updates | yes |
| Reward normalization | absent | no code found | absent | not used | state explicitly that raw CartPole rewards are used | no |
| Reward clipping | absent | no code found | absent | not used | state explicitly absent | no |
| Observation normalization | absent | no code found | absent | not used | state explicitly absent | no |
| Entropy regularization | yes | `rl_a4/ppo.py:171-179` | `ent_coef=0.01` default, `0.001` tuned | encourages exploration and delays policy collapse | entropy is subtracted from the total loss with coefficient `ent_coef` | yes |
| Value-function loss coefficient | yes | `rl_a4/ppo.py:175-179` | `vf_coef=0.5` | balances actor and critic losses | critic MSE is weighted by `vf_coef` | yes |
| Gradient clipping | yes | `rl_a4/ppo.py:183-188` | `max_grad_norm=0.5` | avoids exploding updates | clip global gradient norm before optimizer step | yes |
| Orthogonal initialization | yes | `rl_a4/networks.py:54-58`, `25-30` | hidden `sqrt(2)`, actor `0.01`, critic `1.0` | standard PPO initialization for stable early learning | shared MLP is orthogonally initialized, with small actor-head scale | yes |
| Fixed learning rate | yes | `rl_a4/ppo.py:84`, configs | `3e-4` | simple stable Adam setup | no schedule is used; Adam runs at a fixed rate | no |
| Learning-rate scheduling | absent | no scheduler found | absent | not used | state absent | no |
| Rollout length | yes | `scripts/train_ppo.py:132-138`, configs | default `2048`, tuned `512` | controls batch size and update frequency | collect a fixed number of on-policy steps before updating | no |
| Single environment collection | yes | `scripts/train_ppo.py:112-113` | `1` env | simplest from-scratch setup | data are collected sequentially from one environment instance | no |
| Vectorized environments | absent | no vector env code found | absent | not used | state absent | no |
| Time-limit / truncation handling | yes | `scripts/train_ppo.py:176-189`, `rl_a4/buffers.py:117-126` | separate `done` and `terminated` flags | bootstraps correctly through truncations | terminated controls bootstrap; done stops GAE across resets | yes |
| Bootstrap through truncation | yes | `scripts/train_ppo.py:178` | `next_value = get_value(next_obs)` unless `terminated` | avoids treating truncation as terminal | true terminals only zero out bootstrap | yes |
| Old log-prob storage | yes | `rl_a4/ppo.py:87-103`, `rl_a4/buffers.py:91-103` | stored once at collection time | needed for PPO ratio computation | behavior-policy log-probs are cached in the rollout buffer | yes |
| Approximate KL logging | yes | `rl_a4/ppo.py:152-155` | logged every update | diagnostic for policy drift | KL is monitored but not used for early stopping | yes |
| Clip fraction logging | yes | `rl_a4/ppo.py:152-155` | logged every update | diagnostic for how often clipping activates | useful to interpret trust-region behavior | yes |
| Target KL early stopping | absent | no code found | absent | not used | mention KL is logged only, not enforced | no |
| Explained variance logging | absent | no code found | absent | not used | mention absent if discussing critic diagnostics | no |
| Deterministic evaluation | yes | `rl_a4/evaluate.py:43-66` | greedy `argmax`, `20` episodes every `10k` steps | cleaner performance signal | report eval separately from stochastic training returns | yes |
| Seed averaging | yes | `rl_a4/metrics.py`, `rl_a4/plotting.py` | 3 seeds in ablations, 5 in top-level folders | robustness estimation | aggregate curves and summary metrics over seeds | no |
| Moving-average smoothing | yes | `rl_a4/plotting.py:66`, `75-115`, `156-190` | window `25` | reduces noise in train/baseline curves | note that deterministic PPO eval curves are not smoothed the same way | no |
| Checkpointing model weights | absent | no code found | absent | not used | state that only CSV logs and configs are saved | no |
| Device selection | yes | `rl_a4/utils.py:56-60` | CPU unless `prefer_gpu=True` and CUDA available | optional acceleration | simple runtime device choice | no |
| Dtype handling | yes | `rl_a4/buffers.py:61-68`, `146-150`; `rl_a4/ppo.py:97,109`; `rl_a4/evaluate.py:54` | `float32` tensors, `int64` actions | standard Torch numerical setup | mention only if needed | no |
| Replay buffer | absent | no code found | absent | PPO is on-policy here | explicitly contrast with SAC/DQN-style replay | no |
| Target networks | absent | no code found | absent | not needed for PPO | state absent | no |

