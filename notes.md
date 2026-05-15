# PPO Engineering Notes

All tricks used in this implementation, their motivation, and where they appear in the code.

---

## 1. Clipped Surrogate Objective

**What it is.**  
The core PPO innovation. The policy loss is:

```
L^CLIP(θ) = E_t [ min(r_t · Â_t,  clip(r_t, 1−ε, 1+ε) · Â_t) ]
```

where `r_t = π_θ(a|s) / π_{θ_old}(a|s)` is the probability ratio.

**Why it matters.**  
Without clipping, multiple gradient steps on the same data can push the policy far from where the data was collected (the distribution shift problem). Once the policy diverges too far, the old advantage estimates become unreliable. The clip acts as a first-order trust region: it removes the incentive to move the ratio outside `[1−ε, 1+ε]`.

**Ablation.**  
`ppo_no_clip` removes this clip, effectively running multiple-epoch policy gradient (like reinventing vanilla PG with minibatches). Compare its learning curves to `ppo_full` to see whether clipping improves stability.

**Code location.**  
`rl_a4/ppo.py` – `PPOAgent.update()`, controlled by `use_clip` flag.

---

## 2. Generalised Advantage Estimation (GAE)

**What it is.**  
GAE (Schulman et al., 2015) interpolates between one-step TD and Monte-Carlo advantage:

```
δ_t = r_t + γ · V(s_{t+1}) · (1−done_t) − V(s_t)
A_t = Σ_{l≥0} (γλ)^l · δ_{t+l}
```

**Why it matters.**  
- `λ=0`: pure one-step TD. Low variance but high bias (V may be wrong).  
- `λ=1`: Monte-Carlo. Unbiased but high variance.  
- `λ=0.95`: practical sweet spot. Reduces variance while keeping reasonable bias.

**Ablation.**  
`ppo_lambda0` sets `gae_lambda=0.0`, approximating one-step actor-critic. This should increase variance of updates, potentially slowing or destabilising learning.

**Code location.**  
`rl_a4/buffers.py` – `RolloutBuffer.compute_gae_and_returns()`.

---

## 3. Multiple Epochs over the Same Rollout

**What it is.**  
After collecting a rollout, the same data is reused for `update_epochs=10` passes of minibatch SGD.

**Why it matters.**  
On-policy methods (REINFORCE, A2C) discard each rollout after a single gradient step, wasting experience. PPO's trust-region constraint makes it safe to take multiple steps on the same data — the clip prevents the policy from straying too far.

**Trade-off.**  
More epochs → better sample efficiency, but with diminishing returns and increased risk of overfitting to one rollout if the clip is too loose.

**Code location.**  
`rl_a4/ppo.py` – `update_epochs` loop in `PPOAgent.update()`.

---

## 4. Minibatch SGD with Random Shuffling

**What it is.**  
Each epoch, the rollout buffer (2048 transitions) is randomly shuffled and split into minibatches of size 256 (8 minibatches per epoch → 80 gradient steps per rollout).

**Why it matters.**  
- Minibatches reduce gradient noise compared to full-batch updates.  
- Shuffling decorrelates consecutive transitions, improving stability.  
- Allows using Adam (which works best with minibatches, not full-batch).

**Code location.**  
`rl_a4/buffers.py` – `RolloutBuffer.minibatch_iter()`.

---

## 5. Advantage Normalisation

**What it is.**  
Per-rollout normalisation of advantages:

```
Â_t ← (Â_t − mean(Â)) / (std(Â) + ε)
```

Applied *after* GAE, *per rollout* (not globally across rollouts).

**Why it matters.**  
- Keeps the effective learning rate stable across rollouts regardless of reward scale.  
- Prevents a single large advantage from dominating the gradient.  
- Improves training stability, especially early in learning.

**Ablation.**  
`ppo_no_adv_norm` disables this. Expect higher variance or slower convergence.

**Code location.**  
`rl_a4/buffers.py` – `RolloutBuffer.get_tensors()`, controlled by `normalize_advantages`.

---

## 6. Entropy Bonus

**What it is.**  
A term added to the loss that encourages a higher-entropy (more uniform) policy:

```
L_total = L^CLIP + vf_coef · L^VF − ent_coef · H(π)
```

**Why it matters.**  
Prevents premature collapse to a deterministic policy. On CartPole the optimal policy is near-deterministic, so `ent_coef=0.01` is small — enough to prevent early collapse without preventing eventual exploitation.

**Code location.**  
`rl_a4/ppo.py` – entropy bonus in `PPOAgent.update()`.

---

## 7. Gradient Clipping

**What it is.**  
After computing gradients, all parameter gradients are clipped to have global L2 norm ≤ `max_grad_norm=0.5`:

```python
nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
```

**Why it matters.**  
Prevents exploding gradients, particularly during early training when the policy is far from optimal and advantage estimates have high variance. This is an unconditional safeguard that complements (but does not replace) the PPO clip.

**Code location.**  
`rl_a4/ppo.py` – just before `optimizer.step()`.

---

## 8. Orthogonal Weight Initialisation

**What it is.**  
Network weights are initialised with orthogonal matrices (standard PyTorch orthogonal init):
- Hidden layers: `std = sqrt(2)` (appropriate for Tanh activations).  
- Actor head: `std = 0.01` (near-uniform initial policy).  
- Critic head: `std = 1.0`.

**Why it matters.**  
Orthogonal init preserves gradient norms through deep networks and tends to give faster early learning. The small actor-head init prevents large initial logit differences, which would bias early action selection.

**Code location.**  
`rl_a4/networks.py` – `_layer_init()`.

---

## 9. Deterministic Evaluation

**What it is.**  
Every 10,000 env steps, the policy is evaluated for 20 episodes using the greedy action `argmax(logits)` (no sampling noise). Mean and std return are recorded.

**Why it matters.**  
Training returns are noisy due to exploration (entropy bonus) and the stochastic Categorical policy. Deterministic evaluation gives a cleaner signal of actual policy quality, essential for fair comparison and reporting.

**Code location.**  
`rl_a4/evaluate.py` – `evaluate_policy()`.

---

## 10. Network Architecture: Shared Backbone + Separate Heads

**Choice.**  
Shared two-layer MLP backbone (64 units, Tanh) feeding into separate actor (logits) and critic (scalar) heads.

**Why shared vs separate.**  
Sharing encourages aligned feature representations between policy and value function (consistent with how A2C uses a shared network). Completely separate networks would require more parameters and are less standard for small environments. The separate heads prevent the value scale from corrupting policy gradients.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Single run (quick smoke test)
python scripts/train_ppo.py --config configs/ppo_full.json --seed 0 --total-env-steps 5000

# Single full run (1M steps, seed 0)
python scripts/train_ppo.py --config configs/ppo_full.json --seed 0

# All experiments (4 variants × 5 seeds)
python scripts/run_all.py

# Generate figures
python scripts/plot_results.py

# Compute metrics table
python scripts/compute_metrics.py
```

## Output Layout

```
results/
  ppo_full/
    seed_0/  train.csv  eval.csv  update_log.csv  config.json
    seed_1/  ...
    ...
  ppo_no_clip/ ...
  ppo_lambda0/ ...
  ppo_no_adv_norm/ ...
  metrics.csv
  metrics_latex.tex
figures/
  main_comparison.pdf / .png
  ppo_ablation.pdf / .png
  ppo_eval.pdf
```
