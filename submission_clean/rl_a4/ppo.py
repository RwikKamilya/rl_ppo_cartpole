"""
ppo.py – PPO-clipped update logic.

Proximal Policy Optimisation (Schulman et al., 2017) addresses the instability
of vanilla policy gradient and A2C by constraining how much the policy can
change in a single gradient step.  The clipped surrogate objective:

    L^CLIP(θ) = E_t [ min(r_t(θ)·Â_t,  clip(r_t(θ), 1-ε, 1+ε)·Â_t) ]

where r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)  (probability ratio).

The clip removes the gradient incentive to move the ratio outside [1-ε, 1+ε],
bounding each update to a "trust region" without the second-order optimisation
of TRPO.

Total loss:
    L = -L^CLIP  +  vf_coef · L^VF  −  ent_coef · H

where L^VF = 0.5 · MSE(V(s), R_t)  and  H = E[entropy of π].

Engineering tricks implemented here:
  1. Clipped surrogate objective (core PPO trick).
  2. Multiple epochs over the same rollout data.
  3. Minibatch SGD with random shuffling each epoch.
  4. Gradient clipping (max_grad_norm=0.5) for stability.
  5. Entropy bonus to maintain exploration.
  6. Approximate KL divergence and clip fraction logged as diagnostics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List

from rl_a4.networks import ActorCriticNet
from rl_a4.buffers import RolloutBuffer


class PPOAgent:
    """PPO-clipped agent.

    Parameters
    ----------
    obs_dim : int
    num_actions : int
    hidden_dim : int
    lr : float            Learning rate for Adam.
    clip_coef : float     ε in the clipped surrogate (default 0.2).
    vf_coef : float       Critic loss coefficient.
    ent_coef : float      Entropy bonus coefficient.
    max_grad_norm : float Gradient clipping threshold.
    use_clip : bool       If False, run unclipped policy gradient (ablation).
    device : torch.device
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        clip_coef: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        use_clip: bool = True,
        use_orthogonal_init: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.use_clip = use_clip
        self.device = device

        self.net = ActorCriticNet(
            obs_dim,
            num_actions,
            hidden_dim,
            use_orthogonal_init=use_orthogonal_init,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_step(self, obs_np):
        """Select action for one environment step (no grad needed).

        Returns
        -------
        action       : int
        log_prob     : float
        value        : float
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, value, _ = self.net.get_action_and_log_prob(obs_t)
        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_value(self, obs_np) -> float:
        """Bootstrap value for the state after the last rollout step."""
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.net.get_value(obs_t).item()

    # ------------------------------------------------------------------
    def update(
        self,
        buffer: RolloutBuffer,
        update_epochs: int,
        minibatch_size: int,
        normalize_advantages: bool,
    ) -> Dict[str, float]:
        """Run PPO update over `update_epochs` epochs of minibatch SGD.

        Parameters
        ----------
        buffer : RolloutBuffer
            Buffer with GAE already computed.
        update_epochs : int
            Number of passes over the rollout data.
        minibatch_size : int
        normalize_advantages : bool

        Returns
        -------
        dict with mean losses, approx_kl, clip_fraction, entropy.
        """
        all_policy_losses: List[float] = []
        all_value_losses: List[float] = []
        all_entropies: List[float] = []
        all_approx_kls: List[float] = []
        all_clip_fracs: List[float] = []

        for _ in range(update_epochs):
            for obs_mb, act_mb, lp_old_mb, adv_mb, ret_mb in buffer.minibatch_iter(
                minibatch_size, normalize_advantages
            ):
                # ── Re-evaluate actions under current policy ────────────────
                log_prob_new, value_new, entropy = self.net.evaluate_actions(obs_mb, act_mb)

                # ── Probability ratio ───────────────────────────────────────
                log_ratio = log_prob_new - lp_old_mb
                ratio = log_ratio.exp()

                # ── Approximate KL (for diagnostics) ───────────────────────
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()

                # ── Policy loss ─────────────────────────────────────────────
                surr1 = ratio * adv_mb
                if self.use_clip:
                    # PPO-clip: pessimistic bound via min of two surrogates
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()
                else:
                    # Ablation: unclipped policy gradient (like vanilla PG)
                    policy_loss = -surr1.mean()

                # ── Value loss ──────────────────────────────────────────────
                assert not torch.isnan(value_new).any(), "NaN in value prediction"
                value_loss = 0.5 * ((ret_mb - value_new) ** 2).mean()

                # ── Entropy bonus ───────────────────────────────────────────
                entropy_bonus = entropy.mean()

                # ── Total loss ──────────────────────────────────────────────
                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy_bonus
                )

                assert not torch.isnan(loss), f"NaN in total loss: pl={policy_loss:.4f} vl={value_loss:.4f}"

                # ── Gradient step ───────────────────────────────────────────
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping: prevents catastrophically large updates
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy_bonus.item())
                all_approx_kls.append(approx_kl)
                all_clip_fracs.append(clip_frac)

        def _mean(lst: List[float]) -> float:
            return sum(lst) / max(len(lst), 1)

        return {
            "policy_loss": _mean(all_policy_losses),
            "value_loss": _mean(all_value_losses),
            "entropy": _mean(all_entropies),
            "approx_kl": _mean(all_approx_kls),
            "clip_fraction": _mean(all_clip_fracs),
        }
