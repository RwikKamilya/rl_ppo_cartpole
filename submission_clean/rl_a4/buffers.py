"""
buffers.py – On-policy rollout buffer with Generalised Advantage Estimation (GAE).

Key design decisions
--------------------
* No experience replay – PPO is strictly on-policy.
* The buffer stores a fixed-length trajectory of `rollout_steps` transitions,
  then GAE + returns are computed once before the PPO update.
* GAE (Schulman et al., 2015):
      δ_t = r_t + γ · V(s_{t+1}) · (1-terminal_t) − V(s_t)
      A_t = δ_t + γλ · (1-episode_end_t) · A_{t+1}
  where λ=0 gives one-step TD (low variance but biased) and λ=1 gives
  full Monte-Carlo advantage (unbiased but high variance). λ=0.95 strikes a
  practical balance.
* Returns are R_t = A_t + V(s_t), used as regression targets for the critic.
* Terminated vs Truncated: Gymnasium separates `terminated` (true MDP terminal)
  from `truncated` (time-limit cutoff). PPO should bootstrap through truncation
  but must still stop the GAE recursion across the reset boundary. We therefore
  store both the "episode ended" flag and the "true terminal" flag separately.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Iterator, Tuple


class RolloutBuffer:
    """Fixed-length on-policy rollout buffer.

    Parameters
    ----------
    rollout_steps : int
        Number of environment steps per rollout.
    obs_dim : int
        Observation space dimensionality.
    device : torch.device
        Tensor device.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda (λ). 0 → one-step TD, 1 → Monte-Carlo.
    """

    def __init__(
        self,
        rollout_steps: int,
        obs_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.rollout_steps = rollout_steps
        self.obs_dim = obs_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Pre-allocate storage (CPU numpy, moved to device after GAE)
        self.obs = np.zeros((rollout_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros(rollout_steps, dtype=np.int64)
        self.rewards = np.zeros(rollout_steps, dtype=np.float32)
        self.episode_ends = np.zeros(rollout_steps, dtype=np.float32)
        self.terminated = np.zeros(rollout_steps, dtype=np.float32)
        self.log_probs_old = np.zeros(rollout_steps, dtype=np.float32)
        self.values = np.zeros(rollout_steps, dtype=np.float32)
        self.next_values = np.zeros(rollout_steps, dtype=np.float32)

        # Filled after compute_gae_and_returns()
        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

        self._ptr = 0  # write pointer

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the buffer for the next rollout."""
        self._ptr = 0
        self.advantages = None
        self.returns = None

    # ------------------------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        episode_end: float,
        terminated: float,
        log_prob: float,
        value: float,
        next_value: float,
    ) -> None:
        """Store a single transition."""
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.episode_ends[i] = episode_end
        self.terminated[i] = terminated
        self.log_probs_old[i] = log_prob
        self.values[i] = value
        self.next_values[i] = next_value
        self._ptr += 1

    # ------------------------------------------------------------------
    def compute_gae_and_returns(self) -> None:
        """Run GAE over the stored rollout.
        """
        n_steps = self._ptr
        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0.0

        # Iterate backwards from the final filled transition to the first.
        for t in reversed(range(n_steps)):
            bootstrap_mask = 1.0 - self.terminated[t]
            continuation_mask = 1.0 - self.episode_ends[t]
            delta = (
                self.rewards[t]
                + self.gamma * self.next_values[t] * bootstrap_mask
                - self.values[t]
            )
            # Time-limit truncations bootstrap with V(s_{t+1}) but do not carry
            # A_{t+1} across the environment reset into the next episode.
            last_gae = delta + self.gamma * self.gae_lambda * continuation_mask * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values[:n_steps]  # R_t = A_t + V(s_t)

    # ------------------------------------------------------------------
    def get_tensors(self, normalize_advantages: bool = True):
        """Return all rollout data as tensors, optionally normalising advantages.

        Advantage normalisation: A ← (A − mean(A)) / (std(A) + ε)
        Reduces variance of policy gradient estimates within a rollout.
        """
        assert self.advantages is not None, "Call compute_gae_and_returns first."
        n_steps = self._ptr

        adv = self.advantages.copy()
        if normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(self.obs[:n_steps], dtype=torch.float32, device=self.device)
        act_t = torch.tensor(self.actions[:n_steps], dtype=torch.long, device=self.device)
        lp_t = torch.tensor(self.log_probs_old[:n_steps], dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(self.returns, dtype=torch.float32, device=self.device)

        return obs_t, act_t, lp_t, adv_t, ret_t

    # ------------------------------------------------------------------
    def minibatch_iter(
        self,
        minibatch_size: int,
        normalize_advantages: bool = True,
    ) -> Iterator[Tuple[torch.Tensor, ...]]:
        """Yield randomly shuffled minibatches for the PPO update.

        Yields
        ------
        (obs, actions, log_probs_old, advantages, returns)
        each of shape (minibatch_size, …).
        """
        obs_t, act_t, lp_t, adv_t, ret_t = self.get_tensors(normalize_advantages)
        n_steps = obs_t.shape[0]
        indices = torch.randperm(n_steps, device=self.device)

        for start in range(0, n_steps, minibatch_size):
            mb_idx = indices[start: start + minibatch_size]
            yield (
                obs_t[mb_idx],
                act_t[mb_idx],
                lp_t[mb_idx],
                adv_t[mb_idx],
                ret_t[mb_idx],
            )
