"""
networks.py – Actor-Critic neural network for PPO on CartPole-v1.

Architecture choice: shared backbone + separate heads.
  - Two fully-connected hidden layers (hidden_dim units each) with Tanh activation.
  - Actor head: Linear -> logits over num_actions (used with Categorical distribution).
  - Critic head: Linear -> scalar state-value V(s).

Sharing the backbone encourages aligned representations between policy and value
estimates (similar to A2C), while keeping the output heads separate prevents the
value scale from corrupting policy gradients. Tanh is preferred over ReLU here
because CartPole states are bounded and Tanh avoids dead-neuron saturation at
initialisation.

Weights are initialised with orthogonal initialisation (scale=sqrt(2) for hidden,
scale=0.01 for actor, scale=1.0 for critic), following the standard PPO recipe
from Schulman et al. 2017 and CleanRL.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _layer_init(layer: nn.Linear, std: float = 1.4142135623730951,
                bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight init with controllable output scale."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNet(nn.Module):
    """Shared-backbone actor-critic network.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation vector (4 for CartPole-v1).
    num_actions : int
        Number of discrete actions (2 for CartPole-v1).
    hidden_dim : int
        Width of the two shared hidden layers. Default: 64.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()

        # ── Shared backbone ────────────────────────────────────────────────
        self.backbone = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )

        # ── Actor head (policy logits) ─────────────────────────────────────
        # Small init std keeps initial policy close to uniform.
        self.actor_head = _layer_init(nn.Linear(hidden_dim, num_actions), std=0.01)

        # ── Critic head (state-value) ──────────────────────────────────────
        self.critic_head = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """Return (logits, value) for a batch of observations."""
        features = self.backbone(x)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)  # (B,)
        return logits, value

    # ------------------------------------------------------------------
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the scalar value estimate V(s)."""
        features = self.backbone(x)
        return self.critic_head(features).squeeze(-1)

    # ------------------------------------------------------------------
    def get_action_and_log_prob(self, x: torch.Tensor):
        """Sample action and return (action, log_prob, value, entropy).

        Used during rollout collection.
        """
        logits, value = self.forward(x)
        assert not torch.isnan(logits).any(), "NaN detected in actor logits"
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy

    # ------------------------------------------------------------------
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        """Re-evaluate stored actions under the *current* policy.

        Returns (log_prob, value, entropy) for the PPO update step.
        """
        logits, value = self.forward(x)
        assert not torch.isnan(logits).any(), "NaN in logits during update"
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy

    # ------------------------------------------------------------------
    def get_deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        """Return argmax(logits) – used for deterministic evaluation only."""
        logits, _ = self.forward(x)
        return logits.argmax(dim=-1)
