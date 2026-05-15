"""
evaluate.py – Deterministic policy evaluation.

During evaluation the policy is *greedy*: action = argmax(logits).
Stochasticity is disabled to get a noise-free signal of current policy quality.
Each evaluation checkpoint runs 20 independent episodes and records mean/std
return, which is far more stable than a single episode.

Evaluation is decoupled from training so it never affects learning.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
import torch

from rl_a4.networks import ActorCriticNet


def evaluate_policy(
    env_fn,
    net: ActorCriticNet,
    n_episodes: int = 20,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """Run `n_episodes` greedy episodes and return (mean_return, std_return).

    Parameters
    ----------
    env_fn : callable
        Zero-argument factory that returns a fresh gym.Env.
    net : ActorCriticNet
        The current network (used in eval mode, no grad).
    n_episodes : int
        Number of evaluation episodes.
    device : torch.device

    Returns
    -------
    (mean_return, std_return) over the `n_episodes` episodes.
    """
    net.eval()
    episode_returns = []

    with torch.no_grad():
        for _ in range(n_episodes):
            env = env_fn()
            obs, _ = env.reset()
            ep_return = 0.0
            done = False

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = net.get_deterministic_action(obs_t).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                done = terminated or truncated

            episode_returns.append(ep_return)
            env.close()

    net.train()

    returns = np.array(episode_returns, dtype=np.float32)
    return float(returns.mean()), float(returns.std())
