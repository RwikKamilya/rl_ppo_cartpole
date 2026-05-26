"""
train_ppo.py – Main PPO training script.

Usage:
    python scripts/train_ppo.py --config configs/ppo_full.json --seed 0
    python scripts/train_ppo.py --config configs/ppo_full.json --seed 0 --total-env-steps 5000

The script:
  1. Loads a JSON config (overridden by CLI flags if provided).
  2. Sets reproducibility seeds.
  3. Runs the PPO training loop for `total_env_steps` environment steps.
  4. Saves per-seed CSVs:
       results/<exp_name>/seed_<seed>/train.csv    (episode, step, return)
       results/<exp_name>/seed_<seed>/eval.csv     (step, mean_return, std_return)
       results/<exp_name>/seed_<seed>/update_log.csv
       results/<exp_name>/seed_<seed>/config.json

Training loop outline:
  while global_step < total_env_steps:
      collect rollout_steps transitions
      compute GAE + returns
      run PPO update (update_epochs * n_minibatches gradient steps)
      if (global_step // eval_interval) changed: run deterministic eval
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Allow running as `python scripts/train_ppo.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import gymnasium as gym

from rl_a4.networks import ActorCriticNet
from rl_a4.buffers import RolloutBuffer
from rl_a4.ppo import PPOAgent
from rl_a4.evaluate import evaluate_policy
from rl_a4.utils import set_seed, make_env, save_config, load_config, get_device


# ──────────────────────────────────────────────────────────────────────────────
# Defaults (overridden by config JSON or CLI)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "exp_name": "ppo_full",
    "env_id": "CartPole-v1",
    "total_env_steps": 1_000_000,
    "rollout_steps": 2048,
    "update_epochs": 10,
    "minibatch_size": 256,
    "learning_rate": 3e-4,
    "clip_coef": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "hidden_dim": 64,
    "use_orthogonal_init": True,
    "normalize_advantages": True,
    "use_clip": True,          # False → PPO_no_clip ablation
    "eval_interval": 10_000,
    "n_eval_episodes": 20,
    "eval_seed_offset": 9999,
    "seed_list": [0, 1, 2, 3, 4],
    "prefer_gpu": False,
    "results_dir": "results",
}

CONFIG_ALIASES = {
    "total_steps": "total_env_steps",
    "lambda_gae": "gae_lambda",
    "clip_epsilon": "clip_coef",
    "entropy_coef": "ent_coef",
    "value_coef": "vf_coef",
    "num_epochs": "update_epochs",
    "eval_episodes": "n_eval_episodes",
}

LR_ALIAS_KEYS = ("shared_lr", "policy_lr", "value_lr")


def normalize_config(config: dict) -> dict:
    """Accept report-friendly config names while keeping internal keys stable."""
    config = dict(config)

    for alias, target in CONFIG_ALIASES.items():
        if alias in config and target not in config:
            config[target] = config[alias]

    if "learning_rate" not in config:
        provided_lrs = [float(config[k]) for k in LR_ALIAS_KEYS if k in config]
        if provided_lrs:
            if any(abs(lr - provided_lrs[0]) > 1e-12 for lr in provided_lrs[1:]):
                raise ValueError(
                    "This PPO implementation uses one shared optimizer; provide "
                    "matching policy_lr/value_lr values or use shared_lr."
                )
            config["learning_rate"] = provided_lrs[0]

    # Keep saved configs explicit and synchronized after CLI overrides.
    for alias, target in CONFIG_ALIASES.items():
        if target in config:
            config[alias] = config[target]
    if "learning_rate" in config:
        config["shared_lr"] = config["learning_rate"]

    return config


def assert_ppo_action_modes(agent: PPOAgent, obs_dim: int, device: torch.device) -> None:
    """Sanity-check that rollout collection samples and evaluation is greedy."""
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(12345)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(12345)

        obs = np.zeros(obs_dim, dtype=np.float32)
        sampled_actions = [agent.collect_step(obs)[0] for _ in range(64)]
        assert len(set(sampled_actions)) > 1, (
            "PPO training action selection did not produce stochastic samples. "
            "Rollout collection must sample from Categorical(logits=logits), not argmax."
        )

        obs_t = torch.zeros((1, obs_dim), dtype=torch.float32, device=device)
        with torch.no_grad():
            deterministic_actions = [
                agent.net.get_deterministic_action(obs_t).item()
                for _ in range(8)
            ]
            logits, _ = agent.net(obs_t)
            greedy_action = int(logits.argmax(dim=-1).item())
        assert deterministic_actions == [greedy_action] * len(deterministic_actions), (
            "Evaluation action selection must be deterministic argmax(logits)."
        )
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on CartPole-v1")
    p.add_argument("--config", type=str, default=None,
                   help="Path to JSON config file.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed.")
    p.add_argument("--total-env-steps", type=int, default=None,
                   help="Override total_env_steps from config.")
    p.add_argument("--eval-interval", type=int, default=None,
                   help="Override eval_interval from config.")
    p.add_argument("--n-eval-episodes", type=int, default=None,
                   help="Override n_eval_episodes from config.")
    p.add_argument("--exp-name", type=str, default=None,
                   help="Override exp_name from config.")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Override results_dir from config.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def train(config: dict, seed: int) -> None:
    """Run one training run for a given config and seed."""

    # ── Setup ──────────────────────────────────────────────────────────────
    config = normalize_config(config)
    set_seed(seed)
    device = get_device(config["prefer_gpu"])
    assert config["eval_seed_offset"] != 0, (
        "Evaluation must use a distinct seed offset to keep train and eval environments separate."
    )

    exp_name = config["exp_name"]
    results_dir = Path(config["results_dir"]) / exp_name / f"seed_{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_config(config | {"seed": seed}, results_dir / "config.json")

    env_fn = make_env(config["env_id"], seed)
    env = env_fn()
    obs_dim = env.observation_space.shape[0]   # 4
    num_actions = env.action_space.n            # 2

    # ── Agent + Buffer ─────────────────────────────────────────────────────
    agent = PPOAgent(
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden_dim=config["hidden_dim"],
        lr=config["learning_rate"],
        clip_coef=config["clip_coef"],
        vf_coef=config["vf_coef"],
        ent_coef=config["ent_coef"],
        max_grad_norm=config["max_grad_norm"],
        use_clip=config["use_clip"],
        use_orthogonal_init=config["use_orthogonal_init"],
        device=device,
    )
    assert_ppo_action_modes(agent, obs_dim, device)

    buffer = RolloutBuffer(
        rollout_steps=config["rollout_steps"],
        obs_dim=obs_dim,
        device=device,
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
    )

    # ── CSV writers ────────────────────────────────────────────────────────
    train_csv = open(results_dir / "train.csv", "w", newline="")
    eval_csv = open(results_dir / "eval.csv", "w", newline="")
    upd_csv = open(results_dir / "update_log.csv", "w", newline="")

    train_writer = csv.writer(train_csv)
    eval_writer = csv.writer(eval_csv)
    upd_writer = csv.writer(upd_csv)

    train_writer.writerow(["episode", "step", "return"])
    eval_writer.writerow(["step", "mean_return", "std_return"])
    upd_writer.writerow(["update", "step", "policy_loss", "value_loss",
                          "entropy", "approx_kl", "clip_fraction"])

    # ── Training loop ──────────────────────────────────────────────────────
    obs, _ = env.reset(seed=seed)
    global_step = 0
    episode = 0
    ep_return = 0.0
    last_eval_checkpoint = -1
    update_count = 0
    start_time = time.time()

    print(f"[{exp_name} seed={seed}] Starting training on {device}. "
          f"total_env_steps={config['total_env_steps']}")

    while global_step < config["total_env_steps"]:

        # ── Collect rollout ────────────────────────────────────────────────
        buffer.reset()

        for step in range(config["rollout_steps"]):
            if global_step >= config["total_env_steps"]:
                break

            action, log_prob, value = agent.collect_step(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = float(terminated or truncated)
            next_value = 0.0 if terminated else agent.get_value(next_obs)

            buffer.add(
                obs,
                action,
                float(reward),
                done,
                float(terminated),
                log_prob,
                value,
                next_value,
            )
            obs = next_obs
            global_step += 1
            ep_return += reward

            if terminated or truncated:
                # Log completed episode
                episode += 1
                train_writer.writerow([episode, global_step, ep_return])
                train_csv.flush()
                ep_return = 0.0
                obs, _ = env.reset()

        # ── GAE + Returns ──────────────────────────────────────────────────
        buffer.compute_gae_and_returns()

        # ── PPO Update ────────────────────────────────────────────────────
        update_count += 1
        stats = agent.update(
            buffer,
            update_epochs=config["update_epochs"],
            minibatch_size=config["minibatch_size"],
            normalize_advantages=config["normalize_advantages"],
        )
        upd_writer.writerow([
            update_count, global_step,
            f"{stats['policy_loss']:.6f}",
            f"{stats['value_loss']:.6f}",
            f"{stats['entropy']:.6f}",
            f"{stats['approx_kl']:.6f}",
            f"{stats['clip_fraction']:.4f}",
        ])
        upd_csv.flush()

        if update_count % 10 == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed if elapsed > 0 else 0
            print(
                f"  step={global_step:>8d}  ep={episode:>5d}  "
                f"pol_loss={stats['policy_loss']:+.4f}  "
                f"val_loss={stats['value_loss']:.4f}  "
                f"ent={stats['entropy']:.4f}  "
                f"kl={stats['approx_kl']:.4f}  "
                f"clip={stats['clip_fraction']:.3f}  "
                f"sps={sps:.0f}"
            )

        # ── Periodic evaluation ────────────────────────────────────────────
        eval_checkpoint = global_step // config["eval_interval"]
        if eval_checkpoint > last_eval_checkpoint:
            last_eval_checkpoint = eval_checkpoint
            eval_mean, eval_std = evaluate_policy(
                env_fn=make_env(config["env_id"], seed + config["eval_seed_offset"]),
                net=agent.net,
                n_episodes=config["n_eval_episodes"],
                device=device,
                seed=seed + config["eval_seed_offset"],
            )
            eval_writer.writerow([global_step, f"{eval_mean:.4f}", f"{eval_std:.4f}"])
            eval_csv.flush()
            print(f"  [EVAL] step={global_step}  mean={eval_mean:.1f}  std={eval_std:.1f}")

    # ── Cleanup ────────────────────────────────────────────────────────────
    env.close()
    train_csv.close()
    eval_csv.close()
    upd_csv.close()

    elapsed = time.time() - start_time
    print(f"[{exp_name} seed={seed}] Done. {global_step} steps in {elapsed:.1f}s "
          f"({global_step / elapsed:.0f} sps). Results → {results_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Start from defaults, overlay config file, overlay CLI
    config = DEFAULT_CONFIG.copy()
    if args.config:
        file_cfg = load_config(args.config)
        config.update(file_cfg)
    if args.total_env_steps is not None:
        config["total_env_steps"] = args.total_env_steps
    if args.eval_interval is not None:
        config["eval_interval"] = args.eval_interval
    if args.n_eval_episodes is not None:
        config["n_eval_episodes"] = args.n_eval_episodes
    if args.exp_name is not None:
        config["exp_name"] = args.exp_name
    if args.results_dir is not None:
        config["results_dir"] = args.results_dir
    config = normalize_config(config)

    train(config, seed=args.seed)


if __name__ == "__main__":
    main()
