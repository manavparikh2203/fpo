import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import argparse
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flow_policy import fpo, rollouts


def make_gym_ant_env(env_name: str, num_envs: int):
    base_env = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])

    class GymAntEnv:
        def __init__(self, env):
            self.env = env
            self.action_space = env.single_action_space
            self.observation_space = env.single_observation_space
            self.observation_size = int(np.prod(self.observation_space.shape))
            self.action_size = int(np.prod(self.action_space.shape))

        def reset(self):
            obs, _ = self.env.reset()
            return np.asarray(obs, dtype=np.float32)

        def step(self, action: np.ndarray):
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, self.action_space.low, self.action_space.high)
            obs, reward, terminated, truncated, infos = self.env.step(action)
            return (
                np.asarray(obs, dtype=np.float32),
                np.asarray(reward, dtype=np.float32),
                np.asarray(terminated, dtype=np.bool_),
                np.asarray(truncated, dtype=np.bool_),
                infos,
            )

    return GymAntEnv(base_env)


def extract_episode_stats(
    rewards: np.ndarray,
    truncation: np.ndarray,
    discount: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    episode_rewards = []
    episode_lengths = []
    T, B = rewards.shape
    for b in range(B):
        current_reward = 0.0
        current_length = 0
        for t in range(T):
            current_reward += float(rewards[t, b])
            current_length += 1
            if truncation[t, b] > 0.5 or discount[t, b] == 0.0:
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                current_reward = 0.0
                current_length = 0
    return np.array(episode_rewards, dtype=np.float32), np.array(
        episode_lengths, dtype=np.int32
    )


def collect_transitions(env, agent_state, prng, config):
    obs = jnp.asarray(env.reset(), dtype=jnp.float32)
    obs_seq = []
    next_obs_seq = []
    action_seq = []
    reward_seq = []
    truncation_seq = []
    discount_seq = []
    action_info_seq = []

    for t in range(config.iterations_per_env):
        prng, prng_step = jax.random.split(prng)
        action, action_info = agent_state.sample_action(obs, prng_step, deterministic=False)
        next_obs_np, reward_np, terminated_np, truncated_np, _ = env.step(np.asarray(action))

        next_obs = jnp.asarray(next_obs_np, dtype=jnp.float32)
        reward = jnp.asarray(reward_np, dtype=jnp.float32)
        truncation = jnp.asarray(truncated_np, dtype=jnp.float32)
        discount = 1.0 - jnp.asarray(terminated_np, dtype=jnp.float32)

        obs_seq.append(obs)
        next_obs_seq.append(next_obs)
        action_seq.append(action)
        reward_seq.append(reward)
        truncation_seq.append(truncation)
        discount_seq.append(discount)
        action_info_seq.append(action_info)

        obs = next_obs

    transitions = rollouts.TransitionStruct(
        obs=jnp.stack(obs_seq),
        next_obs=jnp.stack(next_obs_seq),
        action=jnp.stack(action_seq),
        action_info=tree_util.tree_map(lambda *xs: jnp.stack(xs), *action_info_seq),
        reward=jnp.stack(reward_seq),
        truncation=jnp.stack(truncation_seq),
        discount=jnp.stack(discount_seq),
    )
    return transitions, prng


def train_ant_gymnasium(
    env_name: str,
    output_dir: Path,
    num_timesteps: int,
    num_envs: int,
    batch_size: int,
    num_minibatches: int,
    unroll_length: int,
    num_updates_per_batch: int,
    seed: int,
):
    config = fpo.FpoConfig(
        num_timesteps=num_timesteps,
        num_envs=num_envs,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        unroll_length=unroll_length,
        num_updates_per_batch=num_updates_per_batch,
    )

    env = make_gym_ant_env(env_name, num_envs)
    agent_state = fpo.FpoState.init(prng=jax.random.key(seed), env=env, config=config)
    prng = jax.random.key(seed + 1)

    outer_iters = int(num_timesteps // (config.iterations_per_env * num_envs))
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for i in range(outer_iters):
        transitions, prng = collect_transitions(env, agent_state, prng, config)
        agent_state, metrics = agent_state.training_step(transitions)

        reward_np = np.asarray(transitions.reward)
        truncation_np = np.asarray(transitions.truncation)
        discount_np = np.asarray(transitions.discount)
        episode_rewards, episode_lengths = extract_episode_stats(
            reward_np, truncation_np, discount_np
        )

        mean_reward = float(np.mean(reward_np))
        mean_episode_reward = float(np.mean(episode_rewards)) if episode_rewards.size else mean_reward
        mean_episode_length = float(np.mean(episode_lengths)) if episode_lengths.size else 0.0

        row = {
            "step": i,
            "mean_reward": mean_reward,
            "mean_episode_reward": mean_episode_reward,
            "mean_episode_length": mean_episode_length,
            "num_episodes": int(episode_rewards.size),
            **{f"train_{k}": float(np.mean(v)) for k, v in metrics.items()},
        }
        history.append(row)

        print(
            f"Iter {i+1}/{outer_iters} | mean_reward={mean_reward:.3f} "
            f"mean_episode_reward={mean_episode_reward:.3f} "
            f"mean_episode_length={mean_episode_length:.1f} "
            f"episodes={row['num_episodes']}"
        )

        # Save intermediate results every 10 iterations
        if (i + 1) % 10 == 0 or i == outer_iters - 1:
            pd.DataFrame(history).to_csv(train_csv, index=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot([row["step"] for row in history], [row["mean_reward"] for row in history], label="mean_reward")
            ax.plot([row["step"] for row in history], [row["mean_episode_reward"] for row in history], label="mean_episode_reward")
            ax.set_title(f"{env_name} FPO Training (Iter {i+1})")
            ax.set_xlabel("outer iteration")
            ax.set_ylabel("reward")
            ax.legend()
            fig.savefig(output_dir / "reward_curve.png")
            plt.close(fig)
            print(f"Saved intermediate results at iteration {i+1}")

    # Final save is handled in the loop
    print(f"Training complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Ant-v5")
    parser.add_argument("--num-timesteps", type=int, default=1000000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--unroll-length", type=int, default=30)
    parser.add_argument("--num-updates-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="gym_fpo_results")
    args = parser.parse_args()

    train_ant_gymnasium(
        env_name=args.env_name,
        output_dir=Path(args.output_dir) / args.env_name.replace("-", ""),
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        unroll_length=args.unroll_length,
        num_updates_per_batch=args.num_updates_per_batch,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
