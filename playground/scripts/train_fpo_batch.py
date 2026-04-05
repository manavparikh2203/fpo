import csv
import datetime
import time
from pathlib import Path
from typing import Annotated

import jax
import jax_dataclasses as jdc
import matplotlib.pyplot as plt
import numpy as onp
import tyro
from jax import numpy as jnp
from mujoco_playground import locomotion, registry

try:
    import gymnasium as gym
except ImportError:
    gym = None

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore

from flow_policy import fpo, rollouts


class GymVectorEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.single_action_space
        self.observation_space = env.single_observation_space

    def reset(self):
        obs, info = self.env.reset()
        num_envs = obs.shape[0]
        class State:
            def __init__(self, obs):
                self.obs = obs
                self.data = jnp.zeros((num_envs,))  # dummy
                self.done = jnp.zeros(num_envs, dtype=bool)
        return State(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated
        class State:
            def __init__(self, obs):
                self.obs = obs
                self.data = jnp.zeros_like(done)  # dummy
                self.done = done
        return State(obs), reward, done, info


def _extract_episode_stats(
    rewards: onp.ndarray,
    truncation: onp.ndarray,
    discount: onp.ndarray,
) -> tuple[onp.ndarray, onp.ndarray]:
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
    return onp.array(episode_rewards, dtype=onp.float32), onp.array(
        episode_lengths, dtype=onp.int32
    )


def train_one_env(
    env_name: str,
    output_dir: Path,
    wandb_entity: str,
    wandb_project: str,
    config: fpo.FpoConfig,
    seed: int,
    use_wandb: bool,
) -> None:
    if env_name in ['Ant', 'Ant-v5']:
        if gym is None:
            raise RuntimeError("Gymnasium not installed. Install gymnasium.")
        def make_env():
            return gym.make('Ant-v5')
        env = gym.vector.SyncVectorEnv([make_env for _ in range(config.num_envs)])
        env = GymVectorEnvWrapper(env)
    else:
        env_config = registry.get_default_config(env_name)
        env = registry.load(env_name, config=env_config)

    wandb_run = None
    if use_wandb:
        if wandb is None:
            raise RuntimeError(
                "WandB is not installed. Install wandb or run with --use-wandb=False."
            )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=f"fpo_{env_name}_{timestamp}",
            config={
                "env_name": env_name,
                "fpo_config": jdc.asdict(config),
                "seed": seed,
            },
        )

    agent_state = fpo.FpoState.init(prng=jax.random.key(seed), env=env, config=config)
    rollout_state = rollouts.BatchedRolloutState.init(
        env,
        prng=jax.random.key(seed),
        num_envs=config.num_envs,
    )

    outer_iters = config.num_timesteps // (config.iterations_per_env * config.num_envs)
    eval_iters = set(onp.linspace(0, outer_iters - 1, config.num_evals, dtype=int))

    train_history = []
    eval_history = []

    for i in range(int(outer_iters)):
        if i in eval_iters:
            eval_outputs = rollouts.eval_policy(
                agent_state,
                prng=jax.random.fold_in(agent_state.prng, i),
                num_envs=128,
                max_episode_length=config.episode_length,
            )
            s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}
            eval_history.append({"step": i, **{k: float(v) for k, v in s_np.items()}})
            eval_outputs.log_to_wandb(wandb_run, step=i)

        rollout_state, transitions = rollout_state.rollout(
            agent_state,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
        )
        agent_state, metrics = agent_state.training_step(transitions)

        reward_np = onp.array(transitions.reward)
        truncation_np = onp.array(transitions.truncation)
        discount_np = onp.array(transitions.discount)
        episode_rewards, episode_lengths = _extract_episode_stats(
            reward_np, truncation_np, discount_np
        )

        mean_reward = float(onp.mean(reward_np))
        mean_episode_reward = float(onp.mean(episode_rewards)) if episode_rewards.size else mean_reward
        mean_episode_length = float(onp.mean(episode_lengths)) if episode_lengths.size else 0.0
        train_row = {
            "step": i,
            "mean_reward": mean_reward,
            "mean_episode_reward": mean_episode_reward,
            "mean_episode_length": mean_episode_length,
            "num_episodes": int(episode_rewards.size),
            **{f"train_{k}": float(onp.mean(v)) for k, v in metrics.items()},
        }
        train_history.append(train_row)

        if use_wandb and wandb_run is not None:
            log_dict = {
                **train_row,
                "train/reward_histogram": wandb.Histogram(
                    reward_np.flatten()[::16].tolist()
                ),
            }
            if episode_rewards.size:
                log_dict["train/episode_reward_histogram"] = wandb.Histogram(
                    episode_rewards.tolist()
                )
            if episode_lengths.size:
                log_dict["train/episode_length_histogram"] = wandb.Histogram(
                    episode_lengths.tolist()
                )
            wandb_run.log(log_dict, step=i)

    env_dir = output_dir / env_name
    env_dir.mkdir(parents=True, exist_ok=True)

    train_csv = env_dir / "train_metrics.csv"
    with train_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(train_history[0].keys()))
        writer.writeheader()
        writer.writerows(train_history)

    if eval_history:
        eval_csv = env_dir / "eval_metrics.csv"
        with eval_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(eval_history[0].keys()))
            writer.writeheader()
            writer.writerows(eval_history)

    plot_history(env_name, env_dir, train_history, eval_history)

    if use_wandb and wandb_run is not None:
        wandb_run.finish()


def plot_history(
    env_name: str,
    env_dir: Path,
    train_history: list[dict],
    eval_history: list[dict],
) -> None:
    steps = [row["step"] for row in train_history]
    rewards = [row["mean_reward"] for row in train_history]
    episode_rewards = [row["mean_episode_reward"] for row in train_history]
    episode_lengths = [row["mean_episode_length"] for row in train_history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rewards, label="train mean reward")
    ax.plot(steps, episode_rewards, label="train mean episode reward")
    if eval_history:
        eval_steps = [row["step"] for row in eval_history]
        eval_rewards = [row["reward_mean"] for row in eval_history]
        ax.plot(eval_steps, eval_rewards, label="eval reward mean", marker="o")
    ax.set_title(f"{env_name} reward curves")
    ax.set_xlabel("training step")
    ax.set_ylabel("reward")
    ax.legend()
    fig.savefig(env_dir / "reward_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, episode_lengths, label="train mean episode length")
    if eval_history:
        eval_steps = [row["step"] for row in eval_history]
        eval_steps_mean = [row["steps_mean"] for row in eval_history]
        ax.plot(eval_steps, eval_steps_mean, label="eval episode length", marker="o")
    ax.set_title(f"{env_name} episode length curves")
    ax.set_xlabel("training step")
    ax.set_ylabel("episode length")
    ax.legend()
    fig.savefig(env_dir / "episode_length_curve.png")
    plt.close(fig)


def main(
    env_names: Annotated[
        list[str],
        tyro.conf.arg(
            metavar="ENV_NAME",
            help="A list of Mujoco env names to train sequentially.",
        ),
    ] = ["Ant", "Humanoid", "Hopper", "Walker", "HalfCheetah"],
    use_wandb: bool = False,
    wandb_entity: str = "",
    wandb_project: str = "fpo-batch",
    seed: int = 0,
    output_dir: Path = Path("./fpo_results"),
    num_timesteps: int | None = None,
    num_envs: int | None = None,
) -> None:
    config = fpo.FpoConfig()
    if num_timesteps is not None:
        config.num_timesteps = num_timesteps
    if num_envs is not None:
        config.num_envs = num_envs

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for env_name in env_names:
        print(f"=== Training env: {env_name} ===")
        train_one_env(
            env_name=env_name,
            output_dir=output_dir,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            config=config,
            seed=seed,
            use_wandb=use_wandb,
        )


if __name__ == "__main__":
    tyro.cli(main)
