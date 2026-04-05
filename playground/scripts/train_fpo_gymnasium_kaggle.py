"""Kaggle-friendly FPO runner for Gymnasium MuJoCo tasks.

This keeps the baseline FPO implementation unchanged by reusing:
- flow_policy.fpo.FpoState
- flow_policy.rollouts.TransitionStruct

Only rollout collection and evaluation are adapted for Python Gymnasium
environments, which lets you compare against the same FPO update code on
Gymnasium MuJoCo tasks such as Ant-v4.

Notebook usage:
    1. Upload this repo to Kaggle.
    2. Install deps in a notebook cell:
       !pip install -e /kaggle/working/fpo/playground "gymnasium[mujoco]"
    3. Run this file in one notebook cell:
       %run /kaggle/working/fpo/playground/scripts/train_fpo_gymnasium_kaggle.py

The live plots refresh in that same output cell during training.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np

try:
    from IPython.display import clear_output, display
except ImportError:  # pragma: no cover
    clear_output = None
    display = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYGROUND_SRC = REPO_ROOT / "playground" / "src"
if str(PLAYGROUND_SRC) not in sys.path:
    sys.path.insert(0, str(PLAYGROUND_SRC))

from flow_policy import fpo, rollouts


SUPPORTED_GYMNASIUM_TASKS = (
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Humanoid-v4",
)


@dataclass(slots=True)
class KaggleRunConfig:
    """Run config for a notebook-friendly Gymnasium FPO baseline."""

    env_name: str = "Ant-v4"
    seed: int = 0
    num_timesteps: int = 491_520
    num_envs: int = 8
    batch_size: int = 64
    num_minibatches: int = 8
    unroll_length: int = 30
    num_updates_per_batch: int = 4
    num_evals: int = 8
    eval_num_envs: int = 8
    episode_length: int = 1000
    plot_every: int = 1
    save_every: int = 1
    rolling_window: int = 20
    output_dir: str = str(REPO_ROOT / "kaggle_outputs" / "fpo_ant_v4")

    def make_fpo_config(self) -> fpo.FpoConfig:
        if self.env_name not in SUPPORTED_GYMNASIUM_TASKS:
            raise ValueError(
                f"Unsupported env_name={self.env_name!r}. "
                f"Choose one of: {', '.join(SUPPORTED_GYMNASIUM_TASKS)}"
            )

        total_subsequence_steps = (
            self.num_minibatches * self.batch_size * self.unroll_length
        )
        if total_subsequence_steps % self.num_envs != 0:
            raise ValueError(
                "num_minibatches * batch_size * unroll_length must be divisible "
                f"by num_envs. Got {total_subsequence_steps=} and {self.num_envs=}."
            )

        return fpo.FpoConfig(
            num_timesteps=self.num_timesteps,
            num_envs=self.num_envs,
            num_minibatches=self.num_minibatches,
            batch_size=self.batch_size,
            unroll_length=self.unroll_length,
            num_updates_per_batch=self.num_updates_per_batch,
            num_evals=self.num_evals,
            episode_length=self.episode_length,
        )


@dataclass(slots=True)
class EpisodeTracker:
    """Tracks in-flight and completed episode stats across rollouts."""

    returns: np.ndarray
    lengths: np.ndarray

    @classmethod
    def init(cls, num_envs: int) -> "EpisodeTracker":
        return cls(
            returns=np.zeros(num_envs, dtype=np.float32),
            lengths=np.zeros(num_envs, dtype=np.int32),
        )

    def update(
        self,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> tuple[list[float], list[int]]:
        done = np.logical_or(terminated, truncated)
        self.returns += rewards.astype(np.float32)
        self.lengths += 1

        finished_returns = self.returns[done].astype(np.float32).tolist()
        finished_lengths = self.lengths[done].astype(np.int32).tolist()

        self.returns[done] = 0.0
        self.lengths[done] = 0
        return finished_returns, finished_lengths


class GymnasiumBatchEnv:
    """Small Python-side batch env with baseline-style auto-reset."""

    def __init__(self, env_name: str, num_envs: int, seed: int) -> None:
        self.env_name = env_name
        self.num_envs = num_envs
        self._rng = np.random.default_rng(seed)
        self.envs = [gym.make(env_name) for _ in range(num_envs)]

        action_space = self.envs[0].action_space
        observation_space = self.envs[0].observation_space

        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError(
                f"Expected Box action space for {env_name}, got {type(action_space)}"
            )
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError(
                "This runner expects Box observations, got "
                f"{type(observation_space)}"
            )

        self.action_space = action_space
        self.observation_space = observation_space
        self.action_shape = tuple(action_space.shape)
        self.observation_size = int(np.prod(observation_space.shape))
        self.action_size = int(np.prod(action_space.shape))
        self.current_obs: np.ndarray | None = None

    def _next_seed(self) -> int:
        return int(self._rng.integers(0, 2**31 - 1))

    def _format_obs(self, obs: Any) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32).reshape(self.observation_size)

    def _format_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_shape)
        return np.clip(action, self.action_space.low, self.action_space.high)

    def reset(self) -> np.ndarray:
        obs_batch = []
        for env in self.envs:
            obs, _ = env.reset(seed=self._next_seed())
            obs_batch.append(self._format_obs(obs))
        self.current_obs = np.stack(obs_batch, axis=0)
        return self.current_obs.copy()

    def get_obs(self) -> np.ndarray:
        if self.current_obs is None:
            raise RuntimeError("Call reset() before get_obs().")
        return self.current_obs.copy()

    def step(
        self,
        action_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.current_obs is None:
            raise RuntimeError("Call reset() before step().")

        next_obs_batch = np.zeros_like(self.current_obs)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)

        for env_index, env in enumerate(self.envs):
            next_obs, reward, term, trunc, _ = env.step(
                self._format_action(action_batch[env_index])
            )

            next_obs_batch[env_index] = self._format_obs(next_obs)
            rewards[env_index] = float(reward)
            terminated[env_index] = bool(term)
            truncated[env_index] = bool(trunc)

            if term or trunc:
                reset_obs, _ = env.reset(seed=self._next_seed())
                self.current_obs[env_index] = self._format_obs(reset_obs)
            else:
                self.current_obs[env_index] = next_obs_batch[env_index]

        return next_obs_batch, rewards, terminated, truncated

    def close(self) -> None:
        for env in self.envs:
            env.close()


def collect_train_rollout(
    env: GymnasiumBatchEnv,
    agent_state: fpo.FpoState,
    prng: jax.Array,
    tracker: EpisodeTracker,
) -> tuple[rollouts.TransitionStruct[Any], jax.Array, list[float], list[int]]:
    config = agent_state.config

    obs_seq = []
    next_obs_seq = []
    action_seq = []
    reward_seq = []
    truncation_seq = []
    discount_seq = []
    action_info_seq = []
    completed_returns: list[float] = []
    completed_lengths: list[int] = []

    obs = env.get_obs()
    for _ in range(config.iterations_per_env):
        prng, step_key = jax.random.split(prng)
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)
        action, action_info = agent_state.sample_action(
            obs_jax,
            step_key,
            deterministic=False,
        )

        stepped_action = np.tanh(np.asarray(jax.device_get(action), dtype=np.float32))
        next_obs, reward, terminated, truncated = env.step(stepped_action)
        finished_returns, finished_lengths = tracker.update(
            reward,
            terminated,
            truncated,
        )

        completed_returns.extend(finished_returns)
        completed_lengths.extend(finished_lengths)

        obs_seq.append(obs_jax)
        next_obs_seq.append(jnp.asarray(next_obs, dtype=jnp.float32))
        action_seq.append(action)
        reward_seq.append(jnp.asarray(reward, dtype=jnp.float32))
        truncation_seq.append(jnp.asarray(truncated, dtype=jnp.float32))
        discount_seq.append(1.0 - jnp.asarray(terminated, dtype=jnp.float32))
        action_info_seq.append(action_info)

        obs = env.get_obs()

    return (
        rollouts.TransitionStruct(
            obs=jnp.stack(obs_seq, axis=0),
            next_obs=jnp.stack(next_obs_seq, axis=0),
            action=jnp.stack(action_seq, axis=0),
            action_info=jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *action_info_seq),
            reward=jnp.stack(reward_seq, axis=0),
            truncation=jnp.stack(truncation_seq, axis=0),
            discount=jnp.stack(discount_seq, axis=0),
        ),
        prng,
        completed_returns,
        completed_lengths,
    )


def evaluate_policy(
    agent_state: fpo.FpoState,
    env_name: str,
    num_envs: int,
    episode_length: int,
    seed: int,
) -> dict[str, float]:
    envs = [gym.make(env_name) for _ in range(num_envs)]
    try:
        obs = np.stack(
            [
                np.asarray(
                    env.reset(seed=seed + env_index)[0],
                    dtype=np.float32,
                ).reshape(agent_state.env.observation_size)
                for env_index, env in enumerate(envs)
            ],
            axis=0,
        )
        done = np.zeros(num_envs, dtype=np.bool_)
        rewards = np.zeros(num_envs, dtype=np.float32)
        lengths = np.zeros(num_envs, dtype=np.int32)
        prng = jax.random.key(seed)

        for _ in range(episode_length):
            if bool(done.all()):
                break

            prng, step_key = jax.random.split(prng)
            action, _ = agent_state.sample_action(
                jnp.asarray(obs, dtype=jnp.float32),
                step_key,
                deterministic=True,
            )
            action_batch = np.tanh(np.asarray(jax.device_get(action), dtype=np.float32))

            for env_index, env in enumerate(envs):
                if done[env_index]:
                    continue

                env_action = np.clip(
                    action_batch[env_index].reshape(env.action_space.shape),
                    env.action_space.low,
                    env.action_space.high,
                )
                next_obs, reward, terminated, truncated, _ = env.step(env_action)
                rewards[env_index] += float(reward)
                lengths[env_index] += 1
                obs[env_index] = np.asarray(next_obs, dtype=np.float32).reshape(
                    agent_state.env.observation_size
                )
                done[env_index] = bool(terminated or truncated)

        return {
            "reward_mean": float(np.mean(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "reward_std": float(np.std(rewards)),
            "steps_mean": float(np.mean(lengths)),
            "steps_min": float(np.min(lengths)),
            "steps_max": float(np.max(lengths)),
            "steps_std": float(np.std(lengths)),
        }
    finally:
        for env in envs:
            env.close()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    if values.size < window:
        return np.convolve(values, np.ones(values.size) / float(values.size), mode="valid")
    return np.convolve(values, kernel, mode="valid")


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_live_dashboard(
    run_config: KaggleRunConfig,
    train_history: list[dict[str, Any]],
    eval_history: list[dict[str, Any]],
    episode_history: list[dict[str, Any]],
    output_dir: Path,
    outer_iter: int,
    outer_iters: int,
    elapsed_seconds: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    train_steps = np.asarray([row["step"] for row in train_history], dtype=np.int32)
    train_mean_reward = np.asarray(
        [row["mean_step_reward"] for row in train_history],
        dtype=np.float32,
    )
    train_mean_episode_return = np.asarray(
        [row["mean_completed_episode_return"] for row in train_history],
        dtype=np.float32,
    )
    train_mean_episode_length = np.asarray(
        [row["mean_completed_episode_length"] for row in train_history],
        dtype=np.float32,
    )

    eval_steps = np.asarray([row["step"] for row in eval_history], dtype=np.int32)
    eval_rewards = np.asarray(
        [row["reward_mean"] for row in eval_history],
        dtype=np.float32,
    )
    eval_lengths = np.asarray(
        [row["steps_mean"] for row in eval_history],
        dtype=np.float32,
    )

    axes[0, 0].plot(train_steps, train_mean_reward, label="train mean step reward")
    axes[0, 0].plot(
        train_steps,
        train_mean_episode_return,
        label="train mean completed episode return",
    )
    if eval_history:
        axes[0, 0].plot(
            eval_steps,
            eval_rewards,
            marker="o",
            label="eval mean episode return",
        )
    axes[0, 0].set_title("Reward And Return")
    axes[0, 0].set_xlabel("outer iteration")
    axes[0, 0].set_ylabel("reward")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

    episode_indices = np.asarray(
        [row["episode_index"] for row in episode_history],
        dtype=np.int32,
    )
    episode_returns = np.asarray(
        [row["episode_return"] for row in episode_history],
        dtype=np.float32,
    )
    if episode_history:
        axes[0, 1].plot(
            episode_indices,
            episode_returns,
            alpha=0.35,
            label="per-episode return",
        )
        smoothed = moving_average(episode_returns, run_config.rolling_window)
        smooth_x = episode_indices[-smoothed.size :]
        axes[0, 1].plot(
            smooth_x,
            smoothed,
            linewidth=2.0,
            label=f"{run_config.rolling_window}-episode moving average",
        )
    else:
        axes[0, 1].text(0.5, 0.5, "No completed episodes yet", ha="center", va="center")
    axes[0, 1].set_title("Live Episode Returns")
    axes[0, 1].set_xlabel("episode index")
    axes[0, 1].set_ylabel("return")
    if episode_history:
        axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(
        train_steps,
        train_mean_episode_length,
        label="train mean completed episode length",
    )
    if eval_history:
        axes[1, 0].plot(
            eval_steps,
            eval_lengths,
            marker="o",
            label="eval mean episode length",
        )
    axes[1, 0].set_title("Episode Length")
    axes[1, 0].set_xlabel("outer iteration")
    axes[1, 0].set_ylabel("steps")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)

    policy_loss = np.asarray(
        [row["policy_loss"] for row in train_history],
        dtype=np.float32,
    )
    value_loss = np.asarray(
        [row["v_loss"] for row in train_history],
        dtype=np.float32,
    )
    axes[1, 1].plot(train_steps, policy_loss, label="policy_loss")
    axes[1, 1].plot(train_steps, value_loss, label="v_loss")
    axes[1, 1].set_title("Optimization")
    axes[1, 1].set_xlabel("outer iteration")
    axes[1, 1].set_ylabel("loss")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle(
        (
            f"FPO baseline on {run_config.env_name} | "
            f"iter {outer_iter + 1}/{outer_iters} | "
            f"elapsed {elapsed_seconds / 60.0:.1f} min"
        ),
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "live_training_dashboard.png", dpi=150, bbox_inches="tight")

    if clear_output is not None and display is not None:
        clear_output(wait=True)
        display(fig)
        print(
            f"iter {outer_iter + 1}/{outer_iters} | "
            f"env={run_config.env_name} | "
            f"completed_episodes={len(episode_history)} | "
            f"output_dir={output_dir}"
        )
    plt.close(fig)


def save_training_artifacts(
    run_config: KaggleRunConfig,
    train_history: list[dict[str, Any]],
    eval_history: list[dict[str, Any]],
    episode_history: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    write_csv_rows(output_dir / "train_metrics.csv", train_history)
    write_csv_rows(output_dir / "eval_metrics.csv", eval_history)
    write_csv_rows(output_dir / "episode_metrics.csv", episode_history)

    summary = {
        "run_config": dataclasses.asdict(run_config),
        "num_train_rows": len(train_history),
        "num_eval_rows": len(eval_history),
        "num_completed_episodes": len(episode_history),
        "final_train_row": train_history[-1] if train_history else None,
        "final_eval_row": eval_history[-1] if eval_history else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def train_gymnasium_baseline(
    run_config: KaggleRunConfig,
) -> tuple[fpo.FpoState, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = run_config.make_fpo_config()
    output_dir = Path(run_config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_env = GymnasiumBatchEnv(
        env_name=run_config.env_name,
        num_envs=run_config.num_envs,
        seed=run_config.seed,
    )
    train_env.reset()

    tracker = EpisodeTracker.init(run_config.num_envs)
    agent_state = fpo.FpoState.init(
        prng=jax.random.key(run_config.seed),
        env=train_env,
        config=config,
    )
    prng = jax.random.key(run_config.seed + 1)

    outer_iters = config.num_timesteps // (config.iterations_per_env * config.num_envs)
    if outer_iters <= 0:
        raise ValueError(
            "num_timesteps is too small for one outer iteration. "
            f"Need at least {config.iterations_per_env * config.num_envs}, "
            f"got {config.num_timesteps}."
        )
    eval_iters = set(np.linspace(0, outer_iters - 1, config.num_evals, dtype=int))

    train_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    episode_history: list[dict[str, Any]] = []

    start_time = time.time()
    try:
        for outer_iter in range(outer_iters):
            if outer_iter in eval_iters:
                eval_row = {
                    "step": outer_iter,
                    **evaluate_policy(
                        agent_state=agent_state,
                        env_name=run_config.env_name,
                        num_envs=run_config.eval_num_envs,
                        episode_length=config.episode_length,
                        seed=run_config.seed + 10_000 + outer_iter,
                    ),
                }
                eval_history.append(eval_row)

            transitions, prng, batch_episode_returns, batch_episode_lengths = collect_train_rollout(
                env=train_env,
                agent_state=agent_state,
                prng=prng,
                tracker=tracker,
            )
            agent_state, metrics = agent_state.training_step(transitions)

            reward_np = np.asarray(jax.device_get(transitions.reward), dtype=np.float32)
            metric_means = {
                key: float(np.asarray(jax.device_get(value)).mean())
                for key, value in metrics.items()
            }

            for episode_return, episode_length in zip(
                batch_episode_returns,
                batch_episode_lengths,
                strict=False,
            ):
                episode_history.append(
                    {
                        "episode_index": len(episode_history),
                        "outer_iter": outer_iter,
                        "episode_return": float(episode_return),
                        "episode_length": int(episode_length),
                    }
                )

            train_history.append(
                {
                    "step": outer_iter,
                    "elapsed_seconds": float(time.time() - start_time),
                    "mean_step_reward": float(np.mean(reward_np)),
                    "mean_completed_episode_return": (
                        float(np.mean(batch_episode_returns))
                        if batch_episode_returns
                        else float("nan")
                    ),
                    "mean_completed_episode_length": (
                        float(np.mean(batch_episode_lengths))
                        if batch_episode_lengths
                        else float("nan")
                    ),
                    "completed_episodes_in_iter": len(batch_episode_returns),
                    "completed_episodes_total": len(episode_history),
                    **metric_means,
                }
            )

            should_render = (
                outer_iter == 0
                or (outer_iter + 1) % run_config.plot_every == 0
                or outer_iter == outer_iters - 1
            )
            should_save = (
                (outer_iter + 1) % run_config.save_every == 0
                or outer_iter == outer_iters - 1
            )

            if should_render:
                render_live_dashboard(
                    run_config=run_config,
                    train_history=train_history,
                    eval_history=eval_history,
                    episode_history=episode_history,
                    output_dir=output_dir,
                    outer_iter=outer_iter,
                    outer_iters=outer_iters,
                    elapsed_seconds=time.time() - start_time,
                )

            if should_save:
                save_training_artifacts(
                    run_config=run_config,
                    train_history=train_history,
                    eval_history=eval_history,
                    episode_history=episode_history,
                    output_dir=output_dir,
                )
    finally:
        train_env.close()

    return agent_state, train_history, eval_history, episode_history


def main() -> None:
    run_config = KaggleRunConfig(
        env_name="Ant-v4",
        output_dir=str(REPO_ROOT / "kaggle_outputs" / "fpo_ant_v4"),
    )
    train_gymnasium_baseline(run_config)


if __name__ == "__main__":
    main()
