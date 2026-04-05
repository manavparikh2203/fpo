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

import base64
import csv
import dataclasses
import io
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
    from IPython import get_ipython
    from IPython.display import HTML, Video, display
except ImportError:  # pragma: no cover
    HTML = None
    Video = None
    display = None
    get_ipython = None


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
    num_timesteps: int = 10000000
    num_envs: int = 128
    batch_size: int = 32
    num_minibatches: int = 8
    unroll_length: int = 30
    num_updates_per_batch: int = 32
    num_evals: int = 64
    eval_num_envs: int = 4
    episode_length: int = 1000
    plot_every: int = 1
    save_every: int = 1
    rolling_window: int = 20
    show_live_plots: bool = True
    prefer_gpu_if_available: bool = True
    run_eval_at_step_zero: bool = True
    save_final_plots: bool = True
    save_final_video: bool = True
    final_video_fps: int = 30
    final_video_max_steps: int = 1000
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
class TrainEpisodeTracker:
    """Tracks episode returns and lengths across rollout chunks for logging."""

    returns: np.ndarray
    lengths: np.ndarray

    @classmethod
    def init(cls, num_envs: int) -> "TrainEpisodeTracker":
        return cls(
            returns=np.zeros(num_envs, dtype=np.float32),
            lengths=np.zeros(num_envs, dtype=np.int32),
        )

    def update_from_transitions(
        self,
        transitions: rollouts.TransitionStruct[Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(jax.device_get(transitions.reward), dtype=np.float32)
        truncation = np.asarray(jax.device_get(transitions.truncation), dtype=np.float32)
        discount = np.asarray(jax.device_get(transitions.discount), dtype=np.float32)

        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        horizon, _ = rewards.shape

        for t in range(horizon):
            self.returns += rewards[t]
            self.lengths += 1
            done = np.logical_or(truncation[t] > 0.5, discount[t] == 0.0)
            if not bool(done.any()):
                continue
            completed_returns.extend(self.returns[done].astype(np.float32).tolist())
            completed_lengths.extend(self.lengths[done].astype(np.int32).tolist())
            self.returns[done] = 0.0
            self.lengths[done] = 0

        return (
            np.asarray(completed_returns, dtype=np.float32),
            np.asarray(completed_lengths, dtype=np.int32),
        )


@dataclass(slots=True)
class LiveDisplayManager:
    """Notebook-friendly live display that updates in place."""

    enabled: bool = False
    figure_handle: Any | None = None
    status_handle: Any | None = None

    @classmethod
    def create(cls, enabled: bool) -> "LiveDisplayManager":
        notebook_enabled = enabled and _is_notebook_runtime()
        return cls(enabled=notebook_enabled)

    def _make_status_html(self, status_text: str) -> Any:
        assert HTML is not None
        escaped = (
            status_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return HTML(
            "<pre style='white-space: pre-wrap; font-family: monospace; "
            "padding: 10px; border: 1px solid #ddd; border-radius: 8px; "
            "background: #fafafa;'>"
            f"{escaped}"
            "</pre>"
        )

    def _make_figure_html(self, fig: plt.Figure) -> Any:
        assert HTML is not None
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return HTML(
            "<img "
            f"src='data:image/png;base64,{encoded}' "
            "style='max-width: 100%; height: auto; display: block;'/>"
        )

    def update_status(self, status_text: str) -> None:
        if self.enabled and display is not None and HTML is not None:
            status_html = self._make_status_html(status_text)
            if self.status_handle is None:
                self.status_handle = display(status_html, display_id=True)
            else:
                self.status_handle.update(status_html)
            return

        print(status_text, flush=True)

    def update(self, fig: plt.Figure, status_text: str) -> None:
        if self.enabled and display is not None and HTML is not None:
            figure_html = self._make_figure_html(fig)
            status_html = self._make_status_html(status_text)
            if self.figure_handle is None:
                self.figure_handle = display(figure_html, display_id=True)
                self.status_handle = display(status_html, display_id=True)
            else:
                self.figure_handle.update(figure_html)
                assert self.status_handle is not None
                self.status_handle.update(status_html)
            return

        print(status_text, flush=True)


def _is_notebook_runtime() -> bool:
    if get_ipython is None:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


def select_compute_device(prefer_gpu_if_available: bool) -> tuple[jax.Device, str]:
    if prefer_gpu_if_available:
        try:
            gpu_devices = jax.devices("gpu")
        except RuntimeError:
            gpu_devices = []
        if gpu_devices:
            return gpu_devices[0], "gpu"

    cpu_devices = jax.devices("cpu")
    return cpu_devices[0], "cpu"


def make_sample_action_fns() -> tuple[Any, Any]:
    train_sample_fn = jax.jit(
        lambda agent_state, obs, prng: agent_state.sample_action(
            obs,
            prng,
            deterministic=False,
        )
    )
    eval_sample_fn = jax.jit(
        lambda agent_state, obs, prng: agent_state.sample_action(
            obs,
            prng,
            deterministic=True,
        )
    )
    return train_sample_fn, eval_sample_fn


def build_eval_iters(
    outer_iters: int,
    num_evals: int,
    run_eval_at_step_zero: bool,
) -> set[int]:
    if outer_iters <= 0 or num_evals <= 0:
        return set()
    if run_eval_at_step_zero:
        return set(np.linspace(0, outer_iters - 1, min(num_evals, outer_iters), dtype=int))

    start_iter = 1 if outer_iters > 1 else 0
    eval_count = min(num_evals, outer_iters - start_iter)
    if eval_count <= 0:
        return set()
    return set(np.linspace(start_iter, outer_iters - 1, eval_count, dtype=int))


def compute_outer_iters(config: fpo.FpoConfig) -> tuple[int, int]:
    timesteps_per_outer_iter = config.iterations_per_env * config.num_envs
    outer_iters = config.num_timesteps // timesteps_per_outer_iter
    planned_total_timesteps = outer_iters * timesteps_per_outer_iter
    return outer_iters, planned_total_timesteps


class GymnasiumBatchEnv:
    """Small Python-side batch env."""

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
        active_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.current_obs is None:
            raise RuntimeError("Call reset() before step().")
        if active_mask is None:
            active_mask = np.ones(self.num_envs, dtype=np.bool_)

        next_obs_batch = self.current_obs.copy()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)

        for env_index, env in enumerate(self.envs):
            if not active_mask[env_index]:
                continue
            next_obs, reward, term, trunc, _ = env.step(
                self._format_action(action_batch[env_index])
            )

            next_obs_batch[env_index] = self._format_obs(next_obs)
            rewards[env_index] = float(reward)
            terminated[env_index] = bool(term)
            truncated[env_index] = bool(trunc)
        self.current_obs = next_obs_batch.copy()
        return next_obs_batch.copy(), rewards, terminated, truncated

    def reset_where(self, reset_mask: np.ndarray) -> np.ndarray:
        if self.current_obs is None:
            raise RuntimeError("Call reset() before reset_where().")
        for env_index, should_reset in enumerate(reset_mask):
            if should_reset:
                obs, _ = self.envs[env_index].reset(seed=self._next_seed())
                self.current_obs[env_index] = self._format_obs(obs)
        return self.current_obs.copy()

    def close(self) -> None:
        for env in self.envs:
            env.close()


@dataclass(slots=True)
class GymnasiumBatchedRolloutState:
    """Gymnasium analogue of the official batched rollout state."""

    env: GymnasiumBatchEnv
    env_obs: np.ndarray
    steps: np.ndarray
    inactive: np.ndarray
    num_envs: int
    prng: jax.Array

    @classmethod
    def init(
        cls,
        env: GymnasiumBatchEnv,
        prng: jax.Array,
        num_envs: int,
    ) -> "GymnasiumBatchedRolloutState":
        obs = env.reset()
        return cls(
            env=env,
            env_obs=obs,
            steps=np.zeros(num_envs, dtype=np.int32),
            inactive=np.zeros(num_envs, dtype=np.bool_),
            num_envs=num_envs,
            prng=prng,
        )

    def rollout(
        self,
        agent_state: fpo.FpoState,
        sample_action_fn: Any,
        compute_device: jax.Device,
        episode_length: int,
        iterations_per_env: int,
        auto_reset: bool = True,
    ) -> tuple["GymnasiumBatchedRolloutState", rollouts.TransitionStruct[Any]]:
        obs_seq = []
        next_obs_seq = []
        action_seq = []
        reward_seq = []
        truncation_seq = []
        discount_seq = []
        action_info_seq = []

        env_obs = self.env_obs.copy()
        steps = self.steps.copy()
        inactive = (
            np.zeros(self.num_envs, dtype=np.bool_)
            if auto_reset
            else self.inactive.copy()
        )
        prng = self.prng

        for _ in range(iterations_per_env):
            prng_act, prng_next = jax.random.split(prng)
            obs_jax = jax.device_put(jnp.asarray(env_obs, dtype=jnp.float32), compute_device)
            prng_act = jax.device_put(prng_act, compute_device)
            action, action_info = sample_action_fn(agent_state, obs_jax, prng_act)

            action_np = np.tanh(np.asarray(jax.device_get(action), dtype=np.float32))
            step_mask = ~inactive
            next_obs = env_obs.copy()
            reward = np.zeros(self.num_envs, dtype=np.float32)
            terminated = inactive.copy()
            env_truncated = np.zeros(self.num_envs, dtype=np.bool_)

            if bool(step_mask.any()):
                step_obs, step_reward, step_terminated, step_truncated = self.env.step(
                    action_np,
                    active_mask=step_mask,
                )
                next_obs[step_mask] = step_obs[step_mask]
                reward[step_mask] = step_reward[step_mask]
                terminated[step_mask] = step_terminated[step_mask]
                env_truncated[step_mask] = step_truncated[step_mask]

            next_steps = steps + (~inactive).astype(np.int32)
            truncation = np.logical_or(env_truncated, next_steps >= episode_length)
            done_env = terminated.astype(np.bool_)
            done_or_tr = np.logical_or(done_env, truncation)
            discount = 1.0 - done_env.astype(np.float32)

            obs_seq.append(obs_jax)
            next_obs_seq.append(jnp.asarray(next_obs, dtype=jnp.float32))
            action_seq.append(action)
            reward_seq.append(jnp.asarray(reward, dtype=jnp.float32))
            truncation_seq.append(jnp.asarray(truncation.astype(np.float32)))
            discount_seq.append(jnp.asarray(discount, dtype=jnp.float32))
            action_info_seq.append(action_info)

            if auto_reset:
                env_obs = self.env.reset_where(done_or_tr)
                steps = np.where(done_or_tr, 0, next_steps)
                inactive = np.zeros(self.num_envs, dtype=np.bool_)
            else:
                env_obs = next_obs
                steps = next_steps
                inactive = np.logical_or(inactive, done_or_tr)

            prng = prng_next

        next_state = GymnasiumBatchedRolloutState(
            env=self.env,
            env_obs=env_obs,
            steps=steps,
            inactive=inactive,
            num_envs=self.num_envs,
            prng=prng,
        )
        transitions = rollouts.TransitionStruct(
            obs=jnp.stack(obs_seq, axis=0),
            next_obs=jnp.stack(next_obs_seq, axis=0),
            action=jnp.stack(action_seq, axis=0),
            action_info=jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *action_info_seq),
            reward=jnp.stack(reward_seq, axis=0),
            truncation=jnp.stack(truncation_seq, axis=0),
            discount=jnp.stack(discount_seq, axis=0),
        )
        return next_state, transitions


def gymnasium_eval_policy(
    agent_state: fpo.FpoState,
    env_name: str,
    prng: jax.Array,
    num_envs: int,
    max_episode_length: int,
    sample_action_fn: Any,
    compute_device: jax.Device,
) -> dict[str, float]:
    seed = int(np.asarray(jax.device_get(jax.random.key_data(prng))).reshape(-1)[0])
    eval_env = GymnasiumBatchEnv(env_name, num_envs, seed=seed)
    try:
        rollout_state = GymnasiumBatchedRolloutState.init(eval_env, prng, num_envs)
        _, transitions = rollout_state.rollout(
            agent_state=agent_state,
            sample_action_fn=sample_action_fn,
            compute_device=compute_device,
            episode_length=max_episode_length,
            iterations_per_env=max_episode_length,
            auto_reset=False,
        )
        valid_mask = transitions.discount > 0.0
        rewards = jnp.sum(transitions.reward, axis=0)
        steps = jnp.sum(valid_mask, axis=0)

        return {
            "reward_mean": float(np.asarray(jnp.mean(rewards))),
            "reward_min": float(np.asarray(jnp.min(rewards))),
            "reward_max": float(np.asarray(jnp.max(rewards))),
            "reward_std": float(np.asarray(jnp.std(rewards))),
            "steps_mean": float(np.asarray(jnp.mean(steps))),
            "steps_min": float(np.asarray(jnp.min(steps))),
            "steps_max": float(np.asarray(jnp.max(steps))),
            "steps_std": float(np.asarray(jnp.std(steps))),
        }
    finally:
        eval_env.close()


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


def render_placeholder_dashboard(
    run_config: KaggleRunConfig,
    live_display: LiveDisplayManager,
    output_dir: Path,
    title_suffix: str,
    status_text: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panel_titles = (
        "Reward And Return",
        "Live Episode Returns",
        "Episode Length",
        "Optimization",
    )

    for ax, panel_title in zip(axes.flat, panel_titles, strict=False):
        ax.set_title(panel_title)
        ax.grid(alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            title_suffix,
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )

    fig.suptitle(
        f"FPO baseline on {run_config.env_name} | {title_suffix}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "live_training_dashboard.png", dpi=150, bbox_inches="tight")
    live_display.update(fig, f"{status_text}\noutput_dir={output_dir}")
    plt.close(fig)


def render_phase_progress(
    run_config: KaggleRunConfig,
    live_display: LiveDisplayManager,
    output_dir: Path,
    phase_title: str,
    status_text: str,
) -> None:
    render_placeholder_dashboard(
        run_config=run_config,
        live_display=live_display,
        output_dir=output_dir,
        title_suffix=phase_title,
        status_text=status_text,
    )


def render_live_dashboard(
    run_config: KaggleRunConfig,
    train_history: list[dict[str, Any]],
    eval_history: list[dict[str, Any]],
    episode_history: list[dict[str, Any]],
    live_display: LiveDisplayManager,
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

    status_text = (
        f"iter {outer_iter + 1}/{outer_iters}\n"
        f"env={run_config.env_name}\n"
        f"completed_episodes={len(episode_history)}\n"
        f"latest_train_mean_step_reward={train_mean_reward[-1]:.4f}\n"
        f"latest_train_mean_episode_return={train_mean_episode_return[-1]:.4f}\n"
        f"output_dir={output_dir}"
    )
    live_display.update(fig, status_text)
    plt.close(fig)


def render_final_plots(
    run_config: KaggleRunConfig,
    config: fpo.FpoConfig,
    train_history: list[dict[str, Any]],
    eval_history: list[dict[str, Any]],
    episode_history: list[dict[str, Any]],
    live_display: LiveDisplayManager,
    output_dir: Path,
    elapsed_seconds: float,
) -> dict[str, str]:
    if not train_history:
        return {}

    timesteps_per_outer_iter = config.iterations_per_env * config.num_envs
    train_outer_steps = np.asarray([row["step"] for row in train_history], dtype=np.int64)
    train_env_steps = (train_outer_steps + 1) * timesteps_per_outer_iter
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

    eval_outer_steps = np.asarray([row["step"] for row in eval_history], dtype=np.int64)
    eval_env_steps = eval_outer_steps * timesteps_per_outer_iter
    eval_rewards = np.asarray(
        [row["reward_mean"] for row in eval_history],
        dtype=np.float32,
    )
    eval_lengths = np.asarray(
        [row["steps_mean"] for row in eval_history],
        dtype=np.float32,
    )

    episode_indices = np.asarray(
        [row["episode_index"] for row in episode_history],
        dtype=np.int32,
    )
    episode_returns = np.asarray(
        [row["episode_return"] for row in episode_history],
        dtype=np.float32,
    )

    dashboard_path = output_dir / "final_training_dashboard.png"
    reward_plot_path = output_dir / "reward_return_vs_env_steps.png"

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(train_env_steps, train_mean_reward, label="train mean step reward")
    axes[0, 0].plot(
        train_env_steps,
        train_mean_episode_return,
        label="train mean completed episode return",
    )
    if eval_history:
        axes[0, 0].plot(
            eval_env_steps,
            eval_rewards,
            marker="o",
            label="eval mean episode return",
        )
    axes[0, 0].set_title("Reward And Return Vs Env Steps")
    axes[0, 0].set_xlabel("env steps")
    axes[0, 0].set_ylabel("reward")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

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
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, "No completed episodes yet", ha="center", va="center")
    axes[0, 1].set_title("Episode Return")
    axes[0, 1].set_xlabel("episode index")
    axes[0, 1].set_ylabel("return")
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(
        train_env_steps,
        train_mean_episode_length,
        label="train mean completed episode length",
    )
    if eval_history:
        axes[1, 0].plot(
            eval_env_steps,
            eval_lengths,
            marker="o",
            label="eval mean episode length",
        )
    axes[1, 0].set_title("Episode Length Vs Env Steps")
    axes[1, 0].set_xlabel("env steps")
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
    axes[1, 1].plot(train_env_steps, policy_loss, label="policy_loss")
    axes[1, 1].plot(train_env_steps, value_loss, label="v_loss")
    axes[1, 1].set_title("Optimization Vs Env Steps")
    axes[1, 1].set_xlabel("env steps")
    axes[1, 1].set_ylabel("loss")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle(
        (
            f"FPO baseline on {run_config.env_name} | "
            f"final summary | elapsed {elapsed_seconds / 60.0:.1f} min"
        ),
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    live_display.update(
        fig,
        (
            "training complete\n"
            f"env={run_config.env_name}\n"
            f"completed_episodes={len(episode_history)}\n"
            f"final_dashboard={dashboard_path}\n"
            f"output_dir={output_dir}"
        ),
    )
    plt.close(fig)

    reward_fig, reward_ax = plt.subplots(figsize=(11, 6))
    reward_ax.plot(train_env_steps, train_mean_reward, label="train mean step reward")
    reward_ax.plot(
        train_env_steps,
        train_mean_episode_return,
        label="train mean completed episode return",
    )
    if eval_history:
        reward_ax.plot(
            eval_env_steps,
            eval_rewards,
            marker="o",
            label="eval mean episode return",
        )
    reward_ax.set_title(f"Reward And Return Vs Env Steps | {run_config.env_name}")
    reward_ax.set_xlabel("env steps")
    reward_ax.set_ylabel("reward")
    reward_ax.legend()
    reward_ax.grid(alpha=0.2)
    reward_fig.tight_layout()
    reward_fig.savefig(reward_plot_path, dpi=150, bbox_inches="tight")
    plt.close(reward_fig)

    return {
        "final_training_dashboard": str(dashboard_path),
        "reward_return_vs_env_steps": str(reward_plot_path),
    }


def save_final_eval_video(
    run_config: KaggleRunConfig,
    agent_state: fpo.FpoState,
    sample_action_fn: Any,
    compute_device: jax.Device,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        return {
            "status": "skipped",
            "error": f"imageio not installed: {exc}",
        }

    video_env = gym.make(run_config.env_name, render_mode="rgb_array")
    try:
        obs, _ = video_env.reset(seed=run_config.seed + 20_000)
        prng = jax.device_put(jax.random.key(run_config.seed + 20_000), compute_device)
        frames: list[np.ndarray] = []
        episode_return = 0.0
        episode_steps = 0
        done = False

        while not done and episode_steps < run_config.final_video_max_steps:
            frame = video_env.render()
            if frame is not None:
                frames.append(np.asarray(frame))

            obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            obs_jax = jax.device_put(jnp.asarray(obs_batch, dtype=jnp.float32), compute_device)
            prng, step_key = jax.random.split(prng)
            step_key = jax.device_put(step_key, compute_device)
            action, _ = sample_action_fn(agent_state, obs_jax, step_key)
            action_np = np.tanh(np.asarray(jax.device_get(action), dtype=np.float32))[0]
            env_action = np.clip(
                action_np.reshape(video_env.action_space.shape),
                video_env.action_space.low,
                video_env.action_space.high,
            )
            obs, reward, terminated, truncated, _ = video_env.step(env_action)
            episode_return += float(reward)
            episode_steps += 1
            done = bool(terminated or truncated)

        final_frame = video_env.render()
        if final_frame is not None:
            frames.append(np.asarray(final_frame))
    finally:
        video_env.close()

    if not frames:
        return {
            "status": "skipped",
            "error": "rendered video contained no frames",
        }

    video_path = output_dir / "final_policy_eval.mp4"
    try:
        imageio.mimsave(video_path, frames, fps=run_config.final_video_fps)
        return {
            "status": "saved",
            "path": str(video_path),
            "format": "mp4",
            "episode_return": float(episode_return),
            "episode_length": int(episode_steps),
            "num_frames": len(frames),
        }
    except Exception as mp4_error:
        gif_path = output_dir / "final_policy_eval.gif"
        try:
            imageio.mimsave(
                gif_path,
                frames,
                format="GIF",
                duration=1.0 / max(run_config.final_video_fps, 1),
            )
            return {
                "status": "saved",
                "path": str(gif_path),
                "format": "gif",
                "episode_return": float(episode_return),
                "episode_length": int(episode_steps),
                "num_frames": len(frames),
                "mp4_error": str(mp4_error),
            }
        except Exception as gif_error:
            return {
                "status": "failed",
                "error": f"mp4_error={mp4_error}; gif_error={gif_error}",
                "episode_return": float(episode_return),
                "episode_length": int(episode_steps),
                "num_frames": len(frames),
            }


def display_final_video(video_info: dict[str, Any]) -> None:
    if (
        Video is None
        or display is None
        or not _is_notebook_runtime()
        or video_info.get("status") != "saved"
        or video_info.get("format") != "mp4"
        or "path" not in video_info
    ):
        return
    display(Video(video_info["path"], embed=True))


def save_training_artifacts(
    run_config: KaggleRunConfig,
    train_history: list[dict[str, Any]],
    eval_history: list[dict[str, Any]],
    episode_history: list[dict[str, Any]],
    output_dir: Path,
    extra_summary: dict[str, Any] | None = None,
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
    if extra_summary is not None:
        summary.update(extra_summary)
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
    live_display = LiveDisplayManager.create(enabled=run_config.show_live_plots)
    compute_device, compute_backend = select_compute_device(
        run_config.prefer_gpu_if_available
    )
    train_sample_action_fn, eval_sample_action_fn = make_sample_action_fns()
    render_phase_progress(
        run_config=run_config,
        live_display=live_display,
        output_dir=output_dir,
        phase_title="Initializing",
        status_text=(
            f"starting run for {run_config.env_name}\n"
            f"num_envs={run_config.num_envs}\n"
            f"num_timesteps={run_config.num_timesteps}\n"
            f"num_updates_per_batch={run_config.num_updates_per_batch}\n"
            f"jax_backend={compute_backend}\n"
            f"jax_device={compute_device}"
        ),
    )

    render_phase_progress(
        run_config=run_config,
        live_display=live_display,
        output_dir=output_dir,
        phase_title="Creating Environments",
        status_text=(
            f"creating {run_config.num_envs} Gymnasium envs\n"
            f"env={run_config.env_name}"
        ),
    )
    train_env = GymnasiumBatchEnv(
        env_name=run_config.env_name,
        num_envs=run_config.num_envs,
        seed=run_config.seed,
    )
    render_phase_progress(
        run_config=run_config,
        live_display=live_display,
        output_dir=output_dir,
        phase_title="Resetting Environments",
        status_text=(
            f"resetting {run_config.num_envs} Gymnasium envs\n"
            f"env={run_config.env_name}"
        ),
    )
    rollout_state = GymnasiumBatchedRolloutState.init(
        train_env,
        jax.device_put(jax.random.key(run_config.seed + 1), compute_device),
        run_config.num_envs,
    )
    render_phase_progress(
        run_config=run_config,
        live_display=live_display,
        output_dir=output_dir,
        phase_title="Initializing Agent",
        status_text=(
            "building FPO state\n"
            "compiling initial JAX functions"
        ),
    )
    agent_state = fpo.FpoState.init(
        prng=jax.random.key(run_config.seed),
        env=train_env,
        config=config,
    )
    agent_state = jax.device_put(agent_state, compute_device)

    outer_iters, planned_total_timesteps = compute_outer_iters(config)
    if outer_iters <= 0:
        raise ValueError(
            "num_timesteps is too small for one outer iteration. "
            f"Need at least {config.iterations_per_env * config.num_envs}, "
            f"got {config.num_timesteps}."
        )
    eval_iters = build_eval_iters(
        outer_iters=outer_iters,
        num_evals=config.num_evals,
        run_eval_at_step_zero=run_config.run_eval_at_step_zero,
    )
    render_phase_progress(
        run_config=run_config,
        live_display=live_display,
        output_dir=output_dir,
        phase_title="Ready To Train",
        status_text=(
            f"initialized env and agent\n"
            f"outer_iters={outer_iters}\n"
            f"iterations_per_env={config.iterations_per_env}\n"
            f"requested_num_timesteps={config.num_timesteps}\n"
            f"planned_total_timesteps={planned_total_timesteps}\n"
            f"num_updates_per_batch={config.num_updates_per_batch}\n"
            f"eval_num_envs={run_config.eval_num_envs}\n"
            f"jax_backend={compute_backend}\n"
            "collecting first rollout next"
        ),
    )

    train_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []
    episode_history: list[dict[str, Any]] = []
    train_episode_tracker = TrainEpisodeTracker.init(run_config.num_envs)

    start_time = time.time()
    try:
        for outer_iter in range(outer_iters):
            if outer_iter in eval_iters:
                phase_status = (
                    f"iter {outer_iter + 1}/{outer_iters}\n"
                    f"phase=evaluation\n"
                    f"env={run_config.env_name}\n"
                    f"eval_num_envs={run_config.eval_num_envs}\n"
                    f"episode_length={config.episode_length}"
                )
                if not train_history:
                    render_phase_progress(
                        run_config=run_config,
                        live_display=live_display,
                        output_dir=output_dir,
                        phase_title="Evaluating Policy",
                        status_text=phase_status,
                    )
                else:
                    live_display.update_status(phase_status)
                eval_row = {
                    "step": outer_iter,
                    **gymnasium_eval_policy(
                        agent_state=agent_state,
                        env_name=run_config.env_name,
                        prng=jax.device_put(
                            jax.random.key(run_config.seed + 10_000 + outer_iter),
                            compute_device,
                        ),
                        num_envs=run_config.eval_num_envs,
                        max_episode_length=config.episode_length,
                        sample_action_fn=eval_sample_action_fn,
                        compute_device=compute_device,
                    ),
                }
                eval_history.append(eval_row)

            phase_status = (
                f"iter {outer_iter + 1}/{outer_iters}\n"
                f"phase=collect_rollout\n"
                f"iterations_per_env={config.iterations_per_env}\n"
                f"num_envs={run_config.num_envs}\n"
                f"completed_episodes_total={len(episode_history)}"
            )
            if not train_history:
                render_phase_progress(
                    run_config=run_config,
                    live_display=live_display,
                    output_dir=output_dir,
                    phase_title="Collecting Rollout",
                    status_text=phase_status,
                )
            else:
                live_display.update_status(phase_status)
            rollout_state, transitions = rollout_state.rollout(
                agent_state=agent_state,
                sample_action_fn=train_sample_action_fn,
                compute_device=compute_device,
                episode_length=config.episode_length,
                iterations_per_env=config.iterations_per_env,
                auto_reset=True,
            )
            phase_status = (
                f"iter {outer_iter + 1}/{outer_iters}\n"
                f"phase=training_step\n"
                f"completed_episodes_total={len(episode_history)}\n"
                "running FPO update"
            )
            if not train_history:
                render_phase_progress(
                    run_config=run_config,
                    live_display=live_display,
                    output_dir=output_dir,
                    phase_title="Applying FPO Update",
                    status_text=phase_status,
                )
            else:
                live_display.update_status(phase_status)
            agent_state, metrics = agent_state.training_step(transitions)

            batch_episode_returns, batch_episode_lengths = train_episode_tracker.update_from_transitions(
                transitions
            )
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
                        if batch_episode_returns.size > 0
                        else float("nan")
                    ),
                    "mean_completed_episode_length": (
                        float(np.mean(batch_episode_lengths))
                        if batch_episode_lengths.size > 0
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
                    live_display=live_display,
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

    final_artifacts: dict[str, Any] = {}
    if run_config.save_final_plots:
        live_display.update_status(
            f"training finished\nphase=saving_final_plots\noutput_dir={output_dir}"
        )
        final_artifacts["final_plots"] = render_final_plots(
            run_config=run_config,
            config=config,
            train_history=train_history,
            eval_history=eval_history,
            episode_history=episode_history,
            live_display=live_display,
            output_dir=output_dir,
            elapsed_seconds=time.time() - start_time,
        )

    if run_config.save_final_video:
        live_display.update_status(
            f"training finished\nphase=saving_final_video\noutput_dir={output_dir}"
        )
        final_artifacts["final_video"] = save_final_eval_video(
            run_config=run_config,
            agent_state=agent_state,
            sample_action_fn=eval_sample_action_fn,
            compute_device=compute_device,
            output_dir=output_dir,
        )
        display_final_video(final_artifacts["final_video"])

    save_training_artifacts(
        run_config=run_config,
        train_history=train_history,
        eval_history=eval_history,
        episode_history=episode_history,
        output_dir=output_dir,
        extra_summary=final_artifacts,
    )
    final_plots = final_artifacts.get("final_plots", {})
    final_video = final_artifacts.get("final_video", {})
    live_display.update_status(
        "run complete\n"
        f"output_dir={output_dir}\n"
        f"final_dashboard={final_plots.get('final_training_dashboard', 'not_saved')}\n"
        f"reward_plot={final_plots.get('reward_return_vs_env_steps', 'not_saved')}\n"
        f"final_video={final_video.get('path', final_video.get('status', 'not_saved'))}"
    )

    return agent_state, train_history, eval_history, episode_history


def main() -> None:
    run_config = KaggleRunConfig(
        env_name="Ant-v4",
        output_dir=str(REPO_ROOT / "kaggle_outputs" / "fpo_ant_v4"),
    )
    train_gymnasium_baseline(run_config)


if __name__ == "__main__":
    main()
