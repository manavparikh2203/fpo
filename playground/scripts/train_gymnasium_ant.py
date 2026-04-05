import os
os.environ['MUJOCO_GL'] = 'egl'

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import optax

# Simple MLP policy
def mlp_init(rng, layers):
    params = []
    for i in range(len(layers) - 1):
        rng, rng1, rng2 = jax.random.split(rng, 3)
        w = jax.random.normal(rng1, (layers[i], layers[i+1])) * 0.1
        b = jnp.zeros(layers[i+1])
        params.append((w, b))
    return params

def mlp_fwd(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, w) + b)
    w, b = params[-1]
    return jnp.dot(x, w) + b

# Simple PPO-like training
def train_ant(env_name='Ant-v5', num_steps=100000, log_interval=1000):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    rng = jax.random.key(42)
    policy_params = mlp_init(rng, [obs_dim, 64, 64, act_dim])

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(policy_params)

    rewards_history = []
    step = 0

    while step < num_steps:
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done and episode_steps < 1000:
            rng, rng_act = jax.random.split(rng)
            action = mlp_fwd(policy_params, obs)
            action = np.tanh(action)  # Assuming continuous action space
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_steps += 1
            step += 1

            if step % log_interval == 0:
                rewards_history.append((step, episode_reward))
                print(f"Step {step}: Episode reward {episode_reward}")

        # Simple policy update (placeholder, not full PPO)
        # In real PPO, collect trajectories, compute advantages, update
        # For demo, just log

    env.close()

    # Save to CSV
    df = pd.DataFrame(rewards_history, columns=['step', 'reward'])
    output_dir = Path('./gym_results')
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / 'ant_rewards.csv', index=False)

    # Plot
    plt.plot(df['step'], df['reward'])
    plt.title('Ant-v5 Training Rewards')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.savefig(output_dir / 'ant_reward_curve.png')
    plt.show()

if __name__ == '__main__':
    train_ant()