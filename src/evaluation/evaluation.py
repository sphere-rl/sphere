import warnings
from typing import Any, Callable, Optional, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
    if not isinstance(env, VecEnv):
        # Wrap single environment as DummyVecEnv if not vectorized
        env = DummyVecEnv([lambda: env])

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Distribute evaluation episodes as evenly as possible across sub-environments
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()  # Reset environment and get initial observations
    states = None  # RNN states (if applicable)
    episode_starts = np.ones((env.num_envs,), dtype=bool)  # Mark whether each environment episode is just starting

    # --- Main evaluation loop ---
    while (episode_counts < episode_count_targets).any():  # Continue while any environment hasn't reached target episodes
        # Predict actions from model
        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,  # Whether to use deterministic actions
        )

        # --- Environment interaction ---
        new_observations, rewards, dones, infos = env.step(actions)

        # --- Standard logging and episode handling ---
        current_rewards += rewards  # Accumulate current episode rewards
        current_lengths += 1        # Accumulate current episode lengths
        for i in range(n_envs):     # Iterate through each parallel environment
            if episode_counts[i] < episode_count_targets[i]:  # If environment hasn't completed target episodes
                done = dones[i]
                info = infos[i]
                # Safely get success flag, default to False if not present
                is_success = info.get("is_success", info.get("success", False))
                episode_starts[i] = done  # If done=True, next loop's episode_start=True

                # Execute callback function (if provided)
                if callback is not None:
                    callback(locals(), globals())

                # --- Episode termination handling ---
                if dones[i]:
                    episode_rewards.append(float(current_rewards[i]))
                    episode_lengths.append(int(current_lengths[i]))
                    episode_successes.append(float(is_success))
                    episode_counts[i] += 1

                    # Reset current episode counters for this environment
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        # Update observations for next loop iteration
        observations = new_observations

    # --- Calculate and return results ---
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    std_reward = np.std(episode_rewards) if episode_rewards else 0.0

    # Check if mean reward exceeds threshold (if set)
    if reward_threshold is not None:
        if not episode_rewards:  # Handle case where no episodes completed
            if warn:
                warnings.warn(
                    "No evaluation episodes completed; cannot check reward_threshold.",
                    UserWarning,
                )
        else:
            assert mean_reward > reward_threshold, (
                f"Mean reward {mean_reward:.2f} below threshold {reward_threshold:.2f}"
            )

    # Return detailed episode info or mean/std based on parameter
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_successes
    return mean_reward, std_reward, episode_successes  # Also return success list
