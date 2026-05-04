from __future__ import annotations

from collections import deque
import os
from typing import Any, Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback as SB3EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

from src.evaluation.evaluation import evaluate_policy


class LogTrainingStats(BaseCallback):
    def _on_training_start(self) -> None:
        reset_num_timesteps = self.locals["reset_num_timesteps"]
        if reset_num_timesteps:
            self.model.train_stats_buffer = {}
            info_keys = getattr(self.model, "info_keys_to_print", [])
            for key in info_keys:
                self.model.train_stats_buffer[key] = deque(maxlen=self.model._stats_window_size)
        return super()._on_training_start()

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        info_keys = getattr(self.model, "info_keys_to_print", [])
        for info in infos:
            for key in info_keys:
                if key in info:
                    self.model.train_stats_buffer[key].append(info[key])
        return True


class EvalCallback(SB3EvalCallback):
    """Lightweight eval callback: logs reward/length/success and saves `.npz` results."""

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way; "
                        "see SB3 EvalCallback + VecNormalize documentation."
                    ) from e

            episode_rewards, episode_lengths, episode_successes = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=None,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
            std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
            mean_ep_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
            mean_success = float(np.mean(episode_successes)) if episode_successes else 0.0
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f}")
                print(f"Success: {100.0 * mean_success:.2f}%")

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/mean_success", mean_success)

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

