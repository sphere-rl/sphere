import multiprocessing as mp
from typing import Any, Callable, Dict, List, Optional
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv as sb3SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize as sb3VecNormalize
from src.utils.sb3_env_util_shim import is_wrapped


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[Dict[str, Any]] = {}

    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                # convert to SB3 VecEnv api
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    if "render_array" in info:
                        info["terminal_render_array"] = info["render_array"]
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class SubprocVecEnv(sb3SubprocVecEnv):
    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        self.observations = np.zeros((n_envs, ) + observation_space.shape,dtype=observation_space.dtype)
        self.rewards = np.zeros((n_envs,), dtype=np.float32)
        self.dones = np.zeros((n_envs,), dtype=bool)


        super(sb3SubprocVecEnv, self).__init__(len(env_fns), observation_space, action_space)

    def step_wait(self) -> VecEnvStepReturn:
        results = []
        for env_id, remote in enumerate(self.remotes):
            result = remote.recv()
            self.observations[env_id][:] = result[0][:]
            self.rewards[env_id] = result[1]
            self.dones[env_id] = result[2]
            results.append(result[3:])

        self.waiting = False
        infos, self.reset_infos = zip(*results)  # type: ignore[assignment]
        return self.observations.copy(), self.rewards.copy(), self.dones.copy(), infos  # type: ignore[return-value]

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = []
        for env_id, remote in enumerate(self.remotes):
            result = remote.recv()
            self.observations[env_id,:] = result[0]
            results.append(result[1])
        self.reset_infos = results  # type: ignore[assignment]
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self.observations.copy()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                pass
        for process in self.processes:
            process.join(timeout=5.0)
        for process in self.processes:
            if process.is_alive():
                print(f"Process {process.pid} did not exit gracefully, terminating.")
                process.terminate()
                process.join()
        for remote in self.remotes:
            try:
                remote.close()
            except Exception as e:
                print(f"Error closing remote pipe: {e}")

        self.closed = True
    
class VecNormalize(sb3VecNormalize):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.reset_infos
