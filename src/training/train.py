from __future__ import annotations

from datetime import datetime
import multiprocessing as mp
import os
import os.path as osp
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

import env  # noqa: F401  # Registers custom environments
from src.training.callbacks import EvalCallback, LogTrainingStats
from src.training.training_utils import set_egl_env_vars, set_osmesa_env_vars, set_seed
from src.utils.hydra_resolvers import register_omega_resolvers
from src.utils.subproc_vec_env import SubprocVecEnv, VecNormalize

mp.set_start_method("spawn", force=True)
register_omega_resolvers()


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def run_single_task(config: OmegaConf, task: str, task_index: int, prev_agent_path: str = "") -> str:
    if config.rendering_backend == "egl":
        set_egl_env_vars()
    elif config.rendering_backend == "osmesa":
        set_osmesa_env_vars()
    else:
        raise NotImplementedError(f"Unknown rendering_backend: {config.rendering_backend}")

    task_name = config.task_names[task]
    env_kwargs = OmegaConf.to_container(config.env_config, resolve=True) if config.get("env_config") else {}

    train_env = make_vec_env(
        task_name,
        n_envs=int(config.num_envs),
        seed=int(config.seed),
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
        vec_env_kwargs={},
    )
    eval_env = make_vec_env(
        task_name,
        n_envs=2,
        seed=int(config.seed),
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
        vec_env_kwargs={},
    )

    if config.get("use_vecnormalize", False):
        norm_reward = bool(config.get("vecnorm_norm_reward", False))
        if prev_agent_path and bool(config.get("use_crl", False)):
            vecnorm_path = osp.join(osp.dirname(prev_agent_path), "vecnormalize.pkl")
            if osp.exists(vecnorm_path):
                train_env = VecNormalize.load(vecnorm_path, train_env)
                train_env.training = True
                train_env.norm_reward = norm_reward
            else:
                train_env = VecNormalize(train_env, norm_reward=norm_reward)
        else:
            train_env = VecNormalize(train_env, norm_reward=norm_reward)
        eval_env = VecNormalize(eval_env, training=False, norm_reward=norm_reward)
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms

    set_seed(int(config.seed))

    date_time = config.date_time or _now_ts()
    task_run_name = f"{config.run_name_prefix}#{config.algo}#{task}#seed{config.seed}#id{task_index}"
    output_dir = osp.join(config.get("outputs_dir", "outputs"), config.algo, date_time, task_run_name, task_name)
    os.makedirs(output_dir, exist_ok=True)

    callback_list = [
        EvalCallback(
            eval_env=eval_env,
            log_path=output_dir,
            eval_freq=int(config.eval_freq) // int(config.num_envs),
            render=bool(config.eval_render),
            n_eval_episodes=10,
        ),
        LogTrainingStats(),
    ]

    agent_partial = instantiate(config.agent, _convert_="all")
    if prev_agent_path and bool(config.get("use_crl", False)) and str(config.get("mode", "train")) == "train":
        agent_class = agent_partial.func
        agent = agent_class.load(prev_agent_path, env=train_env)
        agent.num_timesteps = 0
    else:
        agent = agent_partial(env=train_env)

    new_logger = configure(osp.join(output_dir, "tb_logs"), ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    log_interval = 10
    learn_cfg = config.get("learn")
    if isinstance(learn_cfg, dict):
        log_interval = int(learn_cfg.get("log_interval", log_interval))
    else:
        log_interval = int(getattr(learn_cfg, "log_interval", log_interval))

    agent.learn(
        total_timesteps=int(config.total_timesteps),
        callback=CallbackList(callback_list),
        log_interval=log_interval,
        progress_bar=True,
    )

    checkpoint_path = osp.join(output_dir, "checkpoints.zip")
    agent.save(checkpoint_path)

    if config.get("use_vecnormalize", False) and isinstance(train_env, VecNormalize):
        train_env.save(osp.join(output_dir, "vecnormalize.pkl"))

    train_env.close()
    eval_env.close()
    return checkpoint_path


def run_task_sequence(config: OmegaConf) -> None:
    if config.get("date_time") is None:
        config.date_time = _now_ts()
    tasks = list(config.get("tasks", []))
    prev_agent_path = ""
    for i, task in enumerate(tasks):
        ckpt = run_single_task(config, task, i, prev_agent_path=prev_agent_path)
        if bool(config.get("use_crl", False)) and str(config.get("mode", "train")) == "train":
            prev_agent_path = ckpt


@hydra.main(config_path="../../configs", config_name="algo/ppo/ppo", version_base="1.1")
def main(config: OmegaConf):
    run_task_sequence(config)


if __name__ == "__main__":
    main()
