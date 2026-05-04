import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dm_control.mujoco import index
from dm_control.mujoco.engine import NamedIndexStructs
from humanoid_bench.dmc_deps.dmc_wrapper import MjDataWrapper, MjModelWrapper
from humanoid_bench.wrappers import (
    SingleReachWrapper,
    DoubleReachAbsoluteWrapper,
    DoubleReachRelativeWrapper,
    BlockedHandsLocoWrapper,
    ObservationWrapper,
)
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 3.0, #5.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": 0, #-20.0,
    "azimuth": 90,
}
DEFAULT_RANDOMNESS = 0.01

from humanoid_bench.env import TASKS, ROBOTS
from humanoid_bench.envs.kitchen import Kitchen
from humanoid_bench.envs.cube import Cube
from humanoid_bench.envs.bookshelf import BookshelfSimple, BookshelfHard
from humanoid_bench import env as orginal_env


class HumanoidEnv(orginal_env.HumanoidEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        robot=None,
        control=None,
        task=None,
        render_mode=None,
        width=256,
        height=256,
        randomness=DEFAULT_RANDOMNESS,
        **kwargs,
    ):
        assert robot and control and task, f"{robot} {control} {task}"
        gym.utils.EzPickle.__init__(self, metadata=self.metadata)

        asset_path = os.path.join(os.path.dirname(orginal_env.__file__), "assets")
        model_path = f"envs/{robot}_{control}_{task}.xml"
        model_path = os.path.join(asset_path, model_path)

        self.robot = ROBOTS[robot](self)
        task_info = TASKS[task](self.robot, None, **kwargs)

        self.obs_wrapper = kwargs.get("obs_wrapper", None)
        if self.obs_wrapper is not None:
            self.obs_wrapper = kwargs.get("obs_wrapper", "False").lower() == "true"
        else:
            self.obs_wrapper = False

        self.blocked_hands = kwargs.get("blocked_hands", None)
        if self.blocked_hands is not None:
            self.blocked_hands = kwargs.get("blocked_hands", "False").lower() == "true"
        else:
            self.blocked_hands = False

        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip=task_info.frame_skip,
            observation_space=task_info.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=task_info.camera_name,
        )

        self.action_high = self.action_space.high
        self.action_low = self.action_space.low
        self.action_space = Box(
            low=-1, high=1, shape=self.action_space.shape, dtype=np.float32
        )
        self.camera_azimuths = kwargs.get("camera_azimuths", [90])
        self.task = TASKS[task](self.robot, self, **kwargs)
        # Wrap task to fix render method BEFORE other wrappers
        self.task = TaskWrapper(self.task)
        if self.blocked_hands:
            self.task = BlockedHandsLocoWrapper(self.task, **kwargs)
            # Re-wrap with TaskWrapper after blocked hands wrapper
            self.task = TaskWrapper(self.task)

        # Wrap for hierarchical control
        if (
            "policy_type" in kwargs
            and kwargs["policy_type"]
            and kwargs["policy_type"] is not None
            and kwargs["policy_type"] != "flat"
        ):
            if kwargs["policy_type"] == "reach_single":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = SingleReachWrapper(self.task, **kwargs)
            elif kwargs["policy_type"] == "reach_double_absolute":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = DoubleReachAbsoluteWrapper(self.task, **kwargs)
            elif kwargs["policy_type"] == "reach_double_relative":
                assert "policy_path" in kwargs and kwargs["policy_path"] is not None
                self.task = DoubleReachRelativeWrapper(self.task, **kwargs)
            else:
                raise ValueError(f"Unknown policy_type: {kwargs['policy_type']}")
            # Re-wrap with TaskWrapper after hierarchical control wrapper
            self.task = TaskWrapper(self.task)
        

        if self.obs_wrapper:
            # Note that observation wrapper is not compatible with hierarchical policy
            self.task = ObservationWrapper(self.task, **kwargs)
            # Re-wrap with TaskWrapper after observation wrapper
            self.task = TaskWrapper(self.task)
            self.observation_space = self.task.observation_space

        # Keyframe
        self.keyframe = (
            self.model.key(kwargs["keyframe"]).id if "keyframe" in kwargs else 0
        )

        self.randomness = randomness
        if isinstance(self.task, (BookshelfHard, BookshelfSimple, Kitchen, Cube)):
            self.randomness = 0

        # Set up named indexing.
        data = MjDataWrapper(self.data)
        model = MjModelWrapper(self.model)
        axis_indexers = index.make_axis_indexers(model)
        self.named = NamedIndexStructs(
            model=index.struct_indexer(model, "mjmodel", axis_indexers),
            data=index.struct_indexer(data, "mjdata", axis_indexers),
        )

        assert self.robot.dof + self.task.dof == len(data.qpos), (
            self.robot.dof,
            self.task.dof,
            len(data.qpos),
        )
        # Episode return accumulator for paper-standard success computation
        self._episode_return = 0.0  # scalar
    def step(self, action):
        """Step with paper-standard success flag at episode end.

        - Accumulates per-episode return (scalar)
        - On termination/truncation, sets info['success'] = (episode_return >= success_bar)
        """
        obs, reward, terminated, truncated, info = super().step(action)
        self._episode_return += float(reward)  # scalar
        success_flag = float(self._episode_return >= self.task.success_bar)
        info["success"] = success_flag
        if terminated or truncated:
            self._episode_return = 0.0
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        """Reset episode accumulators before delegating to parent reset.

        This guarantees that per-episode cumulative return starts from zero
        even when episodes are truncated by an outer TimeLimit wrapper.
        """
        self._episode_return = 0.0
        return super().reset(*args, **kwargs)
# Add Task wrapper to fix render method
class TaskWrapper:
    """Wrapper for humanoid-bench tasks to fix render method incompatibility"""
    
    def __init__(self, task):
        self.task = task
        self._env = task._env
        
    def __getattr__(self, name):
        return getattr(self.task, name)
    
    def render(self):
        # Fixed render method that only passes the render_mode parameter
        if self._env.render_mode is None:
            return None
        return self._env.mujoco_renderer.render(self._env.render_mode)
