from __future__ import annotations

import os
import random
from typing import Optional

import jax.numpy as jnp
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_egl_env_vars() -> None:
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["EGL_PLATFORM"] = "device"


def set_osmesa_env_vars() -> None:
    os.environ["MUJOCO_GL"] = "osmesa"


def get_linear_fn(start: float, end: float, end_fraction: float):
    def func(progress_remaining: float) -> float:
        return jnp.where(
            (1 - progress_remaining) > end_fraction,
            end,
            start + (1 - progress_remaining) * (end - start) / end_fraction,
        )

    return func


def parse_linear_scheduler(value_in_config):
    """Parse linear scheduler strings of the form `lin_start_end_frac`."""
    if isinstance(value_in_config, str):
        if not value_in_config.startswith("lin"):
            raise ValueError("Linear schedule must start with `lin`.")
        parts = value_in_config.split("_")
        start_value = float(parts[1])
        if len(parts) == 2:
            end_value = 0.0
            frac = 1.0
        else:
            end_value = float(parts[2])
            frac = float(parts[3]) if len(parts) >= 4 else 1.0
        return get_linear_fn(start_value, end_value, frac)
    return value_in_config
