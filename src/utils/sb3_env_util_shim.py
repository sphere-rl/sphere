"""SB3 env_util shim to avoid function-level imports.

Provides a stable is_wrapped function imported at module load time,
so callers (e.g., subproc workers) can use it without local imports.
"""

from stable_baselines3.common.env_util import is_wrapped as sb3_is_wrapped  # noqa: F401


def is_wrapped(env, wrapper):
    return sb3_is_wrapped(env, wrapper)
