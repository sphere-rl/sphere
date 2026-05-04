from gymnasium.envs.registration import register

# Minimal environment registration for the experiments used by the provided scripts.
# Keep this module import-light: defer heavy third-party imports until an env is created.

_HUMANOIDBENCH_TASKS = ("stand", "walk", "pole", "slide", "run")
for task in _HUMANOIDBENCH_TASKS:
    register(
        id=f"h1-{task}-customized-v0",
        entry_point="env.customized_humanoid_bench:HumanoidEnv",
        max_episode_steps=1000,
        kwargs={"robot": "h1", "control": "pos", "task": task, "render_mode": None},
    )
