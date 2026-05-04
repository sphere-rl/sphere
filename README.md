<div align="center">

# SPHERE: Mitigating the Loss of Spectral Plasticity in Mixture-of-Experts Reinforcement Learning

**Official implementation of SPHERE**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

## Overview

SPHERE is a spectral-plasticity regularization method for Mixture-of-Experts reinforcement learning. This repository contains the official training code used for the HumanoidBench experiments in the paper.

## Quick Start

### Setup

Create a Python 3.11 environment and install dependencies using your preferred workflow.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync
```

The HumanoidBench dependency is pulled from `https://github.com/liruiluo/humanoid-bench.git` via `pyproject.toml`.

### Run experiments

MoE-PPO baseline:

```bash
bash scripts/moe/humanoidbench/run_moe_ppo_topk.sh
```

MoE-PPO + SPHERE:

```bash
bash scripts/moe/humanoidbench/run_moe_ppo_topk_sphere_gradscale.sh
```

The scripts run five seeds (`0,1,2,3,4`) on five HumanoidBench tasks (`h1_stand`, `h1_walk`, `h1_pole`, `h1_slide`, `h1_run`) by default. Outputs are written under `outputs/`.

Note: project shell scripts automatically source the local virtual environment via `scripts/common/setup_env.sh` (i.e., `source .venv/bin/activate`). Run scripts directly with `bash` without wrapping them in `uv run`.

## Citation

If you find this repository useful, please cite the SPHERE paper. The BibTeX entry will be added after the camera-ready metadata is finalized.
