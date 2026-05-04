#!/bin/bash

# Top-K MoE PPO (HumanoidBench), no SPHERE regularization
# Usage: bash scripts/moe/humanoidbench/run_moe_ppo_topk.sh [additional Hydra params]

set -euo pipefail

source scripts/common/setup_env.sh

DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Using unified timestamp: $DATE_TIME"

env HYDRA_FULL_ERROR=1 \
  python src/training/train.py \
  --config-name=algo/moe/moe_ppo_topk \
  --multirun \
  seed=0,1,2,3,4 \
  tasks=[h1_stand,h1_walk,h1_pole,h1_slide,h1_run] \
  num_envs=128 \
  use_vecnormalize=true \
  vecnorm_norm_reward=false \
  total_timesteps=10000000 \
  date_time=$DATE_TIME \
  agent.n_steps=16 \
  agent.batch_size=64 \
  width=null \
  base_width=256 \
  arch_depth=3 \
  param_mult=1 \
  +agent.target_kl=0.03 \
  'agent.learning_rate=${linear_scheduling:lin_3e-4_1e-4_0.5}' \
  'agent.policy_kwargs.activation_fn=${nn:relu}' \
  run_name_prefix=moe_ppo_topk_h1_crl\${use_crl}_N\${agent.policy_kwargs.n_experts}_k\${agent.policy_kwargs.top_k}_pm\${param_mult}_ln\${agent.policy_kwargs.use_layer_norm}_l2\${agent.policy_kwargs.use_l2_norm}_amo\${agent.policy_kwargs.apply_to.actor}_cmo\${agent.policy_kwargs.apply_to.critic} \
  "$@"

echo "HumanoidBench MoE-PPO TopK runs completed"
