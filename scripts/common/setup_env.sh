#!/bin/bash

# Unified Environment Setup
# Remove duplicated environment variable setup across scripts

# Activate virtual environment
if [[ -f ".venv/bin/activate" ]]; then
  # Local project virtual environment (recommended)
  source .venv/bin/activate
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  # Already running inside a virtual environment
  :
else
  echo "Warning: .venv not found and no active virtualenv detected; using system Python."
fi

# JAX/CPU/GPU environment variables
# Performance notes:
# Tune `OMP_NUM_THREADS` for your machine/workload.
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export JAX_PLATFORM_NAME=gpu
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"
