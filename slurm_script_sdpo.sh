#!/bin/bash
#SBATCH --job-name=sdpo_tri         # ← Must include your alias
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --gres=gpu:2                          # ← Number of GPUs
#SBATCH --cpus-per-task=8                     # ← MINIMUM 4 required
#SBATCH --mem=100G
#SBATCH --time=05:00:00                       # ← Max 12:00:00 for GPU jobs
#SBATCH --output=/scratch/hrishikesh/users/tri/logs/job-%j.log
#SBATCH --error=/scratch/hrishikesh/users/tri/logs/job-%j.err

# ── Identity & paths (hardcoded per student, never inherited from environment) ──
export USER_ALIAS=tri
export ENV_PATH=/scratch/hrishikesh/users/$USER_ALIAS/conda_envs/sdpo_env

# ── Shared model cache (read-only, used by HuggingFace AND vLLM) ──
export HF_HOME=/scratch/hrishikesh/shared_models/huggingface

# ── Per-user caches ──
export PIP_CACHE_DIR=/scratch/hrishikesh/users/$USER_ALIAS/pip_cache
export TMPDIR=/scratch/hrishikesh/users/$USER_ALIAS/tmp

mkdir -p $PIP_CACHE_DIR $TMPDIR
mkdir -p /scratch/hrishikesh/users/$USER_ALIAS/logs

# ── Run your script ──
cd /home/hrishikesh/SDPO
./run_attn_map.sh