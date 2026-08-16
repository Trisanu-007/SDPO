#!/bin/bash
#SBATCH --job-name=sdpo_tri         # ← Must include your alias
#SBATCH --partition=gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --gres=gpu:2                          # ← Number of GPUs
#SBATCH --cpus-per-task=16                     # ← MINIMUM 4 required
#SBATCH --mem=300G
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

module load cuda-12.1.0-gcc-11.2.0-s5o57xp      # CUDA 12.1 — use for all GPU/LLM work
module load anaconda3-2022.05-gcc-11.2.0-od5lltp  # Anaconda — use for conda/pip envs

# ── Run your script ──
cd /home/hrishikesh/SDPO
./run_attn_map.sh