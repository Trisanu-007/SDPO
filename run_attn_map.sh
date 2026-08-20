#!/bin/bash

# =============================================================================
# run_attn_map.sh
#
# Runs SDPO training with attention map extraction enabled.
# Uses:
#   - Model:   Qwen/Qwen2.5-Coder-3B-Instruct
#   - Dataset: datasets/lcb_v6  (LiveCodeBench v6)
#   - Attention maps saved to ATTN_SAVE_DIR (default: ./attention_maps)
#
# Usage:
#   ./run_attn_map.sh [--attn-save-dir <path>] [experiment_name_suffix]
#
# Examples:
#   ./run_attn_map.sh
#   ./run_attn_map.sh --attn-save-dir /scratch/my_maps
#   ./run_attn_map.sh --attn-save-dir /scratch/my_maps my_experiment
#
# The saved .npz files can later be visualised with:
#
#   python - <<'EOF'
#   from verl.utils.attention_map_utils import visualize_attention_maps
#   import glob, os
#   for f in sorted(glob.glob("/path/to/attention_maps/*.npz")):
#       visualize_attention_maps(f, output_dir=os.path.dirname(f))
#   EOF
#
# =============================================================================

set -euo pipefail

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
ATTN_SAVE_DIR="/scratch/hrishikesh/users/tri/sdpo_results/attention_maps_verl"
SUFFIX="attn_map_run"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --attn-save-dir)
            ATTN_SAVE_DIR="$2"
            shift 2
            ;;
        *)
            SUFFIX="$1"
            shift
            ;;
    esac
done

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

CONFIG_NAME="sdpo"
DATA_PATH="datasets/lcb_v6"

MODEL_PATH="Qwen/Qwen3-8B"

# Training hyperparameters (small defaults for analysis runs)
TRAIN_BATCH_SIZE=2
ROLLOUT_BATCH_SIZE=4
LR=1e-6
ALPHA=1.0
DONTS_REPROMPT_ON_SELF_SUCCESS=True

# Attention map extraction settings
ATTN_ENABLED=true
ATTN_TARGET_LAYERS="[35]"  # layer indices to capture (e.g. [28,29,30,31] or [31])
ATTN_SAVE_EVERY=1                   # save at every update_policy call
ATTN_MAX_STEPS=20                   # stop capturing after 20 saves

# Reduce max sequence length for analysis runs:
# output_attentions=True stores (batch,heads,seq,seq) in the autograd graph,
# so memory scales as O(seq^2). 4096 → still OOM with double forward (student+teacher)
# on a single 44 GiB L40S; 2048 is 4x smaller (2048^2 vs 4096^2).
MAX_MODEL_LEN=2048

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${PROJECT_ROOT}"
export PATH="/scratch/hrishikesh/users/tri/conda_envs/sdpo_env/bin:$PATH"
export USER="${USER:-$(whoami)}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"

# HuggingFace cache — use local cache; disable hub verification to avoid
# network hangs when Ray workers spawn without inheriting the shell env.
export HF_HOME="/scratch/hrishikesh/shared_models/huggingface"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Silence Ray FutureWarning about accelerator env var override
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="ATTN-MAP-${MODEL_NAME}-${SUFFIX}"

mkdir -p "$ATTN_SAVE_DIR"

echo "================================================================"
echo "SDPO Attention Map Extraction Run"
echo "================================================================"
echo "  Experiment : $EXP_NAME"
echo "  Model      : $MODEL_PATH"
echo "  Dataset    : $DATA_PATH"
echo "  Save dir   : $ATTN_SAVE_DIR"
echo "  Layers     : $ATTN_TARGET_LAYERS"
echo "  Max saves  : $ATTN_MAX_STEPS"
echo "================================================================"

# ------------------------------------------------------------------
# Build argument string for verl_training.sh
# ------------------------------------------------------------------

# Local path overrides (user.yaml defaults to cluster paths: /users/$USER/SDPO,
# /capstor/scratch/...). We explicitly set every path-sensitive field here so
# the script is self-contained on any local machine.
LOCAL_CKPT_DIR="/scratch/hrishikesh/users/tri/sdpo_results/checkpoints/${EXP_NAME}"
REWARD_FN_PATH="${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py"
TRAIN_PARQUET="${PROJECT_ROOT}/${DATA_PATH}/train.parquet"
VAL_PARQUET="${PROJECT_ROOT}/${DATA_PATH}/test.parquet"

mkdir -p "$LOCAL_CKPT_DIR"

ARGS="\
data.train_batch_size=${TRAIN_BATCH_SIZE} \
data.train_files=[\"${TRAIN_PARQUET}\"] \
data.val_files=[\"${VAL_PARQUET}\"] \
data.apply_chat_template_kwargs={} \
trainer.group_name=SDPO-attn-map \
trainer.logger=[console] \
trainer.test_freq=-1 \
trainer.default_local_dir=${LOCAL_CKPT_DIR} \
actor_rollout_ref.rollout.n=${ROLLOUT_BATCH_SIZE} \
actor_rollout_ref.model.path=${MODEL_PATH} \
actor_rollout_ref.actor.optim.lr=${LR} \
actor_rollout_ref.actor.ppo_mini_batch_size=1 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
actor_rollout_ref.actor.self_distillation.alpha=${ALPHA} \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.attn_map_config.enabled=${ATTN_ENABLED} \
actor_rollout_ref.actor.attn_map_config.save_dir=${ATTN_SAVE_DIR} \
actor_rollout_ref.actor.attn_map_config.target_layers=${ATTN_TARGET_LAYERS} \
actor_rollout_ref.actor.attn_map_config.save_every_n_steps=${ATTN_SAVE_EVERY} \
actor_rollout_ref.actor.attn_map_config.max_steps_to_save=${ATTN_MAX_STEPS} \
custom_reward_function.path=${REWARD_FN_PATH} \
+actor_rollout_ref.model.override_config.attn_implementation=eager \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS_PER_NODE} \
actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
actor_rollout_ref.rollout.enforce_eager=True \
actor_rollout_ref.rollout.val_kwargs.n=4 \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
trainer.total_epochs=1 \
trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
actor_rollout_ref.model.use_remove_padding=true \
max_model_len=${MAX_MODEL_LEN} \
data.max_response_length=1536 \
actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
actor_rollout_ref.model.enable_gradient_checkpointing=true"

# ------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------
bash "${PROJECT_ROOT}/training/verl_training.sh" \
    "${EXP_NAME}" \
    "${CONFIG_NAME}" \
    "${DATA_PATH}" \
    ${ARGS}
