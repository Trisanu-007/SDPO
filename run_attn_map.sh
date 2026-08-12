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
ATTN_SAVE_DIR="/scratch/tris/sdpo_experiments/attention_maps_verl"
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

MODEL_PATH="Qwen/Qwen2.5-Coder-3B-Instruct"

# Training hyperparameters (small defaults for analysis runs)
TRAIN_BATCH_SIZE=4
ROLLOUT_BATCH_SIZE=4
LR=1e-6
ALPHA=1.0
DONTS_REPROMPT_ON_SELF_SUCCESS=True

# Attention map extraction settings
ATTN_ENABLED=true
ATTN_NUM_LAYERS=4         # capture last 4 attention layers
ATTN_SAVE_EVERY=1         # save at every update_policy call
ATTN_MAX_STEPS=20         # stop capturing after 20 saves

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export USER="${USER:-$(whoami)}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"

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
echo "  Last layers: $ATTN_NUM_LAYERS"
echo "  Max saves  : $ATTN_MAX_STEPS"
echo "================================================================"

# ------------------------------------------------------------------
# Build argument string for verl_training.sh
# ------------------------------------------------------------------

ARGS="\
data.train_batch_size=${TRAIN_BATCH_SIZE} \
trainer.group_name=SDPO-attn-map \
actor_rollout_ref.rollout.n=${ROLLOUT_BATCH_SIZE} \
actor_rollout_ref.model.path=${MODEL_PATH} \
actor_rollout_ref.actor.optim.lr=${LR} \
actor_rollout_ref.actor.ppo_mini_batch_size=1 \
actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
actor_rollout_ref.actor.self_distillation.alpha=${ALPHA} \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.attn_map_config.enabled=${ATTN_ENABLED} \
actor_rollout_ref.actor.attn_map_config.save_dir=${ATTN_SAVE_DIR} \
actor_rollout_ref.actor.attn_map_config.num_layers_from_end=${ATTN_NUM_LAYERS} \
actor_rollout_ref.actor.attn_map_config.save_every_n_steps=${ATTN_SAVE_EVERY} \
actor_rollout_ref.actor.attn_map_config.max_steps_to_save=${ATTN_MAX_STEPS} \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=4 \
actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
trainer.total_epochs=1 \
trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"

# ------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------
bash "${PROJECT_ROOT}/training/verl_training.sh" \
    "${EXP_NAME}" \
    "${CONFIG_NAME}" \
    "${DATA_PATH}" \
    ${ARGS}
