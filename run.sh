#!/usr/bin/env bash
set -e

export REPO=/mmfs1/gscratch/weirdlab/dg20/steerable-policies-maniskill

# Load secrets from .env
set -a; source "$REPO/.env"; set +a

# HuggingFace model cache (weights, tokenizers, etc.)
export HF_HOME=$REPO/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

# ManiSkill asset cache (~/.maniskill by default)
export MS_ASSET_DIR=$REPO/.cache/maniskill

# PyTorch hub cache
export TORCH_HOME=$REPO/.cache/torch

mkdir -p $HF_HOME $MS_ASSET_DIR $TORCH_HOME

# Local path to the downloaded .pt checkpoint file.
# load_vla() expects: <run_dir>/checkpoints/<name>.pt
# with config.json + dataset_statistics.json sitting in <run_dir>/.
CHECKPOINT="$REPO/.cache/huggingface/hub/models--Embodied-CoT--steerable-policy-openvla-7b-bridge/snapshots/adfc02fe284d0c643a4f1fdb895db907e31a0d61/checkpoints/step-080000-epoch-09-loss=0.0506.pt"

apptainer run --nv \
    --bind $REPO:/workspace \
    --bind $REPO/.cache/maniskill:/root/.maniskill \
    --pwd /workspace \
    --env HF_HOME=/workspace/.cache/huggingface \
    --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub \
    --env TORCH_HOME=/workspace/.cache/torch \
    --env MS_ASSET_DIR=/root/.maniskill \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
    $REPO/maniskill_vla.sif \
    python /workspace/robot/maniskill/run_maniskill_eval.py \
        --pretrained_checkpoint "$CHECKPOINT" \
        --tasks "PutEggplantInBasketScene-v1,StackGreenCubeOnYellowCubeBakedTexInScene-v1,PutSpoonOnTableClothInScene-v1" \
        --prompt_every_n_steps 20