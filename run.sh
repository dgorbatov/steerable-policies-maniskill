export REPO=/mmfs1/gscratch/weirdlab/dg20/steerable-policies-maniskill

# HuggingFace model cache (weights, tokenizers, etc.)
export HF_HOME=$REPO/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub

# ManiSkill asset cache (~/.maniskill by default)
export MS_ASSET_DIR=$REPO/.cache/maniskill

mkdir -p $HF_HOME $MS_ASSET_DIR $TORCH_HOME


MS_ASSET_DIR=$REPO/.cache/maniskill

apptainer run --nv \
    --bind $REPO:/workspace \
    --bind $REPO/.cache/maniskill:/root/.maniskill \
    --bind /path/to/weights:/weights \
    --env HF_HOME=/workspace/.cache/huggingface \
    --env TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub \
    --env TORCH_HOME=/workspace/.cache/torch \
    maniskill_vla.sif \
    python /workspace/robot/maniskill/run_maniskill_eval.py \
        --model_family prismatic \
        --pretrained_checkpoint /weights/my-checkpoint \
        --tasks PutCarrotOnPlateInScene-v1 \
        --num_trials_per_task 1