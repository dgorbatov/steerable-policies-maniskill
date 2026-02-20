# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

**Steerable Policies** — VLAs (built on OpenVLA/Prismatic) that accept steering commands at multiple levels of abstraction (task-level, sub-task, motion primitives). Trained on Bridge dataset with diverse language labels. The repo supports training, real-robot inference, and ManiSkill 3 simulation evaluation.

## Install

```bash
conda create -n openvla python=3.10 -y && conda activate openvla
pip install -e .
# Flash Attention only needed for training:
pip install packaging ninja && pip install "flash-attn==2.5.5" --no-build-isolation
```

## Common Commands

**Lint/format:**
```bash
make check       # check only (black + ruff)
make autoformat  # fix in place
```

**ManiSkill 3 sim eval (primary eval path in this repo):**
```bash
./run.sh   # configured for the downloaded steerable-policy checkpoint
```
`run.sh` sources `.env` for `HF_TOKEN`, sets all cache dirs under `.cache/`, and runs inside `maniskill_vla.sif`.

**Manual eval invocation (inside container or with deps installed):**
```bash
PYTHONPATH=/path/to/repo python robot/maniskill/run_maniskill_eval.py \
    --pretrained_checkpoint /path/to/checkpoints/step-XXXXXX.pt \
    --tasks "PutCarrotOnPlateInScene-v1,PutEggplantInBasketScene-v1" \
    --num_trials_per_task 50
```

**Download ManiSkill bridge assets (once):**
```bash
MS_ASSET_DIR=.cache/maniskill python -m mani_skill.utils.download_asset bridge_v2_real2sim
```

**Download model weights:**
```bash
HF_HOME=.cache/huggingface python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Embodied-CoT/steerable-policy-openvla-7b-bridge')
"
```

**Build Apptainer container:**
```bash
mkdir -p /mmfs1/gscratch/weirdlab/dg20/.apptainer_tmp
apptainer build --tmpdir /mmfs1/gscratch/weirdlab/dg20/.apptainer_tmp maniskill_vla.sif apptainer.def
```

**Training** (follows OpenVLA instructions; requires Bridge steering annotations):
- Set `PATH_TO_REASONING_DATA` in `prismatic/vla/datasets/datasets.py`
- Run via `vla-scripts/train.py` (same interface as OpenVLA)

## Architecture

### Model: Prismatic VLA
`prismatic/` is the core model library (forked from OpenVLA). Key classes:
- `prismatic/models/vlas/openvla.py` — `OpenVLA`, implements `predict_action()`
- `prismatic/models/load.py` — `load_vla()`, loads from a local `.pt` file at `<run_dir>/checkpoints/<name>.pt`; expects `config.json` and `dataset_statistics.json` in `<run_dir>/`
- `prismatic/vla/datasets/datasets.py` — `RLDSBatchTransform` (standard BC) and `ReasonerRLDSBatchTransform` (chain-of-thought). The key training change vs OpenVLA: steering commands are looked up from `PATH_TO_REASONING_DATA` JSON files by episode+timestep ID and replace default task-level language labels

### Inference Utilities (`robot/`)
- `robot/openvla_utils.py` — `get_prismatic_vla()` / `get_vla()` (model loading), `get_prismatic_vla_action()` (inference). **Note:** `get_prismatic_vla_action` only uses `obs["full_image"]`; the `state` key is ignored.
- `robot/robot_utils.py` — `get_model()`, `get_action()`, `normalize_gripper_action()`, `get_image_resize_size()`
- `robot/maniskill/run_maniskill_eval.py` — ManiSkill 3 eval loop (primary eval script)
- `robot/maniskill/maniskill_utils.py` — image preprocessing, action conversion, task→language map

### ManiSkill 3 Eval
- **Env IDs**: `PutCarrotOnPlateInScene-v1`, `PutEggplantInBasketScene-v1`, `StackGreenCubeOnYellowCubeBakedTexInScene-v1`, `PutSpoonOnTableClothInScene-v1`
- **Control mode**: `arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos` — takes Euler XYZ delta pose directly (no rotation conversion needed from VLA output)
- **Action conversion**: only gripper normalization `[0,1]→[-1,+1]` via `normalize_gripper_action()`
- **Image**: `obs['sensor_data']['3rd_view_camera']['rgb']`, shape `(1, 480, 640, 3)` — batch dim always present; `get_maniskill_img()` squeezes it and applies JPEG→lanczos resize to match bridge training preprocessing
- **`obs['extra']` is empty** for bridge envs — no `tcp_pose` provided

### Checkpoint Format
The downloaded checkpoint lives at:
```
.cache/huggingface/hub/models--Embodied-CoT--steerable-policy-openvla-7b-bridge/
  snapshots/<hash>/
    config.json
    config.yaml
    dataset_statistics.json
    checkpoints/step-080000-epoch-09-loss=0.0506.pt   ← pass this path to --pretrained_checkpoint
```
`load_vla()` expects the path to the `.pt` file directly (not the directory, not the HF Hub ID).

### Import Path Note
`robot/` has no `__init__.py`, but Python 3.3+ namespace packages allow `from robot.X import Y` when the repo root is on `sys.path`. Each eval script inserts its own repo root at import time. The older files under `robot/simpler/`, `robot/libero/`, `robot/bridge/` have broken `from experiments.robot.*` imports — do not rely on them inside the container.

## Secrets & Caches

All caches are kept inside `.cache/` (gitignored). Token is in `.env` (gitignored):
```
HF_TOKEN="hf_..."
```
`run.sh` sources `.env` automatically. `HF_TOKEN` is needed because the Llama-2-7b base model is gated on HuggingFace.

## Container (`maniskill_vla.sif`)

Built from `apptainer.def`, which layers VLA deps on top of `maniskill_base.sif` (at `/mmfs1/gscratch/weirdlab/dg20/maniskill_base.sif`). The repo is never baked in — always bind-mounted at `/workspace` at runtime. Rebuild required if Python deps change.
