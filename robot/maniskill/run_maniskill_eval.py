"""
run_maniskill_eval.py

Evaluates a steerable-policy VLA in ManiSkill 3 bridge dataset digital-twin environments.

Control mode : arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos
Obs mode     : rgb+segmentation
Camera       : 3rd_view_camera  (640x480, uint8)
Sim freq     : 500 Hz  |  Control freq : 5 Hz

Usage:
    # Prismatic (steerable policy):
    python robot/maniskill/run_maniskill_eval.py \
        --model_family prismatic \
        --pretrained_checkpoint /path/to/checkpoint \
        --tasks PutCarrotOnPlateInScene-v1 PutEggplantInBasketScene-v1 \
        --num_trials_per_task 50 \
        --center_crop True

    # OpenVLA:
    python robot/maniskill/run_maniskill_eval.py \
        --model_family openvla \
        --pretrained_checkpoint /path/to/checkpoint \
        --num_trials_per_task 50

    Asset download (run once before evaluating):
        python -m mani_skill.utils.download_asset bridge_v2_real2sim
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

# ---------------------------------------------------------------------------
# Ensure repo root is importable as a package root regardless of cwd.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import draccus
import numpy as np
import tqdm
import wandb

# ManiSkill 3 – import triggers env registration
import mani_skill.envs  # noqa: F401
import gymnasium as gym

from robot.libero.libero_utils import save_rollout_video
from robot.maniskill.maniskill_utils import (
    ALL_TASKS,
    TASK_LANGUAGE_INSTRUCTIONS,
    convert_maniskill3_action,
    get_maniskill_img,
)
from robot.openvla_utils import get_processor
from robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    set_seed_everywhere,
)

# Control mode used by ManiSkill 3 bridge envs
_CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
_OBS_MODE = "rgb+segmentation"
# Bridge data is 5 Hz; 150 steps = 30 seconds of interaction
_MAX_STEPS = 150


@dataclass
class GenerateConfig:
    # fmt: off

    ###########################################################################
    # Model parameters
    ###########################################################################
    model_family: str = "openvla"                    # "openvla" or "prismatic"
    hf_token: str = Path(".hf_token")               # Path to HF token file
    pretrained_checkpoint: Union[str, Path] = ""    # Checkpoint path
    load_in_8bit: bool = False                       # (OpenVLA) 8-bit quant
    load_in_4bit: bool = False                       # (OpenVLA) 4-bit quant

    center_crop: bool = True                         # Center-crop augmentation
    obs_history: int = 1                             # Frame history length
    use_wrist_image: bool = False                    # Include wrist camera feed

    ###########################################################################
    # ManiSkill 3 environment parameters
    ###########################################################################
    tasks: List[str] = field(default_factory=lambda: list(ALL_TASKS))
    sim_backend: str = "physx_cpu"                  # "physx_cpu" or "physx_cuda"
    num_envs: int = 1                               # Parallel envs (physx_cuda only)

    num_steps_wait: int = 0                         # No-op steps at episode start
    num_trials_per_task: int = 50                   # Rollouts per task

    ###########################################################################
    # Logging / utilities
    ###########################################################################
    run_id_note: Optional[str] = None               # Extra tag in run ID
    local_log_dir: str = "./experiments/logs"       # Local log directory
    prefix: str = ""

    use_wandb: bool = False                         # Log to Weights & Biases
    wandb_project: str = "prismatic"
    wandb_entity: Optional[str] = None

    seed: int = 7                                   # Global random seed

    # fmt: on


@draccus.wrap()
def eval_maniskill(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must not be empty!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert not cfg.use_wrist_image, "use_wrist_image is not supported for ManiSkill 3 eval."

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------
    set_seed_everywhere(cfg.seed)

    # ------------------------------------------------------------------
    # Action un-normalization key
    # ------------------------------------------------------------------
    if cfg.model_family == "prismatic":
        cfg.unnorm_key = "bridge_dataset"
    else:
        cfg.unnorm_key = "bridge_orig"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model = get_model(cfg)

    if cfg.model_family in ["openvla", "prismatic"]:
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, (
            f"Action un-norm key '{cfg.unnorm_key}' not found in VLA norm_stats!"
        )

    # ------------------------------------------------------------------
    # [OpenVLA] Processor
    # ------------------------------------------------------------------
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    run_id = f"{cfg.prefix}EVAL-maniskill-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    # ------------------------------------------------------------------
    # Image resize size
    # ------------------------------------------------------------------
    resize_size = get_image_resize_size(cfg)

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    total_episodes, total_successes = 0, 0

    for task_name in tqdm.tqdm(cfg.tasks, desc="Tasks"):
        assert task_name in TASK_LANGUAGE_INSTRUCTIONS, (
            f"Unknown task '{task_name}'. Valid tasks: {list(TASK_LANGUAGE_INSTRUCTIONS.keys())}"
        )
        task_description = TASK_LANGUAGE_INSTRUCTIONS[task_name]

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Instruction: \"{task_description}\"")
        print(f"{'='*60}")
        log_file.write(f"\nTask: {task_name}\nInstruction: {task_description}\n")

        # Create ManiSkill 3 environment
        env = gym.make(
            task_name,
            num_envs=cfg.num_envs,
            obs_mode=_OBS_MODE,
            control_mode=_CONTROL_MODE,
            sim_backend=cfg.sim_backend,
        )

        task_episodes, task_successes = 0, 0

        for trial_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"  {task_name[:30]}"):
            seed = 1000 + trial_idx  # deterministic eval seeds

            print(f"\nEpisode {trial_idx + 1} / {cfg.num_trials_per_task}  (seed={seed})")
            log_file.write(f"Episode {trial_idx + 1}: seed={seed}\n")

            obs, _ = env.reset(seed=seed)

            t = 0
            replay_images = []
            success = False

            # Optional wait for objects to settle
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _, _ = env.step(np.zeros(7))

            while t < _MAX_STEPS:
                # ---------- Preprocess image ----------
                img = get_maniskill_img(obs, resize_size)
                replay_images.append(img)

                # Build history buffer (pad with first frame if needed)
                image_history = replay_images[-cfg.obs_history:]
                if len(image_history) < cfg.obs_history:
                    pad = [replay_images[0]] * (cfg.obs_history - len(image_history))
                    image_history = pad + image_history

                # ---------- Build observation dict ----------
                tcp_pose = obs["extra"]["tcp_pose"]
                if hasattr(tcp_pose, "cpu"):
                    tcp_pose = tcp_pose.cpu().numpy()
                elif hasattr(tcp_pose, "numpy"):
                    tcp_pose = tcp_pose.numpy()
                # Squeeze batch dim: (1, 7) -> (7,)
                if hasattr(tcp_pose, "shape") and tcp_pose.ndim == 2:
                    tcp_pose = tcp_pose[0]

                observation = {
                    "full_image": image_history,
                    "state": tcp_pose,
                }

                # ---------- Query model ----------
                action = get_action(cfg, model, observation, task_description, processor=processor)

                # ---------- Convert and step ----------
                obs, reward, terminated, truncated, info = env.step(convert_maniskill3_action(action))

                # Resolve scalar / tensor booleans safely
                _term = terminated
                if hasattr(_term, "item"):
                    _term = _term.item()
                elif hasattr(_term, "__len__"):
                    _term = bool(_term[0])
                else:
                    _term = bool(_term)

                _trunc = truncated
                if hasattr(_trunc, "item"):
                    _trunc = _trunc.item()
                elif hasattr(_trunc, "__len__"):
                    _trunc = bool(_trunc[0])
                else:
                    _trunc = bool(_trunc)

                # Check success from info dict
                _succ = info.get("success", False)
                if hasattr(_succ, "item"):
                    _succ = _succ.item()
                elif hasattr(_succ, "__len__"):
                    _succ = bool(_succ[0])
                else:
                    _succ = bool(_succ)

                if _succ:
                    success = True

                t += 1

                if _term or _trunc:
                    break

            # ------ Episode bookkeeping ------
            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            print(f"  Success: {success}  |  Steps taken: {t}")
            log_file.write(f"  Success: {success}  Steps: {t}\n")
            log_file.write(f"  Running total: {total_successes}/{total_episodes} "
                           f"({total_successes / total_episodes * 100:.1f}%)\n")

            # Save replay video
            save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
            )

            # W&B video logging (first 5 successes and 5 failures)
            if cfg.use_wandb:
                n_fail = task_episodes - task_successes
                if (success and task_successes <= 5) or (not success and n_fail <= 5):
                    group = "success" if success else "failure"
                    idx = task_successes if success else n_fail
                    wandb.log(
                        {
                            f"{task_description}/{group}/{idx}": wandb.Video(
                                np.array(replay_images).transpose(0, 3, 1, 2)
                            )
                        }
                    )

            log_file.flush()

        # ------ Per-task summary ------
        task_sr = float(task_successes) / float(task_episodes)
        total_sr = float(total_successes) / float(total_episodes)
        print(f"\nTask success rate ({task_name}): {task_sr:.3f}")
        print(f"Overall success rate so far:     {total_sr:.3f}")
        log_file.write(f"Task success rate: {task_sr:.3f}\n")
        log_file.write(f"Overall success rate: {total_sr:.3f}\n")

        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": task_sr,
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

        env.close()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    log_file.close()

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    print(f"\nFinal success rate: {total_successes}/{total_episodes} "
          f"({total_successes / total_episodes * 100:.1f}%)")


if __name__ == "__main__":
    eval_maniskill()
