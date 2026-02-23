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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Ensure repo root is importable as a package root regardless of cwd.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import draccus
import imageio
import numpy as np
import tqdm
import wandb

# ManiSkill 3 – import triggers env registration
import mani_skill.envs  # noqa: F401
import gymnasium as gym

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
_MAX_STEPS = 300


@dataclass
class GenerateConfig:
    # fmt: off

    ###########################################################################
    # Model parameters
    ###########################################################################
    model_family: str = "prismatic"                  # "openvla" or "prismatic"
    hf_token: str = "HF_TOKEN"                      # Name of env var holding HF token (or set HF_TOKEN=""  for public models)
    # Path to the .pt checkpoint file: <run_dir>/checkpoints/<name>.pt
    # config.json and dataset_statistics.json must be in <run_dir>/
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False                       # (OpenVLA) 8-bit quant
    load_in_4bit: bool = False                       # (OpenVLA) 4-bit quant

    # Action un-normalization key; "bridge_orig" is correct for steerable-policy-openvla-7b-bridge.
    # Override with --unnorm_key if using a different checkpoint.
    unnorm_key: str = "bridge_orig"

    center_crop: bool = True                         # Center-crop augmentation
    obs_history: int = 1                             # Frame history length
    use_wrist_image: bool = False                    # Include wrist camera feed

    ###########################################################################
    # ManiSkill 3 environment parameters
    ###########################################################################
    # Comma-separated task env IDs, e.g. "PutCarrotOnPlateInScene-v1,PutSpoonOnTableClothInScene-v1"
    # Defaults to all 4 bridge tasks.
    tasks: str = ",".join(ALL_TASKS)
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

    save_video: bool = True                         # Save full-res (640x480) rollout videos
    video_fps: int = 5                              # Match sim control freq (5 Hz)

    interactive: bool = False                       # On failure: show video path, prompt for a new instruction, and retry same seed
    prompt_every_n_steps: int = 0                   # Pause every N steps, save partial video, and prompt for a new instruction (0 = off, non-Gemini only)

    ###########################################################################
    # Gemini high-level policy (Section V-B of the Steerable Policies paper)
    ###########################################################################
    use_gemini: bool = False                        # Enable Gemini as the high-level policy
    # Prompt mode — mirrors the three experimental conditions in the paper:
    #   "full"         → full reasoning + all steering abstractions (Fig. 22)
    #   "no_reasoning" → single-line output, all abstractions       (Fig. 23)
    #   "saycan"       → subtask-level commands only                (Fig. 24)
    gemini_mode: str = "full"
    gemini_model: str = "gemini-2.5-flash"          # Gemini model name
    gemini_api_key_env: str = "GEMINI_API_KEY"      # Name of env var holding Gemini API key
    gemini_steps: int = 20                          # Env steps between Gemini queries (paper: 20)
    gemini_max_history: int = 10                    # Max within-episode history entries

    seed: int = 7                                   # Global random seed

    # fmt: on


@draccus.wrap()
def eval_maniskill(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must not be empty!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert not cfg.use_wrist_image, "use_wrist_image is not supported for ManiSkill 3 eval."
    assert not (cfg.use_gemini and cfg.interactive), (
        "--use_gemini and --interactive are mutually exclusive. "
        "In Gemini mode the policy automatically retries failed seeds without user input."
    )
    assert not (cfg.use_gemini and cfg.prompt_every_n_steps > 0), (
        "--use_gemini and --prompt_every_n_steps are mutually exclusive."
    )

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------
    set_seed_everywhere(cfg.seed)

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
    # Gemini API key validation (fail early if key is missing)
    # ------------------------------------------------------------------
    gemini_api_key = None
    if cfg.use_gemini:
        gemini_api_key = os.environ.get(cfg.gemini_api_key_env)
        if not gemini_api_key:
            raise ValueError(
                f"Gemini API key not found. Set the '{cfg.gemini_api_key_env}' "
                f"environment variable before running with --use_gemini."
            )
        # Validate mode early
        from robot.maniskill.gemini_highlevel import GeminiHighLevelPolicy
        assert cfg.gemini_mode in GeminiHighLevelPolicy.MODES, (
            f"--gemini_mode must be one of {GeminiHighLevelPolicy.MODES}"
        )
        print(f"[Gemini] High-level policy enabled (mode={cfg.gemini_mode}, "
              f"model={cfg.gemini_model}, steps_per_query={cfg.gemini_steps})")

    # ------------------------------------------------------------------
    # Keypoint resolver (for <tag> → [x, y] in manual prompt paths)
    # ------------------------------------------------------------------
    keypoint_resolver = None
    if cfg.interactive or cfg.prompt_every_n_steps > 0:
        _kp_api_key = os.environ.get(cfg.gemini_api_key_env)
        if _kp_api_key:
            from robot.maniskill.gemini_highlevel import KeypointResolver
            keypoint_resolver = KeypointResolver(
                model_name=cfg.gemini_model,
                api_key=_kp_api_key,
                img_width=640,
                img_height=480,
            )
            print(f"[Keypoint] Resolver enabled — <tag> placeholders will be resolved via Gemini")
        else:
            print(f"[Keypoint] No {cfg.gemini_api_key_env} found; <tag> resolution disabled.")

    # ------------------------------------------------------------------
    # Image resize size
    # ------------------------------------------------------------------
    resize_size = get_image_resize_size(cfg)

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    total_episodes, total_successes = 0, 0
    task_list = [t.strip() for t in cfg.tasks.split(",") if t.strip()]

    for task_name in tqdm.tqdm(task_list, desc="Tasks"):
        assert task_name in TASK_LANGUAGE_INSTRUCTIONS, (
            f"Unknown task '{task_name}'. Valid tasks: {list(TASK_LANGUAGE_INSTRUCTIONS.keys())}"
        )
        task_description = TASK_LANGUAGE_INSTRUCTIONS[task_name]

        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Instruction: \"{task_description}\"")
        print(f"{'='*60}")
        log_file.write(f"\nTask: {task_name}\nInstruction: {task_description}\n")

        # Create Gemini high-level policy for this task (if enabled)
        gemini_policy = None
        if cfg.use_gemini:
            from robot.maniskill.gemini_highlevel import GeminiHighLevelPolicy
            gemini_policy = GeminiHighLevelPolicy(
                task_description=task_description,
                mode=cfg.gemini_mode,
                model_name=cfg.gemini_model,
                api_key=gemini_api_key,
                steps_per_query=cfg.gemini_steps,
                max_history=cfg.gemini_max_history,
                img_width=640,
                img_height=480,
            )

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
            current_instruction = task_description  # may be overridden by user on retry
            attempt = 0
            vid_path = None

            # Retry loop: repeats the same seed with a different prompt if interactive + failed
            while True:
                attempt += 1
                print(f"\nEpisode {trial_idx + 1} / {cfg.num_trials_per_task}  "
                      f"(seed={seed}, attempt={attempt}, prompt=\"{current_instruction}\")")
                log_file.write(f"Episode {trial_idx + 1}: seed={seed} attempt={attempt} "
                               f"prompt=\"{current_instruction}\"\n")

                # Reset Gemini episode state (clears within-episode history)
                if gemini_policy is not None:
                    gemini_policy.reset()

                obs, _ = env.reset(seed=seed)

                t = 0
                replay_images = []
                fullres_frames = []
                success = False
                _task_tag = task_name.replace("-", "_").replace(" ", "_")[:40]
                rollout_dir = f"./rollouts/{DATE_TIME}"
                os.makedirs(rollout_dir, exist_ok=True)

                for _ in range(cfg.num_steps_wait):
                    obs, _, _, _, _ = env.step(np.zeros(7))

                while t < _MAX_STEPS:
                    # ---------- Capture full-res frame (needed for Gemini / video / mid-episode prompt) ----------
                    raw_rgb = obs["sensor_data"]["3rd_view_camera"]["rgb"]
                    if hasattr(raw_rgb, "shape") and len(raw_rgb.shape) == 4:
                        raw_rgb = raw_rgb[0]
                    if hasattr(raw_rgb, "cpu"):
                        raw_rgb = raw_rgb.cpu().numpy()
                    elif hasattr(raw_rgb, "numpy"):
                        raw_rgb = raw_rgb.numpy()
                    raw_rgb = np.asarray(raw_rgb, dtype=np.uint8)

                    if cfg.save_video or cfg.prompt_every_n_steps > 0:
                        fullres_frames.append(raw_rgb)

                    # ---------- Mid-episode interactive prompt (non-Gemini) ----------
                    if cfg.prompt_every_n_steps > 0 and t > 0 and t % cfg.prompt_every_n_steps == 0:
                        partial_path = (
                            f"{rollout_dir}/ep{total_episodes + 1:04d}"
                            f"--attempt{attempt}"
                            f"--step{t:03d}"
                            f"--{_task_tag}_partial.mp4"
                        )
                        writer = imageio.get_writer(partial_path, fps=cfg.video_fps)
                        for frame in fullres_frames:
                            writer.append_data(frame)
                        writer.close()
                        print(f"\n{'─'*60}")
                        print(f"  Step {t}/{_MAX_STEPS}  |  Task: \"{task_description}\"")
                        print(f"  Current instruction: \"{current_instruction}\"")
                        print(f"  Partial video saved: {partial_path}")
                        try:
                            new_prompt = input("  New instruction (Enter to keep current): ").strip()
                        except EOFError:
                            new_prompt = ""
                        if new_prompt:
                            if "<" in new_prompt and keypoint_resolver is not None:
                                new_prompt = keypoint_resolver.resolve(new_prompt, raw_rgb)
                                print(f"  Resolved: \"{new_prompt}\"")
                            current_instruction = new_prompt
                            log_file.write(f"  Step {t}: instruction → \"{new_prompt}\"\n")
                            print(f"  Instruction updated: \"{new_prompt}\"")
                        print(f"{'─'*60}")

                    # ---------- Gemini high-level policy: update steering command ----------
                    if gemini_policy is not None:
                        current_instruction = gemini_policy.step(raw_rgb)

                    # ---------- Preprocess image ----------
                    img = get_maniskill_img(obs, resize_size)
                    replay_images.append(img)

                    # Build history buffer (pad with first frame if needed)
                    image_history = replay_images[-cfg.obs_history:]
                    if len(image_history) < cfg.obs_history:
                        pad = [replay_images[0]] * (cfg.obs_history - len(image_history))
                        image_history = pad + image_history

                    # ---------- Build observation dict ----------
                    # Note: get_prismatic_vla_action only uses "full_image"; state is unused.
                    observation = {"full_image": image_history}

                    # ---------- Query model ----------
                    action = get_action(cfg, model, observation, current_instruction, processor=processor)

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

                final_instruction = current_instruction if gemini_policy is None else "(gemini)"
                print(f"  Success: {success}  |  Steps: {t}  |  Prompt: \"{final_instruction}\"")
                log_file.write(f"  Success: {success}  Steps: {t}\n")

                # Save video for this attempt
                if cfg.save_video and fullres_frames:
                    vid_path = (
                        f"{rollout_dir}/ep{total_episodes + 1:04d}"
                        f"--attempt{attempt}"
                        f"--success={success}"
                        f"--{_task_tag}.mp4"
                    )
                    writer = imageio.get_writer(vid_path, fps=cfg.video_fps)
                    for frame in fullres_frames:
                        writer.append_data(frame)
                    writer.close()
                    print(f"  Video: {vid_path}")
                    log_file.write(f"  Video: {vid_path}\n")

                # Save Gemini trace for this attempt
                if gemini_policy is not None:
                    trace_path = (
                        f"{rollout_dir}/ep{total_episodes + 1:04d}"
                        f"--attempt{attempt}"
                        f"--success={success}"
                        f"--{_task_tag}_trace.txt"
                    )
                    with open(trace_path, "w") as tf:
                        tf.write(gemini_policy.format_trace(task_name, seed, attempt, success, t))
                    print(f"  Gemini trace: {trace_path}")
                    log_file.write(f"  Gemini trace: {trace_path}\n")

                # Interactive mode: prompt user for a new instruction, then retry
                if not success and cfg.interactive:
                    print(f"\n{'─'*60}")
                    print(f"  FAILED. Review the video above, then enter a new prompt to")
                    print(f"  retry the same scene (seed={seed}), or press Enter to move on.")
                    print(f"  Current prompt: \"{current_instruction}\"")
                    try:
                        new_prompt = input("  New prompt (Enter to skip): ").strip()
                    except EOFError:
                        new_prompt = ""
                    print(f"{'─'*60}")
                    if new_prompt:
                        if "<" in new_prompt and keypoint_resolver is not None:
                            new_prompt = keypoint_resolver.resolve(new_prompt, raw_rgb)
                            print(f"  Resolved: \"{new_prompt}\"")
                        log_file.write(f"  >> Retrying with prompt: \"{new_prompt}\"\n")
                        current_instruction = new_prompt
                        continue  # retry same seed

                break  # success, or non-interactive, or user skipped

            # ------ Episode bookkeeping (count once per trial, using final attempt result) ------
            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1

            log_file.write(f"  Running total: {total_successes}/{total_episodes} "
                           f"({total_successes / total_episodes * 100:.1f}%)\n")


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
