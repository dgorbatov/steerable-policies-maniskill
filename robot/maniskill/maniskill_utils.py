"""Utilities specific to ManiSkill 3 bridge dataset evaluation."""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Ensure repo root is on sys.path so robot.* imports work regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot.robot_utils import normalize_gripper_action  # noqa: E402


TASK_LANGUAGE_INSTRUCTIONS = {
    "PutCarrotOnPlateInScene-v1": "put the carrot on the plate",
    "PutEggplantInBasketScene-v1": "put the eggplant in the basket",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1": "stack the green cube on the yellow cube",
    "PutSpoonOnTableClothInScene-v1": "put the spoon on the tablecloth",
}

ALL_TASKS = list(TASK_LANGUAGE_INSTRUCTIONS.keys())


def get_maniskill_img(obs, resize_size) -> np.ndarray:
    """Extract camera image from ManiSkill 3 obs dict and preprocess identically
    to how bridge data was processed at training time (JPEG encode/decode + lanczos resize).

    ManiSkill 3 obs layout (rgb+segmentation mode):
        obs['sensor_data']['3rd_view_camera']['rgb'] -> (B, H, W, 3) uint8 tensor
    """
    assert isinstance(resize_size, int)
    image = obs["sensor_data"]["3rd_view_camera"]["rgb"]

    # Squeeze leading batch dimension present when num_envs >= 1
    if hasattr(image, "shape") and len(image.shape) == 4:
        image = image[0]

    # Convert PyTorch / other tensor types to numpy
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()
    elif hasattr(image, "numpy"):
        image = image.numpy()

    image = np.asarray(image, dtype=np.uint8)

    # Replicate Bridge-dataset preprocessing from simpler_utils.get_simpler_img:
    # JPEG encode/decode followed by two-stage lanczos resize (same as training-time).
    IMAGE_BASE_PREPROCESS_SIZE = 128
    image = tf.image.encode_jpeg(image)
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(
        image, (IMAGE_BASE_PREPROCESS_SIZE, IMAGE_BASE_PREPROCESS_SIZE), method="lanczos3", antialias=True
    )
    image = tf.image.resize(image, (resize_size, resize_size), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image.numpy()


def convert_maniskill3_action(action) -> np.ndarray:
    """Convert 7D VLA action [dx, dy, dz, droll, dpitch, dyaw, gripper] to the format
    expected by the ManiSkill 3 bridge env controller (arm_pd_ee_target_delta_pose_align2).

    No rotation conversion is needed: the controller already consumes Euler XYZ deltas,
    which is exactly what the VLA outputs.  Only the gripper dimension needs mapping:
        [0, 1] (VLA output) -> [-1, +1] (env expects), binarized.
    """
    action = action.copy()
    return normalize_gripper_action(action, binarize=True)
