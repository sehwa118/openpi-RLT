"""Policy transforms for the OMY arm (6 joints) + LEAP hand (16 joints) rig.

The User PC policy runner (SHAPE_UP_Tactile `policy_runner`) sends
`{"state": float32[22], "images": {"cam_wrist": uint8[H, W, 3]}, "prompt": str}` and expects
`actions` of shape (action_horizon, 22) back, as absolute joint targets in the same order and
units (rad) as `state`. Training data comes from the `tactile_rec` LeRobot v2.1 converter.
"""

import dataclasses
from typing import ClassVar

import numpy as np
import torch

import openpi.models.model as _model
import openpi.transforms as transforms

ACTION_DIM = 22

JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "leap_index_mcp_side",
    "leap_index_mcp_flex",
    "leap_index_pip_flex",
    "leap_index_dip_flex",
    "leap_middle_mcp_side",
    "leap_middle_mcp_flex",
    "leap_middle_pip_flex",
    "leap_middle_dip_flex",
    "leap_ring_mcp_side",
    "leap_ring_mcp_flex",
    "leap_ring_pip_flex",
    "leap_ring_dip_flex",
    "leap_thumb_cmc_side",
    "leap_thumb_cmc_flex",
    "leap_thumb_mcp_flex",
    "leap_thumb_ip_flex",
)


def make_manifest(*, cameras: list[str], checkpoint: str, norm_stats_id: str, pred_horizon: int = 50) -> dict:
    """The manifest the policy runner compares key by key at start-up (extra keys are rejected)."""
    return {
        "schema_version": 1,
        "joint_names": list(JOINT_NAMES),
        "fps": 30,
        "pred_horizon": pred_horizon,
        "mount": "wrist_flip",
        "hand_action": "sent",
        "cameras": list(cameras),
        "checkpoint": checkpoint,
        "norm_stats_id": norm_stats_id,
    }


def _parse_image(image) -> np.ndarray:
    """Normalize image layout/type to uint8 HWC."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class OmyLeapInputs(transforms.DataTransformFn):
    """Map rig cameras to the three pi0 image slots.

    - cam_ext   -> base_0_rgb (zero image, masked out, when the rig has no external camera)
    - cam_wrist -> left_wrist_0_rgb (required)
    - right_wrist_0_rgb is always a masked-out zero image.
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    BLANK_SHAPE: ClassVar[tuple[int, int, int]] = (224, 224, 3)

    def __call__(self, data: dict) -> dict:
        in_images = data.get("images")
        if not isinstance(in_images, dict):
            raise ValueError(f"Expected 'images' to be a dict of camera name -> image, got {type(in_images)}")
        if "cam_wrist" not in in_images:
            raise ValueError(f"Missing 'cam_wrist' image; got cameras {sorted(in_images)}")

        wrist = _parse_image(in_images["cam_wrist"])
        blank = np.zeros(self.BLANK_SHAPE, dtype=np.uint8)
        if "cam_ext" in in_images:
            base, base_mask = _parse_image(in_images["cam_ext"]), np.True_
        else:
            base, base_mask = blank, np.False_

        inputs = {
            "state": np.asarray(data["state"]),
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": blank,
            },
            "image_mask": {
                "base_0_rgb": base_mask,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class OmyLeapOutputs(transforms.DataTransformFn):
    """Drop the model's padding dims: (horizon, 32) -> (horizon, 22)."""

    action_dim: int = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
