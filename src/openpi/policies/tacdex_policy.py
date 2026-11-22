"""3DTacDex baseline adapter; both cameras are external, despite model slot names."""

import dataclasses

import numpy as np

from openpi.policies.libero_policy import _parse_image
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class TacdexInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (22,):
            raise ValueError(f"Expected 22 source joints, got {state.shape}")
        result = {
            "state": state,
            "image": {"base_0_rgb": _parse_image(data["main"]), "left_wrist_0_rgb": _parse_image(data["secondary"])},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 22:
                raise ValueError(f"Expected 22 action dimensions, got {actions.shape}")
            result["actions"] = actions
            mask = np.asarray(data["action_valid_mask"], dtype=bool)
            result["action_loss_mask"] = np.pad(mask, ((0, 0), (0, 10)), constant_values=False)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class TacdexOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :22]}
