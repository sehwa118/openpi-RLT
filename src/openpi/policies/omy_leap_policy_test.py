import numpy as np
import pytest

from openpi.policies import omy_leap_policy
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
import openpi.transforms as _transforms

# Keys the policy runner's Manifest dataclass requires; anything extra makes it refuse to start.
RUNNER_MANIFEST_KEYS = {
    "schema_version",
    "joint_names",
    "fps",
    "pred_horizon",
    "cameras",
    "mount",
    "hand_action",
    "checkpoint",
    "norm_stats_id",
}


def runner_request(*, with_ext: bool = False) -> dict:
    """The exact request shape the policy runner sends (align_obs.to_request)."""
    rng = np.random.default_rng(0)
    images = {"cam_wrist": rng.integers(0, 256, (424, 240, 3), dtype=np.uint8)}
    if with_ext:
        images["cam_ext"] = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    return {
        "state": rng.uniform(-1, 1, 22).astype(np.float32),
        "images": images,
        "prompt": "Pick up the cup",
    }


def test_inputs_wrist_only_masks_missing_cameras():
    out = omy_leap_policy.OmyLeapInputs()(runner_request())

    assert out["state"].shape == (22,)
    assert out["prompt"] == "Pick up the cup"
    assert "actions" not in out
    assert out["image"]["left_wrist_0_rgb"].shape == (424, 240, 3)
    assert out["image_mask"] == {
        "base_0_rgb": np.False_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    for slot in ("base_0_rgb", "right_wrist_0_rgb"):
        assert out["image"][slot].dtype == np.uint8
        assert not out["image"][slot].any()


def test_inputs_with_external_camera():
    req = runner_request(with_ext=True)
    out = omy_leap_policy.OmyLeapInputs()(req)

    assert out["image_mask"]["base_0_rgb"] == np.True_
    np.testing.assert_array_equal(out["image"]["base_0_rgb"], req["images"]["cam_ext"])


def test_inputs_accept_lerobot_float_chw_and_task():
    chw = np.full((3, 424, 240), 0.5, dtype=np.float32)
    data = {
        "state": np.zeros(22, np.float32),
        "images": {"cam_wrist": chw},
        "actions": np.zeros((50, 22), np.float32),
        "task": "Pick up the cup",
    }
    out = omy_leap_policy.OmyLeapInputs()(data)

    assert out["image"]["left_wrist_0_rgb"].shape == (424, 240, 3)
    assert out["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert out["actions"].shape == (50, 22)
    assert out["prompt"] == "Pick up the cup"


def test_inputs_require_wrist_camera():
    with pytest.raises(ValueError, match="cam_wrist"):
        omy_leap_policy.OmyLeapInputs()({"state": np.zeros(22), "images": {"cam_ext": np.zeros((4, 4, 3))}})


def test_outputs_drop_padding():
    out = omy_leap_policy.OmyLeapOutputs()({"actions": np.ones((50, 32), np.float32)})
    assert out["actions"].shape == (50, 22)


def test_policy_metadata_matches_runner_manifest():
    manifest = _config.get_config("pi05_omy_leap").policy_metadata["manifest"]

    assert set(manifest) == RUNNER_MANIFEST_KEYS
    assert manifest["joint_names"] == list(omy_leap_policy.JOINT_NAMES)
    assert len(manifest["joint_names"]) == 22
    assert manifest["fps"] == 30
    assert manifest["pred_horizon"] == 50
    assert manifest["cameras"] == ["cam_wrist"]
    assert manifest["mount"] == "wrist_flip"
    assert manifest["hand_action"] == "sent"
    assert isinstance(manifest["checkpoint"], str)
    assert isinstance(manifest["norm_stats_id"], str)


def _fake_norm_stats() -> dict[str, _normalize.NormStats]:
    def stats(dim):
        return _normalize.NormStats(mean=np.zeros(dim), std=np.ones(dim), q01=-np.full(dim, 2.0), q99=np.full(dim, 2.0))

    return {"state": stats(22), "actions": stats(22)}


def test_serving_transform_chain_runner_request_to_absolute_actions():
    """Same transform order as policy_config.create_trained_policy, without the model."""
    config = _config.get_config("pi05_omy_leap")
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = _fake_norm_stats()

    input_transform = _transforms.compose(
        [
            _transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    output_transform = _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )

    request = runner_request()
    model_in = input_transform(request)

    assert model_in["state"].shape == (config.model.action_dim,)
    for slot in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        assert model_in["image"][slot].shape == (224, 224, 3)
    assert "tokenized_prompt" in model_in

    # A model output of normalized zero deltas (quantile midpoint 0) must come back as
    # absolute targets equal to the measured state.
    model_actions = np.zeros((config.model.action_horizon, config.model.action_dim), np.float32)
    out = output_transform({"state": model_in["state"], "actions": model_actions})

    assert set(out) == {"actions"}
    assert out["actions"].shape == (50, 22)
    assert out["actions"].dtype.kind == "f"
    assert np.isfinite(out["actions"]).all()
    np.testing.assert_allclose(out["actions"], np.broadcast_to(request["state"], (50, 22)), atol=1e-4)
