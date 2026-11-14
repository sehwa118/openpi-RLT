"""Small tensor checks; GPU execution is coordinated by the offline smoke runner."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.rl_token import RLTokenModel
from openpi.models.trlt import Fusion
from openpi.models.trlt import SafeAttention
from openpi.models.trlt import _rlt_config
from openpi.models.trlt import build_model
from openpi.models.trlt import fusion_attention_mask
from openpi.models.trlt import tactile_reconstruction_loss
from openpi.trlt.config import ModelConfig


def small_config(mode="trlt_alternating"):
    return ModelConfig(
        mode=mode,
        input_dim=12,
        embed_dim=16,
        num_heads=2,
        num_stages=4,
        decoder_layers=2,
        tactile_resolution=8,
        history_frames=3,
        num_sensors=2,
        cnn_channels=(4, 8),
    )


def synthetic_batch(config=None, *, batch_size=2, length=None):
    cfg = config or small_config()
    length = length or cfg.history_frames
    rng = np.random.default_rng(0)
    b, ns, r = batch_size, cfg.num_sensors, cfg.tactile_resolution
    return {
        "visual_features": jnp.array(rng.normal(size=(b, 6, cfg.input_dim)), dtype=jnp.float32),
        "visual_valid_mask": jnp.ones((b, 6), bool),
        "tactile": jnp.array(rng.random(size=(b, length, ns, r, r, cfg.tactile_channels)), dtype=jnp.float32),
        "tactile_valid_mask": jnp.ones((b, length, ns), bool),
        "tactile_timestamps": jnp.broadcast_to(jnp.arange(length)[None, :, None] / 30, (b, length, ns)),
        "proprioception": jnp.array(rng.normal(size=(b, cfg.state_dim)), dtype=jnp.float32),
        "query_timestamp": jnp.full((b,), (length - 1) / 30),
        "visual_timestamp": jnp.zeros((b,)),
    }


@pytest.mark.parametrize("mode", ["trlt_self_only", "trlt_cross_only", "trlt_alternating"])
def test_fusion_connectivity(mode):
    nv, nu = 2, 3
    for stage in range(4):
        mask = np.array(fusion_attention_mask(mode, stage, nv, nu))
        assert mask[-1].all()
        assert not mask[:-1, -1].any()
        is_cross = mode == "trlt_cross_only" or (mode == "trlt_alternating" and stage % 2 == 1)
        if is_cross:
            assert not mask[:nv, :nv].any()
            assert not mask[nv:-1, nv:-1].any()
            assert mask[:nv, nv:-1].all()
            assert mask[nv:-1, :nv].all()
        elif mode == "trlt_self_only":
            assert mask[:-1, :-1].all()
        else:
            assert mask[:nv, :nv].all()
            assert mask[nv:-1, nv:-1].all()
            assert not mask[:nv, nv:-1].any()
            assert not mask[nv:-1, :nv].any()


def test_attention_all_masked_is_zero():
    layer = SafeAttention(16, 2)
    x = jnp.ones((2, 3, 16))
    mask = jnp.zeros((2, 3, 3), bool)
    params = layer.init(jax.random.key(0), x, x, mask)
    output = layer.apply(params, x, x, mask)
    np.testing.assert_array_equal(output, np.zeros(output.shape))


@pytest.mark.parametrize("mode", ["trlt_self_only", "trlt_cross_only", "trlt_alternating"])
def test_readout_never_changes_stream_hidden_states(mode):
    cfg = small_config(mode)
    fusion = Fusion(cfg)
    v, u = jnp.ones((1, 3, 16)), jnp.ones((1, 4, 16)) * 2
    vm, um = jnp.ones((1, 3), bool), jnp.ones((1, 4), bool)
    params = fusion.init(jax.random.key(0), v, u, vm, um)
    z1, streams1 = fusion.apply(params, v, u, vm, um)
    z2, streams2 = fusion.apply(params, v, u, vm, um, query_override=jnp.arange(16)[None, None].astype(jnp.float32))
    np.testing.assert_array_equal(streams1, streams2)
    assert not np.allclose(z1, z2)


@pytest.mark.parametrize("mode", ["rlt_original", "trlt_self_only", "trlt_cross_only", "trlt_alternating"])
def test_shapes_and_invalid_padding_invariance(mode):
    cfg = small_config(mode)
    batch = synthetic_batch(cfg)
    batch["visual_valid_mask"] = batch["visual_valid_mask"].at[:, 2].set(False)
    batch["tactile_valid_mask"] = batch["tactile_valid_mask"].at[:, 0].set(False).at[:, :, 1].set(False)
    spatial = jnp.ones(batch["tactile"].shape, bool).at[:, :, :, 0].set(False)
    batch["spatial_valid_mask"] = spatial
    model = build_model(cfg)
    params = model.init(jax.random.key(0), batch, train=False)
    loss1, _ = model.apply(params, batch, train=False)
    output = model.apply(params, batch, method=model.reconstruct)
    assert output["z"].shape == (2, 1, cfg.embed_dim)
    assert output["visual"].shape == batch["visual_features"].shape
    if mode != "rlt_original":
        assert output["tactile"].shape == batch["tactile"].shape
    corrupted = dict(batch)
    corrupted["visual_features"] = jnp.where(batch["visual_valid_mask"][..., None], batch["visual_features"], jnp.nan)
    valid = batch["tactile_valid_mask"][..., None, None, None] & spatial
    corrupted["tactile"] = jnp.where(valid, batch["tactile"], jnp.nan)
    loss2, _ = model.apply(params, corrupted, train=False)
    z2 = model.apply(params, corrupted, method=model.encode)
    np.testing.assert_allclose(loss1, loss2, atol=1e-6)
    np.testing.assert_allclose(output["z"], z2, atol=1e-6)
    assert np.isfinite(np.array(loss2))


def test_variable_history_and_all_invalid():
    cfg = small_config()
    model = build_model(cfg)
    batch = synthetic_batch(cfg)
    params = model.init(jax.random.key(0), batch, train=False)
    batch = synthetic_batch(cfg, length=2)
    batch["tactile_valid_mask"] = jnp.zeros_like(batch["tactile_valid_mask"])
    batch["visual_valid_mask"] = jnp.zeros_like(batch["visual_valid_mask"])
    batch["tactile"] = jnp.full_like(batch["tactile"], jnp.nan)
    loss, info = model.apply(params, batch, train=False)
    assert np.isfinite(np.array(loss))
    assert float(info["tactile_loss"]) == 0
    assert float(info["visual_loss"]) == 0
    assert float(info["tactile_valid_fraction"]) == 0


def test_tactile_pixel_then_frame_denominators_and_zero_contact():
    cfg = dataclasses.replace(small_config(), num_sensors=2, history_frames=1, tactile_resolution=2)
    batch = synthetic_batch(cfg, batch_size=1)
    batch["tactile"] = jnp.zeros_like(batch["tactile"])
    spatial = jnp.ones_like(batch["tactile"], bool).at[:, :, 1, 1].set(False)
    batch["spatial_valid_mask"] = spatial
    prediction = jnp.ones_like(batch["tactile"]).at[:, :, 1].set(2)
    loss, info = tactile_reconstruction_loss(prediction, batch)
    assert float(loss) == 2.5  # (MSE 1 + MSE 4)/2, despite differing pixel counts.
    assert float(info["tactile_count"]) == 2
    assert float(info["tactile_valid_fraction"]) == 1  # Black maps are valid observations.


def test_original_baseline_reuses_identical_decoder_and_loss():
    cfg = small_config("rlt_original")
    batch = synthetic_batch(cfg)
    model = build_model(cfg)
    params = model.init(jax.random.key(0), batch, train=False)
    original = RLTokenModel(_rlt_config(cfg, cfg.decoder_layers))
    original_loss, _ = original.apply(params, batch["visual_features"], batch["visual_valid_mask"], train=False)
    loss, _ = model.apply(params, batch, train=False)
    np.testing.assert_allclose(loss, original_loss, rtol=1e-6)


def test_reconstruction_has_no_action_label_dependency():
    cfg = small_config()
    batch = synthetic_batch(cfg)
    model = build_model(cfg)
    params = model.init(jax.random.key(0), batch, train=False)
    loss1, _ = model.apply(params, batch, train=False)
    batch["action_target"] = jnp.full((2, 6, 58), jnp.nan)
    batch["observation.tactile_force"] = jnp.full((2, 10), jnp.nan)
    loss2, _ = model.apply(params, batch, train=False)
    np.testing.assert_array_equal(loss1, loss2)
