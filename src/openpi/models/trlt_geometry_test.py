"""Small CPU tensor tests for geometry edge selection and both encoder variants."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.trlt import build_model
from openpi.models.trlt import fusion_attention_mask
from openpi.models.trlt_geometry import fusion_geometry
from openpi.models.trlt_geometry import geometry_logits
from openpi.models.trlt_geometry import rope3d
from openpi.models.trlt_test import small_config
from openpi.models.trlt_test import synthetic_batch
from openpi.trlt.config import ModelConfig


def _config(variant="rope3d", arch="trlt_cross_only"):
    return replace(
        small_config(arch), num_sensors=10, geometry_available=True, geometry_mode=variant, geometry_rope_axis_dim=2
    )


def _batch(config):
    batch = synthetic_batch(config, batch_size=1)
    points = jnp.array(np.random.default_rng(3).normal(size=(1, config.history_frames, 10, 3)) * 0.05, jnp.float32)
    return dict(batch, tactile_positions=points, geometry_valid_mask=jnp.ones(points.shape[:-1], bool))


def test_rope_norm_equal_position_inner_product_and_translation():
    rng = np.random.default_rng(0)
    q, k = [jnp.array(rng.normal(size=(1, 5, 2, 16)), jnp.float32) for _ in range(2)]
    p = jnp.array(rng.normal(size=(1, 5, 3)), jnp.float32)

    def r(x, p):
        return rope3d(x, p, axis_dim=4, base=10000.0)

    np.testing.assert_allclose(jnp.linalg.norm(r(q, p), axis=-1), jnp.linalg.norm(q, axis=-1), atol=1e-6)
    np.testing.assert_allclose(jnp.sum(r(q, p) * r(k, p), axis=-1), jnp.sum(q * k, axis=-1), atol=2e-6)

    def score(p):
        return jnp.einsum("bqhd,bkhd->bhqk", r(q, p), r(k, p))

    np.testing.assert_allclose(score(p), score(p + jnp.array([1.0, 2.0, 3.0])), atol=5e-6)
    np.testing.assert_array_equal(r(q, jnp.zeros_like(p)), q)


def test_pair_policy_cross_only_and_unpositioned_tokens():
    cfg = _config()
    batch = _batch(cfg)
    _, _, (positions, hands, valid) = fusion_geometry(
        batch, batch["tactile_valid_mask"], batch["visual_valid_mask"], cfg
    )
    n = positions.shape[1]
    rng = np.random.default_rng(9)
    q, k = [jnp.array(rng.normal(size=(1, n, 2, 8)), jnp.float32) for _ in range(2)]
    base = jnp.einsum("bqhd,bkhd->bhqk", q, k)
    geo = geometry_logits(q, k, positions, hands, valid, axis_dim=2, base=10000.0)
    eligible = (np.asarray(hands)[:, None] == np.asarray(hands)[None, :]) & (np.asarray(hands)[:, None] >= 0)
    eligible &= np.asarray(valid)[0, :, None] & np.asarray(valid)[0, None, :]
    np.testing.assert_array_equal(np.array(geo - base)[0, :, ~eligible], 0)
    # Nv=6: head 0:2, left wrist 2:4, right wrist 4:6; tactile begins at6.
    assert not np.allclose(geo[0, :, 2, 6], base[0, :, 2, 6])
    np.testing.assert_array_equal(geo[0, :, 2, 11], base[0, :, 2, 11])
    allowed = np.asarray(fusion_attention_mask("trlt_cross_only", 0, 6, n - 7))
    assert np.max(np.abs(np.array(geo - base)[0, :, allowed])) > 0.01
    # Missing geometry disables rotation for those pairs, not the observation.
    valid = valid.at[:, 6].set(False)
    missing = geometry_logits(q, k, positions, hands, valid, axis_dim=2, base=10000.0)
    np.testing.assert_array_equal(missing[:, :, :, 6], base[:, :, :, 6])


@pytest.mark.parametrize("variant", ["mlp", "rope3d"])
@pytest.mark.parametrize("arch", ["trlt_self_only", "trlt_cross_only", "trlt_alternating"])
def test_geometry_changes_latent_and_masked_garbage_is_ignored(variant, arch):
    cfg = _config(variant, arch)
    model = build_model(cfg)
    batch = _batch(cfg)
    # Include invalid history and explicitly missing geometry.
    batch["tactile_valid_mask"] = batch["tactile_valid_mask"].at[:, 0, 0].set(False)
    batch["geometry_valid_mask"] = batch["geometry_valid_mask"].at[:, 0, 1].set(False)
    variables = model.init(jax.random.key(0), batch)
    encode = jax.jit(lambda b: model.apply(variables, b, method=model.encode))
    a = encode(batch)
    clean_valid = batch["geometry_valid_mask"] & batch["tactile_valid_mask"]
    garbage = dict(batch, tactile_positions=jnp.where(clean_valid[..., None], batch["tactile_positions"], jnp.nan))
    np.testing.assert_allclose(a, encode(garbage), atol=1e-6)
    moved = dict(batch, tactile_positions=batch["tactile_positions"].at[:, :, 0, :].add(0.03))
    assert np.max(np.abs(np.array(a - encode(moved)))) > 1e-6
    loss, _ = model.apply(variables, batch)
    assert np.isfinite(float(loss))
    grads = jax.grad(lambda p: model.apply({"params": p}, batch)[0])(variables["params"])
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree_util.tree_leaves(grads))


def test_none_has_identical_parameters_and_outputs_with_extra_geometry():
    cfg = replace(small_config(), num_sensors=10)
    batch = _batch(cfg)
    model = build_model(cfg)
    clean = {k: v for k, v in batch.items() if k not in ("tactile_positions", "geometry_valid_mask")}
    a = model.init(jax.random.key(0), batch)
    b = model.init(jax.random.key(0), clean)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(
        model.apply(a, batch, method=model.encode), model.apply(b, clean, method=model.encode)
    )


def test_geometry_config_rejects_ambiguous_or_invalid_settings():
    with pytest.raises(ValueError, match="geometry_available"):
        ModelConfig(geometry_mode="mlp")
    with pytest.raises(ValueError, match="three even"):
        ModelConfig(geometry_available=True, geometry_mode="rope3d", geometry_rope_axis_dim=22)
    with pytest.raises(ValueError, match="canonical"):
        ModelConfig(mode="rlt_original", geometry_available=True, geometry_mode="mlp")
