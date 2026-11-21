"""Sensor-ID encoder ablation with the decoder and geometry contracts retained."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.trlt import build_model
from openpi.models.trlt_geometry_test import _batch
from openpi.models.trlt_geometry_test import _config
from openpi.trlt.config import ModelConfig
from openpi.trlt.config import digest
from openpi.trlt.config import model_config_dict


@pytest.mark.parametrize("geometry", ["mlp", "rope3d"])
def test_sensor_id_ablation_equals_zero_embedding_and_keeps_decoder(geometry):
    enabled = _config(geometry, "trlt_alternating")
    disabled = replace(enabled, use_sensor_id_embedding=False)
    batch = _batch(enabled)
    on, off = build_model(enabled), build_model(disabled)
    variables = on.init(jax.random.key(1), batch)
    standalone = off.init(jax.random.key(1), batch)
    assert "sensor_embed" not in standalone["params"]["encoder"]
    assert "sensor_query" in standalone["params"]["tactile_decoder"]
    params = jax.tree.map(lambda a: a, variables["params"])
    params["encoder"]["sensor_embed"] = jnp.zeros_like(params["encoder"]["sensor_embed"])
    zero_result = on.apply({"params": params}, batch, method=on.encode)
    del params["encoder"]["sensor_embed"]
    disabled_result = jax.jit(lambda b: off.apply({"params": params}, b, method=off.encode))(batch)
    np.testing.assert_allclose(zero_result, disabled_result, atol=2e-6, rtol=2e-6)
    # Both encoding candidates still respond to coordinates after ID removal.
    moved = dict(batch, tactile_positions=batch["tactile_positions"].at[:, :, 0, :].add(0.03))
    changed = off.apply({"params": params}, moved, method=off.encode)
    assert np.max(np.abs(np.array(disabled_result - changed))) > 1e-6
    grads = jax.grad(lambda p: off.apply({"params": p}, batch)[0])(params)
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grads))
    assert "sensor_embed" not in grads["encoder"]
    assert "sensor_query" in grads["tactile_decoder"]


@pytest.mark.parametrize("geometry", ["none", "mlp", "rope3d"])
def test_true_is_backward_compatible_config_identity_and_false_is_distinct(geometry):
    from dataclasses import asdict

    cfg = ModelConfig(geometry_available=geometry != "none", geometry_mode=geometry)
    old = asdict(cfg)
    old.pop("use_sensor_id_embedding")
    old.pop("use_modality_embedding")
    old.pop("tactile_format")
    old.pop("num_taxels")
    if geometry == "none":
        for key in (
            "geometry_mode",
            "geometry_spec",
            "geometry_scale_m",
            "geometry_mlp_hidden",
            "geometry_rope_axis_dim",
            "geometry_rope_base",
        ):
            old.pop(key)
    assert digest(model_config_dict(cfg)) == digest(old)
    assert model_config_dict(replace(cfg, use_sensor_id_embedding=False))["use_sensor_id_embedding"] is False
    assert digest(model_config_dict(cfg)) != digest(model_config_dict(replace(cfg, use_sensor_id_embedding=False)))


def test_sensor_flag_requires_boolean():
    with pytest.raises(ValueError, match="boolean"):
        ModelConfig(use_sensor_id_embedding="false")
