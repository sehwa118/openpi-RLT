"""Modality removal preserves other embeddings and legacy enabled behavior."""

from dataclasses import asdict
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.trlt import build_model
from openpi.trlt.config import MODELS
from openpi.trlt.config import ModelConfig
from openpi.trlt.config import digest
from openpi.trlt.config import model_config_dict
from openpi.trlt.config import root_path
from openpi.trlt.tacdex_data_test import native_batch
from openpi.trlt.tacdex_data_test import native_config


@pytest.mark.parametrize("mode", MODELS[1:])
def test_removed_modality_equals_zero_embeddings_and_keeps_other_parameters(mode):
    enabled = native_config(mode)
    disabled = replace(enabled, use_modality_embedding=False)
    batch = native_batch()
    on, off = build_model(enabled), build_model(disabled)
    original = on.init(jax.random.key(1), batch, train=False)
    standalone = off.init(jax.random.key(1), batch, train=False)
    encoder = standalone["params"]["encoder"]
    assert "modality_embed" not in encoder
    for name in ("camera_embed", "patch_embed", "sensor_embed", "time_in", "time_out", "state_in", "state_out"):
        assert name in encoder
    assert "sensor_query" in standalone["params"]["tactile_decoder"]
    params = jax.tree.map(lambda x: x, original["params"])
    params["encoder"]["modality_embed"] = jnp.zeros_like(params["encoder"]["modality_embed"])
    reference = on.apply({"params": params}, batch, method=on.reconstruct, train=False)
    del params["encoder"]["modality_embed"]
    actual = off.apply({"params": params}, batch, method=off.reconstruct, train=False)
    for a, b in zip(jax.tree.leaves(reference), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(a, b)
    value, grad = jax.value_and_grad(lambda p: off.apply({"params": p}, batch, train=False)[0])(params)
    assert np.isfinite(value)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad))
    assert "modality_embed" not in grad["encoder"]


def test_enabled_modality_serialization_is_backward_compatible():
    cfg = ModelConfig()
    previous = asdict(cfg)
    previous.pop("use_modality_embedding")
    # Emulate the previous serializer's existing opt-in/default omission rules.
    previous.pop("use_sensor_id_embedding")
    previous.pop("tactile_format")
    previous.pop("num_taxels")
    for key in (
        "geometry_mode",
        "geometry_spec",
        "geometry_scale_m",
        "geometry_mlp_hidden",
        "geometry_rope_axis_dim",
        "geometry_rope_base",
    ):
        previous.pop(key)
    assert digest(model_config_dict(cfg)) == digest(previous)
    disabled = replace(cfg, use_modality_embedding=False)
    assert model_config_dict(disabled)["use_modality_embedding"] is False
    assert digest(model_config_dict(cfg)) != digest(model_config_dict(disabled))
    with pytest.raises(ValueError, match="boolean"):
        ModelConfig(use_modality_embedding="false")


@pytest.mark.parametrize("mode", MODELS)
def test_assembly_configs_use_8k_effective_batch64_without_modality(mode):
    config = json.loads(root_path(f"configs/trlt/3dtacdex_assembly/{mode}.json").read_text())
    model = ModelConfig(**config["model"])
    assert model.use_modality_embedding is False
    assert model.use_sensor_id_embedding is True
    rep = config["representation"]
    assert rep["max_steps"] == 8000
    assert rep["batch_size"] == 64
    assert rep["microbatch_size"] == 64
    assert config["tracking"]["log_every_steps"] == 10
    assert rep["max_epochs"] is None
    assert rep["patience"] is None
    assert config["protocol_dir"] == "datasets/3dtacdex_assembly/protocol"
    assert config["probe"]["batch_size"] == 256
    assert config["probe"]["max_epochs"] == 100
    assert config["probe"]["patience"] == 10


def test_test_split_cannot_be_used_as_training_validation():
    from types import SimpleNamespace

    from openpi.trlt.runner import _compatible

    train = SimpleNamespace(provenance={"split": "train"})
    validation = SimpleNamespace(provenance={"split": "validation"})
    test = SimpleNamespace(provenance={"split": "test"})
    _compatible(train, validation)
    with pytest.raises(ValueError, match="Training must use train and validation"):
        _compatible(train, test)
    with pytest.raises(ValueError, match="Training must use train and validation"):
        _compatible(test, validation)
