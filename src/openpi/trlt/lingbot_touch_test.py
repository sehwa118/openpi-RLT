"""Causal frozen tactile contract tests; source test split is never loaded."""

from dataclasses import asdict
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.trlt import build_model
from openpi.models.trlt import pool_lingbot_features
from openpi.trlt.config import ModelConfig
from openpi.trlt.config import ProbeConfig
from openpi.trlt.config import TrainConfig
from openpi.trlt.config import digest
from openpi.trlt.config import model_config_dict
from openpi.trlt.lingbot_touch import FIELDS
from openpi.trlt.lingbot_touch import TactileFeatureCache
from openpi.trlt.lingbot_touch import fill_history
from openpi.trlt.runner import OBSERVATION_FIELDS
from openpi.trlt.runner import SyntheticDataset
from openpi.trlt.runner import _check_provenance
from openpi.trlt.runner import collate
from openpi.trlt.runner import export_frozen_latents
from openpi.trlt.runner import load_checkpoint
from openpi.trlt.runner import parameter_hash
from openpi.trlt.runner import train_action_probe
from openpi.trlt.runner import train_representation


def tiny():
    return ModelConfig(
        mode="trlt_self_only",
        input_dim=4,
        embed_dim=4,
        num_heads=1,
        num_stages=1,
        decoder_layers=1,
        rlt_encoder_layers=1,
        history_frames=2,
        num_sensors=2,
        tactile_resolution=2,
        cnn_channels=(2,),
        tactile_encoder="lingbot_touch",
        tactile_feature_dim=8,
    )


class FeatureFixture(SyntheticDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.provenance["tactile_feature_cache_hash"] = "synthetic-frozen-features"

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        rng = np.random.default_rng(1000 + index)
        sample.update(
            tactile_features=rng.normal(size=(1, self.config.num_sensors, self.config.tactile_feature_dim)).astype(
                np.float32
            ),
            tactile_feature_timestamps=np.full((1, self.config.num_sensors), sample["query_timestamp"], np.float32),
            tactile_feature_valid_mask=np.ones((1, self.config.num_sensors), bool),
        )
        return sample


def test_pool_uses_actual_time_masks_and_all_invalid():
    batch = {
        "tactile_features": jnp.array([[[[1.0], [float("nan")]], [[5.0], [float("nan")]], [[900.0], [float("nan")]]]]),
        "tactile_feature_timestamps": jnp.array([[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]]),
        "tactile_feature_valid_mask": jnp.array([[[True, False], [True, False], [True, False]]]),
        "query_timestamp": jnp.array([1.0]),
    }
    pooled, valid, relative = pool_lingbot_features(batch, 0.5)
    expected = (np.exp(-2) + 5) / (np.exp(-2) + 1)
    np.testing.assert_allclose(pooled[0, 0, 0], expected, rtol=1e-6)
    np.testing.assert_array_equal(valid, [[True, False]])
    assert pooled[0, 1, 0] == 0
    assert relative[0, 1] == 0


def test_causal_query_padding_preserves_real_values_and_absent_sensors():
    values = np.arange(4 * 2).reshape(4, 2, 1).astype(np.float32)
    valid = np.array([[False, False], [True, False], [False, False], [True, False]])
    filled = fill_history(values, valid)
    np.testing.assert_array_equal(filled[:, 0, 0], [2, 2, 2, 6])
    np.testing.assert_array_equal(values[:, 0, 0], [0, 2, 4, 6])


def test_cache_integrity_identity_and_partial_completion(tmp_path):
    metadata = {"feature_shape": [1, 2, 8], "encoder": "frozen-v1"}
    cache = TactileFeatureCache.create(tmp_path, metadata)
    values = dict(
        zip(FIELDS, (np.ones((1, 2, 8), np.float32), np.ones((1, 2), bool), np.zeros((1, 2), np.float32)), strict=True)
    )
    cache.put((1, 3), values)
    for key in FIELDS:
        np.testing.assert_array_equal(cache[(1, 3)][key], values[key])
    with pytest.raises(ValueError, match="metadata mismatch"):
        TactileFeatureCache(tmp_path, {**metadata, "encoder": "changed"})
    path = cache.path((1, 3))
    with np.load(path) as file:
        corrupted = dict(file)
    corrupted["tactile_features"][0, 0, 0] += 1
    with path.open("wb") as stream:
        np.savez(stream, **corrupted)
    with pytest.raises(ValueError, match="content"):
        cache[(1, 3)]
    from types import SimpleNamespace

    dataset = SimpleNamespace(provenance={"split": "train"}, queries=[(1, 3)])
    with pytest.raises(FileNotFoundError, match="completed tactile"):
        cache.require_complete(dataset)
    (tmp_path / "train_complete.json").write_text(
        json.dumps({"cache_hash": cache.cache_hash, "query_keys_hash": digest([[1, 4]])})
    )
    with pytest.raises(ValueError, match="query set"):
        cache.require_complete(dataset)


def test_old_config_identity_and_new_config_guards():
    config = ModelConfig()
    expected = asdict(config)
    for key in (
        "tactile_encoder",
        "tactile_feature_dim",
        "tactile_time_tau",
        "use_sensor_id_embedding",
        "geometry_mode",
        "geometry_spec",
        "geometry_scale_m",
        "geometry_mlp_hidden",
        "geometry_rope_axis_dim",
        "geometry_rope_base",
    ):
        expected.pop(key)
    assert model_config_dict(config) == expected
    with pytest.raises(ValueError, match="requires TRLT"):
        replace(config, mode="rlt_original", tactile_encoder="lingbot_touch")
    with pytest.raises(ValueError, match="temporal decay"):
        replace(config, tactile_time_tau=0)
    with pytest.raises(ValueError, match="tactile_feature_cache_hash"):
        _check_provenance({"tactile_feature_cache_hash": "one"}, {"tactile_feature_cache_hash": "two"}, "fixture")


def test_cached_feature_bundle_is_atomic_in_collation():
    sample = FeatureFixture(tiny(), size=1)[0]
    del sample["tactile_feature_timestamps"]
    with pytest.raises(ValueError, match="supplied together"):
        collate([sample], OBSERVATION_FIELDS)


def test_projection_gradient_and_one_sensor_token_per_slot():
    config = tiny()
    sample = collate([FeatureFixture(config, size=1)[0]], OBSERVATION_FIELDS)
    model = build_model(config)
    params = model.init(jax.random.PRNGKey(1), sample)["params"]
    assert "tactile_cnn" not in params["encoder"]
    assert "lingbot_projection" in params["encoder"]

    def loss(p):
        return model.apply({"params": p}, sample)[0]

    grad = jax.grad(loss)(params)
    assert np.isfinite(np.asarray(grad["encoder"]["lingbot_projection"]["kernel"])).all()
    assert np.linalg.norm(np.asarray(grad["encoder"]["lingbot_projection"]["kernel"])) > 0


def test_tiny_stage_a_export_stage_b(tmp_path):
    config = tiny()
    train = FeatureFixture(config, size=2)
    validation = FeatureFixture(config, split="validation", size=1)
    stage_a = train_representation(
        train,
        validation,
        config,
        TrainConfig(batch_size=2, microbatch_size=2, max_steps=1, max_epochs=None),
        tmp_path / "representation",
        max_steps=1,
        num_workers=0,
        log_dir=tmp_path / "a_logs",
        progress=False,
    )
    params, _ = load_checkpoint(stage_a["best_checkpoint"])
    before = parameter_hash(params)
    train_latents = export_frozen_latents(train, stage_a["best_checkpoint"], tmp_path / "train_latents", batch_size=2)
    validation_latents = export_frozen_latents(
        validation, stage_a["best_checkpoint"], tmp_path / "validation_latents", batch_size=1
    )
    probe = train_action_probe(
        train_latents,
        validation_latents,
        ProbeConfig(hidden_dim=4, depth=1, batch_size=2),
        tmp_path / "probe",
        max_steps=1,
        num_workers=0,
        log_dir=tmp_path / "b_logs",
        progress=False,
    )
    unchanged, _ = load_checkpoint(stage_a["best_checkpoint"])
    assert parameter_hash(unchanged) == before
    assert probe["best_checkpoint"]


def test_native96_sampler_matches_upstream_and_frozen_encoder():
    from openpi.trlt.lingbot_touch import FrozenTactileExtractor
    from openpi.trlt.lingbot_touch import import_upstream
    from openpi.trlt.lingbot_touch import sample_native_frame
    from openpi.trlt.lingbot_touch import sampled_map

    import_upstream()
    from data_interfaces.pretraining_sampling import prepare_pretraining_reference_input
    import torch

    torch.set_num_threads(2)
    rng = np.random.default_rng(84)
    raw = rng.integers(0, 256, size=(1, 2, 5, 240, 240), dtype=np.uint8)
    official = prepare_pretraining_reference_input(
        torch.from_numpy(raw), layout_id="visuotactile_deformation240_canonical_right_v1", num_points=96
    )
    indices, weights = sampled_map()
    ours = np.stack([[sample_native_frame(raw[0, t, f], f, indices, weights) for f in range(5)] for t in range(2)])
    np.testing.assert_allclose(ours, official.signal[0, ..., 2].numpy(), atol=2e-7)
    values = np.tile(ours, (5, 2, 1))
    valid = np.ones((10, 10), bool)
    valid[:8, 0] = False
    valid[:, 9] = False
    stamps = np.broadcast_to(np.arange(10, dtype=np.float32)[:, None] / 30, valid.shape).copy()
    stamps[~valid] = 0
    extractor = FrozenTactileExtractor(device="cpu")
    before = [p.detach().clone() for p in extractor.encoder.parameters()]
    result = extractor.encode_windows([(values, valid, stamps, np.float32(0.3))])[0]
    assert result["tactile_features"].shape == (5, 10, 768)
    np.testing.assert_array_equal(result["tactile_feature_valid_mask"][:, 0], [False] * 4 + [True])
    assert not result["tactile_feature_valid_mask"][:, 9].any()
    assert not result["tactile_features"][:, 9].any()
    assert all(not p.requires_grad for p in extractor.encoder.parameters())
    assert all(torch.equal(p, previous) for p, previous in zip(extractor.encoder.parameters(), before, strict=True))
    with pytest.raises(ValueError, match="future"):
        extractor.encode_windows([(values, valid, stamps + 1, np.float32(0.3))])


def test_parquet_timestamp_source_changes_invalidate_cache(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from openpi.trlt import lingbot_touch
    from openpi.trlt.config import TemporalConfig

    source = tmp_path / "source"
    source.mkdir()
    (source / "map.mp4").write_bytes(b"video")
    (source / "timestamps.parquet").write_bytes(b"timestamps")
    dataset = SimpleNamespace(
        root=tmp_path,
        subset={
            "revision": "revision",
            "hash": "subset",
            "episodes": [{"data_path": "timestamps.parquet", "streams": {"sensor": {"path": "map.mp4"}}}],
        },
        split_manifest={"hash": "split"},
        query_manifest={"hash": "queries"},
        provenance={"normalization_hash": "norm"},
        temporal=TemporalConfig(),
    )
    monkeypatch.setattr(lingbot_touch, "bundle_identity", lambda *args: {"encoder": "fixed"})
    first = lingbot_touch.cache_metadata(dataset, {})
    (source / "timestamps.parquet").write_bytes(b"changed timestamps")
    second = lingbot_touch.cache_metadata(dataset, {})
    assert first["source_inventory_hash"] != second["source_inventory_hash"]
