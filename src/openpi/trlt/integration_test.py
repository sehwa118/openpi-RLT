"""Independent integration contracts for caches, representation, and frozen probe.

Tests named ``test_host_*`` perform no model computation. Other tests require the
GPU environment selected by the coordinating smoke invocation.
"""

import copy
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models.rl_token import RLTokenDecoder
from openpi.models.trlt import build_model
from openpi.models.trlt_test import small_config
from openpi.models.trlt_test import synthetic_batch
from openpi.trlt.cache import CachedContext
from openpi.trlt.cache import FeatureCache
from openpi.trlt.cache import checkpoint_identity
from openpi.trlt.config import CAMERAS
from openpi.trlt.config import TemporalConfig


def test_host_feature_cache_revision_and_chunk_invalidation(tmp_path):
    metadata = {"revision": "a" * 40, "caption_hash": "caption", "dtype": "float32"}
    cache = FeatureCache.create(tmp_path / "features", metadata)
    features = np.arange(36, dtype=np.float32).reshape(6, 6)
    mask = np.array([True, True, False, True, True, True])
    cache.put(
        (301, 15), features, mask, caption_hash="caption", visual_timestamp=0.49, camera_timestamps=[0.49, 0.5, 0.5]
    )
    np.testing.assert_allclose(cache[301, 15]["visual_timestamp"], 0.49)
    np.testing.assert_allclose(cache[301, 15]["camera_timestamps"], [0.49, 0.5, 0.5])
    np.testing.assert_array_equal(cache[301, 15]["visual_features"], features)
    assert cache.keys() == [(301, 15)]
    with pytest.raises(ValueError, match="Stale"):
        FeatureCache(tmp_path / "features", {**metadata, "revision": "b" * 40})
    with pytest.raises(ValueError, match="nonnegative"):
        cache[(-1, 0)]
    # Rebuild produces a new namespace and removes owned chunks.
    rebuilt = FeatureCache.create(tmp_path / "features", {**metadata, "dtype": "float16"}, force=True)
    assert rebuilt.keys() == []
    assert rebuilt.cache_hash != cache.cache_hash


def test_host_checkpoint_identity_changes_on_replacement(tmp_path):
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "metadata.json").write_text('{"step": 1}')
    first = checkpoint_identity(path)
    (path / "metadata.json").write_text('{"step": 2}')
    assert checkpoint_identity(path)["inventory_hash"] != first["inventory_hash"]


def _context_fixture():
    temporal = TemporalConfig(tactile_history_frames=3)
    context = CachedContext(temporal)
    visual = {"visual_features": np.zeros((6, 12), np.float32), "visual_valid_mask": np.ones(6, bool)}
    context.update_visual(visual, episode_id=10, timestamp=0.0)
    for frame in range(3):
        context.append_tactile(
            np.full((2, 8, 8, 1), frame / 10, np.float32), np.ones(2, bool), episode_id=10, timestamp=frame / 30
        )
    return context


def test_host_context_future_filter_replan_reset_and_stale_lag():
    context = _context_fixture()
    batch = context.encode(lambda value: value, current_state=np.zeros(58), query_time=2 / 30)
    np.testing.assert_allclose(batch["tactile"][0, :, 0, 0, 0, 0], [0, 0.1, 0.2])
    # Changing future-observation values cannot change a past query input.
    a, b = copy.deepcopy(context), copy.deepcopy(context)
    a.history[-1] = (0.1, np.zeros((2, 8, 8, 1), np.float32), np.ones(2, bool))
    b.history[-1] = (0.1, np.full((2, 8, 8, 1), 100, np.float32), np.ones(2, bool))
    batch_a = a.encode(lambda value: value, current_state=np.zeros(58), query_time=1 / 30)
    batch_b = b.encode(lambda value: value, current_state=np.zeros(58), query_time=1 / 30)
    for key in batch_a:
        np.testing.assert_array_equal(batch_a[key], batch_b[key])
    # Replan replaces all visual tokens while preserving within-episode history.
    new_visual = {"visual_features": np.ones((6, 12), np.float32), "visual_valid_mask": np.ones(6, bool)}
    history_size = len(context.history)
    context.update_visual(new_visual, episode_id=10, timestamp=2 / 30)
    batch = context.encode(lambda value: value, current_state=np.zeros(58), query_time=2 / 30)
    assert len(context.history) == history_size
    np.testing.assert_array_equal(batch["visual_features"], np.ones((1, 6, 12)))
    with pytest.raises(ValueError, match="all three"):
        context.update_visual(new_visual, episode_id=10, timestamp=2 / 30, camera_order=CAMERAS[:2])
    stale = context.encode(lambda value: value, current_state=np.zeros(58), query_time=1.0)
    assert not stale["tactile_valid_mask"].any()
    context.reset(11)
    assert context.visual is None
    assert not context.history
    with pytest.raises(ValueError, match="Missing"):
        context.encode(lambda value: value, current_state=np.zeros(58), query_time=0.0)


@pytest.mark.parametrize("mode", ["trlt_self_only", "trlt_cross_only", "trlt_alternating"])
def test_same_visual_decoder_weights_and_mask_produce_same_output(mode):
    from openpi.models.trlt import _rlt_config

    cfg = small_config(mode)
    batch = synthetic_batch(cfg)
    batch["visual_valid_mask"] = batch["visual_valid_mask"].at[:, 1].set(False)
    model = build_model(cfg)
    variables = model.init(jax.random.key(11), batch, train=False)
    output = model.apply(variables, batch, method=model.reconstruct)
    decoder = RLTokenDecoder(_rlt_config(cfg, cfg.decoder_layers))
    direct = decoder.apply(
        {"params": variables["params"]["decoder"]},
        output["z"],
        6,
        target_valid_mask=batch["visual_valid_mask"],
        train=False,
    )
    np.testing.assert_array_equal(output["visual"], direct)


def test_visual_decoder_optional_mask_preserves_tree_and_padding_isolation():
    from openpi.models.trlt import _rlt_config

    cfg = small_config()
    decoder = RLTokenDecoder(_rlt_config(cfg, cfg.decoder_layers))
    z = jax.random.normal(jax.random.key(0), (2, 1, cfg.embed_dim))
    key = jax.random.key(1)
    legacy = decoder.init(key, z, 6, train=False)
    all_valid = jnp.ones((2, 6), bool)
    masked = decoder.init(key, z, 6, train=False, target_valid_mask=all_valid)
    assert jax.tree_util.tree_structure(legacy) == jax.tree_util.tree_structure(masked)
    for a, b in zip(jax.tree_util.tree_leaves(legacy), jax.tree_util.tree_leaves(masked), strict=True):
        np.testing.assert_array_equal(a, b)
    without = decoder.apply(legacy, z, 6, train=False)
    with_all = decoder.apply(legacy, z, 6, train=False, target_valid_mask=all_valid)
    np.testing.assert_allclose(without, with_all, atol=1e-6)
    valid = all_valid.at[:, 1].set(False)
    reference = decoder.apply(legacy, z, 6, train=False, target_valid_mask=valid)
    changed = copy.deepcopy(legacy)
    changed["params"]["q_embed"] = changed["params"]["q_embed"].at[1].set(1e4)
    corrupted = decoder.apply(changed, z, 6, train=False, target_valid_mask=valid)
    np.testing.assert_allclose(reference[:, [0, 2, 3, 4, 5]], corrupted[:, [0, 2, 3, 4, 5]], atol=1e-6)


def test_finite_gradients_updates_and_shared_parameterization():
    cfg = dataclasses.replace(small_config(), num_stages=2)
    batch = synthetic_batch(cfg, batch_size=1)
    model = build_model(cfg)
    variables = model.init(jax.random.key(0), batch, train=False)

    def objective(params):
        return model.apply({"params": params}, batch, train=False)[0]

    loss, grads = jax.jit(jax.value_and_grad(objective))(variables["params"])
    assert np.isfinite(np.asarray(loss))
    for leaf in jax.tree_util.tree_leaves(grads):
        assert np.isfinite(np.asarray(leaf)).all()
    families = {
        "cnn": grads["encoder"]["tactile_cnn"],
        "fusion": grads["encoder"]["fusion"],
        "query": grads["encoder"]["fusion"]["q_embed"],
        "visual_decoder": grads["decoder"],
        "tactile_decoder": grads["tactile_decoder"],
    }
    for name, tree in families.items():
        assert sum(float(jnp.sum(jnp.square(value))) for value in jax.tree_util.tree_leaves(tree)) > 0, name
    optimizer = optax.adamw(1e-3)
    state = optimizer.init(variables["params"])
    update, _ = optimizer.update(grads, state, variables["params"])
    updated = optax.apply_updates(variables["params"], update)
    assert float(objective(updated)) < float(loss)
    shapes = jax.tree_util.tree_map(lambda value: value.shape, variables["params"])
    for mode in ("trlt_self_only", "trlt_cross_only"):
        other = build_model(dataclasses.replace(cfg, mode=mode)).init(jax.random.key(0), batch, train=False)
        assert jax.tree_util.tree_map(lambda value: value.shape, other["params"]) == shapes


class _FixtureDataset:
    def __init__(self, rows, split):
        self.rows = rows
        self.provenance = {
            "revision": "fixture",
            "split_manifest_hash": "fixture-split",
            "normalization_hash": "fixture-norm",
            "feature_cache_hash": "fixture-features",
            "temporal": {"history_frames": 2},
            "split": split,
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _runner_fixture(config, split):
    tensors = synthetic_batch(config, batch_size=2)
    rows = [{key: np.asarray(value[index]) for key, value in tensors.items()} for index in range(2)]
    for index, row in enumerate(rows):
        row.update(
            action_target=np.full((2, 58), 0.1 + index * 0.05, np.float32),
            action_valid_mask=np.array([True, index == 0]),
            episode_id=np.int32(100 + index + (10 if split == "validation" else 0)),
            anchor_frame_index=np.int32(0),
            query_frame_index=np.int32(index),
            object="ball",
        )
    return _FixtureDataset(rows, split)


def test_actual_runner_resume_latents_probe_and_representation_freeze(tmp_path):
    from openpi.trlt.config import ProbeConfig
    from openpi.trlt.config import TrainConfig
    from openpi.trlt.runner import export_frozen_latents
    from openpi.trlt.runner import file_hash
    from openpi.trlt.runner import load_checkpoint
    from openpi.trlt.runner import parameter_hash
    from openpi.trlt.runner import train_action_probe
    from openpi.trlt.runner import train_representation

    cfg = dataclasses.replace(
        small_config(),
        embed_dim=8,
        num_stages=1,
        decoder_layers=1,
        history_frames=2,
        tactile_resolution=4,
        cnn_channels=(2,),
    )
    training = TrainConfig(batch_size=2, microbatch_size=1, max_epochs=4, patience=4)
    train, validation = _runner_fixture(cfg, "train"), _runner_fixture(cfg, "validation")
    kwargs = {"seed": 7, "log_dir": tmp_path / "logs"}
    partial = train_representation(train, validation, cfg, training, tmp_path / "resumed", max_steps=1, **kwargs)
    assert partial["manifest"]["actual_train_steps"] == 1
    resumed = train_representation(
        train, validation, cfg, training, tmp_path / "resumed", max_steps=2, resume=True, **kwargs
    )
    uninterrupted = train_representation(
        train,
        validation,
        cfg,
        training,
        tmp_path / "continuous",
        max_steps=2,
        seed=7,
        log_dir=tmp_path / "continuous-logs",
    )
    p1, m1 = load_checkpoint(resumed["last_checkpoint"])
    p2, m2 = load_checkpoint(uninterrupted["last_checkpoint"])
    assert m1["step"] == m2["step"] == 2
    assert parameter_hash(p1) == parameter_hash(p2)
    checkpoint = resumed["best_checkpoint"]
    rep_file = tmp_path / "resumed" / "best" / "params.msgpack"
    representation_hash = file_hash(rep_file)
    train_latents = export_frozen_latents(train, checkpoint, tmp_path / "latent-train", batch_size=1)
    validation_latents = export_frozen_latents(validation, checkpoint, tmp_path / "latent-validation", batch_size=1)
    replay = export_frozen_latents(train, checkpoint, tmp_path / "latent-replay", batch_size=1)
    np.testing.assert_array_equal(train_latents[0]["z"], replay[0]["z"])
    assert train_latents.provenance["frozen"]
    assert len(train_latents) == len(train)
    probe = train_action_probe(
        train_latents,
        validation_latents,
        ProbeConfig(hidden_dim=8, horizon=2, batch_size=2, max_epochs=2),
        tmp_path / "probe",
        seed=7,
        max_steps=1,
        log_dir=tmp_path / "probe-logs",
    )
    assert probe["manifest"]["actual_train_steps"] == 1
    assert file_hash(rep_file) == representation_hash
    probe_params, probe_metadata = load_checkpoint(probe["last_checkpoint"])
    assert probe_metadata["stage"] == "probe"
    assert "encoder" not in probe_params
    assert "decoder" not in probe_params
    assert set(probe_params) == {"hidden_0", "hidden_1", "output"}
    train.provenance["normalization_hash"] = "changed"
    with pytest.raises(ValueError, match="Stale|provenance|normalization"):
        export_frozen_latents(train, checkpoint, tmp_path / "latent-train", batch_size=1)


def test_original_latent_is_anchor_only_and_probe_uses_current_state():
    from openpi.trlt.config import ProbeConfig
    from openpi.trlt.runner import ActionProbe

    cfg = small_config("rlt_original")
    batch = synthetic_batch(cfg, batch_size=1)
    model = build_model(cfg)
    params = model.init(jax.random.key(0), batch, train=False)
    original = model.apply(params, batch, method=model.encode)
    updated = dict(
        batch,
        proprioception=batch["proprioception"] + 10,
        tactile=batch["tactile"] * 0,
        query_timestamp=batch["query_timestamp"] + 0.1,
    )
    same_anchor = model.apply(params, updated, method=model.encode)
    np.testing.assert_array_equal(original, same_anchor)
    probe = ActionProbe(ProbeConfig(hidden_dim=8, horizon=2))
    age = batch["query_timestamp"] - batch["visual_timestamp"]
    variables = probe.init(jax.random.key(2), original, batch["proprioception"], age)
    action_before = probe.apply(variables, original, batch["proprioception"], age)
    action_after = probe.apply(variables, same_anchor, updated["proprioception"], age + 0.1)
    assert not np.allclose(action_before, action_after)
