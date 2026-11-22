"""LingBot PaXini signed-signal, pooling, immutable cache and Stage A/B contracts."""

from dataclasses import replace
import json
from pathlib import Path
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.trlt import build_model
from openpi.models.trlt import lingbot_temporal_pool
from openpi.trlt.config import ModelConfig
from openpi.trlt.config import ProbeConfig
from openpi.trlt.config import TrainConfig
from openpi.trlt.config import digest
from openpi.trlt.config import model_config_dict
from openpi.trlt.lingbot_touch import FrozenPaxiniEncoder
from openpi.trlt.lingbot_touch import LingbotCache
from openpi.trlt.lingbot_touch import LingbotDataset
from openpi.trlt.lingbot_touch import extract_cache
from openpi.trlt.lingbot_touch import prepare_windows
from openpi.trlt.lingbot_touch import read_geometry
from openpi.trlt.runner import OBSERVATION_FIELDS
from openpi.trlt.runner import collate
from openpi.trlt.tacdex_data import AssemblyDataset
from openpi.trlt.tacdex_data_test import assembly  # noqa: F401
from openpi.trlt.tacdex_data_test import native_batch
from openpi.trlt.tacdex_data_test import native_config


def sample():
    row = {k: np.asarray(v[0]).copy() for k, v in native_batch().items()}
    row["tactile"] = np.linspace(-1, 1, 3 * 8 * 15 * 3, dtype=np.float32).reshape(3, 8, 15, 1, 3)
    return row


def test_joint_signal_signed_z_pad_membership_and_prefix_counts():
    _, pos, coverage, ids = read_geometry()
    row = sample()
    signal, masks, times, counts = prepare_windows([row], pos, ids)
    assert signal.shape == (1, 4, 5, 30, 3)
    assert masks[0, :4].all()
    assert not masks[0, 4].any()
    assert signal[0, 0, 0, 0, 2] < 0  # negative Z survives unchanged
    np.testing.assert_array_equal(signal[0, :2, 0, :15], np.repeat(row["tactile"][0, 0, :, 0][None], 2, 0))
    np.testing.assert_array_equal(signal[0, :, 0, 15:], row["tactile"][[0, 0, 1, 2], 1, :, 0])
    np.testing.assert_array_equal(counts[0, :, 0], [1, 2])
    np.testing.assert_allclose(times[0, :, 0], [0.0, 0.3])
    row["tactile_valid_mask"][:, 7] = False
    row["tactile"][:, 7] = np.nan
    _, masks, _, counts = prepare_windows([row], pos, ids)
    assert not masks[0, 3, 15:].any()
    assert not counts[0, :, 7].any()
    row["tactile_timestamps"][0, 0] = 1.0
    with pytest.raises(ValueError, match="Future"):
        prepare_windows([row], pos, ids)


def test_boundary_copies_are_not_counted_as_observations():
    _, pos, _, ids = read_geometry()
    row = sample()
    row["tactile_valid_mask"][:2] = False
    row["tactile"][:2] = np.nan
    signal, coverage, times, counts = prepare_windows([row], pos, ids)
    np.testing.assert_array_equal(signal[0, :, 0, :15], np.repeat(row["tactile"][2, 0, :, 0][None], 4, axis=0))
    np.testing.assert_array_equal(counts[0, :, 0], [0, 1])
    assert np.all(times <= row["query_timestamp"])
    assert coverage[0, :4].all()


def test_actual_tiny_signed_z_frozen_single_joint_call():
    encoder = FrozenPaxiniEncoder(device="cpu")
    import torch

    torch.set_num_threads(2)
    before = {k: v.clone() for k, v in encoder.encoder.state_dict().items()}
    result = encoder([sample()])
    assert result["lingbot_features"].shape == (1, 2, 8, 768)
    assert np.isfinite(result["lingbot_features"]).all()
    assert not encoder.encoder.training
    assert not any(p.requires_grad for p in encoder.encoder.parameters())
    for k, v in encoder.encoder.state_dict().items():
        assert torch.equal(v, before[k])


def test_fixed_real_time_weighting_and_all_invalid_nan_safety():
    batch = {
        "lingbot_features": jnp.array([[[[1.0, 3.0]], [[5.0, 7.0]]]]),
        "lingbot_valid_mask": jnp.ones((1, 2, 1), bool),
        "lingbot_timestamps": jnp.array([[[0.0], [0.4]]]),
        "lingbot_observation_counts": jnp.ones((1, 2, 1), jnp.int32),
        "query_timestamp": jnp.array([0.4]),
    }
    pooled, valid, relative = lingbot_temporal_pool(batch, 0.2)
    w = np.exp(-2) / (1 + np.exp(-2))
    np.testing.assert_allclose(pooled, [[[w + 5 * (1 - w), 3 * w + 7 * (1 - w)]]], rtol=1e-6)
    np.testing.assert_allclose(relative, [[-0.4 * w]], rtol=1e-6)
    batch["lingbot_valid_mask"] = jnp.zeros((1, 2, 1), bool)
    batch["lingbot_features"] = jnp.full((1, 2, 1, 2), jnp.nan)
    batch["lingbot_timestamps"] = jnp.full((1, 2, 1), jnp.nan)
    pooled, valid, relative = lingbot_temporal_pool(batch, 0.2)
    assert np.all(pooled == 0)
    assert not valid.any()
    assert np.all(relative == 0)


class FakeEncoder:
    identity: ClassVar[dict] = {"fixture": "frozen"}

    def __call__(self, samples):
        value = np.stack([np.full((2, 8, 768), row["query_timestamp"], np.float32) for row in samples])
        _, pos, _, ids = read_geometry()
        _, _, times, counts = prepare_windows(samples, pos, ids)
        valid = counts > 0
        return {
            "lingbot_features": np.where(valid[..., None], value, 0),
            "lingbot_valid_mask": valid,
            "lingbot_timestamps": times,
            "lingbot_observation_counts": counts,
        }


def test_cache_resume_split_identity_and_corruption(assembly, tmp_path):  # noqa: F811
    data, protocol = assembly
    ds = AssemblyDataset(data, protocol, split="train")
    output = tmp_path / "cache"
    manifest = extract_cache(ds, output, FakeEncoder(), batch_size=2, shard_size=3, max_samples=3, progress=False)
    assert manifest["count"] == 3
    assert not manifest["complete"]
    with pytest.raises(ValueError, match="Incomplete"):
        LingbotDataset(ds, output, identity=FakeEncoder.identity)
    manifest = extract_cache(ds, output, FakeEncoder(), batch_size=2, shard_size=3, progress=False)
    assert manifest["complete"]
    assert manifest["count"] == len(ds)
    wrapped = LingbotDataset(ds, output, identity=FakeEncoder.identity)
    np.testing.assert_array_equal(wrapped[0]["tactile"], ds[0]["tactile"])
    np.testing.assert_array_equal(wrapped[0]["action_target"], ds[0]["action_target"])
    with pytest.raises(ValueError, match="identity"):
        LingbotDataset(ds, output, identity={"different": "encoder"})
    shard = output / "train" / manifest["shards"][0]["file"]
    with shard.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="Corrupted"):
        LingbotCache(output, ds, identity=FakeEncoder.identity)[0]


def cached_batch():
    batch = native_batch()
    batch.update(
        lingbot_features=jnp.arange(2 * 8 * 768, dtype=jnp.float32).reshape(1, 2, 8, 768) / 1000,
        lingbot_valid_mask=jnp.ones((1, 2, 8), bool),
        lingbot_timestamps=jnp.broadcast_to(jnp.array([0.0, 0.3])[None, :, None], (1, 2, 8)),
        lingbot_observation_counts=jnp.broadcast_to(jnp.array([1, 2])[None, :, None], (1, 2, 8)),
    )
    return batch


def test_model_eight_tokens_projection_grad_and_decoder_unchanged():
    cfg = replace(native_config("trlt_self_only"), tactile_encoder="lingbot_touch", use_modality_embedding=False)
    batch = cached_batch()
    model = build_model(cfg)
    params = model.init(jax.random.key(7), batch)["params"]
    assert "lingbot_projection" in params["encoder"]
    assert "tactile_taxels" not in params["encoder"]
    output = model.apply({"params": params}, batch, method=model.reconstruct)
    assert output["z"].shape == (1, 1, 8)
    assert output["tactile"].shape == (1, 3, 8, 15, 1, 3)
    changed = dict(batch, tactile=jnp.full_like(batch["tactile"], 0.75))
    np.testing.assert_array_equal(
        model.apply({"params": params}, batch, method=model.encode),
        model.apply({"params": params}, changed, method=model.encode),
    )
    grads = jax.grad(lambda p: model.apply({"params": p}, batch)[0])(params)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grads))
    assert np.any(np.asarray(grads["encoder"]["lingbot_projection"]["kernel"]) != 0)
    missing = {k: v for k, v in batch.items() if k != "lingbot_timestamps"}
    with pytest.raises(ValueError, match="supplied together"):
        collate([{k: np.asarray(v[0]) for k, v in missing.items()}], OBSERVATION_FIELDS)


def test_default_identity_excludes_lingbot_fields():
    cfg = ModelConfig()
    assert "tactile_encoder" not in model_config_dict(cfg)
    assert "lingbot_feature_dim" not in model_config_dict(cfg)
    assert "lingbot_tau_seconds" not in model_config_dict(cfg)
    assert model_config_dict(replace(cfg, tactile_encoder="lingbot_touch"))["lingbot_tau_seconds"] == 0.2
    with pytest.raises(ValueError, match="LingBot"):
        replace(cfg, tactile_encoder="lingbot_touch", lingbot_tau_seconds=0)


def test_stage_a_export_b_and_single_variant_seal_roundtrip(tmp_path):
    from openpi.trlt.runner import evaluate_action_probe
    from openpi.trlt.runner import export_frozen_latents
    from openpi.trlt.runner import train_action_probe
    from openpi.trlt.runner import train_representation
    from openpi.trlt.tacdex_data import seal
    from openpi.trlt.tacdex_experiment import seal_lingbot_selections
    from openpi.trlt.tacdex_experiment import validate_selection_seal

    class Dataset:
        def __init__(self, split):
            self.normalization = {
                "action": {"mean": [0.0] * 22, "std": [1.0] * 22},
                "state": {"mean": [0.0] * 22, "std": [1.0] * 22},
                "action_groups": {"arm": list(range(6)), "hand": list(range(6, 22))},
            }
            self.normalization["hash"] = digest(self.normalization)
            self.provenance = {
                "dataset_kind": "3dtacdex_assembly",
                "split": split,
                "split_hash": "synthetic",
                "normalization_hash": self.normalization["hash"],
                "query_manifest_hash": "synthetic",
                "lingbot_cache_hash": "synthetic-frozen",
                "lingbot_runtime_hash": "synthetic-runtime",
            }

        def __len__(self):
            return 2

        def __getitem__(self, index):
            row = {k: np.asarray(v[0]).copy() for k, v in cached_batch().items()}
            row.update(
                action_target=np.full((3, 22), 0.1, np.float32),
                action_valid_mask=np.ones((3, 22), bool),
                episode_id=np.int32(index),
                query_frame_index=np.int32(index),
                anchor_frame_index=np.int32(0),
                object="synthetic",
            )
            return row

    cfg = replace(native_config("trlt_self_only"), tactile_encoder="lingbot_touch", use_modality_embedding=False)
    train, val = Dataset("train"), Dataset("validation")
    with jax.disable_jit():
        a = train_representation(
            train,
            val,
            cfg,
            TrainConfig(batch_size=2, microbatch_size=1, max_epochs=1, patience=1),
            tmp_path / "representation",
            max_steps=1,
            num_workers=0,
            progress=False,
            log_dir=tmp_path / "stage_a",
        )
        assert a["manifest"]["actual_train_steps"] == 1
        lt = export_frozen_latents(train, a["best_checkpoint"], tmp_path / "train_latents", batch_size=2)
        lv = export_frozen_latents(val, a["best_checkpoint"], tmp_path / "val_latents", batch_size=2)
        b = train_action_probe(
            lt,
            lv,
            ProbeConfig(action_dim=22, horizon=3, hidden_dim=8, depth=1, batch_size=2, max_epochs=1, patience=1),
            tmp_path / "probe",
            max_steps=1,
            num_workers=0,
            progress=False,
            log_dir=tmp_path / "stage_b",
        )
        assert b["manifest"]["actual_train_steps"] == 1
        report = evaluate_action_probe(lv, b["best_checkpoint"], val.normalization, tmp_path / "val_eval", batch_size=2)
        assert report["metrics"]["valid_elements"] == 2 * 3 * 22
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    (protocol / "protocol.json").write_text(
        json.dumps(seal({k: train.provenance[k] for k in ("split_hash", "normalization_hash", "query_manifest_hash")}))
    )
    path = tmp_path / "selected.json"
    selected = seal_lingbot_selections(a["best_checkpoint"], b["best_checkpoint"], path, protocol_dir=protocol)
    assert set(selected["selections"]) == {"trlt_self_only"}
    validate_selection_seal(path, protocol_dir=protocol, checkpoint=a["best_checkpoint"])
    validate_selection_seal(path, protocol_dir=protocol, checkpoint=b["best_checkpoint"])
    with pytest.raises(ValueError, match="not one"):
        validate_selection_seal(path, protocol_dir=protocol, checkpoint=tmp_path / "wrong")
    with (Path(b["best_checkpoint"]) / "params.msgpack").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        validate_selection_seal(path, protocol_dir=protocol)
