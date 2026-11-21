"""Assembly protocol, causal native-row windows, and masked 22D regression checks."""

import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.models.trlt import build_model
from openpi.trlt.config import ModelConfig
from openpi.trlt.config import digest
from openpi.trlt.config import model_config_dict
from openpi.trlt.metrics import ActionMetrics
from openpi.trlt.runner import _counts
from openpi.trlt.runner import probe_loss
from openpi.trlt.tacdex_data import CAMERAS
from openpi.trlt.tacdex_data import AssemblyDataset
from openpi.trlt.tacdex_data import grouped_split
from openpi.trlt.tacdex_data import inventory
from openpi.trlt.tacdex_data import parquet_path
from openpi.trlt.tacdex_data import prepare_protocol
from openpi.trlt.tacdex_data import read_manifest
from openpi.trlt.tacdex_data import validate_split
from openpi.trlt.tacdex_data import video_path


@pytest.fixture
def assembly(tmp_path):
    root = tmp_path / "lerobot"
    (root / "meta").mkdir(parents=True)
    segments = []
    for eid in range(11):
        # Segments 9 and 10 share one original trajectory: never split them.
        trajectory = min(eid, 9)
        n = 4
        values = {
            "observation.state": np.full((n, 22), trajectory + 1.0),
            "observation.tactile": np.arange(n * 8 * 15 * 3).reshape(n, 8, 15, 3) + trajectory * 10000.0,
            "observation.tactile_valid_mask": np.ones((n, 8), bool),
            "action": np.full((n, 22), trajectory + 2.0),
            "action_valid_mask": np.ones((n, 22), bool),
            "source.elapsed_seconds": np.array([0.0, 0.199, 0.401, 0.6]),
            "source.frame_index": np.arange(n) + (100 if eid == 10 else 0),
            "source.episode_id": np.array([str(trajectory)] * n),
            "source.group": np.array(["v1"] * n),
            "source.label": np.array(["nouse"] * n),
            "source.instruction": np.array(["Assemble the object with the dexterous hand."] * n),
        }
        values["action"][1, 6:] = 1e15
        values["action_valid_mask"][1, 6:] = False
        path = parquet_path(root, eid)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({k: v.tolist() for k, v in values.items()}), path)
        for camera in CAMERAS:
            path = video_path(root, eid, camera)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        segments.append(
            {"length": n, "source_episode_id": str(trajectory), "source_group": "v1", "source_label": "nouse"}
        )
    identity = {"cohort": "all", "source_revision": "fixture", "source_sha256": "fixture"}
    (root / "meta/assembly_conversion.json").write_text(
        json.dumps(
            {"status": "complete", "identity": identity, "identity_hash": digest(identity), "segments": segments}
        )
    )
    protocol = tmp_path / "protocol"
    prepare_protocol(root, protocol)
    return root, protocol


def test_grouped_protocol_reuse_and_train_statistics(assembly):
    root, protocol = assembly
    source = inventory(root)
    split = read_manifest(protocol / "splits.json")
    assert grouped_split(source)["hash"] == split["hash"]
    assert any(9 in part and 10 in part for part in split["splits"].values())
    norm = read_manifest(protocol / "normalization.json")
    mean = np.mean([min(eid, 9) + 2.0 for eid in split["splits"]["train"]])
    np.testing.assert_allclose(norm["action"]["mean"], mean)
    assert norm["action"]["count"][6] == len(split["splits"]["train"]) * 3
    assert norm["action"]["count"][0] == len(split["splits"]["train"]) * 4
    assert "tactile" not in norm  # New protocols fit only state/action statistics.
    assert prepare_protocol(root, protocol)["hash"] == read_manifest(protocol / "protocol.json")["hash"]
    with pytest.raises(ValueError, match="differs"):
        prepare_protocol(root, protocol, seed=7)
    bad = copy.deepcopy(split)
    owner = next(name for name, part in bad["splits"].items() if 9 in part)
    other = next(name for name in bad["splits"] if name != owner)
    bad["splits"][owner].remove(10)
    bad["splits"][other].append(10)
    with pytest.raises(ValueError, match="crosses"):
        validate_split(source, bad)


def test_native_windows_actual_times_boundaries_and_element_masks(assembly):
    root, protocol = assembly
    ds = AssemblyDataset(root, protocol)
    first = ds[0]
    assert first["tactile"].shape == (3, 8, 15, 1, 3)
    assert not first["tactile_valid_mask"][:2].any()
    assert first["tactile_valid_mask"][-1].all()
    assert first["action_valid_mask"][1, :6].all()
    assert not first["action_valid_mask"][1, 6:].any()
    assert np.all(first["action_target"][1, 6:] == 0)
    last = ds[3]
    assert last["action_valid_mask"][0].all()
    assert not last["action_valid_mask"][1:].any()
    np.testing.assert_allclose(last["tactile_timestamps"][:, 0], [0.199, 0.401, 0.6])
    # Actual row history is causal despite clock jitter around 5 Hz.
    assert np.all(last["tactile_timestamps"] <= last["query_timestamp"])
    ep = ds._episode(ds.queries[0][0])  # noqa: SLF001 -- intentional fault injection
    before = copy.deepcopy(ds[1])
    ep["observation.tactile"][2:] = 1e10
    np.testing.assert_array_equal(before["tactile"], ds[1]["tactile"])
    ep["source.elapsed_seconds"][2:] += 1
    assert not ds[3]["tactile_valid_mask"][:1].any()
    assert not ds[1]["action_valid_mask"][1:].any()


def test_source_mutation_and_protocol_tampering_rejected(assembly):
    root, protocol = assembly
    source = inventory(root)
    p = parquet_path(root, 0)
    p.touch()
    if inventory(root)["hash"] == source["hash"]:
        pytest.skip("filesystem timestamp resolution")
    with pytest.raises(ValueError, match="changed"):
        AssemblyDataset(root, protocol)


def test_22d_metrics_and_gradient_mask():
    pred = jnp.ones((2, 3, 22))
    mask = np.ones(pred.shape, bool)
    mask[0, 1, 6:] = False
    mask[1, 2] = False
    target = jnp.where(mask, 0, jnp.nan)
    value, grad = jax.value_and_grad(probe_loss)(pred, target, jnp.asarray(mask))
    assert float(value) == 1
    assert np.all(np.asarray(grad)[~mask] == 0)
    assert (
        _counts([{"action_target": np.asarray(target[0]), "action_valid_mask": mask[0]}], probe=True)["action"]
        == mask[0].sum()
    )
    metric = ActionMetrics(np.zeros(22), np.arange(1, 23), groups={"arm": list(range(6)), "hand": list(range(6, 22))})
    metric.update(pred, target, mask, [10, 10], ["v1", "v1"])
    result = metric.result()
    assert result["primary_normalized_episode_macro_mse"] == 1
    assert result["valid_elements"] == mask.sum()
    np.testing.assert_allclose(result["joint_mse"], np.arange(1, 23) ** 2)
    assert len(result["episodes"]) == 1


def native_config(mode):
    return ModelConfig(
        mode=mode,
        input_dim=8,
        embed_dim=8,
        num_heads=2,
        num_stages=2,
        rlt_encoder_layers=1,
        decoder_layers=1,
        mlp_ratio=1,
        num_sensors=8,
        tactile_format="taxels",
        num_taxels=15,
        tactile_channels=3,
        tactile_resolution=1,
        history_frames=3,
        state_dim=22,
        num_cameras=2,
    )


def native_batch():
    return {
        "visual_features": jnp.ones((1, 4, 8)),
        "visual_valid_mask": jnp.ones((1, 4), bool),
        "tactile": jnp.ones((1, 3, 8, 15, 1, 3)),
        "tactile_valid_mask": jnp.ones((1, 3, 8), bool),
        "tactile_timestamps": jnp.broadcast_to(jnp.array([0.0, 0.2, 0.4])[None, :, None], (1, 3, 8)),
        "proprioception": jnp.zeros((1, 22)),
        "query_timestamp": jnp.array([0.4]),
        "visual_timestamp": jnp.array([0.0]),
    }


@pytest.mark.parametrize("mode", ["rlt_original", "trlt_self_only", "trlt_cross_only", "trlt_alternating"])
def test_native_four_modes_reconstruction_and_mask_invariance(mode):
    cfg = native_config(mode)
    batch = native_batch()
    model = build_model(cfg)
    params = model.init(jax.random.key(1), batch, train=False)
    result = model.apply(params, batch, method=model.reconstruct)
    assert result["z"].shape == (1, 1, 8)
    if mode != "rlt_original":
        assert result["tactile"].shape == batch["tactile"].shape
        assert "tactile_taxels" in params["params"]["encoder"]
    # Invalid tactile observations cannot affect a latent (including arbitrary NaN padding).
    batch["tactile_valid_mask"] = jnp.zeros((1, 3, 8), bool)
    a = model.apply(params, batch, method=model.encode)
    batch["tactile"] = jnp.full_like(batch["tactile"], jnp.nan)
    b = model.apply(params, batch, method=model.encode)
    np.testing.assert_array_equal(a, b)
    loss, _ = model.apply(params, batch, train=False)
    assert np.isfinite(loss)


def test_legacy_config_identity_fields_unchanged():
    cfg = ModelConfig()
    assert "tactile_format" not in model_config_dict(cfg)
    assert "num_taxels" not in model_config_dict(cfg)
    native = native_config("trlt_alternating")
    assert model_config_dict(native)["tactile_format"] == "taxels"


def test_native_training_export_probe_evaluation_roundtrip(tmp_path):
    from openpi.trlt.config import ProbeConfig
    from openpi.trlt.config import TrainConfig
    from openpi.trlt.runner import evaluate_action_probe
    from openpi.trlt.runner import export_frozen_latents
    from openpi.trlt.runner import train_action_probe
    from openpi.trlt.runner import train_representation

    class NativeDataset:
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
            }

        def __len__(self):
            return 2

        def __getitem__(self, index):
            batch = native_batch()
            row = {k: np.asarray(v[0]) for k, v in batch.items()}
            row.update(
                action_target=np.ones((3, 22), np.float32) * 0.1,
                action_valid_mask=np.ones((3, 22), bool),
                episode_id=np.int32(index),
                anchor_frame_index=np.int32(0),
                query_frame_index=np.int32(index),
                object="synthetic",
            )
            row["action_valid_mask"][2, 6:] = False
            row["action_target"][2, 6:] = 0
            return row

    cfg = native_config("trlt_cross_only")
    train, val = NativeDataset("train"), NativeDataset("validation")
    result = train_representation(
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
    assert result["manifest"]["actual_train_steps"] == 1
    a = export_frozen_latents(train, result["best_checkpoint"], tmp_path / "latents_train", batch_size=2)
    b = export_frozen_latents(val, result["best_checkpoint"], tmp_path / "latents_val", batch_size=2)
    assert a[0]["action_valid_mask"].shape == (3, 22)
    probe = train_action_probe(
        a,
        b,
        ProbeConfig(action_dim=22, horizon=3, hidden_dim=8, depth=1, batch_size=2, max_epochs=1, patience=1),
        tmp_path / "probe",
        max_steps=1,
        num_workers=0,
        progress=False,
        log_dir=tmp_path / "stage_b",
    )
    assert probe["manifest"]["actual_train_steps"] == 1
    report = evaluate_action_probe(
        b, probe["best_checkpoint"], val.normalization, tmp_path / "evaluation", batch_size=2
    )
    assert report["metrics"]["valid_elements"] == 2 * (3 * 22 - 16)
    assert np.isfinite(report["metrics"]["primary_normalized_episode_macro_mse"])
    assert set(report["metrics"]["group_mse"]) == {"arm", "hand"}


def test_raw_clip_input_target_masks_and_provenance(assembly):
    from openpi.trlt.runner import _check_provenance

    root, protocol = assembly
    default = AssemblyDataset(root, protocol)
    explicit_default = AssemblyDataset(root, protocol, tactile_preprocessing={"mode": "raw_clip", "scale": 30})
    assert default.provenance == explicit_default.provenance
    np.testing.assert_array_equal(default[1]["tactile"], explicit_default[1]["tactile"])
    clipped = AssemblyDataset(root, protocol, tactile_preprocessing={"mode": "raw_clip", "scale": 30})
    assert clipped.provenance["tactile_preprocessing"] == {"mode": "raw_clip", "scale": 30.0}
    ep = clipped._episode(clipped.queries[0][0])  # noqa: SLF001
    ep["observation.tactile"][0, 0, 0] = [-60, 3, 60]
    ep["observation.tactile"][0, 1, 0, 0] = np.inf
    ep["observation.tactile_valid_mask"][0, 2] = False
    sample = clipped[0]
    np.testing.assert_allclose(sample["tactile"][-1, 0, 0, 0], [-1, 0.1, 1])
    assert not sample["tactile_valid_mask"][:2].any()
    assert not sample["tactile_valid_mask"][-1, 1:3].any()
    assert np.all(sample["tactile"][-1, 1:3] == 0)
    assert np.max(np.abs(sample["tactile"])) <= 1
    np.testing.assert_array_equal(sample["action_target"], default[0]["action_target"])
    legacy_provenance = {k: v for k, v in default.provenance.items() if k != "tactile_preprocessing"}
    for a, b in [(legacy_provenance, clipped.provenance), (clipped.provenance, legacy_provenance)]:
        with pytest.raises(ValueError, match="tactile_preprocessing"):
            _check_provenance(a, b, "test")
    changed = dict(clipped.provenance, tactile_preprocessing={"mode": "raw_clip", "scale": 50.0})
    with pytest.raises(ValueError, match="tactile_preprocessing"):
        _check_provenance(clipped.provenance, changed, "test")
    # Reconstruction target is the same transformed batch field, not native raw.
    from openpi.models.trlt import tactile_reconstruction_loss

    batch = {k: jnp.asarray(v[None]) for k, v in sample.items() if k in ("tactile", "tactile_valid_mask")}
    loss, _ = tactile_reconstruction_loss(batch["tactile"], batch)
    assert float(loss) == 0


@pytest.mark.parametrize("value", [{"mode": "raw_clip", "scale": x} for x in [0, -1, float("inf"), float("nan"), True]] + [{"mode": "standardize"}, {"mode": "unknown"}, {"mode": "raw_clip", "scale": 30, "extra": 1}])
def test_invalid_tactile_preprocessing_rejected(value):
    from openpi.trlt.tacdex_data import tactile_preprocessing_config

    with pytest.raises(ValueError, match="tactile_preprocessing"):
        tactile_preprocessing_config(value)
